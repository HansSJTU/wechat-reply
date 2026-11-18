// SPDX-License-Identifier: GPL-3.0-or-later
// Resident R3 text sender (and image downloader) for WeChat-debug 4.2.0.12 (arm64). Adapted from the route in
// Gusion1017/WeChat-ChatBot-Mac (d823148abfb0524edce256b9b3749d99f4160b6b), hook/script.js and
// hook/versions/4_2_0_12_mac.json; see tools/NATIVE_SEND_NOTICE.md.
//
// Differences from tools/native_send.js (one-shot):
//  - the capture hook stays attached: every text WeChat itself sends refreshes the seed closure;
//  - many submissions per seed, each with its own payload looked up by serializer `this`;
//  - one shared serializer vtable and one destructor hook, both alive until WeChat exits;
//  - download(): fetch a received image's full size, as opening it in the client would.
'use strict';

const EXPECTED_UUID = 'a447d5ec4b4237af90ef7f6848ad780d';
const offsets = {
  holder: 0xa5b58d8, scopeString: 0x997c0ce, stringCtor: 0x5aab7c4,
  scope: 0x5aab178, classVtable: 0x9fdb878, ctor: 0x31cd02c,
  taskVtable: 0x9f33350, closureVtable: 0x9f333a8,
  submit: 0x476096c, opNew: 0x72ad4e4,
  // media download (same upstream table): CdnManager getter, StartDownloadMedia
  cdnGetter: 0x5c1f7c0, startDownload: 0x5c742c0,
};
const SEND_CGI = 0x20a;            // newsendmsg
const SERIALIZER_AT = 0x78;        // embedded serializer object inside the task
const CLOSURE_AT = 0x68;

let base, manager, seed = null, seedAt = 0, seedCount = 0;
let fakeVtable = null, url = null, scopeService = null, cdn = null;
const payloads = new Map();        // serializer `this` (string) -> byte array
const pending = new Map();         // task pointer (string) -> our send id
const keep = [];

function event(type, fields = {}) { send(Object.assign({type, at: Date.now()}, fields)); }
function valid(p, length = 8) {
  if (!p || p.isNull()) return false;
  const r = Process.findRangeByAddress(p);
  return !!r && r.protection.includes('r') && p.add(length).compare(r.base.add(r.size)) <= 0;
}
function at(key) { return base.add(offsets[key]); }
function native(key, result, args) { return new NativeFunction(at(key), result, args); }
function uuidOf(m) {
  let p = m.base.add(32);
  const count = m.base.add(16).readU32();
  for (let i = 0; i < count; i++) {
    const cmd = p.readU32(), size = p.add(4).readU32();
    if (size < 8 || size > 65536) throw new Error('bad load command');
    if (cmd === 0x1b) return Array.from(new Uint8Array(p.add(8).readByteArray(16)))
      .map(x => x.toString(16).padStart(2, '0')).join('');
    p = p.add(size);
  }
  throw new Error('missing module UUID');
}
function cloneClosure(closure) {
  const cloneFn = closure.readPointer().add(0x10).readPointer();
  return new NativeFunction(cloneFn, 'pointer', ['pointer'])(closure);
}

function buildSerializer() {
  const fallback = new NativeCallback(function () { return 1; }, 'int', ['pointer']);
  const size = new NativeCallback(function (self) {
    const b = payloads.get(self.toString());
    return b ? b.length : 0;
  }, 'uint32', ['pointer']);
  const write = new NativeCallback(function (self, dst) {
    const b = payloads.get(self.toString());
    if (!b) return dst;
    dst.writeByteArray(b);
    return dst.add(b.length);
  }, 'pointer', ['pointer', 'pointer']);
  fakeVtable = Memory.alloc(0x80);
  for (let i = 0; i < 16; i++) fakeVtable.add(i * 8).writePointer(fallback);
  fakeVtable.add(0x48).writePointer(size);
  fakeVtable.add(0x58).writePointer(write);
  keep.push(fallback, size, write, fakeVtable);
}

function initialize() {
  const main = Process.enumerateModules()[0];
  if (!main.path.endsWith('/WeChat-debug.app/Contents/MacOS/WeChat'))
    throw new Error('refusing to modify a non-debug WeChat process');
  if (Process.arch !== 'arm64') throw new Error('arm64 required');
  const core = Process.enumerateModules().find(m => m.name === 'wechat.dylib' && m.size > 50 * 1024 * 1024);
  if (!core || uuidOf(core) !== EXPECTED_UUID) throw new Error('unsupported live module UUID');
  base = core.base;
  const scopeString = Memory.alloc(24);
  native('stringCtor', 'pointer', ['pointer', 'pointer'])(scopeString, at('scopeString'));
  scopeService = native('scope', 'pointer', ['pointer'])(scopeString);
  if (!valid(scopeService)) throw new Error('no login service scope');
  const holder = at('holder').readPointer();
  if (!valid(holder)) throw new Error('service holder not ready');
  const getService = holder.readPointer().add(0x18).readPointer();
  const service = new NativeFunction(getService, 'pointer', ['pointer'])(holder);
  if (!valid(service)) throw new Error('service not ready');
  manager = service.sub(0x10);
  if (!manager.readPointer().equals(at('classVtable'))) throw new Error('service vtable mismatch');
  keep.push(scopeString);
  buildSerializer();
  url = Memory.allocUtf8String('/cgi-bin/micromsg-bin/newsendmsg');

  // Seed capture: every newsendmsg task WeChat submits on its own refreshes the seed.
  Interceptor.attach(at('submit'), {
    onEnter(args) {
      try {
        const task = args[1];
        if (pending.has(task.toString())) return;  // ours
        if (!valid(task, 0xe0) || !task.readPointer().equals(at('taskVtable')) || task.add(12).readU32() !== SEND_CGI) return;
        const closure = task.add(CLOSURE_AT).readPointer();
        if (!valid(closure) || !closure.readPointer().equals(at('closureVtable'))) return;
        const fresh = cloneClosure(closure);
        if (!valid(fresh)) return;
        seed = fresh; seedAt = Date.now(); seedCount++;  // old seeds are leaked on purpose (tiny, rare)
        setImmediate(() => event('seed_ready', {seedCount}));
      } catch (e) { setImmediate(() => event('capture_error', {error: String(e)})); }
    }
  });
  // Task destruction (vtable slot 0 = complete destructor): release our payload.
  Interceptor.attach(at('taskVtable').readPointer(), {
    onEnter(args) {
      const key = args[0].toString();
      const id = pending.get(key);
      if (id === undefined) return;
      pending.delete(key);
      payloads.delete(args[0].add(SERIALIZER_AT).toString());
      setImmediate(() => event('task_destroyed', {id}));
    }
  });
  event('initialized', {pid: Process.id, base: base.toString()});
}

// ---- media download ---------------------------------------------------------------------------
// WeChat only fetches a received image's full size when it is opened. download() starts the same
// CDN request (StartDownloadMedia on the mars::cdn::CdnManager singleton); WeChat writes the image,
// decrypted, to `filePath`. Request layout from upstream 4_2_0_12 (its non-zero words; libc++ long
// strings at 0x40 file id, 0x58 cdn url, 0x70 aes key, 0x88 file path; file type at 0xa0).
const DL_SIZE = 0x418;
const DL_TEMPLATE = [
  [0xa0, '03000000ffffffff'], [0xa8, '0000000001aaaaaa'], [0xd8, '0100000000000000'],
  [0x128, '020000000a000000'], [0x148, '1128280000000000'], [0x170, '0000020000aaaaaa'],
  [0x178, '0000000001000000'], [0x180, '000000000a000000'], [0x1a0, '1e000000aaaaaaaa'],
  [0x1b0, '00aaaaaa00000000'], [0x1d0, '00221afe0a000000'], [0x2b8, 'ffffffffffffffff'],
  [0x2c0, '004f56fc0a000000'], [0x318, '010000000a000000'], [0x338, '000000000a000000'],
  [0x378, '010000000a000000'], [0x398, '030000000a000000'], [0x3e0, '0000803f00000000'],
];

function hexBytes(h) { const a = []; for (let i = 0; i < h.length; i += 2) a.push(parseInt(h.substr(i, 2), 16)); return a; }
function longString(req, off, value, hold) {
  const data = Memory.allocUtf8String(value);
  hold.push(data);
  req.add(off).writePointer(data);
  req.add(off + 8).writeU64(value.length);
  req.add(off + 0x10).writeU64(uint64('0x8000000000000000').add(value.length + 1));
}
function cdnManager() {  // resolved on first use: by then WeChat is logged in and delivering messages
  if (cdn) return cdn;
  const ctx = native('cdnGetter', 'pointer', ['pointer'])(scopeService);
  if (!valid(ctx, 0x48)) throw new Error('CdnManager context not readable');
  const m = ctx.add(0x40).readPointer();
  if (!valid(m)) throw new Error('CdnManager not readable');
  return (cdn = m);
}

rpc.exports = {
  initialize,
  // fileType: 1 HD/original, 2 normal (mid), 3 thumbnail. Returns at once; the file appears when done.
  download(receiver, cdnUrl, aesKey, fileType, filePath) {
    if (!base) throw new Error('not initialized');
    if (![1, 2, 3].includes(fileType) || !/^[0-9a-f]{32}$/.test(aesKey) || !/^[0-9a-f]{20,400}$/.test(cdnUrl) ||
        typeof filePath !== 'string' || !filePath.startsWith('/') || filePath.length > 250)
      throw new Error('invalid download request');
    const req = Memory.alloc(DL_SIZE);
    for (const [off, h] of DL_TEMPLATE) req.add(off).writeByteArray(hexBytes(h));
    const hold = [req];
    const fileId = `${receiver}_${Math.floor(Date.now() / 1000)}_${Math.floor(Math.random() * 1001)}_1`;
    longString(req, 0x40, fileId, hold);
    longString(req, 0x58, cdnUrl, hold);
    longString(req, 0x70, aesKey, hold);
    longString(req, 0x88, filePath, hold);
    req.add(0xa0).writeU32(fileType);
    const result = native('startDownload', 'int64', ['pointer', 'pointer'])(cdnManager(), req);
    keep.push(hold);  // never freed: WeChat may still refer to the request (about 1.5 KB per image)
    return {fileId, result: Number(result)};
  },
  status() {
    return {pid: Process.id, initialized: !!manager, seedReady: !!seed, seedAt, seedCount, inFlight: pending.size};
  },
  submit(id, bytes) {
    if (!manager) throw new Error('not initialized');
    if (!seed) throw new Error('no send context yet');
    if (!Array.isArray(bytes) || !bytes.length || bytes.length > 32768) throw new Error('invalid protobuf size');
    const closure = cloneClosure(seed);
    if (!valid(closure)) throw new Error('closure clone failed');
    const desc = Memory.alloc(0x38);
    desc.writeU32(SEND_CGI); desc.add(0xc).writeU32(1); desc.add(0x10).writeU32(SEND_CGI);
    native('stringCtor', 'pointer', ['pointer', 'pointer'])(desc.add(0x18), url);
    desc.add(0x30).writeU8(1);
    const task = native('opNew', 'pointer', ['size_t'])(0xe0);
    native('ctor', 'void', ['pointer', 'pointer'])(task, desc);
    if (!task.readPointer().equals(at('taskVtable'))) throw new Error('constructed task vtable mismatch');
    payloads.set(task.add(SERIALIZER_AT).toString(), bytes);
    pending.set(task.toString(), id);
    task.add(SERIALIZER_AT).writePointer(fakeVtable);
    task.add(CLOSURE_AT).writePointer(closure);
    keep.push(desc);
    const result = native('submit', 'int', ['pointer', 'pointer'])(manager, task);
    const taskId = valid(task, 12) ? task.add(8).readU32() : null;
    return {id, result, taskId, seedAgeMs: Date.now() - seedAt};
  },
};
