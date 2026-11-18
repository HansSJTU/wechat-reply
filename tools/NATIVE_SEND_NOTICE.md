# Native text sender provenance

`wxbot/sender/agent.js` and `wxbot/sender/native.py` are licensed under GPL-3.0-or-later.

The R3 task construction, service lookup, closure capture/clone, and serializer
layout are adapted from the GPL-3.0 project
[Gusion1017/WeChat-ChatBot-Mac](https://github.com/Gusion1017/WeChat-ChatBot-Mac),
commit `d823148abfb0524edce256b9b3749d99f4160b6b`, `hook/script.js`.
That project derives its original WeChat bridge from
[yincongcyincong/wechat_chatter](https://github.com/yincongcyincong/wechat_chatter).

Local changes: a resident sender that refreshes its seed from every manual send
and submits many tasks per seed, exact debug-process and live UUID guards, no
retries, destructor observation, Python
protobuf encoding, bounded context wait, private result JSON, and safe callback
lifetime handling. No upstream application UI, AI service, installer, server,
or automatic reply workflow is included.

The image download (`download()` in `agent.js`: the StartDownloadMedia request layout and the
CdnManager lookup) is adapted from the same commit's `hook/script.js` (`triggerDownload`,
`resolveCdnManager`); unlike upstream it lets WeChat write the decrypted file instead of
capturing CDN chunks.

The address table comes from `hook/versions/4_2_0_12_mac.json` in the same
upstream commit. Preserve this notice and `LICENSE.native-send` when
redistributing this implementation.
