#!/bin/bash
# Make the debuggable WeChat copy that key capture (lldb) and the sender (Frida) attach to:
# an APFS clone of /Applications/WeChat.app, ad-hoc re-signed with get-task-allow and without
# hardened runtime. Run again after every WeChat update.
#   scripts/make_debug_wechat.sh [dest]    default: ~/Applications/WeChat-debug.app
set -euo pipefail
SRC=/Applications/WeChat.app
DEST="${1:-$HOME/Applications/WeChat-debug.app}"

[ -d "$SRC" ] || { echo "no $SRC; install WeChat first" >&2; exit 1; }
case "$DEST" in *.app) ;; *) echo "dest must end in .app" >&2; exit 1 ;; esac
[ "$DEST" != "$SRC" ] || { echo "dest must not be $SRC" >&2; exit 1; }
if pgrep -xq WeChat; then echo "quit WeChat first" >&2; exit 1; fi

ENT="$(mktemp -t wechat-ent)"
trap 'rm -f "$ENT"' EXIT
# Original entitlements, minus the App Store identity an ad-hoc signature can't carry,
# plus get-task-allow so a debugger may attach.
codesign -d --entitlements - --xml "$SRC" > "$ENT" 2>/dev/null
/usr/libexec/PlistBuddy -c 'Delete :com.apple.application-identifier' "$ENT" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Add :com.apple.security.get-task-allow bool true' "$ENT"

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -cR "$SRC" "$DEST"   # APFS clone: instant, no extra disk space
codesign --force --sign - --entitlements "$ENT" "$DEST"

echo "made $DEST (WeChat $(defaults read "$DEST/Contents/Info.plist" CFBundleShortVersionString)," \
     "build $(defaults read "$DEST/Contents/Info.plist" CFBundleVersion))"
