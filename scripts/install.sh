#!/bin/bash
# Build a launcher app and (re)load its LaunchAgent.
#   scripts/install.sh                  install / restart the WeChat monitor (the bot)
#   scripts/install.sh uninstall        stop and remove it
#   scripts/install.sh label            install / restart the labeling page (http://127.0.0.1:8765)
#   scripts/install.sh label uninstall  stop and remove it
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
DOMAIN="gui/$(id -u)"
APPS="$HOME/Library/Application Support/wechat-monitor"

if [ "${1:-}" = label ]; then
  shift
  LABEL=com.local.wechat-label
  NAME="WeChat Label"
  EXE=wechat-label
  SRC="$PROJECT/scripts/label_launcher.c"
  LOG=label
  # needs only the Documents folder (the project), not Full Disk Access
  GRANT="allow its 'access files in your Documents folder' prompt"
else
  LABEL=com.local.wechat-monitor
  NAME="WeChat Monitor"
  EXE=wechat-monitor
  SRC="$PROJECT/scripts/launcher.c"
  LOG=monitor
  GRANT="grant it Full Disk Access, and allow its 'access data from other apps' prompt"
fi
APP="$APPS/$NAME.app"
BIN="$APP/Contents/MacOS/$EXE"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [ "${1:-}" = uninstall ]; then
  rm -rf "$PLIST" "$APP"
  echo "uninstalled $LABEL"
  exit 0
fi

mkdir -p "$APP/Contents/MacOS" "$PROJECT/data"
chmod 700 "$PROJECT/data"

# A real .app bundle: macOS only shows the "access data from other apps" prompt
# for app bundles. Rebuild only when the source changed, since privacy grants
# are tied to the code signature.
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$LABEL</string>
  <key>CFBundleName</key><string>$NAME</string>
  <key>CFBundleDisplayName</key><string>$NAME</string>
  <key>CFBundleExecutable</key><string>$EXE</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>LSBackgroundOnly</key><true/>
</dict>
</plist>
EOF
  clang -O2 -o "$BIN" "$SRC" \
    -DPROJECT_DIR="\"$PROJECT\"" -DUV_PATH="\"$UV\""
  codesign --force --sign - "$APP"
  echo "built $APP"
  echo "  -> $GRANT"
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$BIN</string></array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(dirname "$UV"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$PROJECT/data/$LOG.out</string>
  <key>StandardErrorPath</key><string>$PROJECT/data/$LOG.log</string>
</dict>
</plist>
EOF
# bootout returns before the old job is gone; bootstrap fails (error 5) until it is.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null && break
  sleep 1
done
launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || { echo "failed to load $LABEL" >&2; exit 1; }
echo "loaded $LABEL; log: $PROJECT/data/$LOG.log"
