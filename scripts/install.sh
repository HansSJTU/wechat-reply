#!/bin/bash
# Build the launcher app and (re)load the LaunchAgent.
#   scripts/install.sh            install / restart
#   scripts/install.sh uninstall  stop and remove
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.local.wechat-monitor
APP="$HOME/Library/Application Support/wechat-monitor/WeChat Monitor.app"
BIN="$APP/Contents/MacOS/wechat-monitor"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [ "${1:-}" = uninstall ]; then
  rm -rf "$PLIST" "$APP"
  echo "uninstalled"
  exit 0
fi

mkdir -p "$APP/Contents/MacOS" "$PROJECT/data"
chmod 700 "$PROJECT/data"

# A real .app bundle: macOS only shows the "access data from other apps" prompt
# for app bundles. Rebuild only when the source changed, since privacy grants
# are tied to the code signature.
if [ ! -x "$BIN" ] || [ "$PROJECT/scripts/launcher.c" -nt "$BIN" ]; then
  cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$LABEL</string>
  <key>CFBundleName</key><string>WeChat Monitor</string>
  <key>CFBundleDisplayName</key><string>WeChat Monitor</string>
  <key>CFBundleExecutable</key><string>wechat-monitor</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>LSBackgroundOnly</key><true/>
</dict>
</plist>
EOF
  clang -O2 -o "$BIN" "$PROJECT/scripts/launcher.c" \
    -DPROJECT_DIR="\"$PROJECT\"" -DUV_PATH="\"$UV\""
  codesign --force --sign - "$APP"
  echo "built $APP"
  echo "  -> grant it Full Disk Access, and allow its 'access data from other apps' prompt"
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
  <key>StandardOutPath</key><string>$PROJECT/data/monitor.out</string>
  <key>StandardErrorPath</key><string>$PROJECT/data/monitor.log</string>
</dict>
</plist>
EOF
# bootout returns before the old job is gone; bootstrap fails (error 5) until it is.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null && break
  sleep 1
done
launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || { echo "failed to load $LABEL" >&2; exit 1; }
echo "loaded $LABEL; log: $PROJECT/data/monitor.log"
