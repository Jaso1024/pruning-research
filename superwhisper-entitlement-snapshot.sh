#!/bin/bash
# Superwhisper entitlement snapshot helper (research only, redacts secrets)
set -euo pipefail

redact() {
  perl -pe 's/[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}/<uuid>/g; s/"[A-Za-z0-9_\-\.]{30,}"/"<redacted>"/g'
}

echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo
echo "[defaults]"
defaults read com.superduper.superwhisper licenseValid 2>/dev/null || echo "licenseValid: (missing)"
defaults read com.superduper.superwhisper 2>/dev/null | rg -i 'lastRecordingStatSync|filesync|sync|cloud|trial|org|waitingForSubscription' || true
echo
echo "[argmax files]"
for f in "$HOME/Library/Application Support/com.superduper.superwhisper"/argmax.*; do
  [[ -f "$f" ]] || continue
  printf '%s: ' "$(basename "$f")"
  stat -f 'mtime=%Sm size=%z' -t '%Y-%m-%d %H:%M:%S' "$f"
  plutil -p "$f" 2>/dev/null | redact | sed 's/^/  /'
done
echo
echo "[keychain metadata]"
security find-generic-password -s com.superwhisper.vault 2>&1 | redact | rg '"svce"|"cdat"|"mdat"|"acct"' || echo "  (vault not found)"
echo
echo "[CFURL cache recent]"
DB="$HOME/Library/Caches/com.superduper.superwhisper/Cache.db"
if [[ -f "$DB" ]]; then
  sqlite3 "$DB" "select time_stamp, request_key from cfurl_cache_response order by entry_ID desc limit 10;" 2>/dev/null || true
  python3 - <<'PY'
import sqlite3, re
from pathlib import Path
db = Path.home() / 'Library/Caches/com.superduper.superwhisper/Cache.db'
if not db.exists():
    raise SystemExit
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('select r.request_key, d.receiver_data from cfurl_cache_response r left join cfurl_cache_receiver_data d on r.entry_ID=d.entry_ID order by r.entry_ID desc limit 5')
for key, blob in cur.fetchall():
    text = (blob or b'').decode('latin1', 'replace')
    body = text.split('\r\n\r\n', 1)[-1][:120]
    body = re.sub(r'[A-Za-z0-9_\-]{20,}', '<redacted>', body)
    print(f'  body[{key.strip()}]: {body!r}')
PY
fi
echo
