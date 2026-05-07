#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URLS_FILE="$ROOT_DIR/data/GSE320042_visium/visium_urls_https.txt"
OUT_DIR="$ROOT_DIR/data/GSE320042_visium/files"

if [[ ! -f "$URLS_FILE" ]]; then
  echo "Missing URL manifest: $URLS_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

while IFS= read -r url; do
  url="${url//$'\r'/}"
  [[ -z "$url" ]] && continue
  file_name="${url##*file=}"
  out_path="$OUT_DIR/$file_name"

  if [[ -s "$out_path" ]]; then
    echo "[skip] $file_name"
    continue
  fi

  echo "[get ] $file_name"
  curl -L --fail --retry 5 --retry-delay 3 --continue-at - -o "$out_path" "$url"
done < "$URLS_FILE"

echo "Done. Files in: $OUT_DIR"
