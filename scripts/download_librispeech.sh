#!/usr/bin/env bash
# Download and extract LibriSpeech splits from OpenSLR (resource 12).
# Resumable: re-running skips finished splits and continues partial downloads.
#
# Usage: scripts/download_librispeech.sh [data_dir]   (default: data)
set -euo pipefail

DATA_DIR="${1:-data}"
DL_DIR="$DATA_DIR/downloads"
MIRRORS=("https://openslr.elda.org/resources/12" "https://www.openslr.org/resources/12"
         "https://openslr.trmal.net/resources/12")
# Use the system curl: some conda curls exit 0 on TLS errors, which breaks the mirror fallback.
CURL=/usr/bin/curl
[[ -x $CURL ]] || CURL=curl
# Small eval splits first so the pipeline can be tested while the big one downloads.
SPLITS=(dev-clean test-clean dev-other test-other train-clean-100)

mkdir -p "$DL_DIR"

for split in "${SPLITS[@]}"; do
  if [[ -f "$DATA_DIR/LibriSpeech/$split/.done" ]]; then
    echo "[skip] $split already extracted"
    continue
  fi
  tarball="$DL_DIR/$split.tar.gz"
  ok=0
  for mirror in "${MIRRORS[@]}"; do
    echo "[download] $split from $mirror"
    if "$CURL" -fL --retry 5 --retry-delay 5 -C - -o "$tarball" "$mirror/$split.tar.gz" \
        && [[ -s "$tarball" ]]; then
      ok=1; break
    fi
  done
  [[ $ok == 1 ]] || { echo "[error] could not download $split" >&2; exit 1; }

  echo "[extract] $split"
  tar -xzf "$tarball" -C "$DATA_DIR"   # gzip CRC check catches corrupt downloads
  touch "$DATA_DIR/LibriSpeech/$split/.done"
  echo "[done] $split"
done

echo "All splits ready in $DATA_DIR/LibriSpeech"
