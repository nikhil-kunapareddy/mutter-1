#!/usr/bin/env bash
# Start (or resume) training with the Mac kept awake. Extra flags pass through to mutter.train:
#   scripts/train_overnight.sh                  # 6 h batch, resumes runs/conformer-s if present
#   scripts/train_overnight.sh --max-hours 8   # longer batch
# Keep the lid open and the charger plugged in: macOS sleeps on lid close even with caffeinate.
set -euo pipefail
cd "$(dirname "$0")/.."

# Training stalls when the GPU's memory gets pushed to swap. Warn if memory is already tight.
free_pct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
if [[ -n "${free_pct:-}" && "$free_pct" -lt 40 ]]; then
  echo "WARNING: only ${free_pct}% memory free. Quit browsers/editors first or training may stall."
  read -r -p "Continue anyway? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || exit 1
fi

# -i: no idle sleep, -m: no disk sleep, -s: no system sleep while on AC power
exec caffeinate -ims uv run python -m mutter.train "$@"
