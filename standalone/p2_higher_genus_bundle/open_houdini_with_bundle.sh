#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hip_file="${1:-hero_torus.hipnc}"

if [[ "$hip_file" != /* ]]; then
  hip_file="$bundle_dir/$hip_file"
fi

if [[ ! -f "$hip_file" ]]; then
  echo "HIP file not found: $hip_file" >&2
  exit 1
fi

export RHEIDOS_STANDALONE_BUNDLE="$bundle_dir"
export PYTHONPATH="$bundle_dir/python3.11libs${PYTHONPATH:+:$PYTHONPATH}"
export HOUDINI_PATH="$bundle_dir${HOUDINI_PATH:+:$HOUDINI_PATH}:&"

if command -v houdini >/dev/null 2>&1; then
  exec houdini "$hip_file"
fi

houdini_bin="/Applications/Houdini/Current/Frameworks/Houdini.framework/Versions/Current/Resources/bin/houdini"
if [[ -x "$houdini_bin" ]]; then
  exec "$houdini_bin" "$hip_file"
fi

echo "Could not find Houdini. Add Houdini's bin directory to PATH or edit this script." >&2
exit 1
