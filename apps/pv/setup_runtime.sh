#!/usr/bin/env bash
# Matching C++ headers + SONAME link for the already-installed ONNX Runtime Python wheel.
# Everything generated lives under build/deps/onnxruntime; no edits to the venv or system libraries.
set -euo pipefail
cd "$(dirname "$0")/../.."
read -r version library < <(.venv/bin/python - <<'PY'
from pathlib import Path
import onnxruntime as ort
p = Path(ort.__file__).parent / 'capi'
print(ort.__version__, next(p.glob('libonnxruntime.so.*')).resolve())
PY
)
root="$PWD/build/deps/onnxruntime"
mkdir -p "$root/include" "$root/lib"
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_ep_c_api.h onnxruntime_error_code.h; do
    curl --fail --silent --show-error --location --max-time 30 \
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v${version}/include/onnxruntime/core/session/${header}" \
        -o "$root/include/$header"
done
ln -sfn "$library" "$root/lib/libonnxruntime.so.1"
printf '%s\n' "$version" > "$root/version.txt"
echo "ONNX Runtime $version C++ support ready at $root"
