#!/usr/bin/env bash
# Build the Rust native core and install the PyO3 extension into the source tree.
#
#   scripts/build-native.sh            # release build (default)
#   scripts/build-native.sh --debug    # faster build, used while iterating
#
# The extension is a cdylib named `evoxc_python`; this script copies it to
# `evoxc/_evoxc_native.abi3.so`, which `evoxc/_native.py` loads.  It is a build
# product and gitignored.  `import evoxc` never requires it unless a code path
# delegates to the native core.
set -euo pipefail

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_root=$(cd "$_here/.." && pwd)
# shellcheck disable=SC1091
source "$_here/dev-env.sh"

profile=release
cargo_flags=(--release)
if [ "${1:-}" = "--debug" ]; then
    profile=debug
    cargo_flags=()
fi

cargo build --manifest-path "$_root/native/Cargo.toml" -p evoxc-python "${cargo_flags[@]}"

case "$(uname -s)" in
    Darwin) lib="libevoxc_python.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) lib="evoxc_python.dll" ;;
    *) lib="libevoxc_python.so" ;;
esac
src="$_root/native/target/$profile/$lib"
if [ ! -f "$src" ]; then
    echo "build-native: expected $src (cargo build produced no cdylib)" >&2
    exit 1
fi
dest="$_root/evoxc/_evoxc_native.abi3.so"
cp "$src" "$dest"
echo "build-native: installed $dest"

# Smoke-test the binding through the real Python interpreter when one is
# available.  A broken extension should fail here, not in a test run later.
if [ -x "$_root/.venv/bin/python" ]; then
    "$_root/.venv/bin/python" - <<'PY'
import importlib.util, pathlib, sys
root = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else None
so = pathlib.Path("evoxc/_evoxc_native.abi3.so")
spec = importlib.util.spec_from_file_location("evoxc._evoxc_native", so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("build-native: extension version", mod.version())
PY
fi
