#!/usr/bin/env bash
# Source this file to enter the native/GPU development environment on this host.
#
#   source scripts/dev-env.sh
#   uv sync --locked --all-packages --extra dev --extra iree
#   scripts/build-native.sh
#   .venv/bin/python -m pytest tests/ir -q
#
# The host is NixOS: the uv-managed CPython extension modules link against
# libraries that are not on the default dynamic-loader path, and `~/.cargo` is
# read-only.  This script exports the two things every Rust/Python command
# needs, and nothing else.  It is deliberately a *source* script so variables
# survive into the caller's shell; subagents should source it at the start of
# their command line:
#
#   source scripts/dev-env.sh && .venv/bin/python -m pytest ...
#
# All paths are discovered through `nix-build` on first use and cached in
# `.dev-env.cache` (gitignored) so repeated shells are instant.

_evoxc_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_evoxc_cache="$_evoxc_repo_root/.dev-env.cache"

_evoxc_nix_out() {
    # Print the store path of one nixpkgs attribute, cached per attribute.
    local attr=$1
    local key="nix:${attr}"
    if [ -f "$_evoxc_cache" ]; then
        local hit
        hit=$(grep -F "${key}=" "$_evoxc_cache" | head -n 1 | cut -d= -f2-)
        if [ -n "$hit" ] && [ -e "$hit" ]; then
            printf '%s\n' "$hit"
            return 0
        fi
    fi
    local out
    out=$(nix-build '<nixpkgs>' -A "$attr" --no-out-link 2>/dev/null | tail -n 1)
    if [ -n "$out" ]; then
        printf '%s=%s\n' "$key" "$out" >> "$_evoxc_cache"
        printf '%s\n' "$out"
    fi
}

# Writable cargo home (the Nix profile's ~/.cargo is a read-only store path).
export CARGO_HOME=${CARGO_HOME:-$HOME/.cache/cargo}
mkdir -p "$CARGO_HOME"

# Native libraries required by numpy / iree / torch wheels.
_evoxc_libs=()
for _evoxc_attr in stdenv.cc.cc.lib zlib openssl.out libffi expat; do
    _evoxc_out=$(_evoxc_nix_out "$_evoxc_attr")
    [ -n "$_evoxc_out" ] && [ -d "$_evoxc_out/lib" ] && _evoxc_libs+=("$_evoxc_out/lib")
done
export LD_LIBRARY_PATH=$(IFS=:; echo "${_evoxc_libs[*]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")

# Vulkan loader + the Intel Mesa ANV ICD (the only GPU on this host).
_evoxc_vk=$(_evoxc_nix_out vulkan-loader)
if [ -n "$_evoxc_vk" ] && [ -d "$_evoxc_vk/lib" ]; then
    export LD_LIBRARY_PATH="$_evoxc_vk/lib:$LD_LIBRARY_PATH"
fi
if [ -z "${EVOXC_KEEP_ICD:-}" ]; then
    _evoxc_mesa=$(_evoxc_nix_out mesa)
    if [ -n "$_evoxc_mesa" ] && [ -f "$_evoxc_mesa/share/vulkan/icd.d/intel_icd.x86_64.json" ]; then
        export VK_ICD_FILENAMES="$_evoxc_mesa/share/vulkan/icd.d/intel_icd.x86_64.json"
    fi
fi

export PATH="$_evoxc_repo_root/.venv/bin:$PATH"
export PYTHONPATH="$_evoxc_repo_root${PYTHONPATH:+:$PYTHONPATH}"

unset _evoxc_repo_root _evoxc_cache _evoxc_attr _evoxc_out _evoxc_libs _evoxc_vk _evoxc_mesa
