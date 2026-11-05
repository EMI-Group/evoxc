"""Vulkan device-visibility probe: does IREE enumerate this host's software device?

Answers exactly ONE question, twice — once with ``VK_ICD_FILENAMES`` UNSET and
once pinned at a renamed-ICD copy of Mesa lavapipe (``--icd``) — by calling
``iree.runtime.get_driver("vulkan").query_available_devices()`` and printing the
enumerated device-info dicts (``name`` / ``device_id`` / ``path``) as JSON:

* UNSET is the load-bearing case: IREE 3.11 hard-coded a device-NAME denylist
  (``strstr(name, "llvmpipe") == name`` -> hidden) that hid Mesa lavapipe, while
  the adopted iree-base 3.13.0 no longer does, so the software device enumerates
  with no ICD override at all (``evoxc/backends/adapters/CONTEXT.md`` Known
  Issue 16a).
* The renamed-ICD leg (``--icd``) is the 3.11-era workaround. It also pins this
  box's loader away from its wedged radeon ICD, which is the only remaining
  reason to set the variable here.

Every enumeration runs in a CHILD process under ``--timeout``: with a wedged
vendor ICD dlopen'd, this host's Vulkan loader HANGS instead of erroring, and a
hang is neither a result nor an error (Known Issue 16b — "wrap any Vulkan/DRM
probe in ``timeout``"). A timeout is reported as ``"timed_out": true`` with a
non-zero exit status, never retried away.

Deterministic by construction — there is no RNG and nothing is timed — so a
re-run reproduces a recorded cell exactly. Needs iree runtime (`evoxc[iree]`);
nothing here is imported by ``evoxc`` or collected by pytest.

Usage::

    python evoxc/bench/probe_vulkan_device_visibility.py \\
        --icd "$TMPDIR/vk-lvp/lavapipe_icd.json" --out vulkan_device_visibility.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

#: Child body: enumerate in a fresh process and print ONE JSON line. Failures
#: are reported as JSON (never raised) so the parent can distinguish "the
#: driver said no" from "the loader hung".
_CHILD = """
import json

def _version():
    try:
        import importlib.metadata as md
        return md.version("iree-base-runtime")
    except Exception:
        return None

try:
    import iree.runtime as rt

    infos = rt.get_driver("vulkan").query_available_devices()
except Exception as exc:  # driver/loader failure -> report, do not raise
    print(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}))
else:
    print(json.dumps({
        "iree_base_runtime": _version(),
        "devices": [
            {key: info.get(key) for key in ("name", "device_id", "path")}
            for info in infos
        ],
    }))
"""


def _probe(icd_filenames, timeout_s):
    """Enumerate vulkan devices in a child process, hard-timed.

    Returns the child's parsed record plus provenance: ``icd_env`` (the value
    the child ran with, ``None`` = unset), ``timed_out``, ``returncode`` and
    ``stdout``/``stderr`` tails. A timeout or an unparsable answer is DATA here,
    not an exception — the caller records it verbatim.
    """
    env = dict(os.environ)
    if icd_filenames is None:
        env.pop("VK_ICD_FILENAMES", None)
    else:
        env["VK_ICD_FILENAMES"] = icd_filenames
    record = {"icd_env": icd_filenames, "timed_out": False, "returncode": None}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
        record["error"] = (
            f"enumeration did not finish within {timeout_s:g}s — the Vulkan "
            "loader is HUNG (a wedged vendor ICD); wrap every such probe in a "
            "timeout, see Known Issue 16b"
        )
        return record
    record["returncode"] = proc.returncode
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            break
    if payload is None:
        tail = (proc.stderr or "").strip().splitlines()[-1:]
        record["error"] = (
            f"no parsable child result (exit {proc.returncode}): "
            f"{(tail or [''])[0][:400]}"
        )
        return record
    record.update(payload)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--icd",
        default=None,
        help="renamed-lavapipe ICD json (VK_ICD_FILENAMES value) for the second "
        "leg; omit it to probe only the unset case",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-child enumeration budget in seconds (default 60)",
    )
    parser.add_argument("--out", default=None, help="write the record as JSON here")
    args = parser.parse_args(argv)

    record = {
        "probe": "vulkan_device_visibility",
        "python": sys.version.split()[0],
        "timeout_s": args.timeout,
        "probes": {"unset": _probe(None, args.timeout)},
    }
    if args.icd:
        record["probes"]["renamed_icd"] = _probe(args.icd, args.timeout)

    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    # Non-zero only for an infrastructure failure (a hang or an unparsable
    # child) — "zero devices" is a legitimate, recorded answer.
    failed = any(
        probe.get("timed_out") or probe.get("error") for probe in record["probes"].values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
