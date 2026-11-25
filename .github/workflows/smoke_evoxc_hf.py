"""Install both release wheels into one fresh venv and run a tiny HF forward.

This is the ``evoxc-hf`` counterpart of ``smoke_evoxc.py``: it proves that the
``evoxc`` and ``evoxc-hf`` distributions coexist in one environment, that the
installed ``evoxc-hf`` resolves its ``evoxc>=0.1.0`` dependency to the *local*
release wheel (never a PyPI download), and that a whole HuggingFace-style
forward runs on the numpy backend alone — no torch and no compiler extras.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import venv


CHECK_INSTALLED = """
from importlib.metadata import version
from pathlib import Path
import json
import sys

import evoxc
import evoxc._evoxc_native
import evoxc_hf
import numpy as np

expected_evoxc, expected_hf = sys.argv[1], sys.argv[2]
assert version("evoxc") == evoxc.__version__ == expected_evoxc, (
    version("evoxc"),
    evoxc.__version__,
    expected_evoxc,
)
assert version("evoxc-hf") == evoxc_hf.__version__ == expected_hf, (
    version("evoxc-hf"),
    evoxc_hf.__version__,
    expected_hf,
)
for module in (evoxc, evoxc._evoxc_native, evoxc_hf):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve()), path

# The smallest llama-family config: one layer, hidden 16, 2 heads x head_dim 8,
# vocab 32, silu, no rope_scaling. Written into the child's cwd (the temp root).
CONFIG = {
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
}
Path("config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
cfg = evoxc_hf.load_config(".")

# Deterministic tiny params in sorted key order so the graph inputs are stable.
specs = evoxc_hf.param_specs(cfg)
rng = np.random.default_rng(0)
params = {}
for name in sorted(specs):
    spec = specs[name]
    for dim in spec.shape:
        # Static config => static shapes; a symbolic dim would be a packaging bug.
        if not isinstance(dim, int):
            raise AssertionError(f"param {name!r} has non-static dim {dim!r}")
    params[name] = (
        rng.standard_normal(tuple(spec.shape), dtype=np.float32) * np.float32(0.02)
    ).astype(np.float32, copy=False)

# LoadedModel is a 4-field NamedTuple with no defaults: all four are required.
loaded = evoxc_hf.LoadedModel(
    config=cfg, params=params, family=cfg.family, model_dir=None
)
ids = np.zeros((1, 4), dtype=np.int32)
logits = evoxc_hf.predict(loaded, ids, seq_len=4)
assert logits.shape == (1, 4, 32), logits.shape
oracle = evoxc_hf.reference_logits(cfg, params, ids)
# 1e-4 is a sensible f32 tolerance for numpy-interpreter vs pure-numpy-oracle
# agreement on a 1-layer, 16-hidden model (the two differ only in evaluation order).
np.testing.assert_allclose(logits, oracle, rtol=1e-4, atol=1e-4)
print(
    f"evoxc {expected_evoxc} + evoxc-hf {expected_hf}: "
    f"tiny llama forward OK, logits {logits.shape}, "
    f"max |diff| vs oracle {np.max(np.abs(logits - oracle)):.3g}"
)
"""


def main(archive_dir: str, evoxc_version: str) -> None:
    archives = Path(archive_dir).resolve()
    present = archives.is_dir()
    evoxc_wheels = sorted(archives.glob("evoxc-*.whl")) if present else []
    hf_wheels = sorted(archives.glob("evoxc_hf-*.whl")) if present else []
    if len(evoxc_wheels) != 1 or not evoxc_wheels[0].is_file():
        raise SystemExit(
            f"expected exactly one evoxc-*.whl in {archive_dir}, found {evoxc_wheels}"
        )
    if len(hf_wheels) != 1 or not hf_wheels[0].is_file():
        raise SystemExit(
            f"expected exactly one evoxc_hf-*.whl in {archive_dir}, found {hf_wheels}"
        )
    evoxc_wheel, hf_wheel = evoxc_wheels[0], hf_wheels[0]
    match = re.fullmatch(r"evoxc_hf-(.+)-py3-none-any\.whl", hf_wheel.name)
    if match is None:
        raise SystemExit(
            f"expected evoxc_hf-<version>-py3-none-any.whl, found {hf_wheel.name}"
        )
    hf_version = match.group(1)
    # Do not allow the extension loader's development override to mask a bad build.
    env = os.environ.copy()
    env.pop("EVOXC_NATIVE_LIB", None)
    with tempfile.TemporaryDirectory(prefix="evoxc-hf-smoke-", dir=env.get("RUNNER_TEMP")) as directory:
        root = Path(directory).resolve()
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Both release wheels in ONE resolve, so evoxc-hf's evoxc>=0.1.0 is
        # satisfied by the local wheel instead of a PyPI download.
        subprocess.run(
            [
                python,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-cache-dir",
                "--only-binary=:all:",
                str(evoxc_wheel),
                str(hf_wheel),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [python, "-I", "-c", CHECK_INSTALLED, evoxc_version, hf_version],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
