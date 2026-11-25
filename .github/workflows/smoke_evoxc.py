"""Install one release archive into a fresh venv and exercise the native code."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


CHECK_INSTALLED = """
from importlib.metadata import version
from pathlib import Path
import sys

import evoxc
from evoxc import _evoxc_native
import numpy as np

expected = sys.argv[1]
assert version("evoxc") == evoxc.__version__ == expected, (version("evoxc"), expected)
for module in (evoxc, _evoxc_native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve()), path

@evoxc.defn
def squared_norm(x):
    return evoxc.sum(x * x)

spec = evoxc.TensorSpec((2,), evoxc.float32)
graph = evoxc.trace(squared_norm, spec)
gradient = evoxc.grad(graph, argnums=0)
lowered = evoxc.lower(gradient, backend="native")
executable = evoxc.load(evoxc.compile(lowered))
result = evoxc.run(executable, evoxc.tensor([2.0, -3.0], dtype=evoxc.float32))
np.testing.assert_array_equal(result.numpy(), np.array([4.0, -6.0], dtype=np.float32))

# The same graph through the default numpy backend, staged just as explicitly.
lowered_np = evoxc.lower(gradient, backend="numpy")
executable_np = evoxc.load(evoxc.compile(lowered_np))
result_np = evoxc.run(executable_np, evoxc.tensor([2.0, -3.0], dtype=evoxc.float32))
np.testing.assert_array_equal(result_np.numpy(), np.array([4.0, -6.0], dtype=np.float32))

# Graph.save/Graph.load is the documented persistence round-trip; it delegates
# to persist.save_object/load_object, whose own signatures take encoded payloads.
path = "smoke.graph"
graph.save(path)
loaded = evoxc.Graph.load(path)
assert loaded.digest() == graph.digest(), (loaded.digest(), graph.digest())

print(f"evoxc {expected}: installed native extension at {_evoxc_native.__file__}; "
      "graph OK on native and numpy backends; graph persistence round-trip OK")
"""


def main(archive_dir: str, pattern: str, version: str) -> None:
    archives = sorted(Path(archive_dir).resolve().glob(pattern))
    if len(archives) != 1 or not archives[0].is_file():
        raise SystemExit(f"expected exactly one {pattern} in {archive_dir}, found {archives}")
    archive = archives[0]
    # Do not allow the extension loader's development override to mask a bad build.
    env = os.environ.copy()
    env.pop("EVOXC_NATIVE_LIB", None)
    with tempfile.TemporaryDirectory(prefix="evoxc-smoke-", dir=env.get("RUNNER_TEMP")) as directory:
        root = Path(directory).resolve()
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # A direct sdist path with an empty cache forces a source build. A wheel
        # leg has no reason to build even its dependencies from source.
        binary_only = ["--only-binary=:all:"] if archive.suffix == ".whl" else []
        subprocess.run(
            [python, "-I", "-m", "pip", "--isolated", "install", "--no-cache-dir", *binary_only, str(archive)],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run([python, "-I", "-c", CHECK_INSTALLED, version], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
