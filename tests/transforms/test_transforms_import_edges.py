"""Import-direction pin for ``evoxc/transforms/``.

Pins the binding "Constraints → Imports" bullet in
``evoxc/transforms/CONTEXT.md``: the package may import only ``evoxc.core``,
``evoxc.ir``, ``evoxc.trace``, ``evoxc.ops``, the private native seam
(``evoxc._native`` / ``evoxc._native_codec`` / …) and its own submodules —
NEVER ``evoxc.backends`` / ``evoxc.pipeline`` / ``evoxc.dist`` /
``evoxc.block``. Any absolute non-``evoxc`` import must be the standard library
or numpy.

Modelled on ``tests/formats/test_surface.py``: the sources are parsed with `ast`
(never imported) so the pin does not depend on the native extension or any
optional backend.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import evoxc.transforms as transforms

TRANSFORMS_DIR = pathlib.Path(transforms.__file__).parent

# The layered imports the package is allowed to make.
ALLOWED_EVOXC = {
    "evoxc.core",
    "evoxc.ir",
    "evoxc.ops",
    "evoxc.trace",
}
# The documented forbidden edges (CONTEXT.md "Constraints").
FORBIDDEN_EVOXC = (
    "evoxc.backends",
    "evoxc.pipeline",
    "evoxc.dist",
    "evoxc.block",
)


def _import_edges():
    """Yield ``(filename, module, level)`` for every import in the package.

    ``from evoxc import core, ir`` is unfolded to one ``evoxc.core`` /
    ``evoxc.ir`` edge each, so the allow-list sees the real submodule.
    """
    for path in sorted(TRANSFORMS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path.name, alias.name, 0
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    yield path.name, module, node.level
                elif module == "evoxc":
                    for alias in node.names:
                        yield path.name, f"evoxc.{alias.name}", 0
                else:
                    yield path.name, module, 0


def _is_allowed_evoxc(module: str) -> bool:
    # a documented layer or one of its submodules (e.g. evoxc.trace._tree)
    for allowed in ALLOWED_EVOXC:
        if module == allowed or module.startswith(allowed + "."):
            return True
    # the package's own submodules
    if module == "evoxc.transforms" or module.startswith("evoxc.transforms."):
        return True
    # the private native seam (_native, _native_codec, _native_rules, …)
    if module.startswith("evoxc._"):
        return True
    return False


class TestImportDirection:
    def test_imports_stay_within_the_documented_layers(self):
        for filename, module, level in _import_edges():
            if level:
                # a relative import resolves inside the package (level 1 =
                # evoxc.transforms, level 2 = evoxc); anything deeper would
                # escape the package.
                assert 1 <= level <= 2, f"{filename}: relative import level {level}"
                continue
            assert module, f"{filename}: bare/empty import"
            root = module.split(".")[0]
            if root == "evoxc":
                assert _is_allowed_evoxc(module), (
                    f"{filename} imports {module}, outside the documented layers"
                )
            else:
                assert root in sys.stdlib_module_names or root == "numpy", (
                    f"{filename} imports {module}"
                )

    def test_the_forbidden_modules_are_imported_nowhere(self):
        modules = [module for _file, module, _level in _import_edges()]
        for forbidden in FORBIDDEN_EVOXC:
            for module in modules:
                assert not (module == forbidden or module.startswith(forbidden + ".")), (
                    f"forbidden import of {module}"
                )

    def test_the_walk_sees_the_real_imports(self):
        """A positive control: the allow-list above is not vacuously true."""
        modules = {module for _file, module, _level in _import_edges()}
        assert "evoxc.core" in modules
        assert "evoxc.trace" in modules
        assert any(module.startswith("evoxc.transforms") for module in modules)
