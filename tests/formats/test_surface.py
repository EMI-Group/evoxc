"""Public surface and dependency direction of ``evoxc.formats``.

Pins the binding statements in ``evoxc/formats/CONTEXT.md``: the package exports
exactly the six documented names (and nothing at ``evoxc.*`` top level), the
descriptor lives in ``tensorinfo``, and the modules import only ``evoxc.core``,
the standard library and numpy — never the package root or another evoxc
submodule, and never a third-party dependency.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import evoxc
import evoxc.formats as formats
from evoxc.formats import gguf, safetensors, tensorinfo

EXPECTED_NAMES = {
    "GGUFFile",
    "SafetensorsFile",
    "TensorInfo",
    "load_gguf",
    "load_safetensors",
    "save_safetensors",
}

FORMATS_DIR = pathlib.Path(formats.__file__).parent


def _imports():
    """Yield ``(file, module, level)`` for every import in the package sources."""
    for path in sorted(FORMATS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path.name, alias.name.split(".")[0], 0
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                yield path.name, module, node.level


class TestModuleSurface:
    def test_all_is_exactly_the_documented_surface(self):
        assert set(formats.__all__) == EXPECTED_NAMES
        assert len(formats.__all__) == len(EXPECTED_NAMES)
        for name in EXPECTED_NAMES:
            assert hasattr(formats, name), name

    def test_names_are_the_objects_of_their_defining_modules(self):
        assert formats.GGUFFile is gguf.GGUFFile
        assert formats.load_gguf is gguf.load_gguf
        assert formats.SafetensorsFile is safetensors.SafetensorsFile
        assert formats.load_safetensors is safetensors.load_safetensors
        assert formats.save_safetensors is safetensors.save_safetensors
        assert formats.TensorInfo is tensorinfo.TensorInfo

    def test_no_top_level_evoxc_names_are_added(self):
        for name in EXPECTED_NAMES:
            assert not hasattr(evoxc, name), name

    def test_imports_only_core_stdlib_and_numpy(self):
        core_only = (("safetensors.py", "core", 2), ("gguf.py", "core", 2),
                     ("tensorinfo.py", "core", 2), ("mmio.py", "core", 2))
        for filename, module, level in _imports():
            if level == 0:
                assert module in sys.stdlib_module_names or module == "numpy", (
                    f"{filename} imports {module}" if module else f"{filename}: bare import"
                )
            elif level == 1:
                # ``from . import mmio`` / ``from .tensorinfo import TensorInfo``
                pass
            else:
                assert (filename, module, level) in core_only, (
                    f"{filename} imports a non-core evoxc submodule: "
                    f"{'.' * level}{module}"
                )

    def test_the_descriptor_module_has_no_evoxc_imports(self):
        imports = [entry for entry in _imports() if entry[0] == "tensorinfo.py"]
        assert all(level == 0 for _file, _module, level in imports)
        assert all(module in sys.stdlib_module_names for _file, module, _level in imports)
