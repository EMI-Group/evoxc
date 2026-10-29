# src — the evoxc-hf src layout root

## Intent

`src/` is the src-layout root of the `evoxc-hf` distribution: its single child `./evoxc_hf/` is the import package, and nothing else belongs under `src/`.
`uv_build` maps this directory to the package via `module-root = "src"` / `module-name = "evoxc_hf"` in `../pyproject.toml`, which is why the package is not at the repository-relative root of this project.

## API Surface

Nothing importable lives at this level — there is no `src/__init__.py` and no top-level module, so `src` is never an import path component.

## Constraints

- The import name is `evoxc_hf` and never changes; the distribution name is `evoxc-hf`.
- Tests stay outside the src tree at `../tests/`, documentation in `../CONTEXT.md`, packaging metadata in `../pyproject.toml`.
- One concern per module; the per-family graph builders stay in `./evoxc_hf/models/`.

## Routing Table

| Path | Area |
|---|---|
| `./evoxc_hf/` | The import package (all 13 modules + `models/`); its binding contract is the parent `../CONTEXT.md`. |
| `./evoxc_hf/models/` | Per-family graph builders plus the shared block assembly; see its own `CONTEXT.md`. |
