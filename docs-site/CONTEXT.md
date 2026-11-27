# evoxc documentation site — compiler.evox.group

## Intent

The public documentation site for **EvoX Compiler** (`evoxc`), published at **https://compiler.evox.group**.
It is a Nimbus (Astro) static site deployed as a Cloudflare Worker and built by the shared reusable workflows in `EMI-Group/evox-docs-infra`.
Every factual claim here is derived from this repository's contracts — the root `../CONTEXT.md`, `../evoxc/CONTEXT.md`, the per-module `CONTEXT.md` files, `../README.md`, `../RELEASING.md` — and never invented.
Measured numbers may only be copied from those files or from a committed bench record (`../evoxc/bench/`).
The site documents the public API; it never documents private modules (`evoxc._native`, `evoxc._inspect`, …) or promises an implementation detail the contracts mark as deferred.

## Authoring rules (binding for every page)

- Frontmatter: `title` (required, unique), `description` (one sentence, used for meta/OG/search), `sidebar.order` (group pages also set `sidebar.group.label`, page order ascending).
- The page H1 comes from `title`: never repeat it as a `#` heading in the body — start at `##`.
- Markdown-first: `##`/`###` headings, fenced code blocks with a language (`python`, `sh`, `text`, `toml`, `yaml`, `json`), tables for reference data, `<Aside>` for caveats.
- Components usable in `.mdx` without any import (registered in `src/components.ts`): `Aside`, `Card`, `CardGrid`, `PackageManagers`, `Step`/`Steps`, `Tabs`/`TabItem`, `Render`.
  `<Aside type="note|tip|caution|danger" title="…">` is the callout form — those are the only types `ui/aside/Aside.astro` styles, so any other value renders as a Note.
- Internal links are absolute site paths with a trailing slash, e.g. `/guides/graph-transforms/`.
  The `nimbus/internal-link` rule is an error, so a link to a page that does not exist breaks the build.
- Code examples must call the real public API exactly as `../evoxc/CONTEXT.md` defines it — real names, real keyword arguments, no invented parameters or return shapes.
- A page is finished when no `TODO`, placeholder or authoring note remains and both `npx @cloudflare/nimbus-docs check` and `npm run build` pass.
- Prefer explaining *why* a design is the way it is (explicitness, no hidden tracing, physical devices) over restating an API signature that the generated reference already shows; for signatures point readers at [`/reference/api-overview/`](/reference/api-overview/) rather than guessing a `/api/…` path.

## Build and verification

```sh
npm ci                                          # once
npx @cloudflare/nimbus-docs check               # build-free preflight: structure, authoring, types
npm run build                                   # astro build → dist/
```

The generated Python API tree (`src/content/docs/api/`) is produced from the repository root and is git-ignored:

```sh
node <path-to>/evox-docs-infra/bin/docs-infra.mjs python-api --root .
```

The config for that generation is the repository-root `../docs-infra.yml`; the doc-site build workflow regenerates it before every build, so never hand-edit files under `src/content/docs/api/`.

## Routing table

| Path | Area |
|---|---|
| `./astro.config.ts` | Nimbus site configuration: canonical origin, title/description, GitHub link, sidebar items (one labelled group per docs directory, so the generated `api/` group keeps its place between Reference and the HuggingFace page), lint rules, MDX features |
| `./wrangler.jsonc` | Cloudflare Worker name + static-assets config used by the deploy workflow |
| `./src/content/docs/` | The English `docs` collection — `getting-started/`, `guides/`, `reference/`, `huggingface.mdx`, `development.mdx` |
| `./src/content/docs/api/` | GENERATED Python API reference pages (git-ignored, regenerated before each build — never hand-edit) |
| `./src/components/` | Site shell components: `Logo`/`logo.ts` (the evoxc mark), `Header`, `AgentDirective`, `WebMCP`, and the vendored Nimbus `ui/` set |
| `./src/layouts/` | `BaseLayout` (head/OG/agent directives) and `DocsLayout` (sidebar, TOC, breadcrumbs) |
| `./src/pages/` | Route generators: per-page Markdown alternates, `llms.txt`/`llms-full.txt`, `robots.txt`, OG images |
| `./src/styles/` | Tailwind entry and prose styles |
| `./public/` | Static files served as-is: `favicon.svg` (the evoxc mark on a rounded card) with `favicon.ico` as the raster fallback, `_headers`, and verbatim copies of the README diagrams — `evoxc-mark.svg`, `evoxc-stack.svg` (embeds on the landing page) and `evoxc-hardware.svg` (referenced from the compiler-adapters guide); copies of `../assets/`, never edited here |
| `./AGENT.md` | Nimbus's own authoring/upgrade guidance for this starter (delegated to by `CLAUDE.md`) |

## Page inventory (cross-link targets)

| Page | Title | Area |
|---|---|---|
| `/` | EvoX Compiler | Landing page: what it is, the four defining properties, the pipeline, quickstart |
| `/getting-started/` | Getting started | Group landing page |
| `/getting-started/install/` | Install | PyPI, source build, extras, hardware prerequisites |
| `/getting-started/quickstart/` | Quickstart | One traced graph through every explicit stage |
| `/getting-started/core-concepts/` | Core concepts | Explicit staging, the four value kinds, no hidden magic |
| `/guides/` | Guides | Group landing page |
| `/guides/graphs-and-tracing/` | Graphs and tracing | `defn`, `trace`, `Graph`, locations, digest |
| `/guides/graph-transforms/` | Graph transforms | `vectorize`, `vmap`, `grad`, `jvp`, `vjp`, `batch_functions` |
| `/guides/control-flow/` | In-graph control flow | `cond`, `while_loop`, `scan` |
| `/guides/tensors-and-devices/` | Tensors and devices | Physical devices, `Tensor.to`, DLPack, `DeviceError` |
| `/guides/backends/` | Backends and options | numpy/native/compiler backends, the option surface |
| `/guides/compiler-adapters/` | Compiler adapters | iree/xla/tvm, device kinds, target flags, validation state |
| `/guides/collectives/` | Collectives | Groups, `all_reduce`, `all_gather`, `reduce_scatter`, `rank`/`world_size` |
| `/guides/custom-blocks/` | Custom ops and kernels | `block` ops, `external_call`, `custom_call`/`inline_stablehlo`/`compile_text` |
| `/guides/persistence-and-caching/` | Persistence and caching | Save/load container, `Cache`, `build_cached` |
| `/guides/tensor-formats/` | Tensor formats | safetensors, GGUF, `TensorInfo` |
| `/guides/sparse-tensors/` | Sparse tensors | Value model, 16 ops, transform rules, v1 limits |
| `/guides/inspection-and-errors/` | Inspection and errors | Stage inspection, `digest`, error taxonomy |
| `/reference/` | Reference | Group landing page |
| `/reference/api-overview/` | Python API at a glance | Top-level namespace by area, links into `/api/` |
| `/reference/pipeline-objects/` | Pipeline objects | `Defn`, `Graph`, `LoweredProgram`, `CompiledArtifact`, `Executable`, `Tensor` |
| `/reference/ops/` | Op catalogue | Frontend ops by area with dtype/shape notes |
| `/reference/options/` | Options | Compile/lower options, env defaults, per-backend support |
| `/reference/architecture/` | Architecture | EvoXIR, native Rust core, lowering boundary, import layers |
| `/reference/deferrals/` | Limitations and deferrals | What v1 refuses, and the error it raises |
| `/huggingface/` | HuggingFace models | The separate `evoxc-hf` distribution |
| `/development/` | Development | Workspace build, gates, docs maintenance |
| `/api/` | Python API reference | GENERATED index (git-ignored; regenerated before each build) |

## Known issues

- Pages avoid `/api/…` deep links: that tree is generated and git-ignored, so a deep link can point at nothing; `/api/` itself is safe and the site builds green with the `api/` collection absent (28 pages).
- A fresh clone therefore builds with no generation step (`npm ci` then `npm run build`); a full build regenerates the API pages first, which is what makes the `Python API` sidebar group non-empty.
- While the `api/` tree is absent, `npx @cloudflare/nimbus-docs check` still reports every `/api/` link (`development.mdx`, `reference/api-overview.mdx`, `reference/index.mdx`) as `nimbus/internal-link` broken; the build itself passes.
- A group index page must carry a `sidebar.order` strictly lower than every other page in its group — equal orders are not reliably broken — so `guides/index.mdx` uses `order: 0` and leads its group; `reference/index.mdx` sits at `order: 3` and its Overview link renders mid-group, after Op catalogue.
- The generated API pages join the plain `docs` collection, so they are mounted as a labelled directory group (`{ label: "Python API", autogenerate: { directory: "api" } }`) rather than through a separate `api` collection; that is what keeps their position in the sidebar top level after Reference.
- Nimbus discovers `public/favicon.svg` before `public/favicon.ico`, so the head links the SVG and the `.ico` only answers a bare `/favicon.ico` request; both are the evoxc mark — regenerate the `.ico` from the SVG if the mark changes.
- The toolkit derives page URLs by lowercasing symbol names, so `evoxc.ir.OpDef` and `evoxc.ir.opdef` collide on one URL; `../docs-infra.yml` excludes the function and keeps the class page.
- The top-level `evoxc` namespace is not generated: its `__all__` is computed at import time, which Griffe's static analysis cannot evaluate, so `reference/api-overview.mdx` is hand-written and links into the definition pages.
- A new public top-level name therefore never appears without hand edits, and the transform inventory (`vectorize`/`vmap`/`grad`/`jvp`/`vjp`/`batch_functions`) is duplicated in `reference/api-overview.mdx` (Transforms table), `reference/index.mdx` (pipeline chain), `reference/pipeline-objects.mdx` (the `Graph` producer diagram and "Produced by"), `reference/architecture.mdx` (layer diagram, crate table, the "six builtin transforms" sentence), `guides/graph-transforms.mdx`, `guides/index.mdx`, `index.mdx`, `getting-started/quickstart.mdx`, `getting-started/install.mdx` and `development.mdx`; the sidebar needs no edit because `astro.config.ts` autogenerates every group by directory.
- The generated pages carry the toolkit's back-link and signature text verbatim; when a docstring changes the page changes too, so never patch a generated page — fix the docstring.

<!-- genesis:local -->

## Deployment (read only by agents assigned to `./docs-site`)

- Worker name `evoxc-docs`, `workers_dev: false`, static assets `./dist`; the Custom Domain `compiler.evox.group` (zone `evox.group`) is attached to that Worker in the Cloudflare dashboard, not in `wrangler.jsonc`.
- The GitHub environment `docs-production` carries the `CLOUDFLARE_API_TOKEN` secret and the `CLOUDFLARE_ACCOUNT_ID` variable; the token needs Workers Scripts edit rights on this Worker.
- `../.github/workflows/docs-site.yml` calls `EMI-Group/evox-docs-infra/.github/workflows/{build,deploy}.yml@v0.1.0` with `site_directory: docs-site`, `output_directory: docs-site/dist` and `wrangler_config: docs-site/wrangler.jsonc`: it builds on pull requests and deploys on pushes to `main`.
- The optional `DOCS_INFRA_READ_TOKEN` repository secret lets a private checkout of the toolkit be read; with the public toolkit the caller falls back to the job token.
- Local deploy: `npm run build` then `npx wrangler deploy` with `CLOUDFLARE_API_TOKEN` exported.
- Translations are not enabled (`../docs-infra.yml` has `locales: []`); enabling them means adding locale IDs there, registering a Nimbus `docs-<locale>` collection here, and adding the toolkit's `docs-translate.yml` caller.
- Upgrading the Nimbus starter follows `./AGENT.md` (`npx @cloudflare/nimbus-docs outdated`, `diff`, `migrate --dry-run`); keep `nimbus.json` committed and never edit it by hand.
