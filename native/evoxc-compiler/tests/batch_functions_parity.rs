//! Python-driven differential parity gate for the native `batch_functions` merge.
//!
//! This is the end-to-end gate `src/batch_functions/`'s structural unit tests
//! cannot be: it traces real populations in the Python frontend and exercises
//! the merge through the explicit numpy pipeline.  `tests/parity_transform/`
//! owns the corpus (`batch_cases.py`) and the harness
//! (`run_batch_parity.py`), which
//!
//!   * runs today with the PyO3 entry point ABSENT (Python-only self-checks:
//!     report shape/types, per-row semantics, the no-`select` / `bands=1`
//!     invariants and the pinned auto costs) and prints an explicit SKIP line,
//!   * and, once `evoxc._evoxc_native.batch_functions` is wired
//!     (`native/evoxc-python`), compares the native module structurally, the
//!     numbers leaf by leaf and the 12-key `BatchReport` against the Python
//!     merge.
//!
//! It needs the two Python-side artefacts, so this test SKIPS (and says why)
//! when either is missing:
//!
//!   source scripts/dev-env.sh && uv sync --locked --all-packages --extra dev
//!   source scripts/dev-env.sh && scripts/build-native.sh --debug
//!
//! Reproduce by hand:
//!
//!   source scripts/dev-env.sh && .venv/bin/python \
//!       native/evoxc-compiler/tests/parity_transform/run_batch_parity.py

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    // <repo>/native/evoxc-compiler -> <repo>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives at <repo>/native/evoxc-compiler")
        .to_path_buf()
}

const SETUP_HINT: &str = "run: source scripts/dev-env.sh && uv sync --locked --all-packages --extra dev \
                          && scripts/build-native.sh --debug";

#[test]
fn native_batch_functions_match_the_python_merge() {
    let root = repo_root();
    let python = root.join(".venv/bin/python");
    let extension = root.join("evoxc/_evoxc_native.abi3.so");

    let mut missing = Vec::new();
    if !python.is_file() {
        missing.push(format!("{} (the Python dev environment)", python.display()));
    }
    if !extension.is_file() {
        missing.push(format!(
            "{} (the PyO3 extension built by scripts/build-native.sh)",
            extension.display()
        ));
    }
    if !missing.is_empty() {
        println!(
            "SKIP batch_functions_parity: missing prerequisite(s): {}\n\
             SKIP batch_functions_parity: {SETUP_HINT}",
            missing.join(", ")
        );
        return;
    }

    // `scripts/dev-env.sh` exports CARGO_HOME / library paths / .venv/bin, and
    // the harness imports `evoxc` plus its sibling modules, so the child must
    // run from the repository root with repo-relative paths.
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(
            "source scripts/dev-env.sh && exec .venv/bin/python \
             native/evoxc-compiler/tests/parity_transform/run_batch_parity.py",
        )
        .current_dir(&root)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("failed to start the batch_functions parity harness");
    let status = child
        .wait()
        .expect("failed to wait for the batch_functions parity harness");
    println!("batch_functions_parity harness exit status: {status}");
    assert!(
        status.success(),
        "the native-vs-Python batch_functions parity harness failed with {status}; \
         it needs {SETUP_HINT}"
    );
}
