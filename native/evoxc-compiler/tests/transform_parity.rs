//! Python-driven differential parity gate for the native transform engines.
//!
//! This is the end-to-end gate the crate's structural unit tests cannot be: it
//! traces real graphs in the Python frontend, converts them into the
//! `evoxc-native-ir` envelope, runs `evoxc._evoxc_native.transform` on them, and
//! compares the numerically evaluated result against the Python transform of the
//! SAME graph (plus the documented `TransformError` deferrals on both sides).
//!
//! The corpus lives in `tests/parity_transform/` (`convert.py` is the envelope
//! conversion, `run_parity.py` the harness). It needs the two Python-side
//! artefacts, so this test SKIPS (and says why) when either is missing:
//!
//!   source scripts/dev-env.sh && uv sync --locked --all-packages --extra dev
//!   source scripts/dev-env.sh && scripts/build-native.sh --debug
//!
//! Reproduce by hand:
//!
//!   source scripts/dev-env.sh && .venv/bin/python \
//!       native/evoxc-compiler/tests/parity_transform/run_parity.py

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
fn native_transforms_match_the_python_transforms() {
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
            "SKIP transform_parity: missing prerequisite(s): {}\n\
             SKIP transform_parity: {SETUP_HINT}",
            missing.join(", ")
        );
        return;
    }

    // `scripts/dev-env.sh` exports CARGO_HOME / library paths / .venv/bin, and
    // the harness imports `evoxc` plus its own sibling module, so the child must
    // run from the repository root with repo-relative paths.
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(
            "source scripts/dev-env.sh && exec .venv/bin/python \
             native/evoxc-compiler/tests/parity_transform/run_parity.py",
        )
        .current_dir(&root)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("failed to start the python parity harness");
    let status = child.wait().expect("failed to wait for the parity harness");
    println!("transform_parity harness exit status: {status}");
    assert!(
        status.success(),
        "the native-vs-Python transform parity harness failed with {status}; \
         it needs {SETUP_HINT}"
    );
}
