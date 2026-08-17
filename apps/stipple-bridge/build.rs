fn main() {
    // CMake passes the real, fully-resolved list (basalt/include plus whatever
    // -I dirs the linked basalt::opencv/TBB::tbb/basalt::basalt-headers targets
    // carry). Standalone `cargo build`/rust-analyzer (no CMake configure) fall
    // back to paths relative to this crate for the repo-vendored headers
    // (basalt/include, basalt-headers, Eigen, Sophus), plus a best-effort
    // pkg-config lookup for the system packages (OpenCV/TBB) that have no
    // fixed relative location. If pkg-config can't find them either, this
    // still won't compile standalone -- set BASALT_INCLUDE_DIRS manually.
    let basalt_include_dirs = std::env::var("BASALT_INCLUDE_DIRS").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut dirs: Vec<String> = [
            "../../../../include",
            "../../../../thirdparty/basalt-headers/include",
            "../../../../thirdparty/basalt-headers/thirdparty/eigen",
            "../../../../thirdparty/basalt-headers/thirdparty/Sophus",
        ]
        .into_iter()
        .map(|rel| format!("{manifest_dir}/{rel}"))
        .collect();

        dirs.extend(
            pkg_config_include_dirs("opencv4")
                .or_else(|| pkg_config_include_dirs("opencv"))
                .unwrap_or_default(),
        );
        dirs.extend(pkg_config_include_dirs("tbb").unwrap_or_default());

        dirs.join(":")
    });

    let mut build = cxx_build::bridge("src/lib.rs");
    // CMake's harvested list can contain stray empty segments (an
    // $<INSTALL_INTERFACE:...> generator expression evaluates to empty
    // during a normal, non-install build) -- skip them rather than pass
    // cxx_build a bogus `-I` for the current directory.
    for dir in basalt_include_dirs.split(':').filter(|dir| !dir.is_empty()) {
        build.include(dir);
    }
    build.compile("brush-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gpu_mutex.rs");
    println!("cargo:rerun-if-env-changed=BASALT_INCLUDE_DIRS");
}

/// Best-effort `-I` lookup via pkg-config, for the standalone-build fallback
/// only (the CMake-driven build passes real paths via BASALT_INCLUDE_DIRS).
/// Returns `None` (rather than an empty Vec) when the lookup itself failed,
/// so callers can `.or_else()` an alternate package name.
fn pkg_config_include_dirs(package: &str) -> Option<Vec<String>> {
    let output = std::process::Command::new("pkg-config")
        .args(["--cflags", package])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .split_whitespace()
            .filter_map(|tok| tok.strip_prefix("-I"))
            .map(str::to_owned)
            .collect(),
    )
}
