fn main() {
    cxx_build::bridge("src/lib.rs").compile("brush-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gpu_mutex.rs");
}
