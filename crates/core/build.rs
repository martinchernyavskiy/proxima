// Compiles the CUDA k-NN kernel only when the `cuda` feature is enabled. Without
// it (e.g. on a machine with no NVIDIA toolchain), this is a no-op and the crate
// builds as pure-CPU Rust — so `cargo build` / `cargo test` work everywhere.
fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rerun-if-changed=cuda/knn.cu");

    // nvcc compiles the .cu into a static lib linked into the crate. The `cc`
    // crate picks the right host compiler (MSVC on Windows, gcc/clang on Linux).
    cc::Build::new()
        .cuda(true)
        // RTX 4070 Ti is Ada Lovelace = compute capability 8.9 (sm_89).
        .flag("-gencode=arch=compute_89,code=sm_89")
        .file("cuda/knn.cu")
        .compile("searchforge_knn");

    // Link the CUDA runtime. CUDA_PATH is set by the Windows installer; on Linux
    // the toolkit is usually at /usr/local/cuda.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib/x64"); // Windows
        println!("cargo:rustc-link-search=native={cuda_path}/lib64"); // Linux
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}
