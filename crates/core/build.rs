fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rerun-if-changed=cuda/knn.cu");
    println!("cargo:rerun-if-env-changed=PX_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let arch = std::env::var("PX_CUDA_ARCH").unwrap_or_else(|_| "120".to_string());
    let sass = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
    let ptx = format!("-gencode=arch=compute_{arch},code=compute_{arch}");

    cc::Build::new()
        .cuda(true)
        .flag(&sass)
        .flag(&ptx)
        .file("cuda/knn.cu")
        .compile("proxima_knn");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib/x64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}
