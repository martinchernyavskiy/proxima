fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rerun-if-changed=cuda/knn.cu");

    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_89,code=sm_89")
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
