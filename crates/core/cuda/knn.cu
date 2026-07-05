// CUDA brute-force (exact) k-NN — the GPU-accelerated exact-search baseline
// (milestone M1). This is the "ground truth, but fast" path: it computes exact
// nearest neighbors, just massively parallel across queries.
//
// Design: the base matrix is uploaded to the device once (via knn_gpu_create)
// and reused across many search calls, like a real index. Each search launches
// one thread per query; the thread streams the entire base matrix, maintaining a
// private sorted top-k in local memory. This is embarrassingly parallel across a
// query batch and saturates the GPU when the batch is large — exactly the
// workload GPUs win at, and simple enough to be obviously correct.
//
// Distances are reported in the engine's "smaller key = closer" convention so
// the Rust side matches the CPU FlatIndex exactly:
//   metric 0 (L2)           -> key = squared L2 distance
//   metric 1 (InnerProduct) -> key = -dot(query, base)

#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>

// Max k supported by the per-thread top-k buffer. k must be <= this.
#define SF_MAX_K 100

extern "C" {

// Opaque handle holding the device-resident base matrix.
struct SfGpuIndex {
    float* d_base; // N * D, row-major, on device
    int n;
    int dim;
};

__global__ void knn_kernel(const float* __restrict__ base,
                           const float* __restrict__ queries,
                           int n, int dim, int nq, int k, int metric,
                           long long* out_ids, float* out_keys) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    const float* query = queries + (size_t)q * dim;

    float best_key[SF_MAX_K];
    int best_id[SF_MAX_K];
    for (int i = 0; i < k; ++i) { best_key[i] = FLT_MAX; best_id[i] = -1; }

    for (int j = 0; j < n; ++j) {
        const float* v = base + (size_t)j * dim;
        float key;
        if (metric == 0) {
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) { float diff = query[d] - v[d]; s += diff * diff; }
            key = s;
        } else {
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) { s += query[d] * v[d]; }
            key = -s;
        }
        // Insertion into the sorted (ascending-key) top-k, if it beats the worst.
        if (key < best_key[k - 1]) {
            int i = k - 1;
            while (i > 0 && best_key[i - 1] > key) {
                best_key[i] = best_key[i - 1];
                best_id[i] = best_id[i - 1];
                --i;
            }
            best_key[i] = key;
            best_id[i] = j;
        }
    }
    for (int i = 0; i < k; ++i) {
        out_ids[(size_t)q * k + i] = best_id[i];
        out_keys[(size_t)q * k + i] = best_key[i];
    }
}

// Upload the base matrix once; returns NULL on failure.
SfGpuIndex* knn_gpu_create(const float* base, int n, int dim) {
    SfGpuIndex* idx = (SfGpuIndex*)malloc(sizeof(SfGpuIndex));
    if (!idx) return NULL;
    idx->n = n;
    idx->dim = dim;
    size_t bytes = (size_t)n * dim * sizeof(float);
    if (cudaMalloc(&idx->d_base, bytes) != cudaSuccess) { free(idx); return NULL; }
    if (cudaMemcpy(idx->d_base, base, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(idx->d_base);
        free(idx);
        return NULL;
    }
    return idx;
}

// Search a batch of queries. out_ids / out_keys are host buffers of nq*k.
// Returns 0 on success, else a nonzero CUDA error code. Requires k <= SF_MAX_K.
int knn_gpu_search(const SfGpuIndex* idx, const float* queries, int nq, int k,
                   int metric, long long* out_ids, float* out_keys) {
    if (k < 1 || k > SF_MAX_K) return -1;
    float* d_q = NULL;
    long long* d_ids = NULL;
    float* d_keys = NULL;
    size_t qbytes = (size_t)nq * idx->dim * sizeof(float);
    size_t ibytes = (size_t)nq * k * sizeof(long long);
    size_t kbytes = (size_t)nq * k * sizeof(float);

    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&d_q, qbytes)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&d_ids, ibytes)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&d_keys, kbytes)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(d_q, queries, qbytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    {
        int threads = 128;
        int blocks = (nq + threads - 1) / threads;
        knn_kernel<<<blocks, threads>>>(idx->d_base, d_q, idx->n, idx->dim, nq, k,
                                        metric, d_ids, d_keys);
    }
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(out_ids, d_ids, ibytes, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(out_keys, d_keys, kbytes, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;

cleanup:
    if (d_q) cudaFree(d_q);
    if (d_ids) cudaFree(d_ids);
    if (d_keys) cudaFree(d_keys);
    return (int)err;
}

void knn_gpu_free(SfGpuIndex* idx) {
    if (!idx) return;
    if (idx->d_base) cudaFree(idx->d_base);
    free(idx);
}

} // extern "C"
