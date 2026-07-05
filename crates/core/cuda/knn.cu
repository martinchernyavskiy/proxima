// CUDA brute-force (exact) k-NN — the GPU-accelerated exact-search baseline
// (milestone M1). Exact nearest neighbors, massively parallel across queries.
//
// Performance design (this is what makes the GPU actually win):
//   * ONE BLOCK PER QUERY (not one thread) — 100k+ resident threads instead of
//     a few thousand, so the GPU is saturated and memory latency is hidden.
//   * The base matrix is stored TRANSPOSED on the device (dim x N). A warp's 32
//     threads process 32 consecutive base vectors, so at each step they read 32
//     consecutive floats — fully COALESCED global loads, which is the whole game
//     for a memory-bandwidth-bound scan.
//   * Each thread keeps a private top-k; a block reduction merges them.
//
// The base is uploaded once (knn_gpu_create) and reused across searches.
//
// Distances use the engine's "smaller key = closer" convention so the Rust side
// matches the CPU FlatIndex exactly:
//   metric 0 (L2)           -> key = squared L2 distance
//   metric 1 (InnerProduct) -> key = -dot(query, base)

#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>

#define SF_MAX_K 100
#define SF_BLOCK 128 // threads per block (one block handles one query)

extern "C" {

// Opaque handle: the device-resident base matrix, stored TRANSPOSED (dim x N).
struct SfGpuIndex {
    float* d_baseT; // dim * N, column-major over base vectors (baseT[d*N + j])
    int n;
    int dim;
};

// out_ids / out_keys are (nq * k). Dynamic shared memory holds the cached query
// (dim floats) followed by the per-thread top-k reduction scratch.
__global__ void knn_kernel(const float* __restrict__ baseT,
                           const float* __restrict__ queries,
                           int n, int dim, int k, int metric,
                           long long* out_ids, float* out_keys) {
    int q = blockIdx.x;
    int t = threadIdx.x;
    const float* query = queries + (size_t)q * dim;

    extern __shared__ float smem[];
    float* s_query = smem;                        // [dim]
    float* r_key = smem + dim;                     // [blockDim * k]
    int* r_id = (int*)(r_key + blockDim.x * k);    // [blockDim * k]

    for (int d = t; d < dim; d += blockDim.x) s_query[d] = query[d];
    __syncthreads();

    float best_key[SF_MAX_K];
    int best_id[SF_MAX_K];
    for (int i = 0; i < k; ++i) { best_key[i] = FLT_MAX; best_id[i] = -1; }

    // Strided over base vectors: consecutive threads -> consecutive columns j ->
    // coalesced reads of baseT[d*n + j] within each warp.
    for (int j = t; j < n; j += blockDim.x) {
        float key;
        if (metric == 0) {
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = s_query[d] - baseT[(size_t)d * n + j];
                s += diff * diff;
            }
            key = s;
        } else {
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) s += s_query[d] * baseT[(size_t)d * n + j];
            key = -s;
        }
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

    // Publish each thread's top-k, then thread 0 merges into the block's top-k.
    // (Each global top-k element is in some thread's local top-k, so the union
    // is guaranteed to contain the true top-k.)
    for (int i = 0; i < k; ++i) {
        r_key[t * k + i] = best_key[i];
        r_id[t * k + i] = best_id[i];
    }
    __syncthreads();

    if (t == 0) {
        float m_key[SF_MAX_K];
        int m_id[SF_MAX_K];
        for (int i = 0; i < k; ++i) { m_key[i] = FLT_MAX; m_id[i] = -1; }
        int total = blockDim.x * k;
        for (int c = 0; c < total; ++c) {
            int id = r_id[c];
            if (id < 0) continue;
            float key = r_key[c];
            if (key < m_key[k - 1]) {
                int i = k - 1;
                while (i > 0 && m_key[i - 1] > key) {
                    m_key[i] = m_key[i - 1];
                    m_id[i] = m_id[i - 1];
                    --i;
                }
                m_key[i] = key;
                m_id[i] = id;
            }
        }
        for (int i = 0; i < k; ++i) {
            out_ids[(size_t)q * k + i] = m_id[i];
            out_keys[(size_t)q * k + i] = m_key[i];
        }
    }
}

// Upload the base matrix once, transposing to (dim x N) for coalesced access.
// Returns NULL on failure.
SfGpuIndex* knn_gpu_create(const float* base, int n, int dim) {
    SfGpuIndex* idx = (SfGpuIndex*)malloc(sizeof(SfGpuIndex));
    if (!idx) return NULL;
    idx->n = n;
    idx->dim = dim;
    size_t bytes = (size_t)n * dim * sizeof(float);

    float* baseT = (float*)malloc(bytes);
    if (!baseT) { free(idx); return NULL; }
    for (int j = 0; j < n; ++j)
        for (int d = 0; d < dim; ++d)
            baseT[(size_t)d * n + j] = base[(size_t)j * dim + d];

    if (cudaMalloc(&idx->d_baseT, bytes) != cudaSuccess) { free(baseT); free(idx); return NULL; }
    cudaError_t err = cudaMemcpy(idx->d_baseT, baseT, bytes, cudaMemcpyHostToDevice);
    free(baseT);
    if (err != cudaSuccess) { cudaFree(idx->d_baseT); free(idx); return NULL; }
    return idx;
}

// Search a batch of queries. out_ids / out_keys are host buffers of nq*k.
// Returns 0 on success, else a nonzero code. Requires k <= SF_MAX_K.
int knn_gpu_search(const SfGpuIndex* idx, const float* queries, int nq, int k,
                   int metric, long long* out_ids, float* out_keys) {
    if (k < 1 || k > SF_MAX_K) return -1;
    size_t shmem = ((size_t)idx->dim + 2 * (size_t)SF_BLOCK * k) * sizeof(float);
    if (shmem > 48000) return -2; // exceeds default shared-memory budget

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

    knn_kernel<<<nq, SF_BLOCK, shmem>>>(idx->d_baseT, d_q, idx->n, idx->dim, k, metric,
                                        d_ids, d_keys);
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
    if (idx->d_baseT) cudaFree(idx->d_baseT);
    free(idx);
}

} // extern "C"
