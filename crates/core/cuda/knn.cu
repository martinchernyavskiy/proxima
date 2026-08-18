#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>

#define PX_MAX_K 46
#define PX_BLOCK 128

static __device__ __forceinline__ bool key_less(float a, float b) {
    if (isnan(a)) return false;
    if (isnan(b)) return true;
    return a < b;
}

extern "C" {

struct PxGpuIndex {
    float* d_baseT;
    int n;
    int dim;
};

__global__ void knn_kernel(const float* __restrict__ baseT,
                           const float* __restrict__ queries,
                           int n, int dim, int k, int metric,
                           long long* out_ids, float* out_keys) {
    int q = blockIdx.x;
    int t = threadIdx.x;
    const float* query = queries + (size_t)q * dim;

    extern __shared__ float smem[];
    float* s_query = smem;
    float* r_key = smem + dim;
    int* r_id = (int*)(r_key + blockDim.x * k);

    for (int d = t; d < dim; d += blockDim.x) s_query[d] = query[d];
    __syncthreads();

    float best_key[PX_MAX_K];
    int best_id[PX_MAX_K];
    for (int i = 0; i < k; ++i) { best_key[i] = FLT_MAX; best_id[i] = -1; }
    int best_count = 0;

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
        if (best_count < k) {
            int i = best_count;
            while (i > 0 && key_less(key, best_key[i - 1])) {
                best_key[i] = best_key[i - 1];
                best_id[i] = best_id[i - 1];
                --i;
            }
            best_key[i] = key;
            best_id[i] = j;
            ++best_count;
        } else if (key_less(key, best_key[k - 1])) {
            int i = k - 1;
            while (i > 0 && key_less(key, best_key[i - 1])) {
                best_key[i] = best_key[i - 1];
                best_id[i] = best_id[i - 1];
                --i;
            }
            best_key[i] = key;
            best_id[i] = j;
        }
    }

    for (int i = 0; i < k; ++i) {
        r_key[t * k + i] = best_key[i];
        r_id[t * k + i] = best_id[i];
    }
    __syncthreads();

    if (t == 0) {
        float m_key[PX_MAX_K];
        int m_id[PX_MAX_K];
        for (int i = 0; i < k; ++i) { m_key[i] = FLT_MAX; m_id[i] = -1; }
        int m_count = 0;
        int total = blockDim.x * k;
        for (int c = 0; c < total; ++c) {
            int id = r_id[c];
            if (id < 0) continue;
            float key = r_key[c];
            if (m_count < k) {
                int i = m_count;
                while (i > 0 && key_less(key, m_key[i - 1])) {
                    m_key[i] = m_key[i - 1];
                    m_id[i] = m_id[i - 1];
                    --i;
                }
                m_key[i] = key;
                m_id[i] = id;
                ++m_count;
            } else if (key_less(key, m_key[k - 1])) {
                int i = k - 1;
                while (i > 0 && key_less(key, m_key[i - 1])) {
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

PxGpuIndex* knn_gpu_create(const float* base, int n, int dim) {
    PxGpuIndex* idx = (PxGpuIndex*)malloc(sizeof(PxGpuIndex));
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

int knn_gpu_search(const PxGpuIndex* idx, const float* queries, int nq, int k,
                   int metric, long long* out_ids, float* out_keys) {
    if (k < 1 || k > PX_MAX_K) return -1;
    size_t shmem = ((size_t)idx->dim + 2 * (size_t)PX_BLOCK * k) * sizeof(float);
    if (shmem > 48000) return -2;

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

    knn_kernel<<<nq, PX_BLOCK, shmem>>>(idx->d_baseT, d_q, idx->n, idx->dim, k, metric,
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

void knn_gpu_free(PxGpuIndex* idx) {
    if (!idx) return;
    if (idx->d_baseT) cudaFree(idx->d_baseT);
    free(idx);
}

}
