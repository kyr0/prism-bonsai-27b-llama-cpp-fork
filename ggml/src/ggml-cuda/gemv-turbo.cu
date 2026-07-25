// see gemv-turbo.cuh for the design notes
#include "gemv-turbo.cuh"

#include <mutex>
#include <unordered_map>

namespace {

// ---------------------------------------------------------------------------
// activation/weight contracts
//
// weights (per 128-element group): fp16 scale + permuted code words.
//   Q2_0: 8 words, each holding 16 2-bit codes, code 4i+b at bits [8b+2i]
//   Q1_0: 4 words, each holding 32 1-bit codes, code 4i+b at bit  [8b+i]
// (exactly the bonsai-turbo retile contract, so its dp4a schedule applies)
//
// activations: ggml block_q8_1 (36 B / 32 el): ds.x = d8 scale, ds.y =
// d8*sum(qs) = sum(act) in real units. Corrections per q8_1 block:
//   Q2_0 (symbol code-1): sumi*d8 - ds.y
//   Q1_0 (symbol 2b-1):   2*sumi*d8 - ds.y
// ---------------------------------------------------------------------------

__device__ __forceinline__ int dp4a_u8s8(uint32_t codes, uint32_t acts, int acc) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(codes), "r"(acts), "r"(acc));
    return acc;
}

// ---------------------------------------------------------------------------
// one-time re-tile kernels
// ---------------------------------------------------------------------------

__global__ void retile_q2_kernel(const block_q2_0 * __restrict__ src,
                                 uint8_t * __restrict__ codes, __half * __restrict__ scales,
                                 long nblocks) {
    const long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const block_q2_0 * blk = src + b;
    *reinterpret_cast<ggml_half *>(scales + b) = blk->d;
    uint32_t * out = reinterpret_cast<uint32_t *>(codes) + b * 8;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t src_word;
        memcpy(&src_word, blk->qs + 4 * w, 4);
        uint32_t dst = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            // code j = 4i + b lands at bits [8b + 2i]
            dst |= ((src_word >> (2 * j)) & 0x3u) << (8 * (j % 4) + 2 * (j / 4));
        }
        out[w] = dst;
    }
}

__global__ void retile_q1_kernel(const block_q1_0 * __restrict__ src,
                                 uint8_t * __restrict__ codes, __half * __restrict__ scales,
                                 long nblocks) {
    const long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const block_q1_0 * blk = src + b;
    *reinterpret_cast<ggml_half *>(scales + b) = blk->d;
    uint32_t * out = reinterpret_cast<uint32_t *>(codes) + b * 4;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t src_word;
        memcpy(&src_word, blk->qs + 4 * w, 4);
        uint32_t dst = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            // code j = 4i + b lands at bit [8b + i]
            dst |= ((src_word >> j) & 0x1u) << (8 * (j % 4) + (j / 4));
        }
        out[w] = dst;
    }
}

struct RetiledW {
    uint8_t * codes;
    __half *  scales;
};

struct RetileKey {
    int          device;
    const void * wdata;
    long         N;
    long         K;
    int          wbits;

    bool operator==(const RetileKey & o) const {
        return device == o.device && wdata == o.wdata && N == o.N && K == o.K && wbits == o.wbits;
    }
};

struct RetileKeyHash {
    size_t operator()(const RetileKey & k) const {
        size_t h = std::hash<const void *>()(k.wdata);
        h ^= std::hash<long>()(k.N) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<long>()(k.K) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>()((k.device << 4) | k.wbits) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

// one-time per (tensor, device); weights are static for the process lifetime,
// so repacked buffers are never freed (same lifetime policy as get_dense_w)
RetiledW get_retiled_w(const void * wdata, long N, long K, int wbits, int device, cudaStream_t stream) {
    static std::unordered_map<RetileKey, RetiledW, RetileKeyHash> cache;
    static std::mutex cache_mutex;
    const RetileKey key{ device, wdata, N, K, wbits };

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const long nb = N * (K / 128);
    RetiledW d{};
    CUDA_CHECK(cudaMalloc(&d.codes, nb * (wbits == 2 ? 32 : 16)));
    CUDA_CHECK(cudaMalloc(&d.scales, nb * sizeof(__half)));
    const long blocks = (nb + 255) / 256;
    if (wbits == 2) {
        retile_q2_kernel<<<(unsigned) blocks, 256, 0, stream>>>((const block_q2_0 *) wdata, d.codes, d.scales, nb);
    } else {
        retile_q1_kernel<<<(unsigned) blocks, 256, 0, stream>>>((const block_q1_0 *) wdata, d.codes, d.scales, nb);
    }
    // publish only after the retile is observably complete: a consumer GEMV
    // on a different stream would otherwise read uninitialized buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cache.emplace(key, d);
    return d;
}

// ---------------------------------------------------------------------------
// GEMV: 32 rows per block (8 warps x 4 rows), activations staged in smem.
// Each row is covered by 8 lanes streaming contiguous 16B code chunks:
//   Q2: chunk = 64 codes  = 2 q8_1 blocks
//   Q1: chunk = 128 codes = 4 q8_1 blocks
// gridDim.y splits the K range for small-M shapes; partials combine via
// atomicAdd on the (pre-zeroed) destination.
// ---------------------------------------------------------------------------

template <int NBITS, bool SPLIT>
__global__ void gemv_turbo_kernel(const uint8_t * __restrict__ codes,
                                  const __half * __restrict__ w_scale,
                                  const block_q8_1 * __restrict__ acts,
                                  int s11_blocks,
                                  float * __restrict__ y, int M, int K) {
    constexpr int kCodesPerChunk = NBITS == 2 ? 64 : 128;
    const int code_row_bytes = NBITS == 2 ? (K >> 2) : (K >> 3);
    const int chunks = code_row_bytes / 16;

    // even so Q2 splits (2 chunks per 128-group) never start mid-group
    const int chunks_per_split = ((chunks + gridDim.y - 1) / gridDim.y + 1) & ~1;
    const int c0 = blockIdx.y * chunks_per_split;
    if (c0 >= chunks) return;  // uniform per block: safe before __syncthreads
    const int c1 = min(chunks, c0 + chunks_per_split);
    const int act_block0 = c0 * kCodesPerChunk / 32;  // first q8_1 block of the slice
    const int act_blocks = (c1 - c0) * kCodesPerChunk / 32;

    extern __shared__ uint8_t smem[];
    block_q8_1 * s_acts = reinterpret_cast<block_q8_1 *>(smem);

    for (int i = threadIdx.x; i < act_blocks; i += blockDim.x) {
        memcpy(s_acts + i, acts + act_block0 + i, sizeof(block_q8_1));
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int row0 = blockIdx.x * 32 + warp * 4;
    if (row0 >= M) return;
    const int nrows = min(4, M - row0);

    const uint4 * rc[4];
    const __half * rs[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row0 + (r < nrows ? r : 0);  // clamp; extras discarded
        rc[r] = reinterpret_cast<const uint4 *>(codes + (size_t) row * code_row_bytes);
        rs[r] = w_scale + (size_t) row * (K >> 7);
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c = c0 + lane; c < c1; c += 32) {
        const int abase = c * kCodesPerChunk / 32 - act_block0;  // q8_1 block idx in smem
        if (NBITS == 2) {
            // 16B codes = 4 words of 16 codes each = 2 q8_1 blocks
            const block_q8_1 * a0 = s_acts + abase;
            const block_q8_1 * a1 = s_acts + abase + 1;
            const float2 ds0 = __half22float2(a0->ds);
            const float2 ds1 = __half22float2(a1->ds);
            const int g = c >> 1;  // 2 chunks per 128-group
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const uint4 cw = rc[r][c];
                const uint32_t w[4] = {cw.x, cw.y, cw.z, cw.w};
                int sumi0 = 0, sumi1 = 0;
#pragma unroll
                for (int wnum = 0; wnum < 2; ++wnum) {
                    const uint32_t word = w[wnum];
                    const uint32_t u0 = ((const uint32_t *) a0->qs)[4 * wnum + 0];
                    const uint32_t u1 = ((const uint32_t *) a0->qs)[4 * wnum + 1];
                    const uint32_t u2 = ((const uint32_t *) a0->qs)[4 * wnum + 2];
                    const uint32_t u3 = ((const uint32_t *) a0->qs)[4 * wnum + 3];
                    sumi0 = dp4a_u8s8(word & 0x03030303u, u0, sumi0);
                    sumi0 = dp4a_u8s8((word >> 2) & 0x03030303u, u1, sumi0);
                    sumi0 = dp4a_u8s8((word >> 4) & 0x03030303u, u2, sumi0);
                    sumi0 = dp4a_u8s8((word >> 6) & 0x03030303u, u3, sumi0);
                }
#pragma unroll
                for (int wnum = 2; wnum < 4; ++wnum) {
                    const uint32_t word = w[wnum];
                    const uint32_t u0 = ((const uint32_t *) a1->qs)[4 * (wnum - 2) + 0];
                    const uint32_t u1 = ((const uint32_t *) a1->qs)[4 * (wnum - 2) + 1];
                    const uint32_t u2 = ((const uint32_t *) a1->qs)[4 * (wnum - 2) + 2];
                    const uint32_t u3 = ((const uint32_t *) a1->qs)[4 * (wnum - 2) + 3];
                    sumi1 = dp4a_u8s8(word & 0x03030303u, u0, sumi1);
                    sumi1 = dp4a_u8s8((word >> 2) & 0x03030303u, u1, sumi1);
                    sumi1 = dp4a_u8s8((word >> 4) & 0x03030303u, u2, sumi1);
                    sumi1 = dp4a_u8s8((word >> 6) & 0x03030303u, u3, sumi1);
                }
                const float part = (sumi0 * ds0.x - ds0.y) + (sumi1 * ds1.x - ds1.y);
                acc[r] += __half2float(rs[r][g]) * part;
            }
        } else {
            // 16B codes = 4 words of 32 codes each = 4 q8_1 blocks
            const int g = c;  // 1 chunk per 128-group
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const uint4 cw = rc[r][c];
                const uint32_t w[4] = {cw.x, cw.y, cw.z, cw.w};
                float part = 0.f;
#pragma unroll
                for (int wnum = 0; wnum < 4; ++wnum) {
                    const block_q8_1 * a = s_acts + abase + wnum;
                    const float2 ds = __half22float2(a->ds);
                    const uint32_t word = w[wnum];
                    const uint32_t * uq = (const uint32_t *) a->qs;
                    int sumi = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        sumi = dp4a_u8s8((word >> k) & 0x01010101u, uq[k], sumi);
                    }
                    part += 2.f * sumi * ds.x - ds.y;
                }
                acc[r] += __half2float(rs[r][g]) * part;
            }
        }
    }

    // full-warp reduce per row
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xFFFFFFFFu, acc[r], off);
        }
        if (lane == 0 && r < nrows) {
            if constexpr (SPLIT) {
                atomicAdd(&y[row0 + r], acc[r]);
            } else {
                y[row0 + r] = acc[r];
            }
        }
    }
}

struct GemvPlan {
    dim3 grid;
    int smem;
    int splits;
};

GemvPlan plan_gemv_turbo(int nbits, int M, int K) {
    const int row_blocks = (M + 31) / 32;
    const int chunks = (nbits == 2 ? K / 4 : K / 8) / 16;
    int splits = row_blocks >= 256 ? 1 : 1024 / row_blocks;
    splits = std::max(1, std::min(splits, chunks / 32));
    const int cps = ((chunks + splits - 1) / splits + 1) & ~1;
    const int act_blocks = cps * (nbits == 2 ? 64 : 128) / 32;
    return {dim3(row_blocks, splits), act_blocks * (int) sizeof(block_q8_1), splits};
}

bool turbo_gemv_disabled_by_env() {
    static const bool disabled = [] {
        const char * v = getenv("GGML_CUDA_TURBO_GEMV");
        return v && atoi(v) == 0;
    }();
    return disabled;
}

}  // namespace

bool ggml_cuda_gemv_turbo_try(
    const void * src0_data, ggml_type src0_type, int64_t nb00, int64_t nb01,
    const block_q8_1 * acts, int64_t s11_blocks,
    float * dst_d, int64_t ne00, int64_t ne01,
    int device, cudaStream_t stream) {

    if (turbo_gemv_disabled_by_env()) {
        return false;
    }
    const int wbits = src0_type == GGML_TYPE_Q2_0 ? 2 : (src0_type == GGML_TYPE_Q1_0 ? 1 : 0);
    if (wbits == 0) {
        return false;
    }
    // only plain contiguous row-major weight tensors: the retile assumes
    // block-contiguous rows of ne00 columns (nb01 = bytes per row)
    const size_t ts = ggml_type_size(src0_type);
    if ((size_t) nb00 != ts || nb01 != (int64_t) (ts * (ne00 / ggml_blck_size(src0_type)))) {
        return false;
    }
    if (ne00 % 128 != 0) {
        return false;
    }

    const RetiledW w = get_retiled_w(src0_data, ne01, ne00, wbits, device, stream);
    static bool announced = getenv("GGML_CUDA_TURBO_GEMV_DEBUG") != nullptr;
    if (announced) {
        announced = false;
        fprintf(stderr, "gemv-turbo: engaged (M=%lld K=%lld wbits=%d)\n",
                (long long) ne01, (long long) ne00, wbits);
    }
    const GemvPlan p = plan_gemv_turbo(wbits, (int) ne01, (int) ne00);
    if (p.splits > 1) {
        CUDA_CHECK(cudaMemsetAsync(dst_d, 0, (size_t) ne01 * sizeof(float), stream));
    }
    if (wbits == 2) {
        if (p.splits > 1) {
            gemv_turbo_kernel<2, true><<<p.grid, 256, p.smem, stream>>>(
                w.codes, w.scales, acts, (int) s11_blocks, dst_d, (int) ne01, (int) ne00);
        } else {
            gemv_turbo_kernel<2, false><<<p.grid, 256, p.smem, stream>>>(
                w.codes, w.scales, acts, (int) s11_blocks, dst_d, (int) ne01, (int) ne00);
        }
    } else {
        if (p.splits > 1) {
            gemv_turbo_kernel<1, true><<<p.grid, 256, p.smem, stream>>>(
                w.codes, w.scales, acts, (int) s11_blocks, dst_d, (int) ne01, (int) ne00);
        } else {
            gemv_turbo_kernel<1, false><<<p.grid, 256, p.smem, stream>>>(
                w.codes, w.scales, acts, (int) s11_blocks, dst_d, (int) ne01, (int) ne00);
        }
    }
    return true;
}
