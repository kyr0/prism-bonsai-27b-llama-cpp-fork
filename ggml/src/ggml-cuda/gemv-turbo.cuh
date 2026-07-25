// gemv-turbo: batch-1 decode GEMV for Q1_0/Q2_0 (ternary/binary packs),
// ported from bonsai-turbo (github.com/kyr0/bonsai-turbo, src/cuda/gemv.cu)
// and adapted to ggml's q8_1 activation contract (see mmvq.cu).
//
// Two parts:
//   1. one-time lazy weight re-tile: interleaved block_q1_0/block_q2_0 ->
//      dp4a-friendly permuted code words + fp16 group scales, cached per
//      (device, wdata, N, K, wbits) -- mirrors get_dense_w (mmq-hopper-q1.cu)
//   2. a 32-rows-per-block GEMV consuming ggml's standard block_q8_1
//      activations with the same deferred affine correction the stock
//      vec_dot_q1_0_q8_1 / vec_dot_q2_0_q8_1 use (vecdotq.cuh), so numerics
//      stay within the existing int8 activation noise floor.
//
// Engaged from ggml_cuda_mul_mat_vec_q for Q1_0/Q2_0, ncols_dst == 1,
// no ids, no fusion; everything else falls through to the stock path.
// Disable with GGML_CUDA_TURBO_GEMV=0.
#pragma once

#include "common.cuh"

// returns true when the operation was handled (or definitively rejected by
// env), false to fall through to the stock mmvq path
bool ggml_cuda_gemv_turbo_try(
    const void * src0_data, ggml_type src0_type, int64_t nb00, int64_t nb01,
    const block_q8_1 * acts, int64_t s11_blocks,
    float * dst_d, int64_t ne00, int64_t ne01,
    int device, cudaStream_t stream);
