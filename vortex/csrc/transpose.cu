#include "vortex.h"
#include <cub/cub.cuh>
__global__ void chunkwise_NH2HN_transpose_kernel(
const __nv_bfloat16* __restrict__ x,
const int*  __restrict__ indptr,
const uint16_t*  __restrict__ batch_table,
const int num_kv_heads,
const int num_attention_groups,
const int head_dim,
__nv_bfloat16* __restrict__ output
){

const int bx = blockIdx.x;
const int by = blockIdx.y;
const int bz = blockIdx.z;
const int tx = threadIdx.x;
const int batch_idx =  static_cast<int>(batch_table[bx]);

const int batch_offset = indptr[batch_idx];
const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
const int token_offset = bx - indptr[batch_idx];

const __nv_bfloat16* src = x + bx * num_kv_heads * num_attention_groups * head_dim +
            by * num_attention_groups * head_dim + bz * head_dim;

__nv_bfloat16* dst = output + batch_offset * num_kv_heads * num_attention_groups * head_dim +
            + by * batch_q_len * num_attention_groups * head_dim + token_offset * num_attention_groups * head_dim +
            bz * head_dim;

if (tx < head_dim) {
    dst[tx] = src[tx];
}

}

void ChunkwiseNH2HNTransposeKernelLauncher(
at::Tensor x,
at::Tensor indptr,
at::Tensor output,
const uint16_t* batch_table,
const int num_qo_heads,
const int num_kv_heads,
const int head_dim
){

const int q_len = x.size(0);
assert(num_qo_heads == x.size(1));
assert(head_dim == x.size(2));
const int num_attention_groups = num_qo_heads / num_kv_heads;


dim3 nblks(q_len, num_kv_heads, num_attention_groups);
dim3 nthrs(head_dim);
chunkwise_NH2HN_transpose_kernel<<<nblks, nthrs>>>(
    reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
    indptr.data_ptr<int>(),
    batch_table,
    num_kv_heads,
    num_attention_groups,
    head_dim,
    reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
);

}

__global__ void chunkwise_HN2NH_transpose_kernel(
const __nv_bfloat16* __restrict__ x,
const float* __restrict__ y,
const int*  __restrict__ indptr,
const uint16_t*  __restrict__ batch_table,
const int num_kv_heads,
const int num_attention_groups,
const int head_dim,
__nv_bfloat16* __restrict__ x_output,
float* __restrict__ y_output
){

const int bx = blockIdx.x;
const int by = blockIdx.y;
const int bz = blockIdx.z;
const int tx = threadIdx.x;

const int batch_idx =  static_cast<int>(batch_table[bx]);

const int batch_offset = indptr[batch_idx];
const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
const int token_offset = bx - indptr[batch_idx];

const __nv_bfloat16* src_x = x + batch_offset * num_kv_heads * num_attention_groups * head_dim +
            + by * batch_q_len * num_attention_groups * head_dim + token_offset * num_attention_groups * head_dim +
            bz * head_dim;

__nv_bfloat16* dst_x = x_output + bx * num_kv_heads * num_attention_groups * head_dim +
            by * num_attention_groups * head_dim + bz * head_dim;


const float* src_y = y + batch_offset * num_kv_heads * num_attention_groups +
            + by * batch_q_len * num_attention_groups + token_offset * num_attention_groups + bz;

float* dst_y = y_output + bx * num_kv_heads * num_attention_groups +
            by * num_attention_groups + bz;

if (tx < head_dim) {
    dst_x[tx] = src_x[tx];
}

if (tx == 0) {
    *dst_y = *src_y;
}

}

void ChunkwiseHN2NHTransposeKernelLauncher(
at::Tensor x,
at::Tensor y,
at::Tensor indptr,
at::Tensor x_output,
at::Tensor y_output,
const uint16_t* batch_table,
const int num_qo_heads,
const int num_kv_heads,
const int head_dim
){

const int num_attention_groups = x.size(1);
const int q_len = x_output.size(0);

assert(q_len == y_output.size(0));
assert(num_attention_groups == y.size(1));
assert(num_qo_heads == x_output.size(1));
assert(num_qo_heads == y_output.size(1));
assert(q_len * num_kv_heads == x.size(0));
assert(q_len * num_kv_heads == y.size(0));
assert(num_attention_groups * num_kv_heads == num_qo_heads);
assert(head_dim == x.size(2));
assert(head_dim == x_output.size(2));

dim3 nblks(q_len, num_kv_heads, num_attention_groups);
dim3 nthrs(head_dim);
chunkwise_HN2NH_transpose_kernel<<<nblks, nthrs>>>(
    reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
    y.data_ptr<float>(),
    indptr.data_ptr<int>(),
    batch_table,
    num_kv_heads,
    num_attention_groups,
    head_dim,
    reinterpret_cast<__nv_bfloat16*>(x_output.data_ptr<at::BFloat16>()),
    y_output.data_ptr<float>()
);

}