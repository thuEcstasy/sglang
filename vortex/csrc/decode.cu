#include "vortex.h"
#include <cub/cub.cuh>

__global__ void DecodeMHAScoreKernel(
const __nv_bfloat16* __restrict__ query,
const __nv_bfloat16* __restrict__ landmarks,
const int* __restrict__ dense_kv_indices,
const WorkInfo* __restrict__ work_info,
const PlanInfo* __restrict__ plan_info,
const int head_dim,
float* __restrict__ score
){

const int num_sms = gridDim.x;
const int bx = blockIdx.x;
const int bdx = blockDim.x;
const int bdy = blockDim.y;
const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int tile_h = head_dim / bdx;
const int num_workload = plan_info->num_workload;

const int per_sm_workload = num_workload / num_sms;
const int r = num_workload % num_sms;
const int start = bx * per_sm_workload + min(bx, r);
const int end = start + per_sm_workload + (bx < r ? 1 : 0);

extern __shared__ __nv_bfloat16 q_shared[];

uint16_t curr_q_idx = 0;

for (int w = start; w < end; ++w){

    const int* kv_indices = dense_kv_indices + work_info[w].kv_offset;
    const uint16_t kv_len = work_info[w].kv_len;
    const uint16_t q_idx = work_info[w].q_idx;
    const __nv_bfloat16* q = query + q_idx * head_dim;
    float* score_output = score + work_info[w].score_offset;
    if ((curr_q_idx != q_idx) || (w==start)){
        if (ty == 0){
            for (int j = tx; j < head_dim; j+=bdx){
                    q_shared[j] = q[j];
            }
        }
        curr_q_idx = q_idx;
        __syncthreads();
    }

    __nv_bfloat16* q_tile = q_shared + tile_h * tx;

    for (int i = ty; i < kv_len; i+=bdy){

        float acc = 0.0f;
        const int kv_index = kv_indices[i];
        const __nv_bfloat16* lmk_tile = landmarks + kv_index * head_dim + tile_h * tx;
        for (int j = 0; j < tile_h; ++j){
                acc += __bfloat162float(lmk_tile[j]) * __bfloat162float(q_tile[j]);
            }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if(tx == 0){
                score_output[i] = acc;
            }
    }
}

}


void DecodeMHAScoreKernelLauncher(
at::Tensor query,
at::Tensor landmarks,
at::Tensor dense_kv_indices,
const WorkInfo* work_info,
const PlanInfo* plan_info,
float* score,
const int head_dim,
const int num_sms
){

assert(query.size(1) == head_dim);
assert(landmarks.size(1) == head_dim);


dim3 nblks(num_sms);
dim3 nthrs(32, 32);
size_t smem_size = head_dim * sizeof(__nv_bfloat16);
DecodeMHAScoreKernel<<<nblks, nthrs, smem_size>>>(
    reinterpret_cast<__nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
    reinterpret_cast<__nv_bfloat16*>(landmarks.data_ptr<at::BFloat16>()),
    dense_kv_indices.data_ptr<int>(),
    work_info,
    plan_info,
    head_dim,
    score
);

}


template <int NUM_THREADS, int ITEM_PER_THREAD, int PARTIAL_SIZE>
__global__ void TwoPathDecodeTopKKernel(
const __nv_bfloat16* __restrict__ score,
const int* __restrict__ dense_kv_indptr,
const int* __restrict__ sparse_kv_indptr,
const int* __restrict__ dense_kv_indices,
int* __restrict__ sparse_kv_indices,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){


const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int blk_size = blockDim.x;
constexpr int items_per_iteration = NUM_THREADS * ITEM_PER_THREAD;

const int num_scores_blk = dense_kv_indptr[bx + 1] - dense_kv_indptr[bx] - page_reserved_bos - page_reserved_eos;
if (num_scores_blk <= topk_val) return;


float key[ITEM_PER_THREAD];
int value[ITEM_PER_THREAD];
__shared__ float partial_score[PARTIAL_SIZE];
__shared__ int partial_idx[PARTIAL_SIZE];

using BlockRadixSort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;
__shared__ typename BlockRadixSort::TempStorage temp_storage;


const int*  kv_indices_blk = dense_kv_indices + dense_kv_indptr[bx] + page_reserved_bos;
int* output_indices_blk = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;
const __nv_bfloat16* score_blk = score + dense_kv_indptr[bx] + page_reserved_bos;
const int num_loops = (num_scores_blk + items_per_iteration - 1) / items_per_iteration;


for (int i = 0; i < num_loops; ++i){
        int offset = i * items_per_iteration + tx * ITEM_PER_THREAD;
        for (int j = 0; j < ITEM_PER_THREAD; ++j){
                key[j] = ((offset + j) < num_scores_blk) ? __bfloat162float(score_blk[offset + j]):-INFINITY;
                value[j] = ((offset + j) < num_scores_blk) ? kv_indices_blk[offset + j]:0;
        }

        BlockRadixSort(temp_storage).SortDescending(key, value);
        for (int j = 0; j < ITEM_PER_THREAD; ++j){
                if ((tx * ITEM_PER_THREAD + j) < topk_val){
                    partial_score[i * topk_val + tx * ITEM_PER_THREAD + j] = key[j];
                    partial_idx[i * topk_val + tx * ITEM_PER_THREAD + j] = value[j];
                }
        }
        __syncthreads();
        
}

for (int j = 0; j < ITEM_PER_THREAD; ++j){
            key[j] = ((tx * ITEM_PER_THREAD + j) < num_loops * topk_val) ? partial_score[tx * ITEM_PER_THREAD + j]:-INFINITY;
            value[j] = ((tx * ITEM_PER_THREAD + j) < num_loops * topk_val) ? partial_idx[tx * ITEM_PER_THREAD + j]:0;
}

BlockRadixSort(temp_storage).SortDescending(key, value);
for (int j = 0; j < ITEM_PER_THREAD; ++j){
    if ((tx * ITEM_PER_THREAD + j) < topk_val){
                output_indices_blk[tx * ITEM_PER_THREAD + j] = value[j];
        }
}

}



template <int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void OnePathDecodeTopKKernel(
const float* __restrict__ score,
const int* __restrict__ dense_kv_indptr,
const int* __restrict__ sparse_kv_indptr,
const int* __restrict__ dense_kv_indices,
int* __restrict__ sparse_kv_indices,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){


const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int blk_size = blockDim.x;
constexpr int items_per_iteration = NUM_THREADS * ITEM_PER_THREAD;

const int num_scores_blk = dense_kv_indptr[bx + 1] - dense_kv_indptr[bx] - page_reserved_bos - page_reserved_eos;
if (num_scores_blk <= topk_val) return;


float key[ITEM_PER_THREAD];
int value[ITEM_PER_THREAD];


using BlockRadixSort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;
__shared__ typename BlockRadixSort::TempStorage temp_storage;

const int*  kv_indices_blk = dense_kv_indices + dense_kv_indptr[bx] + page_reserved_bos;
int* output_indices_blk = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;
const float* score_blk = score + dense_kv_indptr[bx] + page_reserved_bos;
int offset = tx * ITEM_PER_THREAD;
for (int j = 0; j < ITEM_PER_THREAD; ++j){
                key[j] = ((offset + j) < num_scores_blk) ? score_blk[offset + j]:-INFINITY;
                value[j] = ((offset + j) < num_scores_blk) ? kv_indices_blk[offset + j]:0;
}
BlockRadixSort(temp_storage).SortDescending(key, value);

for (int j = 0; j < ITEM_PER_THREAD; ++j){
    if ((tx * ITEM_PER_THREAD + j) < topk_val){
                output_indices_blk[tx * ITEM_PER_THREAD + j] = value[j];
        }
}

}

void DecodeTopKKernelLauncher(
const float* score,
at::Tensor dense_kv_indptr,
at::Tensor sparse_kv_indptr,
at::Tensor dense_kv_indices,
at::Tensor sparse_kv_indices,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos,
const int max_score_lengths,
const int batch_size
){


dim3 nblks(batch_size);
if (max_score_lengths <= 128){
    OnePathDecodeTopKKernel<128, 1><<<nblks, 128>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 256){
    OnePathDecodeTopKKernel<128, 2><<<nblks, 128>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 512 && topk_val <= 64){
    OnePathDecodeTopKKernel<128, 4><<<nblks, 128>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 512 && topk_val <= 256){
    OnePathDecodeTopKKernel<128, 4><<<nblks, 128>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

} else if (max_score_lengths <= 1024 && topk_val <= 64){
    OnePathDecodeTopKKernel<256, 4><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 1024 && topk_val <= 128){
    OnePathDecodeTopKKernel<256, 4><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

} else if (max_score_lengths <= 1024 && topk_val <= 256){
    OnePathDecodeTopKKernel<256, 4><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

}  else if (max_score_lengths <= 2048 && topk_val <= 32){
    OnePathDecodeTopKKernel<256, 8><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 2048 && topk_val <= 64){
    OnePathDecodeTopKKernel<256, 8><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 2048 && topk_val <= 128){
    OnePathDecodeTopKKernel<256, 8><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 2048 && topk_val <= 256){
    OnePathDecodeTopKKernel<256, 8><<<nblks, 256>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 4096 && topk_val <= 32){
    OnePathDecodeTopKKernel<512, 8><<<nblks, 512>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 4096 && topk_val <= 64){
    OnePathDecodeTopKKernel<512, 8><<<nblks, 512>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 4096 && topk_val <= 128){
    OnePathDecodeTopKKernel<512, 8><<<nblks, 512>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
} else if (max_score_lengths <= 4096 && topk_val <= 256){
    OnePathDecodeTopKKernel<512, 8><<<nblks, 512>>>(
        score,
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
}

}

__global__ void DecodeQuestMHAScoreKernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ landmarks,  // [num_blocks, 2, head_dim]
    const int* __restrict__ dense_kv_indices,
    const WorkInfo* __restrict__ work_info,
    const PlanInfo* __restrict__ plan_info,
    const int head_dim,
    float* __restrict__ score
){
    const int num_sms = gridDim.x;
    const int bx = blockIdx.x;
    const int bdx = blockDim.x;
    const int bdy = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_h = head_dim / bdx;
    const int num_workload = plan_info->num_workload;

    const int per_sm_workload = num_workload / num_sms;
    const int r = num_workload % num_sms;
    const int start = bx * per_sm_workload + min(bx, r);
    const int end = start + per_sm_workload + (bx < r ? 1 : 0);

    extern __shared__ __nv_bfloat16 q_shared[];

    uint16_t curr_q_idx = 0;

    for (int w = start; w < end; ++w){
        const int* kv_indices = dense_kv_indices + work_info[w].kv_offset;
        const uint16_t kv_len = work_info[w].kv_len;
        const uint16_t q_idx = work_info[w].q_idx;
        const __nv_bfloat16* q = query + q_idx * head_dim;
        float* score_output = score + work_info[w].score_offset;
        if ((curr_q_idx != q_idx) || (w==start)){
            if (ty == 0){
                for (int j = tx; j < head_dim; j+=bdx){
                    q_shared[j] = q[j];
                }
            }
            curr_q_idx = q_idx;
            __syncthreads();
        }

        __nv_bfloat16* q_tile = q_shared + tile_h * tx;

        for (int i = ty; i < kv_len; i+=bdy){
            float acc = 0.0f;
            const int kv_index = kv_indices[i];
            const __nv_bfloat16* lmk_max = landmarks + kv_index * 2 * head_dim + tile_h * tx;
            const __nv_bfloat16* lmk_min = lmk_max + head_dim;
            
            for (int j = 0; j < tile_h; ++j){
                float q_val = __bfloat162float(q_tile[j]);
                float max_val = __bfloat162float(lmk_max[j]);
                float min_val = __bfloat162float(lmk_min[j]);
                float max_score = q_val * max_val;
                float min_score = q_val * min_val;
                acc += fmaxf(max_score, min_score);
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }

            if(tx == 0){
                score_output[i] = acc;
            }
        }
    }
}

void DecodeQuestMHAScoreKernelLauncher(
    at::Tensor query,
    at::Tensor landmarks,
    at::Tensor dense_kv_indices,
    const WorkInfo* work_info,
    const PlanInfo* plan_info,
    float* score,
    const int head_dim,
    const int num_sms
){
    assert(query.size(1) == head_dim);
    assert(landmarks.size(2) == head_dim);  // landmarks shape: [num_blocks, 2, head_dim]

    dim3 nblks(num_sms);
    dim3 nthrs(32, 32);
    size_t smem_size = (head_dim * sizeof(__nv_bfloat16) + 2 * head_dim * sizeof(float));
    
    DecodeQuestMHAScoreKernel<<<nblks, nthrs, smem_size>>>(
        reinterpret_cast<__nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(landmarks.data_ptr<at::BFloat16>()),
        dense_kv_indices.data_ptr<int>(),
        work_info,
        plan_info,
        head_dim,
        score
    );
}