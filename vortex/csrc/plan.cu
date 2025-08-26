#include "vortex.h"
#include <cub/cub.cuh>
__global__ void PlanPrefillIndptrKernel(
    const int*  __restrict__ cached_seq_lens,
    const int*  __restrict__ input_seq_lens,
    int*  __restrict__ kv_indptr,
    int*  __restrict__ qo_indptr_ragged,
    int*  __restrict__ qo_indptr_paged,
    const int batch_size,
    const int num_kv_heads,
    const int page_size
)
{   

    const int tx = threadIdx.x;
    using BlockScan = cub::BlockScan<int, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    int input_seq_cumsum = (tx < batch_size) ? input_seq_lens[tx] : 0;
    //int num_cached_pages = (cached_seq_lens[tx] + page_size - 1) / page_size;
    int num_cached_pages = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    int cached_seq_cumsum = (tx < batch_size) ? num_cached_pages : 0;

    BlockScan(temp_storage).InclusiveSum(input_seq_cumsum, input_seq_cumsum);
    __syncthreads();
    BlockScan(temp_storage).InclusiveSum(cached_seq_cumsum, cached_seq_cumsum);
    __syncthreads();

    
    if (tx < batch_size){
        qo_indptr_ragged[tx + 1] = input_seq_cumsum;
        for (int i = 0; i < num_kv_heads; ++i){
            qo_indptr_paged[num_kv_heads * (tx + 1) - i] = input_seq_cumsum * num_kv_heads
                - i * input_seq_lens[tx];
            
            kv_indptr[num_kv_heads * (tx + 1) - i] = cached_seq_cumsum * num_kv_heads
                - i * num_cached_pages;
        }

    }

    if(tx == 0){
        qo_indptr_ragged[0] = 0;
        qo_indptr_paged[0] = 0;
        kv_indptr[0] = 0;
    }

}

void PlanPrefillIndptrKernelLauncher(
at::Tensor cached_seq_lens,
at::Tensor input_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor qo_indptr_ragged,
at::Tensor qo_indptr_paged,
const int num_kv_heads,
const int page_size
){

    const int batch_size = cached_seq_lens.size(0);

    TORCH_CHECK(batch_size == input_seq_lens.size(0));
    TORCH_CHECK((batch_size + 1) == qo_indptr_ragged.size(0));
    TORCH_CHECK((batch_size * num_kv_heads + 1) == qo_indptr_paged.size(0));
    TORCH_CHECK((batch_size * num_kv_heads + 1) == dense_kv_indptr.size(0));
    TORCH_CHECK(batch_size <= 1024);
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(input_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_ragged.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_paged.dtype() == torch::kInt32);

    PlanPrefillIndptrKernel<<<1, 1024>>>(
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        qo_indptr_paged.data_ptr<int>(),
        batch_size,
        num_kv_heads,
        page_size
    );
}

__global__ void PlanPrefillIndicesBatchTableKernel(
    const int*  __restrict__ req_to_token,
    const long*  __restrict__ req_indices,
    const int*  __restrict__ cache_seq_lens,
    const int*  __restrict__ input_seq_lens,
    const int*  __restrict__ kv_indptr,
    const int*  __restrict__ qo_indptr_ragged,
    const int page_size,
    const int num_kv_heads,
    const int req_to_token_stride,
    int*  __restrict__ kv_last_page_len,
    int*  __restrict__ kv_indices,
    uint16_t*  __restrict__ batch_table
){
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    // const int num_cached_pages = (cache_seq_lens[bx] + page_size - 1) / page_size;
    const int num_cached_pages = cache_seq_lens[bx];
    int* output = kv_indices + kv_indptr[bx * num_kv_heads + by];
    //const int last_len = cache_seq_lens[bx] % page_size;
    const int last_len = 1;
    kv_last_page_len[bx * num_kv_heads + by] = (last_len == 0)? page_size:last_len;

    int pos = tx;
    while(pos < num_cached_pages){
        int data = token_indices[pos];
        output[pos] = (data / page_size) * (page_size * num_kv_heads) + by * page_size + data % page_size;
        pos += block_size;
    }

    const int qo_len = input_seq_lens[bx];
    uint16_t* batch_table_output = batch_table + qo_indptr_ragged[bx];
    pos = tx;
    if (by == 0){
        while(pos < qo_len){
        batch_table_output[pos] = static_cast<uint16_t>(bx);
        pos += block_size;
    }
    }

}

void PlanPrefillIndicesBatchTableKernelLauncher(
at::Tensor req_to_token,
at::Tensor req_indices,
at::Tensor cached_seq_lens,
at::Tensor input_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor qo_indptr_ragged,
at::Tensor dense_kv_indices,
at::Tensor kv_last_page_len,
uint16_t* batch_table,
const int page_size,
const int num_kv_heads
){

    const int req_to_token_stride = req_to_token.size(1);
    const int batch_size = req_indices.size(0);
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(128);
    PlanPrefillIndicesBatchTableKernel<<<nblks, nthrs>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        page_size,
        num_kv_heads,
        req_to_token_stride,
        kv_last_page_len.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        batch_table
    );
}


__global__ void PlanDecodeIndptrKernel(
const int*  __restrict__ cached_seq_lens,
int*  __restrict__ dense_kv_indptr,
int*  __restrict__ sparse_kv_indptr,
const int batch_size,
const int num_kv_heads,
const int page_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_page_len = (cached_seq_len + page_size - 1) / page_size;
    using BlockScan = cub::BlockScan<int, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int dense_cumsum = (tx < batch_size) ? cached_page_len : 0;
    int sparse_cumsum = (tx < batch_size) ? min(kv_budget, cached_page_len) : 0;
    
    BlockScan(temp_storage).InclusiveSum(dense_cumsum, dense_cumsum);
    __syncthreads();
    BlockScan(temp_storage).InclusiveSum(sparse_cumsum, sparse_cumsum);
    __syncthreads();
   
    if (tx < batch_size){
        for (int i = 0; i < num_kv_heads; ++i){
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] = dense_cumsum * num_kv_heads
                - i * cached_page_len;
            
            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] = sparse_cumsum * num_kv_heads
                - i * min(kv_budget, cached_page_len);
        }

    }

    if(tx == 0){
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }

}

void PlanDecodeIndptrKernelLauncher(
at::Tensor cached_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor sparse_kv_indptr,
const int num_kv_heads,
const int page_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

const int batch_size = cached_seq_lens.size(0);
TORCH_CHECK((batch_size * num_kv_heads + 1) == dense_kv_indptr.size(0));
TORCH_CHECK((batch_size * num_kv_heads + 1) == sparse_kv_indptr.size(0));
TORCH_CHECK(batch_size <= 1024);
PlanDecodeIndptrKernel<<<1, 1024>>>(
    cached_seq_lens.data_ptr<int>(),
    dense_kv_indptr.data_ptr<int>(),
    sparse_kv_indptr.data_ptr<int>(),
    batch_size,
    num_kv_heads,
    page_size,
    topk_val,
    page_reserved_bos,
    page_reserved_eos
);

}


__global__ void PlanDecodeIndicesKernel(
const int*  __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*  __restrict__ cache_seq_lens,
const int*  __restrict__ dense_kv_indptr,
const int*  __restrict__ sparse_kv_indptr,
int*  __restrict__ dense_kv_indices,
int*  __restrict__ sparse_kv_indices,
int*  __restrict__ kv_last_page_len,
const int req_to_token_stride,
const int page_size,
const int num_kv_heads,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
const int block_size = blockDim.x;
const int bx = blockIdx.x;
const int by = blockIdx.y;
const int tx = threadIdx.x;
const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
const int kv_len = cache_seq_lens[bx];
const int page_len = (kv_len + page_size - 1) / page_size;
const int last_len = kv_len % page_size;
kv_last_page_len[bx * num_kv_heads + by] = (last_len == 0)? page_size:last_len;
int* dense_output = dense_kv_indices + dense_kv_indptr[bx * num_kv_heads + by];
int* sparse_output = sparse_kv_indices + sparse_kv_indptr[bx * num_kv_heads + by];

int pos = tx;
while(pos < page_len){
        int data = token_indices[pos * page_size];
        dense_output[pos] = (data / page_size) * (num_kv_heads) + by;
        pos += block_size;
}

if(page_len <= kv_budget){
    int pos = tx;
    while(pos < page_len){
        sparse_output[pos] = dense_output[pos];
        pos += block_size;
    }
}else{
    int pos = tx;
    while(pos < page_reserved_bos){
        sparse_output[pos] = dense_output[pos];
        pos += block_size;
    }

    pos = tx;
    while(pos < page_reserved_eos){
        int data = token_indices[(page_len-pos-1) * page_size];
        sparse_output[kv_budget-pos-1] = (data / page_size) * (num_kv_heads) + by;
        pos += block_size;
    }

}

}


void PlanDecodeIndicesKernelLauncher(
at::Tensor req_to_token,
at::Tensor req_indices,
at::Tensor cached_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor sparse_kv_indptr,
at::Tensor dense_kv_indices,
at::Tensor sparse_kv_indices,
at::Tensor kv_last_page_len,
const int page_size,
const int num_kv_heads,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

    const int req_to_token_stride = req_to_token.size(1);
    const int batch_size = req_indices.size(0);
    TORCH_CHECK(batch_size == cached_seq_lens.size(0));
    TORCH_CHECK((batch_size * num_kv_heads + 1) == dense_kv_indptr.size(0));
    TORCH_CHECK(
        (batch_size * num_kv_heads + 1) == sparse_kv_indptr.size(0),
        "Mismatch in sizes: batch_size = ", batch_size,
        ", num_kv_heads = ", num_kv_heads,
        ", expected sparse_kv_indptr.size(0) = ", (batch_size * num_kv_heads + 1),
        ", actual = ", sparse_kv_indptr.size(0)
    );


    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(512);
    PlanDecodeIndicesKernel<<<nblks, nthrs>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        req_to_token_stride,
        page_size,
        num_kv_heads,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

}


template <int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void PlanDecodeWorkLoadKernel(
    const int* __restrict__ dense_kv_indptr,
    WorkInfo* __restrict__ work_info,
    PlanInfo* __restrict__ plan_info,
    const int max_chunk_size,
    const int min_chunk_size,
    const int batch_size,
    const int topk_val,
    const int page_reserved_bos,
    const int page_reserved_eos
) {
    const int tx = threadIdx.x;

    using BlockScan = cub::BlockScan<int, NUM_THREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    uint16_t page_count[ITEM_PER_THREAD];
    int chunked_page_count_prefix_sum[ITEM_PER_THREAD + 1];
    int tx_offset = tx * ITEM_PER_THREAD;

    chunked_page_count_prefix_sum[0] = 0;
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        int16_t w = ((tx_offset + i) < batch_size) ? 
            (dense_kv_indptr[tx_offset+i+1] - dense_kv_indptr[tx_offset+i] 
            - page_reserved_bos - page_reserved_eos): 0;
    
        page_count[i] = (w > topk_val) ? w : 0;
        chunked_page_count_prefix_sum[i + 1] =  int((page_count[i] + max_chunk_size - 1) / max_chunk_size);
    }

    BlockScan(temp_storage).InclusiveSum(chunked_page_count_prefix_sum, chunked_page_count_prefix_sum);
    
    if (tx == NUM_THREADS - 1){
        plan_info->num_workload = chunked_page_count_prefix_sum[ITEM_PER_THREAD];
        plan_info->chunk_size = max_chunk_size;
    }
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        if((tx_offset + i) < batch_size){
        const int start = chunked_page_count_prefix_sum[i];
        const int end = chunked_page_count_prefix_sum[i+1];
        int last_len = int(page_count[i] % max_chunk_size);
        if (last_len == 0) last_len = max_chunk_size;
        for (int j = start; j < end; ++j){
                work_info[j].q_idx = tx_offset + i;
                work_info[j].kv_len = (j!=end-1)?(max_chunk_size):(last_len);
                work_info[j].kv_offset = dense_kv_indptr[tx_offset + i] + (j - start) * max_chunk_size + page_reserved_bos;
                work_info[j].score_offset = dense_kv_indptr[tx_offset + i] + (j - start) * max_chunk_size + page_reserved_bos;
        }

        }

    }
}


void PlanDecodeWorkLoadKernelLauncher(
at::Tensor dense_kv_indptr,
WorkInfo*  work_info,
PlanInfo*  plan_info,
const int max_chunk_size,
const int min_chunk_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){


const int batch_size = dense_kv_indptr.size(0) - 1;
TORCH_CHECK(batch_size <= 8192);
dim3 nblks(1);
dim3 nthrs(1024);

if(batch_size <= 1024){
    PlanDecodeWorkLoadKernel<1024, 1><<<nblks, nthrs>>>(
    dense_kv_indptr.data_ptr<int>(),
    work_info,
    plan_info,
    max_chunk_size,
    min_chunk_size,
    batch_size,
    topk_val,
    page_reserved_bos,
    page_reserved_eos
);
} else if (batch_size <= 2048){
    PlanDecodeWorkLoadKernel<1024, 2><<<nblks, nthrs>>>(
    dense_kv_indptr.data_ptr<int>(),
    work_info,
    plan_info,
    max_chunk_size,
    min_chunk_size,
    batch_size,
    topk_val,
    page_reserved_bos,
    page_reserved_eos
);
} else if (batch_size <= 4096){
    PlanDecodeWorkLoadKernel<1024, 4><<<nblks, nthrs>>>(
    dense_kv_indptr.data_ptr<int>(),
    work_info,
    plan_info,
    max_chunk_size,
    min_chunk_size,
    batch_size,
    topk_val,
    page_reserved_bos,
    page_reserved_eos
);
} else if (batch_size <= 8192){
    PlanDecodeWorkLoadKernel<1024, 8><<<nblks, nthrs>>>(
    dense_kv_indptr.data_ptr<int>(),
    work_info,
    plan_info,
    max_chunk_size,
    min_chunk_size,
    batch_size,
    topk_val,
    page_reserved_bos,
    page_reserved_eos
);
}
}