#pragma once

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include <optional>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

            
enum SparseAlgorithm {
    DENSE,
    STREAMING_LLM,
    BLOCK_TOPK,
    QUEST,
    NSA,
    MANUAL
};


inline SparseAlgorithm parse_sparse_algo(const std::string& name) {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::toupper);
    if (n == "DENSE") return SparseAlgorithm::DENSE;
    if (n == "STREAMING_LLM") return SparseAlgorithm::STREAMING_LLM;
    if (n == "BLOCK_TOPK") return SparseAlgorithm::BLOCK_TOPK;
    if (n == "QUEST") return SparseAlgorithm::QUEST;
    if (n == "NSA") return SparseAlgorithm::NSA;
    if (n == "MANUAL") return SparseAlgorithm::MANUAL;
    return SparseAlgorithm::DENSE;
}

inline int next_power_of_2(int n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

struct ModelConfig{

        int head_dim;
        int num_kv_heads;
        int num_qo_heads;
        
        //init function
        ModelConfig(int head_dim = 128, int num_kv_heads = 8, int num_qo_heads = 32)
        : head_dim(head_dim), num_kv_heads(num_kv_heads), num_qo_heads(num_qo_heads) {}
};


struct EngineConfig{

        int page_size;
        int max_batch_size;
        int max_seq_lengths;
        int max_prefill_lengths;
        int max_num_tokens;
        int num_selected_pages;
        int page_reserved_bos;
        int page_reserved_eos;
        //init function
        EngineConfig(int page_size = 16, int max_batch_size = 4097, int max_seq_lengths = 32768, int max_prefill_lengths = 32768,
        int max_num_tokens = 1048576, int num_selected_pages = 16, int page_reserved_bos = 1, int page_reserved_eos = 1)
        : page_size(page_size), max_batch_size(max_batch_size), max_seq_lengths(max_seq_lengths), max_prefill_lengths(max_prefill_lengths), max_num_tokens(max_num_tokens), num_selected_pages(num_selected_pages), page_reserved_bos(page_reserved_bos), page_reserved_eos(page_reserved_eos){}
};


struct __align__(16) WorkInfo {
    int score_offset;  // 4 bytes
    int kv_offset;     // 4 bytes
    int q_idx;         // 4 bytes
    int kv_len;        // 4 bytes

    __host__ __device__
    WorkInfo(int kv_offset = 0, uint16_t q_idx = 0, uint16_t kv_len = 0)
        : kv_offset(kv_offset), q_idx(q_idx), kv_len(kv_len) {}
};


struct __align__(8) PlanInfo {
    int num_workload;     // 4 bytes
    int chunk_size;         // 4 bytes

    __host__ __device__
    PlanInfo(int num_workload = 0, int chunk_size = 0)
        : num_workload(num_workload), chunk_size(chunk_size) {}
};


void PlanPrefillIndptrKernelLauncher(
at::Tensor cached_seq_lens,
at::Tensor input_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor qo_indptr_ragged,
at::Tensor qo_indptr_paged,
const int num_kv_heads,
const int page_size
);

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
);

void ChunkwiseNH2HNTransposeKernelLauncher(
at::Tensor x,
at::Tensor indptr,
at::Tensor output,
const uint16_t* batch_table,
const int num_qo_heads,
const int num_kv_heads,
const int head_dim
);


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
);

void PlanDecodeIndptrKernelLauncher(
at::Tensor cached_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor sparse_kv_indptr,
const int num_kv_heads,
const int page_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
);

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
);

void PlanDecodeWorkLoadKernelLauncher(
at::Tensor dense_kv_indptr,
WorkInfo*  work_info,
PlanInfo*  plan_info,
const int max_chunk_size,
const int min_chunk_size,
const int kv_budget,
const int page_reserved_bos,
const int page_reserved_eos
);

void DecodeMHAScoreKernelLauncher(
at::Tensor query,
at::Tensor landmarks,
at::Tensor dense_kv_indices,
const WorkInfo* work_info,
const PlanInfo* plan_info,
float* score,
const int head_dim,
const int num_sms
);

void DecodeQuestMHAScoreKernelLauncher(
at::Tensor query,
at::Tensor landmarks,
at::Tensor dense_kv_indices,
const WorkInfo* work_info,
const PlanInfo* plan_info,
float* score,
const int head_dim,
const int num_sms
);

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
);




class SparseAttentionServer{

        public:
            SparseAttentionServer(
                int head_dim,
                int num_kv_heads,
                int num_qo_heads,
                int page_size,
                int max_batch_size,
                int max_seq_lengths,
                int max_prefill_lengths,
                int max_num_tokens,
                int min_chunk_size,
                int max_chunk_size,
                int num_selected_pages,
                int page_reserved_bos, 
                int page_reserved_eos,
                const std::string& algo_name
            );
            ~SparseAttentionServer();

            void get_sparse_kv_indices(
                at::Tensor query,
                at::Tensor landmarks,
                at::Tensor dense_kv_indptr,
                at::Tensor dense_kv_indices,
                at::Tensor sparse_kv_indptr,
                at::Tensor sparse_kv_indices
            );
            at::Tensor chunkwise_NH2HN_transpose(
                at::Tensor x,
                at::Tensor indptr
                );

            std::tuple<at::Tensor, at::Tensor> chunkwise_HN2NH_transpose(
                at::Tensor x,
                at::Tensor y,
                at::Tensor indptr
            );

            void plan_prefill(
                at::Tensor cached_seq_lens,
                at::Tensor dense_kv_indptr,
                at::Tensor dense_kv_indices,
                at::Tensor input_seq_lens,
                at::Tensor qo_indptr_ragged,
                at::Tensor qo_indptr_paged,
                at::Tensor kv_last_page_len,
                at::Tensor req_to_token,
                at::Tensor req_indices
            );
            
            void plan_decode(
                at::Tensor cached_seq_lens,
                at::Tensor dense_kv_indptr,
                at::Tensor dense_kv_indices,
                at::Tensor sparse_kv_indptr,
                at::Tensor sparse_kv_indices,
                at::Tensor kv_last_page_len,
                at::Tensor req_to_token,
                at::Tensor req_indices
            );

        private:
            ModelConfig model_config;
            EngineConfig engine_config;
            int min_chunk_size;
            int max_chunk_size;
            float* score_buffer;
            uint16_t* prefill_batch_table;
            int max_score_lengths_this_iteration;
            int num_sms;
            SparseAlgorithm algo;
            WorkInfo* work_info;
            PlanInfo* plan_info;
            WorkInfo* h_work_info;
            PlanInfo* h_plan_info;
};


