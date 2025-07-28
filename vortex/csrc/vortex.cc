#include "vortex.h"

SparseAttentionServer::SparseAttentionServer(
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
    const std::string& algo_name){

        this->model_config.head_dim = head_dim;
        this->model_config.num_kv_heads = num_kv_heads;
        this->model_config.num_qo_heads = num_qo_heads;

        this->engine_config.page_size = page_size;
        this->engine_config.max_batch_size = max_batch_size;
        this->engine_config.num_selected_pages = num_selected_pages;
        this->engine_config.page_reserved_bos = page_reserved_bos;
        this->engine_config.page_reserved_eos = page_reserved_eos;
        this->engine_config.max_seq_lengths = max_seq_lengths;
        this->engine_config.max_prefill_lengths = max_prefill_lengths;
        this->engine_config.max_num_tokens = max_num_tokens;
        this->min_chunk_size = min_chunk_size;
        this->max_chunk_size = max_chunk_size;
        this->max_score_lengths_this_iteration = 0;
        this->algo = parse_sparse_algo(algo_name);
        
        TORCH_CHECK(page_reserved_eos >= 1);
        int maximum_num_pages =  (max_batch_size * max_seq_lengths * num_kv_heads / page_size) + max_batch_size * num_kv_heads;
        int maximum_num_workloads = (maximum_num_pages / min_chunk_size) + max_batch_size * num_kv_heads;

        
        cudaMalloc((void**)&this->score_buffer, maximum_num_pages * sizeof(float));
        cudaMalloc((void**)&this->work_info, maximum_num_workloads * sizeof(WorkInfo));
        cudaMalloc((void**)&this->plan_info, sizeof(PlanInfo));
        cudaMalloc((void**)&this->prefill_batch_table, max_prefill_lengths * sizeof(uint16_t));
        
        this->h_work_info = new WorkInfo[maximum_num_workloads];
        this->h_plan_info = new PlanInfo;


        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        this->num_sms = prop.multiProcessorCount;

        float total_allocated_memory = float(maximum_num_pages * sizeof(float) + maximum_num_workloads * sizeof(WorkInfo) +
            + sizeof(PlanInfo) + max_prefill_lengths * sizeof(uint16_t)) / (1024 * 1024);
        
        
        std::cout << " max_batch_size: " << this->engine_config.max_batch_size << std::endl; 
        std::cout << " page_size: " << this->engine_config.page_size << std::endl; 
        std::cout << " max_seq_lengths: " << this->engine_config.max_seq_lengths << std::endl;
        std::cout << " max_num_tokens: " << this->engine_config.max_num_tokens << std::endl;


        std::cout << " head_dim: " << this->model_config.head_dim << std::endl;
        std::cout << " num_qo_heads: " << this->model_config.num_qo_heads << std::endl;
        std::cout << " num_kv_heads: " << this->model_config.num_kv_heads << std::endl;

        std::cout << " num_selected_pages: " << this->engine_config.num_selected_pages << std::endl;
        std::cout << " page_reserved_bos: " << this->engine_config.page_reserved_bos << std::endl;
        std::cout << " page_reserved_eos: " << this->engine_config.page_reserved_eos << std::endl;
        
        std::cout << " min_chunk_size: " << this->min_chunk_size << std::endl;
        std::cout << " max_chunk_size: " << this->max_chunk_size << std::endl;
        


        std::cout << " Num Streaming Processors: " << this->num_sms << std::endl; 
        std::cout << " maximum_num_pages: " << maximum_num_pages << std::endl;
        std::cout << " maximum_num_workloads: " << maximum_num_workloads << std::endl;
        std::cout << " max_prefill_lengths: " << max_prefill_lengths << std::endl;
        std::cout << " Vortex Allocated: " << total_allocated_memory << " MB " << std::endl; 
        
}       



SparseAttentionServer::~SparseAttentionServer(){
    cudaFree(this->score_buffer);
    cudaFree(this->work_info);
    cudaFree(this->plan_info);
    cudaFree(this->prefill_batch_table);
    delete[] this->h_work_info;
    delete this->h_plan_info;
}


at::Tensor SparseAttentionServer::chunkwise_NH2HN_transpose(
at::Tensor x,
at::Tensor indptr
){
    const int q_len = x.size(0);
    const int batch_size = indptr.size(0) - 1;
    const int num_attention_groups = this->model_config.num_qo_heads / this->model_config.num_kv_heads;
    at::Tensor output = torch::empty(
    {q_len * this->model_config.num_kv_heads, num_attention_groups, this->model_config.head_dim}, x.options()
        );

    ChunkwiseNH2HNTransposeKernelLauncher(
    x,
    indptr,
    output,
    this->prefill_batch_table,
    this->model_config.num_qo_heads,
    this->model_config.num_kv_heads,
    this->model_config.head_dim
    );
        
    return output;
    
}

std::tuple<at::Tensor, at::Tensor> SparseAttentionServer::chunkwise_HN2NH_transpose(
at::Tensor x,
at::Tensor y,
at::Tensor indptr
){
    const int q_len = x.size(0) / this->model_config.num_kv_heads;
    
    at::Tensor x_output = torch::empty(
    {q_len, this->model_config.num_qo_heads, this->model_config.head_dim}, x.options()
        );
    
    at::Tensor y_output = torch::empty(
    {q_len, this->model_config.num_qo_heads}, y.options()
        );

    ChunkwiseHN2NHTransposeKernelLauncher(
        x,
        y,
        indptr,
        x_output,
        y_output,
        this->prefill_batch_table,
        this->model_config.num_qo_heads,
        this->model_config.num_kv_heads,
        this->model_config.head_dim
    );

    return std::make_tuple(x_output, y_output);

}


void SparseAttentionServer::plan_prefill(
at::Tensor cached_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor dense_kv_indices,
at::Tensor input_seq_lens,
at::Tensor qo_indptr_ragged,
at::Tensor qo_indptr_paged,
at::Tensor kv_last_page_len,
at::Tensor req_to_token,
at::Tensor req_indices
){

    TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
    TORCH_CHECK(req_indices.dtype() == torch::kInt64);
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(input_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_ragged.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_paged.dtype() == torch::kInt32);

    PlanPrefillIndptrKernelLauncher(
        cached_seq_lens,
        input_seq_lens,
        dense_kv_indptr,
        qo_indptr_ragged,
        qo_indptr_paged,
        this->model_config.num_kv_heads,
        this->engine_config.page_size
    );

    PlanPrefillIndicesBatchTableKernelLauncher(
        req_to_token,
        req_indices,
        cached_seq_lens,
        input_seq_lens,
        dense_kv_indptr,
        qo_indptr_ragged,
        dense_kv_indices,
        kv_last_page_len,
        this->prefill_batch_table,
        this->engine_config.page_size,
        this->model_config.num_kv_heads
    );

}


void SparseAttentionServer::plan_decode(
at::Tensor cached_seq_lens,
at::Tensor dense_kv_indptr,
at::Tensor dense_kv_indices,
at::Tensor sparse_kv_indptr,
at::Tensor sparse_kv_indices,
at::Tensor kv_last_page_len,
at::Tensor req_to_token,
at::Tensor req_indices
){

TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
TORCH_CHECK(req_indices.dtype() == torch::kInt64);
TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
TORCH_CHECK(sparse_kv_indptr.dtype() == torch::kInt32);
TORCH_CHECK(sparse_kv_indices.dtype() == torch::kInt32);
TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32);
    

this->max_score_lengths_this_iteration = (cached_seq_lens.max().item<int>() + this->engine_config.page_size - 1) / this->engine_config.page_size
    - this->engine_config.page_reserved_bos - this->engine_config.page_reserved_eos;

PlanDecodeIndptrKernelLauncher(
cached_seq_lens,
dense_kv_indptr,
sparse_kv_indptr,
this->model_config.num_kv_heads,
this->engine_config.page_size,
this->engine_config.num_selected_pages,
this->engine_config.page_reserved_bos,
this->engine_config.page_reserved_eos
);


PlanDecodeIndicesKernelLauncher(
req_to_token,
req_indices,
cached_seq_lens,
dense_kv_indptr,
sparse_kv_indptr,
dense_kv_indices,
sparse_kv_indices,
kv_last_page_len,
this->engine_config.page_size,
this->model_config.num_kv_heads,
this->engine_config.num_selected_pages,
this->engine_config.page_reserved_bos,
this->engine_config.page_reserved_eos
);


PlanDecodeWorkLoadKernelLauncher(
dense_kv_indptr,
this->work_info,
this->plan_info,
this->max_chunk_size,
this->min_chunk_size,
this->engine_config.num_selected_pages,
this->engine_config.page_reserved_bos,
this->engine_config.page_reserved_eos
);
}




void SparseAttentionServer::get_sparse_kv_indices(
                at::Tensor query,
                at::Tensor landmarks,
                at::Tensor dense_kv_indptr,
                at::Tensor dense_kv_indices,
                at::Tensor sparse_kv_indptr,
                at::Tensor sparse_kv_indices
){
if (this->algo == SparseAlgorithm::BLOCK_TOPK){
DecodeMHAScoreKernelLauncher(
query,
landmarks,
dense_kv_indices,
this->work_info,
this->plan_info,
this->score_buffer,
this->model_config.head_dim,
this->num_sms
);

// float host_score[10];
// cudaMemcpy(host_score, this->score_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
// printf("BLOCK_TOPK scores: ");
// for (int i = 0; i < 10; i++) {
//     printf("%.4f ", host_score[i]);
// }
// printf("\n");

DecodeTopKKernelLauncher(
this->score_buffer,
dense_kv_indptr,
sparse_kv_indptr,
dense_kv_indices,
sparse_kv_indices,
this->engine_config.num_selected_pages,
this->engine_config.page_reserved_bos,
this->engine_config.page_reserved_eos,
this->max_score_lengths_this_iteration,
query.size(0)
);
}
else if (this->algo == SparseAlgorithm::QUEST){
DecodeQuestMHAScoreKernelLauncher(
query,
landmarks,
dense_kv_indices,
this->work_info,
this->plan_info,
this->score_buffer,
this->model_config.head_dim,
this->num_sms
);

// float host_score[10];
// cudaMemcpy(host_score, this->score_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
// printf("QUEST scores: ");
// for (int i = 0; i < 10; i++) {
//     printf("%.4f ", host_score[i]);
// }
// printf("\n");

DecodeTopKKernelLauncher(
    this->score_buffer,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    this->engine_config.num_selected_pages,
    this->engine_config.page_reserved_bos,
    this->engine_config.page_reserved_eos,
    this->max_score_lengths_this_iteration,
    query.size(0)
    );

}
else if (this->algo == SparseAlgorithm::MANUAL){
DecodeMHAScoreKernelLauncher(
query,
landmarks,
dense_kv_indices,
this->work_info,
this->plan_info,
this->score_buffer,
this->model_config.head_dim,
this->num_sms
);

// float host_score[10];
// cudaMemcpy(host_score, this->score_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
// printf("QUEST scores: ");
// for (int i = 0; i < 10; i++) {
//     printf("%.4f ", host_score[i]);
// }
// printf("\n");

DecodeTopKKernelLauncher(
    this->score_buffer,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    this->engine_config.num_selected_pages,
    this->engine_config.page_reserved_bos,
    this->engine_config.page_reserved_eos,
    this->max_score_lengths_this_iteration,
    query.size(0)
    );

}

}


PYBIND11_MODULE(vortex_C, m) {
    py::class_<SparseAttentionServer>(m, "SparseAttentionServer")
    .def(py::init<
        int, int, int, int, int,
        int, int, int, int, int,
        int, int, int, std::string>(),
        py::arg("head_dim"),
        py::arg("num_kv_heads"),
        py::arg("num_qo_heads"),
        py::arg("page_size"),
        py::arg("max_batch_size"),
        py::arg("max_seq_lengths"),
        py::arg("max_prefill_lengths"),
        py::arg("max_num_tokens"),
        py::arg("min_chunk_size"),
        py::arg("max_chunk_size"),
        py::arg("num_selected_pages"),
        py::arg("page_reserved_bos"),
        py::arg("page_reserved_eos"),
        py::arg("algo_name")
    )
    .def("chunkwise_NH2HN_transpose", &SparseAttentionServer::chunkwise_NH2HN_transpose,
         py::arg("x"), py::arg("indptr"))
    .def("chunkwise_HN2NH_transpose", &SparseAttentionServer::chunkwise_HN2NH_transpose,
         py::arg("x"), py::arg("y"), py::arg("indptr"))
    .def("plan_prefill", &SparseAttentionServer::plan_prefill,
         py::arg("cached_seq_lens"),
         py::arg("dense_kv_indptr"),
         py::arg("dense_kv_indices"),
         py::arg("input_seq_lens"),
         py::arg("qo_indptr_ragged"),
         py::arg("qo_indptr_paged"),
         py::arg("kv_last_page_len"),
         py::arg("req_to_token"),
         py::arg("req_indices"))
    .def("plan_decode", &SparseAttentionServer::plan_decode,
         py::arg("cached_seq_lens"),
         py::arg("dense_kv_indptr"),
         py::arg("dense_kv_indices"),
         py::arg("sparse_kv_indptr"),
         py::arg("sparse_kv_indices"),
         py::arg("kv_last_page_len"),
         py::arg("req_to_token"),
         py::arg("req_indices"))
    .def("get_sparse_kv_indices", &SparseAttentionServer::get_sparse_kv_indices,
         py::arg("query"),
         py::arg("landmarks"),
         py::arg("dense_kv_indptr"),
         py::arg("dense_kv_indices"),
         py::arg("sparse_kv_indptr"),
         py::arg("sparse_kv_indices"));

}
