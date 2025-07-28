import torch
from vortex import SparseAttentionServer, set_kv_buffer_launcher, update_landmark_launcher

vortex_server = SparseAttentionServer(
    head_dim = 128,
    num_kv_heads = 8,
    num_qo_heads = 64,
    page_size = 16,
    max_batch_size = 1024,
    max_seq_lengths = 32768,
    max_prefill_lengths = 32768,
    max_num_tokens = 1024576,
    min_chunk_size = 8,
    max_chunk_size = 32,
    num_selected_pages = 32,
    page_reserved_bos = 1, 
    page_reserved_eos = 1,
    algo_name = "BLOCK_TOPK"
)

