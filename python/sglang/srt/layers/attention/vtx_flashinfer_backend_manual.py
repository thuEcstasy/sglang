from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

from operator import truediv
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.mem_cache.vtx_memory_pool import VTXTokenToKVPool
from sglang.srt.mem_cache.vtx_memory_pool_manual import VTXTokenToKVPoolManual
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state

from vortex import SparseAttentionServer

@dataclass
class DecodeMetadata:
    use_sparsity: bool


@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VTXFlashInferAttnBackendManual(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.decode_use_tensor_cores = True
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 1
        self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.num_attn_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        
        assert self.q_data_type == torch.bfloat16
        assert self.data_type == torch.bfloat16
        
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.layers_skip =  model_runner.server_args.vortex_layers_skip
        
        self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                ),
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                ),
            ]
        
        self.kv_indices = [
                torch.zeros(
                    (
                        (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1),), 
                        dtype=torch.int32, device=model_runner.device
                ),
                torch.zeros(
                    (
                        (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1),), 
                        dtype=torch.int32, device=model_runner.device
                ),
            ]
        
        self.kv_last_page_len = [
                torch.ones(
                (max_bs * self.num_kv_heads,), dtype=torch.int32, device=model_runner.device
            ),
                torch.ones(
                (max_bs * self.num_kv_heads,), dtype=torch.int32, device=model_runner.device
            )
        ]
        
        self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                ),
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                ),
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                    )
    
        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )           
            ]
        
         
        self.vtx_api = SparseAttentionServer(
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                num_qo_heads=self.num_qo_heads,
                page_size=model_runner.server_args.page_size,
                max_batch_size=max_bs,
                max_seq_lengths=model_runner.model_config.context_len,
                max_prefill_lengths=model_runner.server_args.max_prefill_tokens,
                max_num_tokens=model_runner.max_total_num_tokens,
                min_chunk_size=8,
                max_chunk_size=32,
                num_selected_pages=model_runner.server_args.vortex_num_selected_pages,
                page_reserved_bos=model_runner.server_args.vortex_page_reserved_bos, 
                page_reserved_eos=model_runner.server_args.vortex_page_reserved_eos,
                algo_name=model_runner.server_args.vortex_sparse_attention_algorithm
        )
        
        
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        
        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()
        
        if forward_batch.forward_mode.is_decode_or_idle():
            
            bs = len(forward_batch.req_pool_indices)
            self.vtx_api.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                dense_kv_indptr=self.kv_indptr[1][:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices[1],
                sparse_kv_indptr=self.kv_indptr[0][:bs*self.num_kv_heads+1],
                sparse_kv_indices=self.kv_indices[0],
                kv_last_page_len=self.kv_last_page_len[1][:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices
            )
            
            self.decode_wrappers[0].plan(
                indptr=self.kv_indptr[0][:bs*self.num_kv_heads+1],
                indices=self.kv_indices[0],
                last_page_len=self.kv_last_page_len[1][:bs*self.num_kv_heads],
                num_qo_heads=self.num_attn_groups,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_wrappers[1].plan(
                indptr=self.kv_indptr[1][:bs*self.num_kv_heads+1],
                indices=self.kv_indices[1],
                last_page_len=self.kv_last_page_len[1][:bs*self.num_kv_heads],
                num_qo_heads=self.num_attn_groups,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
    
            self.forward_metadata = DecodeMetadata(use_sparsity=True)
            
        
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            self.vtx_api.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr[1][:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices[1],
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len[0][:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices
            )

            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs+1],
                self.qo_indptr[0][:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            
            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indices[1],
                self.kv_last_page_len[0][:bs*self.num_kv_heads],
                self.num_attn_groups,
                1,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=False,
            )
            

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        raise NotImplementedError
    
    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        raise NotImplementedError

    def get_cuda_graph_seq_len_fill_value(self):
        
        raise NotImplementedError

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        assert isinstance(forward_batch.token_to_kv_pool, VTXTokenToKVPoolManual)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        
        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        if self.forward_metadata.extend_no_prefix:
            o = self.prefill_wrapper_ragged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            
            q_t = self.vtx_api.chunkwise_NH2HN_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0]
            )
            
            
            
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = self.vtx_api.chunkwise_HN2NH_transpose(
                o2, s2, self.qo_indptr[0]
            )
            
            o, _ = merge_state(o1, s1, o2_t, s2_t)


        if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):  
        assert isinstance(forward_batch.token_to_kv_pool, VTXTokenToKVPoolManual)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        
        use_sparsity = (self.forward_metadata.use_sparsity) and (layer.layer_id not in self.layers_skip)
        
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
        k, v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k = k.view(-1, self.page_size, 1, self.head_dim)
        v = v.view(-1, self.page_size, 1, self.head_dim)
        
        if use_sparsity:            
            q_compress = q.contiguous().view(-1, self.num_attn_groups, layer.head_dim).sum(dim=-2)
            landmarks = forward_batch.token_to_kv_pool.get_landmark_buffer(layer.layer_id)
            self.vtx_api.get_sparse_kv_indices(
                query=q_compress,
                landmarks=landmarks.view(-1, layer.head_dim),
                dense_kv_indptr=self.kv_indptr[1],
                dense_kv_indices=self.kv_indices[1],
                sparse_kv_indptr=self.kv_indptr[0],
                sparse_kv_indices=self.kv_indices[0]
            )
            
            self.decode_wrappers[0]._paged_kv_indices_buf = self.kv_indices[0]
            o = self.decode_wrappers[0].forward(
                q.contiguous().view(-1, self.num_attn_groups, layer.head_dim),
                (k, v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
            o_dense = self.decode_wrappers[1].forward(
                q.contiguous().view(-1, self.num_attn_groups, layer.head_dim),
                (k, v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
            v_output_dir = f"layer_{layer.layer_id}_dense"
            o_manual = self.forward_manual(q, k, v, self.decode_wrappers[1]._paged_kv_indices_buf, self.decode_wrappers[1]._paged_kv_indptr_buf, self.decode_wrappers[0]._paged_kv_indices_buf, self.decode_wrappers[0]._paged_kv_indptr_buf, forward_batch.seq_lens, v_output_dir)
            # print(o_dense.view(-1, layer.tp_q_head_num * layer.head_dim), o.view(-1, layer.tp_q_head_num * layer.head_dim), o_manual, flush=True)
            # compute MSE error between o and o_dense, using 4 digits after the decimal point
            print(forward_batch.seq_lens, flush=True)
            mse = torch.mean((o - o_dense) ** 2)
            denominator = torch.mean(o_dense ** 2)
            rel_error = mse / denominator
            # print(f"******", flush=True)
            # print(f"MSE error between o and o_dense: {mse.item():.10f}", flush=True)
            # print(f"Relative error between o and o_dense: {rel_error.item():.10f}", flush=True)
            # print(f"******", flush=True)
            
        else:
            o = self.decode_wrappers[1].forward(
                q.contiguous().view(-1, self.num_attn_groups, layer.head_dim),
                (k, v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
    def forward_manual(
        self,
        q: torch.Tensor,         # [1, D]
        k: torch.Tensor,         # [N, head_dim]
        v: torch.Tensor,         # [N, head_dim]
        indices: torch.Tensor,   # CSR index
        indptr: torch.Tensor,     # CSR ptr
        sparse_indices: torch.Tensor,  # CSR index for sparse KV
        sparse_indptr: torch.Tensor,   # CSR ptr for sparse KV
        seq_lens: torch.Tensor,  # [1]
        v_output_dir: str,
    ):
        head_dim = v.shape[-1]
        k = k.view(-1, head_dim)
        v = v.view(-1, head_dim)
        assert q.shape[0] == 1, "Only supports single-token decoding"
        assert q.shape[1] == self.num_qo_heads * head_dim

        # Reshape q -> [num_qo_heads, head_dim]
        q = q.view(self.num_qo_heads, head_dim)

        outputs = []

        for kv_head_id in range(self.num_kv_heads):
            start, end = indptr[kv_head_id], indptr[kv_head_id + 1]
            page_indices = indices[start:end]  # [num_pages]
    
            token_indices = torch.cat([
                torch.arange(p * self.page_size, (p + 1) * self.page_size, device=q.device)
                for p in page_indices
            ])  # [num_pages * page_size]
            
            sparse_start, sparse_end = sparse_indptr[kv_head_id], sparse_indptr[kv_head_id + 1]
            sparse_page_indices = sparse_indices[sparse_start:sparse_end]  # [num_sparse
            sparse_token_indices = torch.cat([
                torch.arange(p * self.page_size, (p + 1) * self.page_size, device=q.device)
                for p in sparse_page_indices
            ])  # [num_sparse * page_size]

            k_i = k[token_indices]  # [N_i, D]
            v_i = v[token_indices]  # [N_i, D]
    
            if seq_lens.item() == 4000:
                import time
                timestamp = int(time.time() * 1000)  # 毫秒级时间戳
                q_output_file = os.path.join(v_output_dir, f"q_head_{kv_head_id}_{timestamp}.pt")
                k_output_file = os.path.join(v_output_dir, f"k_head_{kv_head_id}_{timestamp}.pt")
                v_output_file = os.path.join(v_output_dir, f"v_head_{kv_head_id}_{timestamp}.pt")
                # print all inner product if seq_len = 4000
                if not os.path.exists(v_output_dir):
                    os.makedirs(v_output_dir, exist_ok=True)
                torch.save(q, q_output_file)
                torch.save(k_i, k_output_file)
                torch.save(v_i, v_output_file)
            
            token_indices_cpu = token_indices.cpu().tolist()
            sparse_token_indices_cpu = set(sparse_token_indices.cpu().tolist())

            # mask是长度N_i的bool列表，True表示对应token_indices的token是sparse的
            mask_list = [idx in sparse_token_indices_cpu for idx in token_indices_cpu]

            # 转成tensor，放回GPU设备
            mask = torch.tensor(mask_list, dtype=torch.bool, device=token_indices.device)  # [N_i]
    
            start_q = kv_head_id * self.num_attn_groups
            end_q = (kv_head_id + 1) * self.num_attn_groups
            q_group = q[start_q:end_q]         # [num_attn_groups, D]
            
            q_group = q_group.to(torch.float64)
            k_i = k_i.to(torch.float64)
            v_i = v_i.to(torch.float64)
            
            attn_scores = torch.matmul(q_group, k_i.T) / (head_dim ** 0.5)   # [num_attn_groups, N_i]
            attn_weights = torch.softmax(attn_scores, dim=-1)               # [num_attn_groups, N_i]
            
            mask_expanded = mask.unsqueeze(0).expand(attn_weights.shape[0], -1)  # [num_attn_groups, N_i]

            sparse_attn_weights = torch.zeros_like(attn_weights)

            sparse_attn_weights[mask_expanded] = attn_weights[mask_expanded]
                
            # compute the sum of attention weights for each group
            sparse_attn_cumsum = torch.sum(sparse_attn_weights, dim=-1)  # [num_attn_groups]
            dense_attn_cumsum = torch.sum(attn_weights, dim=-1)
            
            # compute the theoretical attn cumsum by selecting top-k tokens
            # print(seq_lens, flush=True)
            # print((3 + 16) * self.page_size + seq_lens % 16, flush=True)
            
            
            topk = min((3 + 16) * self.page_size + seq_lens % 16 , attn_weights.shape[-1])  # or choose top-k manually
            # print("Choosing", topk.item(), flush=True)
            topk_mask = torch.zeros_like(attn_weights)
            topk_scores, topk_indices = torch.topk(attn_weights, k=topk, dim=-1)
            topk_mask.scatter_(dim=-1, index=topk_indices, value=1.0)
            attn_weights_topk = attn_weights * topk_mask
            topk_attn_cumsum = torch.sum(attn_weights_topk, dim=-1)
            print(sparse_attn_cumsum, topk_attn_cumsum, flush=True)
            
            # print(attn_weights, flush=True)
            o_i = torch.matmul(attn_weights, v_i)                           # [num_attn_groups, D]

            o_i = o_i.to(torch.bfloat16)

            outputs.append(o_i)

        o = torch.cat(outputs, dim=0)
        o = o.view(1, -1)
        return o



