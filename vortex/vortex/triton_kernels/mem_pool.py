import torch
import triton
import triton.language as tl

@triton.jit
def set_kv_buffer_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr)
    src_v = tl.load(new_v + src_ptr)
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, src_k)
    tl.store(dst_v_ptr, src_v)


def set_kv_buffer_launcher(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]
    
    set_kv_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache,
        v_cache,
        new_k,
        new_v,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )


@triton.jit
def update_landmark_buffer_kernel(
    k_cache,
    landmark,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE !=0:
        return
    
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    k_cache_offset = position_trans + 1 - PAGE_SIZE
    dim = tl.arange(0, HEAD_DIM)
    page = tl.arange(0, PAGE_SIZE)
    src_ptr = k_cache + k_cache_offset * HEAD_DIM + page[:,None] * HEAD_DIM + dim
    page_block = tl.load(src_ptr)                         # [PAGE_SIZE, HEAD_DIM]
    lmk = tl.sum(page_block, axis=0)
    lmk = lmk.to(tl.bfloat16)  
    
    dst_ptr = landmark + ((token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id) * HEAD_DIM + dim
    tl.store(dst_ptr, lmk)

def update_landmark_launcher(
    k_cache: torch.Tensor,
    landmark: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    num_kv_head: int,
    head_dim: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_head
    HEAD_DIM = head_dim
    
    update_landmark_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        k_cache,
        landmark,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )

@triton.jit
def update_landmark_buffer_kernel_quest(
    k_cache,
    landmark,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return
    
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position % PAGE_SIZE
    
    k_cache_offset = position_trans + 1 - PAGE_SIZE
    dim = tl.arange(0, HEAD_DIM)
    page = tl.arange(0, PAGE_SIZE)
    src_ptr = k_cache + k_cache_offset * HEAD_DIM + page[:,None] * HEAD_DIM + dim
    page_block = tl.load(src_ptr)                         # [PAGE_SIZE, HEAD_DIM]
    
    max_vals = tl.max(page_block, axis=0)  # [HEAD_DIM]
    min_vals = tl.min(page_block, axis=0)  # [HEAD_DIM]
    
    max_vals = max_vals.to(tl.bfloat16)
    min_vals = min_vals.to(tl.bfloat16)
    
    # [num_blocks, 2, head_dim]
    block_offset = ((token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id) * 2
    dst_ptr_max = landmark + block_offset * HEAD_DIM + dim
    dst_ptr_min = landmark + (block_offset + 1) * HEAD_DIM + dim
    
    tl.store(dst_ptr_max, max_vals)
    tl.store(dst_ptr_min, min_vals)

def update_landmark_launcher_quest(
    k_cache: torch.Tensor,
    landmark: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    num_kv_head: int,
    head_dim: int
):
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_head
    HEAD_DIM = head_dim
    
    update_landmark_buffer_kernel_quest[(NNZ, NUM_KV_HEAD)](
        k_cache,
        landmark,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )