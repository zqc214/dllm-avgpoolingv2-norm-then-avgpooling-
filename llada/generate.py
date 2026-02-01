# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx






def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens







def get_keep_indices_by_avgpool(key_states, keep_ratio, kernel_size=3):
    """
    [修改版] 先 Norm 再 AvgPool 的筛选策略。
    key_states: (Batch, Num_Heads, Seq_Len, Head_Dim)
    """
    if key_states.shape[2] == 0:
        return torch.empty(key_states.shape[0], 0, device=key_states.device, dtype=torch.long)

    # 1. 预处理：聚合多头 (B, H, L, D) -> (B, L, D)
    # 我们可以先对 Head 维度取平均，或者也可以先算每个 Head 的 Norm 再平均
    # 这里保持和你原逻辑一致：先融合多头特征
    x = key_states.mean(dim=1) # (B, L, D)

    # --- [核心修改开始] ---

    # 2. 先计算能量 (L2 Norm)
    # 此时 x 还是原始的 Key 向量，包含了 RoPE 旋转
    # 但 Norm 操作是旋转不变的，所以这一步完美提取了每个 Token 的“纯粹力度”
    # energy shape: (B, L)
    energy = torch.norm(x, p=2, dim=-1)

    # 3. 再进行平均池化 (平滑能量包络)
    # energy 需要扩展维度才能进 avg_pool1d: (B, L) -> (B, 1, L)
    energy = energy.unsqueeze(1)

    pad = kernel_size // 2
    
    # 对能量曲线进行低通滤波
    # 这会平滑掉那些孤立的噪声峰值，保留连续的高能量区域
    # output shape: (B, 1, L)
    smoothed_energy = F.avg_pool1d(
        energy, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=pad, 
        count_include_pad=False
    )
    
    # 还原形状: (B, 1, L) -> (B, L)
    scores = smoothed_energy.squeeze(1)

    # --- [核心修改结束] ---
    
    # 4. Top-K 筛选 (保持不变)
    L = key_states.shape[2]
    k_keep = max(1, int(L * keep_ratio))
    
    _, indices = torch.topk(scores, k=k_keep, dim=1)
    
    # 5. 必须排序
    indices, _ = torch.sort(indices, dim=1)
    
    return indices

# llada/generate.py

def prune_dual_kv_cache(past_key_values, current_start, current_end, prefix_ratio=0.5, suffix_ratio=0.5,
                        keep_first_n=0, keep_last_n=0, prompt_length=0):
    """
    对 Dual Cache 进行分区剪枝，并剔除 Current Block。
    
    Args:
        keep_first_n: 保留 prompt 的前 N 个 token（不参与剪枝）
        keep_last_n: 保留当前 block 前一个 block 的最后 N 个 token（不参与剪枝）
        prompt_length: prompt 的长度
    """
    layer_idx = len(past_key_values) // 2
    sample_key = past_key_values[layer_idx][0] # (B, H, Total_Len, D)
    
    B, H, Total_Len, D = sample_key.shape
    device = sample_key.device
    
    # [Prefix: 0~s] | [Current: s~e] | [Suffix: e~end]
    prefix_keys = sample_key[:, :, :current_start, :]
    suffix_keys = sample_key[:, :, current_end:, :]
    
    # ============================================================
    # [新增] 确定需要保护的 token 范围
    # ============================================================
    # 1. Prompt 的前 N 个 token（全局坐标）
    protected_first_start = 0
    protected_first_end = min(keep_first_n, prompt_length)
    
    # 2. Current block 前面的最后 N 个 token（全局坐标）
    #    - 如果是第一个 block，这部分就是 prompt 的最后 N 个
    #    - 否则是前一个 block 的最后 N 个
    protected_last_start = max(0, current_start - keep_last_n)
    protected_last_end = current_start
    
    # 生成保护的索引（全局坐标）
    protected_indices_list = []
    
    # 添加前 N 个 token
    if protected_first_end > protected_first_start:
        first_n_indices = torch.arange(protected_first_start, protected_first_end, device=device).unsqueeze(0).expand(B, -1)
        protected_indices_list.append(first_n_indices)
    
    # 添加最后 N 个 token（在 current block 之前）
    if protected_last_end > protected_last_start:
        last_n_indices = torch.arange(protected_last_start, protected_last_end, device=device).unsqueeze(0).expand(B, -1)
        protected_indices_list.append(last_n_indices)
    
    # ============================================================
    # [修改] 对 prefix 和 suffix 分别进行剪枝（排除保护的 token）
    # ============================================================
    
    # 分别计算保留索引（这些函数会在所有 token 上计算能量，包括保护的）
    prefix_indices = get_keep_indices_by_avgpool(prefix_keys, prefix_ratio) 
    suffix_indices = get_keep_indices_by_avgpool(suffix_keys, suffix_ratio)
    
    # 加上偏移量，转换成全局坐标
    suffix_indices = suffix_indices + current_end 
    
    # ============================================================
    # [新增] 将保护的索引添加到最终的保留列表中
    # ============================================================
    indices_to_keep = []
    
    # 1. 添加保护的 token
    if protected_indices_list:
        indices_to_keep.extend(protected_indices_list)
    
    # 2. 添加剪枝后保留的 prefix token
    if prefix_indices.shape[1] > 0:
        indices_to_keep.append(prefix_indices)
    
    # 3. 添加剪枝后保留的 suffix token
    if suffix_indices.shape[1] > 0:
        indices_to_keep.append(suffix_indices)
    
    # 合并所有保留的索引
    if indices_to_keep:
        global_indices = torch.cat(indices_to_keep, dim=1)
        
        # 去重（因为保护的 token 可能与剪枝保留的重复）并排序
        # 使用 unique 来去重
        global_indices_list = []
        for b in range(B):
            unique_indices = torch.unique(global_indices[b])
            # 排序
            unique_indices, _ = torch.sort(unique_indices)
            global_indices_list.append(unique_indices)
        
        # 找到最大长度，进行 padding（用 -1 标记）
        max_len = max(idx.shape[0] for idx in global_indices_list)
        global_indices = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        for b in range(B):
            global_indices[b, :global_indices_list[b].shape[0]] = global_indices_list[b]
    else:
        # 如果没有任何保留的 token，返回空
        global_indices = torch.empty((B, 0), dtype=torch.long, device=device)
    
    # ============================================================
    # 执行物理剪枝（处理 padding）
    # ============================================================
    new_past_key_values = []
    
    if global_indices.shape[1] > 0:
        # 创建有效 mask（-1 表示 padding）
        valid_mask = global_indices >= 0  # (B, max_len)
        
        # 为每个 batch 单独处理
        batch_results = []
        for b in range(B):
            valid_idx = global_indices[b][valid_mask[b]]  # 去除 padding
            batch_results.append(valid_idx)
        
        # 再次找到最大长度并 padding
        max_valid_len = max(idx.shape[0] for idx in batch_results)
        padded_indices = torch.zeros((B, max_valid_len), dtype=torch.long, device=device)
        for b in range(B):
            padded_indices[b, :batch_results[b].shape[0]] = batch_results[b]
        
        gather_idx = padded_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, max_valid_len, D)
        
        for layer in past_key_values:
            k, v = layer 
            k_pruned = torch.gather(k, 2, gather_idx)
            v_pruned = torch.gather(v, 2, gather_idx)
            new_past_key_values.append((k_pruned, v_pruned))
    else:
        # 空的情况
        for layer in past_key_values:
            k, v = layer
            k_empty = torch.empty((B, H, 0, D), dtype=k.dtype, device=device)
            v_empty = torch.empty((B, H, 0, D), dtype=v.dtype, device=device)
            new_past_key_values.append((k_empty, v_empty))
        
    return tuple(new_past_key_values)







@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    prefix_keep_ratio=0.5, suffix_keep_ratio=0.5,
    keep_first_n=0, keep_last_n=0
):
    """
    Args:
        keep_first_n: 保留 prompt 的前 N 个 token（不参与剪枝）
        keep_last_n: 保留当前 block 前面最近的 N 个 token（不参与剪枝）
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    # [新增] 初始化全局计时累加器 (单位: 毫秒)
    total_prune_time = 0.0
    total_model_time = 0.0
    
    # [新增] 创建 Event 对象
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # ============================================================
        # [修改] 测量主要的模型推理时间 (Block Warmup)
        # 这通常是计算量最大的一步，也是享受上一轮剪枝红利的地方
        # ============================================================
        t0.record()

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        
        t1.record()
        torch.cuda.synchronize()
        total_model_time += t0.elapsed_time(t1)
        
        past_key_values = out_full.past_key_values
        nfe += 1

        # ============================================================
        # 1. 在剪枝前：获取并打印原始形状
        # ============================================================
        # 我们只在主进程 (Rank 0) 且第 0 个 Block 打印，避免刷屏
        if nb == 0 and int(os.environ.get("RANK", 0)) == 0:
            # 获取形状并赋值给变量
            shape_before = past_key_values[0][0].shape 
            print(f"[DEBUG] Block {nb} Before Pruning: {shape_before}")

        # ============================================================
        # [修改] Token Pruning 逻辑与计时
        # ============================================================
        # 只有当有东西可以剪的时候才执行
        if past_key_values[0][0].shape[2] > block_length:

            t0.record() # 开始计时
            
            past_key_values = prune_dual_kv_cache(
                past_key_values, 
                current_start=s, 
                current_end=e, 
                prefix_ratio=prefix_keep_ratio, 
                suffix_ratio=suffix_keep_ratio,
                keep_first_n=keep_first_n,
                keep_last_n=keep_last_n,
                prompt_length=Lp
            )

            t1.record() # 结束计时
            torch.cuda.synchronize() # 等待 GPU 执行完
            total_prune_time += t0.elapsed_time(t1) # 累加时间

            # ============================================================
            # 3. 在剪枝后：获取并打印新形状 (移入 if 内部)
            # ============================================================
            if nb == 0 and int(os.environ.get("RANK", 0)) == 0:
                # 获取新的形状
                shape_after = past_key_values[0][0].shape
                print(f"[DEBUG] Block {nb} After  Pruning: {shape_after}")
                
                # 计算剪掉了多少
                reduced = shape_before[2] - shape_after[2]
                print(f"[DEBUG] Pruned {reduced} tokens in total.")
             
        # [修改结束] 此时 past_key_values 的长度变短了 (物理减少显存)
        # ============================================================

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            # ============================================================
            # [修改] 测量 Refinement Loop 中的模型推理时间
            # ============================================================
            t0.record() # 开始计时

            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            t1.record() # 结束计时
            torch.cuda.synchronize()
            total_model_time += t0.elapsed_time(t1)

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    # [新增] 在函数返回前打印统计信息
    # 只在主进程打印，且打印一次即可
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"\n[Performance Profile]")
        print(f"Total Pruning Time: {total_prune_time:.2f} ms")
        print(f"Total Model   Time: {total_model_time:.2f} ms")
        if total_model_time > 0:
            ratio = (total_prune_time / total_model_time) * 100
            print(f"Pruning Overhead: {ratio:.2f}% relative to Inference")

    return x, nfe




def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
