import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import os

from verl.utils.reward_score import default_compute_score

def main():
    output_path = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-dense_rollout/global_step_120/actor/huggingface/math500_output_prompt_len_512_num_pages_16_BLOCK_TOPK.jsonl"
    output_path = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-dense_rollout/global_step_120/actor/huggingface/math500_output_prompt_len_512_num_pages_dense_BLOCK_TOPK.jsonl"
    output_path = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-block-topk-sparse_rollout/global_step_120/actor/huggingface/math500_output_prompt_len_512_num_pages_16_BLOCK_TOPK.jsonl"
    output_path = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-block-topk-sparse_rollout/global_step_120/actor/huggingface/math500_output_prompt_len_512_num_pages_dense_BLOCK_TOPK.jsonl"
    # evaluate
    with open(output_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    total = len(results)
    correct = 0
    for res in results:
        pred = res["pred_output"]
        gold = res["gold_output"]
        score = default_compute_score(
            data_source="HuggingFaceH4/MATH-500",
            solution_str=pred,
            ground_truth=gold,
            extra_info=None,
            sandbox_fusion_url=None,
            concurrent_semaphore=None,
            memory_limit_mb=None,
        )
        if isinstance(score, dict):
            score = score.get("score", 0.0)
        if score >= 1.0:
            correct += 1
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.4f}")
if __name__ == "__main__":
    main()
