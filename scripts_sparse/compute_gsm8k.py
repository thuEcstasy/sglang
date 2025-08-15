import json
import re

json_file = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-dense_rollout/global_step_200/gsm8k_output_prompt_len_4096_num_pages_16_BLOCK_TOPK.jsonl"
json_file = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-dense_rollout/global_step_200/gsm8k_output_prompt_len_4096_num_pages_dense_BLOCK_TOPK.jsonl"
json_file = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-block-topk-sparse_rollout/global_step_200/gsm8k_output_prompt_len_4096_num_pages_16_BLOCK_TOPK.jsonl"
json_file = "/home/haizhonz/Zhaofeng/sglang/output/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-block-topk-sparse_rollout/global_step_200/gsm8k_output_prompt_len_4096_num_pages_dense_BLOCK_TOPK.jsonl"

json_file = "/home/haizhonz/Zhaofeng/sglang/output/Qwen/Qwen2.5-Math-1.5B/gsm8k_output_prompt_len_4096_num_pages_dense_BLOCK_TOPK.jsonl"

def extract_last_boxed(s):
    numbers = re.findall(r'\d+(?:\.\d+)?', s)
    if numbers:
        return numbers[-1]  # 最后一个数字通常是答案
    return None

def extract_answer_after_hashes(s):
    # 提取所有以 #### 开头的行
    matches = re.findall(r'####\s*([^\n]*)', s)
    return matches[-1].strip() if matches else None

total = 0
correct = 0
all_scores = []

with open(json_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        pred_ans = extract_last_boxed(data['pred_output'])
        gold_ans = extract_answer_after_hashes(data['gold_output'])
        
        if pred_ans is not None and gold_ans is not None:
            score = 100 if pred_ans == gold_ans else 0
            all_scores.append(score)
            correct += score == 100
            total += 1
        else:
            print("Missing boxed answer in:", data)
            score = 0
            all_scores.append(score)
            total += 1

average = sum(all_scores) / len(all_scores) if all_scores else 0
print(f"Average score: {average:.2f} (Correct: {correct}/{total})")


