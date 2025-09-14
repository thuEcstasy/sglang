import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import os

from verl.utils.reward_score import default_compute_score

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_len", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_dense_kv", action="store_true")
    parser.add_argument("--num_pages", type=int, default=4096)
    parser.add_argument("--algo", type=str, default="BLOCK_TOPK")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name
    num_pages = args.num_pages
    if args.use_dense_kv:
        llm = sgl.Engine(model_path=model_name, 
                        disable_cuda_graph=True, 
                        attention_backend="flashinfer",
        )
    else:
        llm = sgl.Engine(model_path=model_name, 
                        disable_cuda_graph=True, 
                        page_size=16,
                        vortex_num_selected_pages=num_pages,     
                        vortex_sparse_attention_algorithm=args.algo,
                        disable_overlap_schedule=True,
                        attention_backend="flashinfer",
                        enable_vortex_sparsity=True,
                        vortex_page_reserved_bos=2,
                        vortex_page_reserved_eos=2,
                        vortex_layers_skip=[0, 1],
                        )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    ruler_dataset_path = args.dataset_dir
    with open(ruler_dataset_path, "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line) for line in f]
    
    bs = 32768 // args.prompt_len
    output_path = f"../output/{model_name}/aime_output_prompt_len_{args.prompt_len}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}_{args.algo}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output path: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ruler_data), bs)):

            batch = ruler_data[i:i+bs]
            inputs = [item["Problem"] for item in batch]
            # apply template: Please reason step by step, and put your final answer within \boxed{}
            # if "Qwen2-1.5B" not in model_name and "Qwen2.5-Math" not in model_name:
            #     inputs = [f"Please reason step by step, and put your final answer within \\boxed{{}}. {item}" for item in inputs]
            sampling_params = {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                # "temperature": 1,
                # "top_p": 1,
                # "top_k": -1,
                "max_new_tokens": 8192
            }
            inputs = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': item}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
            ) for item in inputs]
            print(f"Inputs: {inputs[0]}")
            from time import time
            start_time = time()
            predictions = llm.generate(inputs, sampling_params)
            # tokenize the prediction and print the length
            end_time = time()
            print(f"Time for batch {i//bs}: {end_time - start_time} seconds", flush=True)
            input_tokens = tokenizer.encode(inputs[0])
            pred_tokens = tokenizer.encode(predictions[0]["text"])
            print(len(input_tokens), len(pred_tokens), flush=True)
            for _ , (item, ruler_item, pred) in enumerate(zip(inputs, batch, predictions)):
                output_item = {
                    "input": item,
                    "gold_output": ruler_item["Answer"],
                    "pred_output": pred["text"]
                }
                json.dump(output_item, fout, ensure_ascii=False)
                fout.write("\n")
                
    # evaluate
    with open(output_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    total = len(results)
    correct = 0
    for res in results:
        pred = res["pred_output"]
        gold = str(res["gold_output"])
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
