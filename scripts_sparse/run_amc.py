import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import os

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ruler_dataset_path = args.dataset_dir
    with open(ruler_dataset_path, "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line) for line in f]
    
    bs = 32768 // args.prompt_len
    output_path = f"../output/{model_name}/amc_output_prompt_len_{args.prompt_len}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}_{args.algo}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output path: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ruler_data), bs)):
            batch = ruler_data[i:i+bs]
            inputs = [item["question"] for item in batch]
            # apply template: Please reason step by step, and put your final answer within \boxed{}
            inputs = [f"Please reason step by step, and put your final answer within \\boxed{{}}. {item}" for item in inputs]
            if "Llama" in model_name:
                sampling_params = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_new_tokens": 32768,
                    "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                } 
                # we don't apply chat template in RULER for Llama, so we need to apply it here    
                inputs = [tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': item}],
                    tokenize=False,
                ) for item in inputs]
            else:
                sampling_params = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_new_tokens": 32768
                }    
                inputs = [tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': item}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
                ) for item in inputs]
                print(f"Inputs: {inputs[0]}")
            predictions = llm.generate(inputs, sampling_params)
            # tokenize the prediction and print the length
            input_tokens = tokenizer.encode(inputs[0])
            pred_tokens = tokenizer.encode(predictions[0]["text"])
            print(len(input_tokens), len(pred_tokens))
            for _ , (item, ruler_item, pred) in enumerate(zip(inputs, batch, predictions)):
                output_item = {
                    "input": item,
                    "gold_output": ruler_item["answer"],
                    "id": ruler_item["id"],
                    "pred_output": pred["text"]
                }
                json.dump(output_item, fout, ensure_ascii=False)
                fout.write("\n")
if __name__ == "__main__":
    main()
