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
    parser.add_argument("--dataset", type=str, required=True)
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
    ruler_dataset_path = f"{args.dataset_dir}/{args.dataset}/validation.jsonl"
    with open(ruler_dataset_path, "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line) for line in f]
    
    bs = 32768 // args.request_len
    output_path = f"../output/{model_name}/{args.dataset}/ruler_output_prefill_len_{args.request_len - 4096}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}_{args.algo}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output path: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ruler_data), bs)):
            batch = ruler_data[i:i+bs]
            inputs = [item["input"] for item in batch]
            if "Llama" in model_name:
                sampling_params = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_new_tokens": 4096,
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
                    "max_new_tokens": 4096
                }    
            predictions = llm.generate(inputs, sampling_params)
            # tokenize the prediction and print the length
            input_tokens = tokenizer.encode(inputs[0])
            pred_tokens = tokenizer.encode(predictions[0]["text"])
            assert len(input_tokens) + len(pred_tokens) <= args.request_len
            print(len(input_tokens), len(pred_tokens))
            for _ , (item, ruler_item, pred) in enumerate(zip(inputs, batch, predictions)):
                output_item = {
                    "input": item,
                    "gold_output": ruler_item["outputs"],
                    "pred_output": pred["text"]
                }
                json.dump(output_item, fout, ensure_ascii=False)
                fout.write("\n")
if __name__ == "__main__":
    main()
