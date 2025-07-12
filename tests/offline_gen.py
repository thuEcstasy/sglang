import json
import sys
sys.path.append("../")
import python.sglang as sgl
from transformers import AutoTokenizer
def main():
    model_name = "/home/zhuominc/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=True, 
                    page_size=16,
                    vortex_num_selected_pages=32,       
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=True,
                    vortex_page_reserved_bos=2,
                    vortex_page_reserved_eos=2,
                    vortex_layers_skip=[0, 1]
                    )
    
    with open('story.txt', "r", encoding="utf-8") as file:
        content = file.read()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [
        [{"role":"user","content":"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}],
        [{"role":"user","content":"If the original price of a shirt is $25, a discount of 20% is applied. How much will you pay for the shirt after the discount?"}],
        [{"role":"user","content":"Tell me about Reinforcement Learning in 200 words."}],
        [{"role":"user","content": content}],
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ) for text in texts
    ]
    
    prompts = prompts * 8

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 4096}

    outputs = llm.generate(prompts, sampling_params)
    with open("output.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
if __name__ == "__main__":
    main()
