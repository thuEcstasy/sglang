from datasets import load_dataset
import json

# Login using e.g. `huggingface-cli login` to access this dataset

ds = load_dataset("math-ai/amc23")

# save to local jsonl file
for split in ds.keys():
    with open(f"amc23_{split}.jsonl", 'w', encoding='utf-8') as f:
        for item in ds[split]:
            f.write(json.dumps(item) + '\n')
            
# ds = load_dataset("math-ai/olympiadbench")

# # save to local jsonl file
# for split in ds.keys():
#     with open(f"olympiadbench_{split}.jsonl", 'w', encoding='utf-8') as f:
#         for item in ds[split]:
#             f.write(json.dumps(item) + '\n')
            
# ds = load_dataset("HuggingFaceH4/MATH-500")

# # save to local jsonl file
# for split in ds.keys():
#     with open(f"math500_{split}.jsonl", 'w', encoding='utf-8') as f:
#         for item in ds[split]:
#             f.write(json.dumps(item) + '\n')
