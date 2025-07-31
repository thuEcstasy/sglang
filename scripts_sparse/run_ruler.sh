#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

PROMPT_LEN=8192
DATASET="niah_single_1"
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_DIR="/home/haizhonz/Zhaofeng/RULER/scripts/benchmark_root/Qwen3-4B/synthetic/4096/data"
NUM_PAGES=16
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

# Get total number of lines (samples)
TOTAL=$(wc -l < ${DATASET_DIR}/${DATASET}/validation.jsonl)

# clean the output file before running
#     output_path = f"../output/{model_name}/{args.dataset}/ruler_output_prompt_len_{args.prompt_len}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}.jsonl"

OUTPUT_FILE="../output/${MODEL_NAME}/${DATASET}/ruler_output_prompt_len_${PROMPT_LEN}_num_pages_${NUM_PAGES}_${ALGO}.jsonl"
rm -f "$OUTPUT_FILE"

for (( start=0; start<TOTAL; start+=BS )); do
    end=$((start + BS))
    echo "Running batch from $start to $end"
    python run_ruler.py \
        --prompt_len $PROMPT_LEN \
        --model_name $MODEL_NAME \
        --dataset $DATASET \
        --dataset_dir $DATASET_DIR \
        --num_pages $NUM_PAGES \
        --start $start \
        --end $end \
        --algo $ALGO
done
