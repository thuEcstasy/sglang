export CUDA_LAUNCH_BLOCKING=1
module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=256
MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/gsm8k_test_2.jsonl"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

compute-sanitizer --tool memcheck python run_gsm8k.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO \
