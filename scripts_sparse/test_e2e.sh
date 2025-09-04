module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=256
MODEL_NAME="Qwen/Qwen3-1.7B-Base"
DATASET_DIR="/home/haizhonz/Zhaofeng/RULER/scripts/benchmark_root/Qwen3-4B/synthetic/4096/data"
DATASET="niah_single_1"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

python run_ruler_orig.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --dataset $DATASET \
    --num_pages $NUM_PAGES \
    --algo $ALGO
