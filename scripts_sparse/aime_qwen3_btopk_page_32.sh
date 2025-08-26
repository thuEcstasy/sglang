module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=512
MODEL_NAME="Qwen/Qwen3-1.7B"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/aime24_train_32.jsonl"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

python run_aime.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO




