module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=4096
MODEL_NAME="/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-dense_rollout/global_step_200"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/gsm8k_test.jsonl"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

python run_gsm8k.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --use_dense_kv




