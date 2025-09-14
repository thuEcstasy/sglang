module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=128
MODEL_NAME="/home/haizhonz/Zhaofeng/checkpoints/huggingface"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/aime24_train.jsonl"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

python run_deepscaleR_eval.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --use_dense_kv