export CUDA_VISIBLE_DEVICES=2

PROMPT_LEN=8192
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_DIR="/home/haizhonz/Zhaofeng/dataset/amc23_test.jsonl"
NUM_PAGES=16
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

export MSE_FILE="/home/haizhonz/Zhaofeng/sglang/output/${MODEL_NAME}/amc/block_topk_16_pass4.txt"
echo $MSE_FILE
python run_amc.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO




