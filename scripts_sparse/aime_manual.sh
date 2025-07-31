export CUDA_VISIBLE_DEVICES=5

PROMPT_LEN=32768
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_DIR="/home/haizhonz/Zhaofeng/dataset/aime24_train.jsonl"
NUM_PAGES=16
ALGO="MANUAL"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

export MSE_FILE="/home/haizhonz/Zhaofeng/sglang/output/${MODEL_NAME}/aime/manual_16_pass4.txt"
echo $MSE_FILE
python run_aime_manual.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO




