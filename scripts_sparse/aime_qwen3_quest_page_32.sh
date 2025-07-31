export CUDA_VISIBLE_DEVICES=3

PROMPT_LEN=4096
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_DIR="/home/haizhonz/Zhaofeng/dataset/aime24_train.jsonl"
NUM_PAGES=16
ALGO="QUEST"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

export MSE_FILE="/home/haizhonz/Zhaofeng/sglang/output/${MODEL_NAME}/aime/quest_32_pass4.txt"
echo $MSE_FILE
python run_aime.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO






