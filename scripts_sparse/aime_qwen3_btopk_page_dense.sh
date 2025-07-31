export CUDA_VISIBLE_DEVICES=1

PROMPT_LEN=32768
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_DIR="/home/haizhonz/Zhaofeng/dataset/aime24_train.jsonl"
NUM_PAGES=4096
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

# Get total number of lines (samples)
TOTAL=$(wc -l < ${DATASET_DIR})

if [ "$USE_DENSE_KV" = true ]; then
  NUM_PAGES_FOR_NAME="dense"
else
  NUM_PAGES_FOR_NAME="$NUM_PAGES"
fi

for (( start=0; start<TOTAL; start+=BS )); do
    end=$((start + BS))
    echo "Running batch from $start to $end"
    export MSE_FILE="/home/haizhonz/Zhaofeng/sglang/output/${MODEL_NAME}/aime/dense.txt"
    echo $MSE_FILE
    python run_aime.py \
        --prompt_len $PROMPT_LEN \
        --model_name $MODEL_NAME \
        --dataset_dir $DATASET_DIR \
        --num_pages $NUM_PAGES \
        --start $start \
        --end $end \
        --algo $ALGO

done



