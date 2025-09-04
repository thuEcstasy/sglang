module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=512
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/math500_test.jsonl"
NUM_PAGES=16
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))
export USE_DYNAMIC_NUM_PAGES=1

python run_math500_eval.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO
    
python run_math500_eval.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --use_dense_kv 



