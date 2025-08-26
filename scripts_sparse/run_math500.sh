module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=512
MODEL_NAME="/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-block-topk-sparse_rollout/global_step_120/actor/huggingface"
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




MODEL_NAME="/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/Deepseek_r1_distill_1.5B-dense_rollout/global_step_120/actor/huggingface"

python run_math500_eval.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --use_dense_kv 

python run_math500_eval.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO



