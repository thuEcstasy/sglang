module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=128
MODEL_NAME="/home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8_GSPO1_clipped95/global_step_320/actor/huggingface"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/amc23_test.jsonl"
NUM_PAGES=32
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

# export TEST_E2E=True

python test_e2e_sparse.py \
    --prompt_len $PROMPT_LEN \
    --model_name $MODEL_NAME \
    --dataset_dir $DATASET_DIR \
    --num_pages $NUM_PAGES \
    --algo $ALGO \
    --use_dense_kv
