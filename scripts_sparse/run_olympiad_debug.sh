export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
module load cuda12.4/toolkit/12.4.1
PROMPT_LEN=2048
MODEL_NAME="/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-dense_rollout/global_step_200"
DATASET_DIR="/home/haizhonz/Zhaofeng/sglang/data/olympiadbench_test.jsonl"
NUM_PAGES=16
ALGO="BLOCK_TOPK"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

# /cm/shared/apps/cuda12.2/visual-tools/12.2.2/bin/nsys profile \
#   --trace=cuda,nvtx \
#   --sample=none \
#   --output=profile.nsys-rep \
python run_olympiad.py \
  --prompt_len $PROMPT_LEN \
  --model_name $MODEL_NAME \
  --dataset_dir $DATASET_DIR \
  --num_pages $NUM_PAGES \
  --algo $ALGO




