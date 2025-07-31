export CUDA_VISIBLE_DEVICES=2

PROMPT_LEN=1200
MODEL_NAME="NousResearch/Meta-Llama-3.1-8B-Instruct"
DATASET_DIR="/home/haizhonz/Zhaofeng/dataset/gsm8k_test.jsonl"
NUM_PAGES=4
ALGO="QUEST"
# Compute batch size
BS=$((32768 / PROMPT_LEN))

# Get total number of lines (samples)
TOTAL=$(wc -l < ${DATASET_DIR})

# clean the output file before running
#     output_path = f"../output/{model_name}/{args.dataset}/ruler_output_prompt_len_{args.prompt_len}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}.jsonl"

OUTPUT_FILE="../output/${MODEL_NAME}/gsm8k_output_prompt_len_${PROMPT_LEN}_num_pages_${NUM_PAGES}_${ALGO}.jsonl"
rm -f "$OUTPUT_FILE"

for (( start=0; start<TOTAL; start+=BS )); do
    end=$((start + BS))
    echo "Running batch from $start to $end"
    python run_gsm8k.py \
        --prompt_len $PROMPT_LEN \
        --model_name $MODEL_NAME \
        --dataset_dir $DATASET_DIR \
        --num_pages $NUM_PAGES \
        --start $start \
        --end $end \
        --algo $ALGO
done



