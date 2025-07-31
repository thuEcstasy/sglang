export CUDA_VISIBLE_DEVICES=1

DATASETS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multiquery"
    "niah_multivalue"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
    "vt"
)

for dataset in ${DATASETS[@]}; do

    PROMPT_LEN=8192
    DATASET=$dataset
    MODEL_NAME="NousResearch/Meta-Llama-3.1-8B-Instruct"
    DATASET_DIR="/home/haizhonz/Zhaofeng/RULER/scripts/benchmark_root/base/synthetic/4096/data"
    NUM_PAGES=16
    ALGO="BLOCK_TOPK"
    # Compute batch size
    BS=$((32768 / PROMPT_LEN))

    # Get total number of lines (samples)
    TOTAL=$(wc -l < ${DATASET_DIR}/${DATASET}/validation.jsonl)

    # clean the output file before running
    #     output_path = f"../output/{model_name}/{args.dataset}/ruler_output_prompt_len_{args.prompt_len}_num_pages_{num_pages if not args.use_dense_kv else 'dense'}.jsonl"

    OUTPUT_FILE="../output/${MODEL_NAME}/${DATASET}/ruler_output_prompt_len_${PROMPT_LEN}_num_pages_${NUM_PAGES}_${ALGO}.jsonl"
    rm -f "$OUTPUT_FILE"

    for (( start=0; start<TOTAL; start+=BS )); do
        end=$((start + BS))
        echo "Running batch from $start to $end"
        python run_ruler.py \
            --prompt_len $PROMPT_LEN \
            --model_name $MODEL_NAME \
            --dataset $DATASET \
            --dataset_dir $DATASET_DIR \
            --num_pages $NUM_PAGES \
            --start $start \
            --end $end \
            --algo $ALGO
    done

done



