#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

PROMPT_LEN=8183
MODEL_NAME="NousResearch/Meta-Llama-3.1-8B-Instruct"
DATASET_DIR="/home/haizhonz/Zhaofeng/RULER/scripts/benchmark_root/Qwen3-4B/synthetic/4096/data"
NUM_PAGES=16
ALGO="BLOCK_TOPK"

# 定义所有数据集
DATASETS=(
    # "niah_single_1"
    # "niah_single_2"
    # "niah_single_3"
    # "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    # "niah_multiquery"
    # "niah_multivalue"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
    "vt"
)

# Compute batch size
BS=$((32768 / PROMPT_LEN))

echo "Starting evaluation for ${#DATASETS[@]} datasets with batch size: $BS"
echo "Model: $MODEL_NAME"
echo "Algorithm: $ALGO"
echo "Prompt length: $PROMPT_LEN"
echo "Number of pages: $NUM_PAGES"
echo "=================="

# 遍历所有数据集
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Processing dataset: $DATASET"
    echo "=================="
    
    # 检查数据集文件是否存在
    if [ ! -f "${DATASET_DIR}/${DATASET}/validation.jsonl" ]; then
        echo "Warning: Dataset file not found: ${DATASET_DIR}/${DATASET}/validation.jsonl"
        echo "Skipping dataset: $DATASET"
        continue
    fi
    
    # Get total number of lines (samples)
    TOTAL=$(wc -l < ${DATASET_DIR}/${DATASET}/validation.jsonl)
    echo "Total samples in $DATASET: $TOTAL"
    
    # 定义输出文件
    OUTPUT_FILE="../output/${MODEL_NAME}/${DATASET}/ruler_output_prompt_len_${PROMPT_LEN}_num_pages_${NUM_PAGES}_${ALGO}.jsonl"
    
    # 创建输出目录（如果不存在）
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    # 清理输出文件
    rm -f "$OUTPUT_FILE"
    
    # 批处理运行
    for (( start=0; start<TOTAL; start+=BS )); do
        end=$((start + BS))
        echo "Running batch from $start to $end for dataset: $DATASET"
        export MSE_FILE="/home/haizhonz/Zhaofeng/sglang/output/${MODEL_NAME}/${DATASET}/${ALGO}_16.txt"
        echo $MSE_FILE
        python run_ruler.py \
            --prompt_len $PROMPT_LEN \
            --model_name $MODEL_NAME \
            --dataset $DATASET \
            --dataset_dir $DATASET_DIR \
            --num_pages $NUM_PAGES \
            --start $start \
            --end $end \
            --algo $ALGO
        
        # 检查命令是否成功执行
        if [ $? -ne 0 ]; then
            echo "Error: Failed to process batch $start-$end for dataset: $DATASET"
            echo "Continuing with next batch..."
        fi
    done
    
    echo "Completed dataset: $DATASET"
    echo "Output saved to: $OUTPUT_FILE"
done

echo ""
echo "=================="
echo "All datasets processing completed!"
echo "Total datasets processed: ${#DATASETS[@]}"