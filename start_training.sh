#!/bin/bash

# 定义默认参数
DATA_DIR=""
LABELS_DIR=""
NUM_EPOCHS=100
LR=0.0001
BATCH_SIZE=64
WEIGHT_DECAY=0.00005
TEST_RATIO=0.2
VAL_RATIO=0.1
RANDOM_SEED=42
DEVICE=""
OUTPUT_MODEL_DIR=""
OUTPUT_RESULT_DIR=""
VERBOSE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --labels_dir)
            LABELS_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --test_ratio)
            TEST_RATIO="$2"
            shift 2
            ;;
        --val_ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output_model_dir)
            OUTPUT_MODEL_DIR="$2"
            shift 2
            ;;
        --output_result_dir)
            OUTPUT_RESULT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 启动 Python 脚本
python main.py \
    --data_dir "$DATA_DIR" \
    --labels_dir "$LABELS_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --weight_decay "$WEIGHT_DECAY" \
    --test_ratio "$TEST_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --device "$DEVICE" \
    --output_model_dir "$OUTPUT_MODEL_DIR" \
    --output_result_dir "$OUTPUT_RESULT_DIR" \
    $VERBOSE