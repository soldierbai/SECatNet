@echo off

:: 定义默认参数
set DATA_DIR=
set LABELS_DIR=
set NUM_EPOCHS=100
set LR=0.0001
set BATCH_SIZE=64
set WEIGHT_DECAY=0.00005
set TEST_RATIO=0.2
set VAL_RATIO=0.1
set RANDOM_SEED=42
set DEVICE=
set OUTPUT_MODEL_DIR=
set OUTPUT_RESULT_DIR=
set VERBOSE=

:: 解析命令行参数
:parse_args
if "%1"=="" goto end_parse
if "%1"=="--data_dir" (
    set DATA_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--labels_dir" (
    set LABELS_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--num_epochs" (
    set NUM_EPOCHS=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--lr" (
    set LR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--batch_size" (
    set BATCH_SIZE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--weight_decay" (
    set WEIGHT_DECAY=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--test_ratio" (
    set TEST_RATIO=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--val_ratio" (
    set VAL_RATIO=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--random_seed" (
    set RANDOM_SEED=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--device" (
    set DEVICE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--output_model_dir" (
    set OUTPUT_MODEL_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--output_result_dir" (
    set OUTPUT_RESULT_DIR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--verbose" (
    set VERBOSE=--verbose
    shift
    goto parse_args
)
echo 未知参数: %1
exit /b 1

:end_parse

:: 启动 Python 脚本
python main.py ^
    --data_dir "%DATA_DIR%" ^
    --labels_dir "%LABELS_DIR%" ^
    --num_epochs %NUM_EPOCHS% ^
    --lr %LR% ^
    --batch_size %BATCH_SIZE% ^
    --weight_decay %WEIGHT_DECAY% ^
    --test_ratio %TEST_RATIO% ^
    --val_ratio %VAL_RATIO% ^
    --random_seed %RANDOM_SEED% ^
    --device "%DEVICE%" ^
    --output_model_dir "%OUTPUT_MODEL_DIR%" ^
    --output_result_dir "%OUTPUT_RESULT_DIR%" ^
    %VERBOSE%