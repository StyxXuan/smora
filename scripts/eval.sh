#!/bin/bash

# Define variables
BASE_MODEL_PATH=PATH_TO_BASE_MODEL
LORA_MODELS_DIR=PATH_TO_LORA_MODELS
JSON_PATH="datasets/combined_test.json"
OUTPUT_DIR=PATH_TO_OUTPUT_DIR

# Default evaluation parameters
BATCH_SIZE=8
MAX_LENGTH=1024
MAX_NEW_TOKENS=200
TEMPERATURE=0.1

# If task argument is provided, use it
TASK_ARGS=""
if [ "$1" != "" ]; then
    TASK_ARGS="--tasks $1"
    echo "Evaluating only task: $1"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting evaluation of task-specific LoRA models..."
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA models directory: $LORA_MODELS_DIR"
echo "Dataset: $JSON_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max input length: $MAX_LENGTH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"

# Run evaluation script
python evaluate_task_specific_lora.py \
    --base_model_path $BASE_MODEL_PATH \
    --lora_models_dir $LORA_MODELS_DIR \
    --json_path $JSON_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    $TASK_ARGS

echo "Evaluation completed, results saved in $OUTPUT_DIR" 