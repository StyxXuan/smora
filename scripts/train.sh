
#!/bin/bash

# Change to the LLaMA-Factory directory
cd "$(dirname "$0")/../packages/LLaMA-Factory" || exit

# Run training with an example YAML config
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
