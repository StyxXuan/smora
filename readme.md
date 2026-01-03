# KDD-2026-SMoRA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.02913-b31b1b.svg)](https://arxiv.org/pdf/2302.10911.pdf)

>**This is the official implementation of the KDD 2026 paper: *Each Rank Could be an Expert: Single-Ranked Mixture of Experts LoRA for Multi-task Learning***



**TODO:**

- [x] Release source code
- [ ] Code needs to be organized
- [ ] Publish model parameters

<!-- A comprehensive framework for training and evaluating task-specific LoRA models using LLaMA-Factory and OpenCompass. -->

## Overview

This project provides a complete pipeline for:
- **Training**: Fine-tuning models with LoRA using LLaMA-Factory
- **Evaluation**: Assessing model performance using OpenCompass
- **SRMoLE**: Custom single-ranked mixture of low-rank experts implementation

## Installation
### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/StyxXuan/smora.git
   cd smora
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install packages in development mode**
   ```bash
   # Install LLaMA-Factory
   cd packages/LLaMA-Factory
   pip install -e .
   cd ../..

   # Install OpenCompass
   cd packages/opencompass
   pip install -e .
   cd ../..

   # Install PEFT (with SRMoLE support)
   cd packages/peft
   pip install -e .
   cd ../..
   ```

4. **Download the training datasets**
   Some large dataset files are excluded from version control due to GitHub size limits.
   To download the complete dataset package:

   ```bash
   cd packages/LLaMA-Factory
   # Download from Google Drive
   wget "https://drive.google.com/uc?export=download&id=1VgKYdlxgjkS8yqdupK4hR3MFWAod7fUE" -O datasets.tar.gz
   tar -xzf datasets.tar.gz
   rm datasets.tar.gz
   ```

   This will extract all dataset files to the `data/` directory.

## Project Structure

```
smora/
├── packages/
│   ├── LLaMA-Factory/          # Training framework
│   ├── opencompass/            # Evaluation framework
│   └── peft/                   # Parameter-efficient fine-tuning (includes SRMoLE)
├── scripts/
│   ├── train.sh               # Training script
│   ├── eval.sh                # Evaluation script
│   └── training_configs/      # Training configurations
├── datasets/                  # Dataset files
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Usage

### Training

We leverage LLaMA-Factory for training LoRA models. The demonstration configurations are available in `scripts/training_configs/`.

#### Configuration Types
- **Flan Tasks**: Use `scripts/training_configs/config_flan/`
- **Multi-domain Tasks**: Use `scripts/training_configs/config_multidomain/`

#### Training Command
```bash
# Basic training
llamafactory-cli train config_path/config.yaml

# Example with specific config
llamafactory-cli train scripts/training_configs/config_flan/llama3_lora_sft.yaml
```

### Evaluation

#### Using OpenCompass
```bash
# Basic evaluation
./scripts/eval.sh

# Evaluate specific task
./scripts/eval.sh task_name
```

#### Custom Evaluation
1. **Update model configurations** in OpenCompass configs
2. **Run evaluation** on downstream tasks:
   ```bash
   cd packages/opencompass
   opencompass --models your_model_config --datasets your_dataset
   ```

## SRMoLE Implementation

This project includes a custom implementation of SRMoLE (Single-Ranked Mixture of LoRA Experts) in the PEFT package.

### Usage
```python
from peft import SRMoLEConfig, SRMoLEModel

config = SRMoLEConfig(
    peft_type="SRMOLE",
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = SRMoLEModel(base_model, config, "default")
```

## Configuration

### Training Configuration
Edit YAML files in `scripts/training_configs/` to customize:
- Model architecture
- Training parameters
- Dataset paths
- LoRA settings

### Evaluation Configuration
Modify OpenCompass configs in `packages/opencompass/configs/` for:
- Model evaluation settings
- Dataset configurations
- Output formats

## Examples

### Training a Flan Task Model
```bash
# Configure your dataset and model paths
# Edit scripts/training_configs/config_flan/your_config.yaml

# Run training
llamafactory-cli train scripts/training_configs/config_flan/your_config.yaml
```

### Evaluating with OpenCompass
```bash
# Set up model and dataset paths
# Edit packages/opencompass/configs/models/your_model.py

# Run evaluation
cd packages/opencompass
opencompass --models your_model --datasets your_dataset
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{zhao2025each,
  title={Each rank could be an expert: Single-ranked mixture of experts lora for multi-task learning},
  author={Zhao, Ziyu and Zhou, Yixiao and Zhang, Zhi and Zhu, Didi and Shen, Tao and Li, Zexi and Yang, Jinluan and Wang, Xuwu and Su, Jing and Kuang, Kun and others},
  journal={arXiv preprint arXiv:2501.15103},
  year={2025}
}
```
