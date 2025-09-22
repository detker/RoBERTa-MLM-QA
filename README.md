# RoBERTa for MLM and Question Answering

## 📋 Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Training](#training)
- [Notebooks](#notebooks)

## 🔎 Introduction

This repository implements a RoBERTa-based model fine-tuned with LoRA (Low-Rank Adaptation) for question answering tasks. The project utilizes Hugging Face's `transformers` library and the `accelerate` framework for efficient training and evaluation. Project is also provided with RoBERTa for MLM training script in case user wants to train his very own base RoBERTa rather than using pretrained Hugging Face's weights.

### Highlights
- **LoRA Fine-Tuning**: Efficiently fine-tune large models with low-rank adaptation for QA.
- **Question Answering**: Designed for extractive QA tasks.
- **Masked Language Modeling**: Designed also for MLM tasks.
- **Customizable Training**: Easily modify hyperparameters and configurations.
- **Pretrained Weights**: Leverages pretrained RoBERTa models for initialization.

### 📂 Project Structure
```
.
├── src/
├── notebooks/
├── wandb/
├── data/
│   ├── roberta_data/
│   └── squad_data/
├── working_directory/
│   ├── mlm_experiment_name/
│   │   └── checkpoints/
│   └── qa_experiment_name/
│       └── checkpoints/
```

## ⚙️ Setup

### Prerequisites
Ensure the following dependencies are installed:
- Python 3.11.4
- Conda 23.7.3
- PyTorch (compatible with your hardware)

### Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/detker/RoBERTa-MLM-QA
cd RoBERTa-MLM-QA
conda create -n roberta_mlmqa python=3.11.4
conda activate roberta_mlmqa
pip install -r requirements.txt
```

### Dataset Preparation
Prepare the dataset (wikipedia + bookcorpus) for base (MLM) using the `prepare_data.py` script:
```bash
python prepare_data.py
```
This will preprocess and save the dataset in the `data/` directory.
Dataset (SQuAD) preparation for QA finetuning with LoRA is already implemented in the training script leveraging Hugging Face's datasets library.

## 🚀 Training

Train the base model using the `train_mlm.sh` script. Adjust the parameters in the script as needed. Example:
```bash
chmod +x train_mlm.sh
./train_mlm.sh
```

Train the finetuned model for QA with LoRA using the `train_qa.sh` script. Adjust the parameters in the script as needed. Example:
```bash
chmod +x train_qa.sh
./train_mlm.sh
```

Train script for QA as a parameter offers selecting very own trained weights from `train_mlm.sh` or loading Hugging Face's RoBERTa weights. 

Training QA parameters include:

| **Parameter**               | **Description**                                                                 | **Default**       | **Type**            |
|-----------------------------|---------------------------------------------------------------------------------|-------------------|---------------------|
| `--experiment_name`         | Name of the experiment being launched                                           | **Required**      | `str`               |
| `--working_directory`       | Directory for checkpoints and logs                                             | **Required**      | `str`               |
| `--hf_model_name`           | Hugging Face model name or path                                                | **Required**      | `str`               |
| `--hf_dataset`              | Hugging Face dataset name                                                      | **Required**      | `str`               |
| `--use_lora`                | Whether to use LoRA                                                            | `False`           | `bool`              |
| `--train_head_only`         | Whether to train only the classification head                                  | `False`           | `bool`              |
| `--lora_rank`               | Rank of the LoRA adaptation matrices                                           | `8`               | `int`               |
| `--lora_alpha`              | Alpha scaling factor for LoRA                                                  | `8`               | `int`               |
| `--lora_use_rslora`         | Whether to use RS-LoRA                                                         | `False`           | `bool`              |
| `--lora_dropout`            | Dropout rate for LoRA layers                                                   | `0.1`             | `float`             |
| `--lora_bias`               | Bias configuration for LoRA                                                    | `'none'`          | `str` (choices: `none`, `lora_only`, `all`) |
| `--lora_target_modules`     | Comma-separated list of target modules for LoRA                                | **None**          | `list`              |
| `--lora_exclude_modules`    | Comma-separated list of modules to exclude from LoRA                           | **None**          | `list`              |
| `--max_grad_norm`           | Maximum norm for gradient clipping                                             | `1.0`             | `float`             |
| `--per_gpu_batch_size`      | Effective batch size                                                           | `32`              | `int`               |
| `--warmup_steps`            | Number of warmup steps for the learning rate scheduler                         | `0`               | `int`               |
| `--epochs`                  | Number of training epochs                                                      | `3`               | `int`               |
| `--num_workers`             | Number of workers for data loading                                             | `4`               | `int`               |
| `--learning_rate`           | Learning rate for the optimizer                                                | `5e-5`            | `float`             |
| `--weight_decay`            | Weight decay for the optimizer                                                 | `0.0`             | `float`             |
| `--gradient_checkpointing`  | Whether to use gradient checkpointing                                          | `False`           | `bool`              |
| `--adam_beta1`              | Beta1 parameter for Adam optimizer                                             | `0.9`             | `float`             |
| `--adam_beta2`              | Beta2 parameter for Adam optimizer                                             | `0.999`           | `float`             |
| `--adam_epsilon`            | Epsilon parameter for Adam optimizer                                           | `1e-8`            | `float`             |
| `--seed`                    | Random seed for reproducibility                                                | `42`              | `int`               |
| `--wandb`                   | Whether to use Weights & Biases for logging                                    | `False`           | `bool`              |
| `--loading_from_checkpoint` | Whether to load weights from the latest checkpoint                             | `False`           | `bool`              |
| `--max_no_of_checkpoints`   | Max number of latest checkpoints to store on disk                              | `10`              | `int`               |
| `--path_to_pretrained_backbone` | Path to pretrained backbone weights                                       | **None**          | `str`               |
| `--pretrained_backbone`     | Type of pretrained backbone to use (`pretrained`, `pretrained_huggingface`, `random`) | **None**          | `str`               |
| `--path_to_cache_dir`       | Path to Hugging Face cache directory                                           | **None**          | `str`               |

Checkpoints are saved in the `{working_directory}/{experiment_name}/checkpoints/` directory at regular intervals.

## 🧪 Notebooks 

Explore the `notebooks/` directory for a quick inference demo.
