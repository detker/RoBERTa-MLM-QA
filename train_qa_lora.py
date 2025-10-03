import argparse
import os
import shutil

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from safetensors.torch import save_file, safe_open
from transformers import (get_cosine_schedule_with_warmup,
                          DefaultDataCollator,
                          RobertaTokenizerFast)

from src.lora import LoRAModel, LoRAConfig
from src.utils import RobertaConfig, QAProcessor
from src.model import RobertaForQA

import warnings

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--experiment_name',
                        type=str,
                        required=True,
                        help='Name of the experiment')

    parser.add_argument('--working_directory',
                        type=str,
                        required=True,
                        help='Directory to save experiment outputs')

    parser.add_argument('--checkpoint_weights_dir',
                        type=str,
                        required=True,
                        help='Directory to save model checkpoints')

    parser.add_argument('--hf_model_name',
                        type=str,
                        required=True,
                        help='Hugging Face model name')

    parser.add_argument('--hf_dataset',
                        type=str,
                        required=True,
                        help='Hugging Face dataset name.')

    parser.add_argument('--path_to_cache_dir',
                        help='Path to Hugging Face cache',
                        default=None,
                        type=str)

    parser.add_argument('--use_lora',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to use LoRA.')

    parser.add_argument('--train_head_only',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to train only the classification head.')

    parser.add_argument('--lora_rank',
                        type=int,
                        default=8,
                        help='Rank of the LoRA adaptation matrices')

    parser.add_argument('--lora_alpha',
                        type=int,
                        default=8,
                        help='LoRA Alpha scaling factor')

    parser.add_argument('--lora_use_rslora',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to use RS-LoRA')

    parser.add_argument('--lora_dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate for LoRA')

    parser.add_argument('--lora_bias',
                        type=str,
                        default='none',
                        choices=['none', 'lora_only', 'all'],
                        help='Bias configuration for LoRA.')

    parser.add_argument('--lora_target_modules',
                        type=lambda x: [s.strip() for s in x.split(',')],
                        help='Comma-separated list of target modules for LoRA.')

    parser.add_argument('--lora_exclude_modules',
                        type=lambda x: [s.strip() for s in x.split(',')],
                        help='Comma-separated list of modules to exclude from LoRA.')

    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='Maximum gradient norm for gradient clipping')

    parser.add_argument('--per_gpu_batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU')

    parser.add_argument('--warmup_steps',
                        type=int,
                        default=0,
                        help='Number of warmup steps')

    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Number of training epochs')

    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for DataLoader')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-5,
                        help='Learning rate for the optimizer')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay for the optimizer')

    parser.add_argument('--gradient_checkpointing',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to use gradient checkpointing')

    parser.add_argument('--adam_beta1',
                        type=float,
                        default=0.9,
                        help='Beta1 parameter for Adam optimizer')

    parser.add_argument('--adam_beta2',
                        type=float,
                        default=0.999,
                        help='Beta2 parameter for Adam optimizer')

    parser.add_argument('--adam_epsilon',
                        type=float,
                        default=1e-8,
                        help='Epsilon parameter for Adam optimizer')

    parser.add_argument('--wandb',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Log metrics to Weight & Biases')

    parser.add_argument('--loading_from_checkpoint',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to load weights from latest checkpoint in working directory folder.')

    parser.add_argument('--max_no_of_checkpoints',
                        type=int,
                        default=10,
                        help='Max number of latest checkpoints to store on disk.')

    parser.add_argument('--pretrained_backbone',
                        help='Do you want want a `pretrained` backbone from `train_base.py`, \
                             `pretrained_huggingface` backbone, or `random` initialized backbone',
                        choices=('pretrained', 'pretrained_huggingface', 'random'),
                        type=str)

    parser.add_argument('--path_to_pretrained_backbone',
                        help='Path to model weights stored from `train_base.py` to initialize the backbone',
                        default=None,
                        type=str)


def main():
    ## parser
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ## create accelerator and experiment
    ## wandb logger init
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    if not os.path.isdir(path_to_experiment):
        os.makedirs(path_to_experiment)
    accelerator = Accelerator(project_dir=path_to_experiment, log_with='wandb' if args.wandb else None)
    if args.wandb:
        accelerator.init_trackers(args.experiment_name)

    ## load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.hf_model_name)
    config = RobertaConfig(pretrained_backbone=args.pretrained_backbone,
                           path_to_pretrained_model=args.path_to_pretrained_backbone,
                           gradient_checkpointing=args.gradient_checkpointing)

    ## load data
    data = load_dataset(args.hf_dataset, cache_dir=args.path_to_cache_dir)

    ## tokenize dataset
    with accelerator.main_process_first():
        tokenized_dataset = data.map(QAProcessor(), batched=True, remove_columns=data['train'].column_names)

    ## create train test dataloaders, init data collator
    collate_func = DefaultDataCollator()
    trainloader = DataLoader(tokenized_dataset['train'], batch_size=args.per_gpu_batch_size, shuffle=True,
                             collate_fn=collate_func, num_workers=args.num_workers)
    testloader = DataLoader(tokenized_dataset['validation'], batch_size=args.per_gpu_batch_size, shuffle=False,
                            collate_fn=collate_func, num_workers=args.num_workers)

    ## load model
    model = RobertaForQA(config)

    ## check if lora or head only, wrap model in lora
    if not args.use_lora and args.train_head_only:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    if args.use_lora:
        ## create config file
        lora_config = LoRAConfig(**{k: v for k, v in args.__dict__.items() if 'lora_' in k})
        model = LoRAModel(model, config=lora_config)

    ## move model to device
    model = model.to(accelerator.device)
    accelerator.print(model)

    ## init optim, scheduler, loss (categorical crossentropy)
    optimizer = optim.AdamW([x for x in model.parameters() if x.requires_grad],
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps * accelerator.num_processes,
                                                num_training_steps=args.epochs * len(
                                                    trainloader) * accelerator.num_processes)
    accelerator.register_for_checkpointing(scheduler)

    completed_epochs = 0
    ## check if we're loading from checkpoint (checkpoint format: checkpoint_n.pt)
    if args.loading_from_checkpoint:
        path = os.path.join(args.working_directory, args.experiment_name, args.checkpoint_weights_dir)

        weights_type = None
        if args.use_lora:
            weights_type = 'lora'
        elif not args.use_lora and args.train_head_only:
            weights_type = 'headonly'
        else:
            weights_type = 'fulltrain'

        checkpoints_sorted = sorted(filter(lambda x: weights_type in x, os.listdir(path)),
                                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(checkpoints_sorted) == 0:
            raise Exception('No checkpoints found')
        checkpoint_path = os.path.join(path, checkpoints_sorted[-1])
        with accelerator.main_process_first():
            new_state_dict = model.state_dict()
            # with safe_open('work_dir/LoRA_RoBERTa_QA/checkpoints/checkpoint_lora_1.safetensors', framework='pt') as f:
            with safe_open(checkpoint_path, framework='pt') as f:
                for k in f.keys():
                    new_state_dict['model.' + k] = f.get_tensor(k)
            model.load_state_dict(new_state_dict)
        completed_epochs = int(checkpoints_sorted[-1].split('_')[-1].split('.')[0])

        accelerator.print(f'Checkpoint {completed_epochs} loaded.')

    ## accelerate.prepare
    model, optimizer, scheduler, trainloader, testloader = accelerator.prepare(model, optimizer, scheduler, trainloader,
                                                                               testloader)

    accelerator.print('TRAINING THE MODEL...')

    completed_train_steps = completed_epochs * len(trainloader)
    completed_test_steps = completed_epochs * len(testloader)

    for epoch in range(completed_epochs, args.epochs):
        accelerator.print(f'Epoch: {epoch + 1}/{args.epochs}')

        pbar = tqdm(range(0, len(trainloader)), disable=not accelerator.is_local_main_process)

        loss_train = []
        loss_test = []

        model.train()
        for batch in trainloader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            loss, start_logits, end_logits = model(**batch)
            accelerator.print(loss)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            gathered_loss = accelerator.gather_for_metrics(loss)

            loss_train.append(torch.mean(gathered_loss).item())

            if completed_train_steps % 10 == 0:
                accelerator.log({"training_loss": np.array(loss_train).mean()},
                                step=completed_train_steps)

            pbar.update(1)
            completed_train_steps += 1

        model.eval()
        accelerator.print('Evaluating model...')
        for batch in testloader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                loss, start_logits, end_logits = model(**batch)

            gathered_loss = accelerator.gather_for_metrics(loss)

            loss_test.append(torch.mean(gathered_loss).item())

            if completed_test_steps % 10 == 0:
                accelerator.log({"test_loss": np.array(loss_test).mean()},
                                step=completed_test_steps)

            completed_test_steps += 1

        epoch_train_loss = np.mean(loss_train)
        epoch_test_loss = np.mean(loss_test)

        accelerator.print(f'Training Loss: {epoch_train_loss}')
        accelerator.print(f'Testing Loss: {epoch_test_loss}')

        accelerator.print('Saving checkpoint...')
        accelerator.wait_for_everyone()

        path_to_save = os.path.join(args.working_directory, args.experiment_name, args.checkpoint_weights_dir)

        if accelerator.is_main_process:
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            if args.use_lora:
                if epoch + 1 == args.epochs:
                    accelerator.unwrap_model(model).save_weights(
                        os.path.join(path_to_save, f'checkpoint_merged_{epoch + 1}.safetensors'))
                accelerator.unwrap_model(model).save_weights(
                    os.path.join(path_to_save, f'checkpoint_lora_{epoch + 1}.safetensors'), merge_weights=False)
            elif not args.use_lora and args.train_head_only:
                save_file(accelerator.unwrap_model(model).state_dict(),
                          os.path.join(path_to_save, f'checkpoint_headonly_{epoch + 1}.safetensors'))
            else:
                save_file(accelerator.unwrap_model(model).state_dict(),
                          os.path.join(path_to_save, f'checkpoint_fulltrain_{epoch + 1}.safetensors'))

            checkpoints = os.listdir(path_to_save)
            if len(checkpoints) > args.max_no_of_checkpoints:
                checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoints_to_delete = checkpoints_sorted[:-args.max_no_of_checkpoints]
                for checkpoint_to_delete in checkpoints_to_delete:
                    path_to_checkpoint_to_delete = os.path.join(path_to_save, checkpoint_to_delete)
                    if os.path.isdir(path_to_checkpoint_to_delete):
                        shutil.rmtree(path_to_checkpoint_to_delete)

        accelerator.wait_for_everyone()
        accelerator.print(f'Saved. End of epoch {epoch + 1}.')

    accelerator.print('END OF TRAINING.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
