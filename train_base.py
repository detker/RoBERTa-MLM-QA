import os
import argparse
from tqdm import tqdm
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from datasets import load_from_disk
from transformers import RobertaTokenizerFast, get_scheduler, set_seed
from accelerate import Accelerator

from src.utils import RobertaConfig, RobertaMaskedLMCollateFun
from src.model import RoBERTaForMLM


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--experiment_name',
        required=True,
        type=str
    )

    parser.add_argument(
        '--working_directory',
        required=True,
        type=str
    )

    parser.add_argument(
        '--hf_model_name',
        help='Huggingface model name we want to use for the tokenizer',
        default='FacebookAI/roberta-base',
        type=str
    )

    parser.add_argument(
        '--path_to_prepared_data',
        required=True,
        type=str
    )

    parser.add_argument(
        '--context_length',
        help='Max sequence length we want the model to accept',
        default=512,
        type=int
    )

    parser.add_argument(
        '--masking_probability',
        help='Probability of token to be selected to be masked',
        default=0.15,
        type=float
    )

    parser.add_argument(
        '--num_workers',
        help='Number of workers for dataloading',
        default=24,
        type=int
    )

    parser.add_argument(
        '--hidden_dropout_p',
        help='Dropout probability on all linear layers',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--attention_dropout_p',
        help='Dropout probability on attention matrix',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--num_transformer_blocks',
        help='Number of transformer blocks in model',
        default=12,
        type=int
    )

    parser.add_argument(
        '--num_attention_heads',
        help='Number of heads of attention',
        default=12,
        type=int
    )

    parser.add_argument(
        '--embedding_dimension',
        help='Transformer embedding dimension',
        default=768,
        type=int
    )

    parser.add_argument(
        '--mlp_ratio',
        help='Hidden layer expansion factor for feed forward layers',
        default=4,
        type=int
    )

    parser.add_argument(
        '--layer_norm_eps',
        help='error added to layer norm to avoid divide by zero',
        default=1e-5,
        type=float
    )

    parser.add_argument(
        '--initializer_range',
        help='Standard deviation of linear layers initialized as normal distribution',
        default=0.02,
        type=float
    )

    parser.add_argument(
        '--per_gpu_batch_size',
        help='Overall batch size per gpu during training',
        default=128,
        type=int
    )

    parser.add_argument(
        '--gradient_accumulation_steps',
        help='Splits per_gpu_batch_size by gradient_accumulation_steps',
        default=4,
        type=int
    )

    parser.add_argument(
        '--num_training_steps',
        help='Number of training steps to take',
        default=250000,
        type=int
    )

    parser.add_argument(
        '--max_grad_norm',
        help='Max gradient norm used for stabilizing training with gradient clipping',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--num_warmup_steps',
        type=int,
        default=20000,
        help='Number of steps for the warmup in the lr scheduler.'
    )

    parser.add_argument(
        '--lr_scheduler_type',
        type=str,
        default='linear',
        help='The scheduler type to use.',
        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
    )

    parser.add_argument(
        '--logging_steps',
        help='Number of iterations for every log of metrics to wandb',
        default=1,
        type=int
    )

    parser.add_argument(
        '--evaluation_interval',
        help='Number of iterations for every evaluation and plotting',
        default=2500,
        type=int
    )

    parser.add_argument(
        '--checkpoint_interval',
        help='Number of iterations for checkpointing',
        default=2500,
        type=int
    )

    parser.add_argument(
        '--learning_rate',
        help='Max learning rate for all Learning Rate Schedulers',
        default=6e-4,
        type=float
    )

    parser.add_argument(
        '--bias_weight_decay',
        help='Apply weight decay to bias',
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--norm_weight_decay',
        help='Apply weight decay to normalization weight and bias',
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--weight_decay',
        help='Weight decay constant for AdamW optimizer',
        default=0.01,
        type=float
    )

    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='Beta1 for AdamW optimizer',
    )

    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.98,
        help='Beta2 for AdamW optimizer',
    )

    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=1e-6,
        help='Epsilon for AdamW optimizer',
    )

    parser.add_argument(
        '--num_keep_checkpoints',
        help='Number of Checkpoints to Keep, if None, all checkpoints will be saved',
        default=None,
        type=int
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        help='Checkpoint folder for model to resume training from, inside the experiment folder',
        default=None,
        type=str
    )

    parser.add_argument(
        '--wandb',
        help='Flag to enable logging to wandb',
        default=False,
        action=argparse.BooleanOptionalAction
    )


# parse args
parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

# set seed
if args.seed is not None:
    set_seed(args.seed)

# set experiment w/w accelerator & wandb platform logging
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, log_with='wandb' if args.wandb else None)
if args.wandb:
    accelerator.init_trackers(args.experiment_name)

# load roberta tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(args.hf_model_name)

# prepare config from args
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    start_token=tokenizer.bos_token_id,
    end_token=tokenizer.eos_token_id,
    pad_token=tokenizer.pad_token_id,
    mask_token=tokenizer.mask_token_id,
    embd_dim=args.embedding_dimension,
    n_transformer_blocks=args.num_transformer_blocks,
    n_heads=args.num_attention_heads,
    mlp_ratio=args.mlp_ratio,
    layer_norm_eps=args.layer_norm_eps,
    hidden_drop_p=args.hidden_dropout_p,
    att_drop_p=args.attention_dropout_p,
    context_length=args.context_length,
    masking_prob=args.masking_probability,
    hf_model_name=args.hf_model_name
)

# load model
model = RoBERTaForMLM(config)
model = model.to(accelerator.device)

# prepare dataloaders
tokens = load_from_disk(args.path_to_prepared_data)

collate_fn = RobertaMaskedLMCollateFun(config)
minibatch_size = args.per_gpu_batch_size // args.gradient_accumulation_steps

trainloader = DataLoader(tokens['train'],
                         batch_size=minibatch_size,
                         shuffle=True,
                         collate_fn=collate_fn)

testloader = DataLoader(tokens['test'],
                        batch_size=minibatch_size,
                        shuffle=False,
                        collate_fn=collate_fn)

# optim&scheduler setup
optimizer = None
if (not args.bias_weight_decay) or (not args.norm_weight_decay):
    accelerator.print('Disabling weight decay on some parameters')
    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name and not args.bias_weight_decay:
                non_weight_decay_params.append(param)
            elif 'groupnorm' in name and not args.norm_weight_decay:
                non_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {'params': weight_decay_params, 'weight_decay': args.weight_decay},
        {'params': non_weight_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = optim.AdamW(optimizer_group,
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon)

else:
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_epsilon,
                            weight_decay=args.weight_decay)

scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps
)

# prep all
model, optimizer, trainloader, testloader = accelerator.prepare([model, optimizer, trainloader, testloader])
accelerator.register_for_checkpointing(scheduler)

accuracy_f = Accuracy(task='multiclass', num_classes=config.vocab_size, ignore_index=-100).to(accelerator.device)

### training

print(f"training on: {'cuda' if torch.cuda.is_available() else 'cpu'}).")

completed_steps = 0
if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)

    completed_steps = int(args.resume_from_checkpoint.split('_')[-1])
    accelerator.print(f'Checkpoint detected. Resuming from iteration {completed_steps}.')

train = True
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_local_main_process)

while train:
    accum_steps = 0
    accum_loss = 0
    accum_acc = 0
    for batch in trainloader:
        batch = {k:v.to(accelerator.device) for k,v in batch.items()}
        hidd_states, logits, loss = model(**batch)
        loss = loss / args.gradient_accumulation_steps
        accum_loss += loss

        accelerator.backward(loss)

        labels = batch['labels'].reshape(-1)
        acc = accuracy_f(logits, labels) / args.gradient_accumulation_steps
        accum_acc += acc

        accum_steps += 1

        # update model
        if accum_steps % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            if completed_steps % args.logging_steps == 0:
                accum_loss = accum_loss.detach()
                accum_acc = accum_acc.detach()

                if accelerator.state.num_processes > 1:
                    accum_loss = torch.mean(accelerator.gather_for_metrics(accum_loss))
                    accum_acc = torch.mean(accelerator.gather_for_metrics(accum_acc))

                log = {'train_loss': accum_loss,
                       'train_acc': accum_acc,
                       'lr': scheduler.get_last_lr()[0]}
                if accelerator.is_main_process:
                    progress_bar.write(f'[{completed_steps}/{args.num_training_steps}] steps. Loss: {accum_loss:.4f}, Accuracy: {accum_acc:.4f}')
                if args.wandb:
                    accelerator.log(log, step=completed_steps)

            if completed_steps % args.evaluation_interval == 0 and completed_steps > 0:
                if accelerator.is_main_process:
                    progress_bar.write('Evaluating model')

                model.eval()

                log = {'test_loss': 0, 'test_acc': 0}
                num_losses = 0
                for batch in tqdm(testloader, disable=not accelerator.is_local_main_process):
                    batch = {k:v.to(accelerator.device) for k,v in batch.items()}
                    with torch.no_grad():
                        hidd_states, logits, loss = model(**batch)

                    loss = loss.detatch()
                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))

                    labels = batch['labels'].reshape(-1)
                    accuracy = accuracy_f(logits, labels)
                    if accelerator.num_processes > 1:
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                    log['test_loss'] += loss
                    log['test_acc'] += accuracy
                    num_losses += 1

                log['test_loss'] /= num_losses
                log['test_acc'] /= num_losses

                if accelerator.is_main_process:
                    progress_bar.write(f"[{completed_steps}/{args.num_training_steps}] Test loss: {log['test loss']:.4f}, Test accuracy: {log['test acc']:.4f}.")

                if args.wandb:
                    accelerator.log(log, step=completed_steps)

                model.train()

            if completed_steps % args.checkpoint_interval == 0:
                path_to_checkpoint = os.path.join(path_to_experiment, f'checkpoint_{completed_steps}')

                if accelerator.is_main_process:
                    progress_bar.write(f'saving checkpoint to {path_to_checkpoint}...')

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    accelerator.save_state(output_dir=path_to_checkpoint)

                if args.num_keep_checkpoints is not None:
                    if accelerator.is_main_process:
                        checkpoints = os.listdir(path_to_experiment)
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('.')[0].split('_')[-1]))

                        if len(checkpoints) > args.num_keep_checkpoints:
                            checkpoints_to_delete = checkpoints[:-args.num_keep_checkpoints]

                            for checkpoint_to_delete in checkpoints_to_delete:
                                path_to_checkpoint_to_delete = os.path.join(path_to_experiment, checkpoint_to_delete)
                                if os.path.isdir(path_to_checkpoint_to_delete):
                                    shutil.rmtree(path_to_checkpoint_to_delete)

                accelerator.wait_for_everyone()

            if completed_steps >= args.num_training_steps:
                train = False
                if accelerator.is_main_process:
                    progress_bar.write('end of training')
                break

            completed_steps += 1
            progress_bar.update(1)

            accum_acc = 0
            accum_loss = 0
            torch.cuda.empty_cache()

accelerator.end_training()
