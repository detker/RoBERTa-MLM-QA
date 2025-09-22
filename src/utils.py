import torch
import numpy as np
from tqdm import tqdm
from typing import Literal
from transformers import RobertaTokenizerFast
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
import random

# hf typical config prep
@dataclass
class RobertaConfig:
    vocab_size: int = 50265
    start_token: int = 0
    end_token: int = 2
    pad_token: int = 1
    mask_token: int = 50264

    embd_dim: int = 768
    n_transformer_blocks: int = 12
    n_heads: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-6
    hidden_drop_p: float = 0.1
    att_drop_p: float = 0.1
    context_length: int = 512

    masking_prob: int = 0.15 # % of tokens that are candidates for masking

    hf_model_name: str = 'FacebookAI/roberta-base'
    pretrained_backbone: Literal['pretrained', 'pretrained_huggingface', 'random'] = 'pretrained_huggingface'
    path_to_pretrained_model: str = None
    gradient_checkpointing: bool = False

    def to_dict(self):
        return asdict(self)

def mask_tokens(tokens, special_tokens_mask, vocab_size, special_ids, masking_prob, mask_token):
    non_special_tokens = list(set(range(vocab_size)) - set(special_ids))

    uniform_sample = torch.rand(*tokens.shape)
    uniform_sample[special_tokens_mask == 1] = 1.0

    mask = (uniform_sample < 0.15)

    labels = torch.full(tokens.shape, fill_value=-100)
    labels[mask] = tokens[mask]

    candidates_indexes = mask.nonzero()

    uniform_sample = torch.rand(len(candidates_indexes))
    chosen_tokens_to_mask = (uniform_sample < 0.8)
    chosen_idxs_to_mask = candidates_indexes[chosen_tokens_to_mask]
    not_chosen_idxs_to_mask = candidates_indexes[~chosen_tokens_to_mask]

    uniform_sample = torch.rand(len(not_chosen_idxs_to_mask))
    chosen_tokens_to_fill = (uniform_sample < 0.5)
    chosen_idxs_to_fill = not_chosen_idxs_to_mask[chosen_tokens_to_fill]
    chosen_idxs_to_ignore = not_chosen_idxs_to_mask[~chosen_tokens_to_fill]

    if len(chosen_idxs_to_mask) > 0:
        tokens[chosen_idxs_to_mask[:,0], chosen_idxs_to_mask[:, 1]] = mask_token

    if len(chosen_idxs_to_fill) > 0:
        sampled_tokens_to_fill = torch.tensor(random.sample(non_special_tokens, len(chosen_idxs_to_fill)))
        tokens[chosen_idxs_to_fill[:,0], chosen_idxs_to_fill[:,1]] = sampled_tokens_to_fill

    return tokens, labels


def RobertaMaskedLMCollateFun(config):
    tokenizer = RobertaTokenizerFast.from_pretrained(config.hf_model_name)

    def _collate_fn(batch):
        tokens = [torch.tensor(sample['input_ids']) for sample in batch]
        padding_att_mask = [torch.ones(sample.shape) for sample in tokens]
        padding_att_mask = torch.nn.utils.rnn.pad_sequence(padding_att_mask, padding_value=0, batch_first=True)
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, padding_value=config.pad_token, batch_first=True)
        special_tokens_mask = torch.isin(tokens, torch.tensor(tokenizer.all_special_ids))

        tokens, labels = mask_tokens(tokens=tokens,
                                     special_tokens_mask=special_tokens_mask,
                                     vocab_size=config.vocab_size,
                                     special_ids=tokenizer.all_special_ids,
                                     masking_prob=config.masking_prob,
                                     mask_token=config.mask_token)

        return {
            'input_ids': tokens,
            'attention_mask': padding_att_mask,
            'labels': labels
        }

    return _collate_fn

def QAProcessor():
    tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')

    def chars2tokens(examples):
        questions = [q.strip() for q in examples['question']]

        inputs = tokenizer(
            text=questions,
            text_pair=examples['context'],
            max_length=512,
            truncation='only_second',
            return_offsets_mapping=True,
            padding='max_length'
        )

        offsets = inputs.pop('offset_mapping')
        answers = examples['answers']
        starting_tokens_idxs = []
        ending_tokens_idxs = []

        for i, offset in enumerate(offsets):
            answer = answers[i]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])

            seq_ids = inputs.sequence_ids(i)

            seq_ids_np = np.array([1 if x is not None and x == 1 else 0 for x in seq_ids])
            context_start, context_end = np.argwhere(seq_ids_np == 1)[[0, -1]]
            context_start, context_end = context_start[0], context_end[0]
            context_start_char = offset[context_start][0]
            context_end_char = offset[context_end][-1]

            if context_start_char <= start_char and context_end_char >= end_char:
                starting_token_idx = None
                ending_token_idx = None
                for idx, (off, seq_id) in enumerate(zip(offset, seq_ids)):
                    if seq_id == 1:
                        if start_char in range(off[0], off[-1]+1):
                            starting_token_idx = idx
                        if end_char in range(off[0], off[-1]+1):
                            ending_token_idx = idx
                starting_tokens_idxs.append(starting_token_idx)
                ending_tokens_idxs.append(ending_token_idx)
            else:
                starting_tokens_idxs.append(0)
                ending_tokens_idxs.append(0)

        inputs['start_pos'] = starting_tokens_idxs
        inputs['end_pos'] = ending_tokens_idxs

        return inputs

    return chars2tokens


if __name__ == '__main__':
    # path_to_data_storage = 'data/roberta_data'
    # data = load_from_disk(path_to_data_storage)
    #
    # config = RobertaConfig()
    # collate_fn = RobertaMaskedLMCollateFun(config)
    # trainloader = DataLoader(data['train'], batch_size=4, collate_fn=collate_fn)
    #
    # for sample in tqdm(trainloader):
    #     print(sample['input_ids'])
    #     print(sample['attention_mask'])
    #     print(sample['labels'])
    #     break
    from datasets import load_dataset
    dataset = load_dataset('squad')
    processor = QAProcessor()
    processor(dataset['train'][:4])

