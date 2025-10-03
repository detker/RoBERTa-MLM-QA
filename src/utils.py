import random
from typing import Literal
from dataclasses import dataclass, asdict

import numpy as np

import torch
from transformers import RobertaTokenizerFast


@dataclass
class RobertaConfig:
    """
    Configuration class for the RoBERTa model.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param start_token: Token ID for the start token.
    :type start_token: int
    :param end_token: Token ID for the end token.
    :type end_token: int
    :param pad_token: Token ID for the padding token.
    :type pad_token: int
    :param mask_token: Token ID for the mask token.
    :type mask_token: int
    :param embd_dim: Dimension of the embeddings.
    :type embd_dim: int
    :param n_transformer_blocks: Number of transformer blocks.
    :type n_transformer_blocks: int
    :param n_heads: Number of attention heads.
    :type n_heads: int
    :param mlp_ratio: Ratio for the feed-forward layer expansion.
    :type mlp_ratio: int
    :param layer_norm_eps: Epsilon value for layer normalization.
    :type layer_norm_eps: float
    :param hidden_drop_p: Dropout probability for hidden layers.
    :type hidden_drop_p: float
    :param att_drop_p: Dropout probability for attention layers.
    :type att_drop_p: float
    :param context_length: Maximum context length.
    :type context_length: int
    :param masking_prob: Probability of masking tokens.
    :type masking_prob: float
    :param hf_model_name: Name of the Hugging Face model.
    :type hf_model_name: str
    :param pretrained_backbone: Type of pretrained backbone to use.
    :type pretrained_backbone: Literal['pretrained', 'pretrained_huggingface', 'random']
    :param path_to_pretrained_model: Path to the pretrained model.
    :type path_to_pretrained_model: str
    :param gradient_checkpointing: Whether to enable gradient checkpointing.
    :type gradient_checkpointing: bool
    """
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
    """
    Masks tokens for masked language modeling.

    :param tokens: Input tensor of token IDs.
    :type tokens: torch.Tensor
    :param special_tokens_mask: Mask indicating special tokens.
    :type special_tokens_mask: torch.Tensor
    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param special_ids: List of special token IDs.
    :type special_ids: list
    :param masking_prob: Probability of masking tokens.
    :type masking_prob: float
    :param mask_token: Token ID for the mask token.
    :type mask_token: int
    :return: Tuple containing masked tokens and labels.
    :rtype: tuple
    """
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

    if len(chosen_idxs_to_mask) > 0:
        tokens[chosen_idxs_to_mask[:,0], chosen_idxs_to_mask[:, 1]] = mask_token

    if len(chosen_idxs_to_fill) > 0:
        sampled_tokens_to_fill = torch.tensor(random.sample(non_special_tokens, len(chosen_idxs_to_fill)))
        tokens[chosen_idxs_to_fill[:,0], chosen_idxs_to_fill[:,1]] = sampled_tokens_to_fill

    return tokens, labels


def RobertaMaskedLMCollateFun(config):
    """
    Creates a collate function for preparing batches of data for masked language modeling.

    :param config: Configuration object containing model parameters.
    :type config: RobertaConfig
    :return: Collate function for DataLoader.
    :rtype: function
    """
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
    """
    Creates a processor for question-answering tasks. Tokenizes input questions and contexts,
    aligns character-level answer spans to token-level indices, and prepares the data for training.

    :return: Function to process question-answering datasets.
    :rtype: function
    """
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

