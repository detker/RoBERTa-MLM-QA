from datasets import load_dataset, concatenate_datasets
from transformers import RobertaTokenizerFast

import warnings
warnings.filterwarnings('ignore')

train_test_split_ratio = 0.005 # ratio of test samples
context_length = 512
path_to_data_storage = 'data/roberta_data'
huggingface_cache_dir = 'data/huggingface_cache'
seed = 42
n_workers = 32
huggingface_model = 'FacebookAI/roberta-base'

tokenizer = RobertaTokenizerFast.from_pretrained(huggingface_model)

wiki_dataset = load_dataset('wikipedia', '20220301.en', cache_dir=huggingface_cache_dir, trust_remote_code=True)
books_dataset = load_dataset('bookcorpus/bookcorpus', cache_dir=huggingface_cache_dir, trust_remote_code=True)

wiki_dataset = wiki_dataset.select_columns(['text'])
books_dataset = books_dataset.select_columns(['text'])
wiki_dataset, books_dataset = wiki_dataset['train'], books_dataset['train']

dataset = concatenate_datasets([wiki_dataset, books_dataset])

dataset = dataset.train_test_split(test_size=train_test_split_ratio, seed=42)

def clean_data(examples):
    ret = [' '.join(example.replace('\n', ' ').replace('\t', ' ').split()) for example in examples['text']]
    return tokenizer(ret, return_attention_mask=False)

clean_dataset = dataset.map(
    function=clean_data,
    batched=True,
    num_proc=n_workers,
    remove_columns='text'
)


def group(batch):
    concat_sents = []
    for sent in batch['input_ids']:
        concat_sents.extend(sent)

    ret = [concat_sents[i:i+context_length] for i in range(0, len(concat_sents), context_length)]

    data = {'input_ids': ret}

    return data

tokens = clean_dataset.map(
    function=group,
    batched=True,
    num_proc=8
)

tokens.save_to_disk(path_to_data_storage)

