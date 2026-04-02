import sys
sys.path.append('../')

import torch
from transformers import RobertaTokenizerFast
try:
    from inference.hf_pretrained_model import RobertaConfigHF, RobertaForQAHF
except ModuleNotFoundError:
    from hf_pretrained_model import RobertaConfigHF, RobertaForQAHF
from transformers import AutoModel, AutoConfig

import warnings
warnings.filterwarnings('ignore')


class Inference:
    def load_model(self):
        AutoConfig.register('roberta-qa', RobertaConfigHF)
        AutoModel.register(RobertaConfigHF, RobertaForQAHF)
        self.config = AutoConfig.from_pretrained('detker/roberta-qa-125M')
        self.model = AutoModel.from_pretrained('detker/roberta-qa-125M',
                                        trust_remote_code=True,
                                        use_safetensors=True,
                                        config=self.config)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.config.hf_model_name)
        self.model.eval()

    def predict(self, question, context):
        inputs = self.tokenizer(
            text=question,
            text_pair=context,
            max_length=self.config.context_length,
            truncation='only_second',
            return_tensors='pt'
        )

        with torch.no_grad():
            start_logits, end_logits = self.model(inputs)

        start_token_idx = start_logits.squeeze().argmax().item()
        end_token_idx = end_logits.squeeze().argmax().item()

        tokens = inputs['input_ids'].squeeze()[start_token_idx:end_token_idx+1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        return start_token_idx, end_token_idx, answer

