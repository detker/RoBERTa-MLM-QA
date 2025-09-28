import sys
sys.path.append('../')

from typing import Literal
from transformers import PretrainedConfig, PreTrainedModel

from src.model import RobertaForQA

class RobertaConfigHF(PretrainedConfig):
    model_type = 'roberta-qa'

    def __init__(self,
                 vocab_size: int = 50265,
                 start_token: int = 0,
                 end_token: int = 2,
                 pad_token: int = 1,
                 mask_token: int = 50264,
                 embd_dim: int = 768,
                 n_transformer_blocks: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 layer_norm_eps: float = 1e-6,
                 hidden_drop_p: float = 0.1,
                 att_drop_p: float = 0.1,
                 context_length: int = 512,
                 masking_prob: int = 0.15,
                 hf_model_name: str = 'FacebookAI/roberta-base',
                 pretrained_backbone: Literal['pretrained', 'pretrained_huggingface', 'random'] = 'pretrained_huggingface',
                 path_to_pretrained_model: str = None,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.embd_dim = embd_dim
        self.n_transformer_blocks = n_transformer_blocks
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.hidden_drop_p = hidden_drop_p
        self.att_drop_p = att_drop_p
        self.context_length = context_length
        self.masking_prob = masking_prob
        self.hf_model_name = hf_model_name
        self.pretrained_backbone = pretrained_backbone
        self.path_to_pretrained_model = path_to_pretrained_model
        self.gradient_checkpointing = gradient_checkpointing

class RobertaForQAHF(PreTrainedModel):
    config_class = RobertaConfigHF

    def __init__(self, config):
        super().__init__(config)
        self.model = RobertaForQA(config)

    def forward(self, x):
        return self.model(x)
