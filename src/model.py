import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import RobertaModel as RobertaHF

import src.utils


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.pos_embd = nn.Embedding(config.context_length, config.embd_dim)
        self.norm = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_drop_p)


    def forward(self, x):
        # (B, S) -> (B, S, E)
        batch_size, seq_len = x.shape
        x = self.embd(x)

        seq_len_indicies = torch.arange(start=0, end=seq_len, dtype=torch.long, device=x.device)
        pos = self.pos_embd(seq_len_indicies).unsqueeze(0)

        x = x + pos

        x = self.norm(x)
        x = self.dropout(x)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.embd_dim % config.n_heads == 0
        self.head_dim = config.embd_dim // config.n_heads

        self.q = nn.Linear(config.embd_dim, config.embd_dim)
        self.k = nn.Linear(config.embd_dim, config.embd_dim)
        self.v = nn.Linear(config.embd_dim, config.embd_dim)

        self.proj_out = nn.Linear(config.embd_dim, config.embd_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embd_dim = x.shape

        # q,k,v: (B, S, E) -> (B, n_heads, S, head_dim)
        q = self.q(x).reshape(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k(x).reshape(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v(x).reshape(batch_size, seq_len, self.config.n_heads, self.head_dim).transpose(1, 2).contiguous()

        # (B, n_heads, S, head_dim) -> (B, n_heads, S, head_dim)
        out_att = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.config.att_drop_p if self.training else 0.0
        )

        # (B, n_heads, S, head_dim) -> (B, S, E)
        out_att = out_att.transpose(1, 2).reshape(batch_size, seq_len, embd_dim).contiguous()

        # (B, S, E) -> (B, S, E)
        out = self.proj_out(out_att)

        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config.embd_dim, config.embd_dim*config.mlp_ratio)
        self.dropout1 = nn.Dropout(config.hidden_drop_p)
        self.activation_f = nn.GELU()
        self.linear2 = nn.Linear(config.embd_dim*config.mlp_ratio, config.embd_dim)
        self.dropout2 = nn.Dropout(config.hidden_drop_p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_f(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config.att_drop_p)
        self.norm1 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)

        self.ff = FeedForwardLayer(config)
        self.norm2 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attention(x, attention_mask))
        x = self.norm1(x)

        x = x + self.ff(x)
        x = self.norm2(x)

        return x

class RoBERTaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(0, config.n_transformer_blocks)
        ])

    def forward(self, x, attention_mask=None):
        _, seq_len, _ = x.shape
        if attention_mask is not None:
            # (B, S) -> (B, 1, 1, S) -> (B, 1, S, S) ### for FlashAtttention :3
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x

class HeadForMaskedLanguageModeling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(config.embd_dim, config.embd_dim)
        self.activation_f = nn.GELU()
        self.norm = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.embd_dim, config.vocab_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation_f(x)
        x = self.norm(x)
        x = self.decoder(x)

        return x

class RoBERTa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = Embedding(config)
        self.encoder = RoBERTaEncoder(config)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)  # (B, S) -> (B, S, E)
        x = self.encoder(x, attention_mask)  # (B, S, E) -> (B, S, E)
        return x

class RoBERTaForMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.roberta = RoBERTa(config)
        self.head = HeadForMaskedLanguageModeling(config)

        self.apply(init_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        roberta_out_hidden = self.roberta(input_ids, attention_mask)
        logits = self.head(roberta_out_hidden)

        loss = None
        if labels is not None:
            logits = logits.reshape(batch_size*seq_len, -1)
            labels = labels.reshape(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            return roberta_out_hidden, logits, loss

        return roberta_out_hidden, logits

def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(0.0, 0.02)
        # if module.padding_idx is not None:
        #     module.weight.data[module.padding_idx].zero_()

class RobertaForQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = None
        self.config = config
        self.build_model()
        self.head = nn.Linear(config.embd_dim, 2)

    def build_model(self):
        if self.config.pretrained_backbone == 'pretrained_huggingface':
            self.model = RobertaHF.from_pretrained(self.config.hf_model_name)
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model = RoBERTa(self.config)
            if self.config.pretrained_backbone == 'pretrained':
                state_dict = load_file(self.config.path_to_pretrained_model)
                new_state_dict = {}

                for k,v in state_dict.items():
                    if 'roberta.' in k:
                        new_state_dict[k.replace('roberta.', '')] = v
                self.model.load_state_dict(new_state_dict)

    def forward(self,
                input_ids,
                attention_mask=None,
                start_pos=None,
                end_pos=None):
        if self.config.pretrained_backbone == 'pretrained_huggingface':
            output = self.model(input_ids, attention_mask).last_hidden_state
        else:
            output = self.model(input_ids, attention_mask)

        logits = self.head(output)
        start_logits, end_logits = torch.chunk(logits, chunks=2, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        if start_pos is not None and end_pos is not None:
            if len(start_pos.shape) > 1:
                start_pos = start_pos.flatten()
            if len(end_pos.shape) > 1:
                end_pos = end_pos.flatten()

            ignored_idxs_start = logits.shape[1]
            start_pos = start_pos.clamp(0, ignored_idxs_start)
            end_pos = end_pos.clamp(0, ignored_idxs_start)

            loss_start_pos = F.cross_entropy(start_logits, start_pos, ignore_index=ignored_idxs_start)
            loss_end_pos = F.cross_entropy(end_logits, end_pos, ignore_index=ignored_idxs_start)

            total_loss = (loss_start_pos+loss_end_pos) / 2

            return total_loss, start_logits, end_logits
        else:
            return start_logits, end_logits


if __name__ == '__main__':
    x = torch.randint(0, 1000, size=(4, 8))
    config = utils.RobertaConfig()
    model = RoBERTaForMLM(config)
    hidden_states, logits = model(x)
    print(hidden_states)
    print(hidden_states.shape)
    print(logits)
    print(logits.shape)
    test = RobertaForQA(config)
    start_pos = torch.tensor([1,2,2,4]).unsqueeze(-1)
    end_pos = torch.tensor([3,4,5,6]).unsqueeze(-1)
    print(test(x, start_pos=start_pos, end_pos=end_pos))
