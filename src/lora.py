from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import bitsandbytes.nn as bnn
from safetensors.torch import save_file


@dataclass
class LoRAConfig:
    """
    Configuration class for LoRA / QLoRA (Low-Rank Adaptation).

    When ``use_qlora`` is True the targeted Linear/Conv2d base weights are frozen
    in 4-bit precision via bitsandbytes (QLoRA), otherwise they stay in full
    precision (plain LoRA). The trainable lora_A / lora_B adapters are always full
    precision.

    :param lora_target_modules: List of module names to apply LoRA or a single module name.
    :type lora_target_modules: Optional[Union[list[str], str]]
    :param lora_exclude_modules: List of module names to exclude from LoRA or a single module name.
    :type lora_exclude_modules: Optional[Union[list[str], str]]
    :param lora_bias: Type of bias handling. Options: "none", "all", "lora_only".
    :type lora_bias: Literal["none", "all", "lora_only"]
    :param lora_rank: Rank of the low-rank decomposition.
    :type lora_rank: int
    :param lora_alpha: Scaling factor for LoRA.
    :type lora_alpha: float
    :param lora_dropout: Dropout probability for LoRA layers.
    :type lora_dropout: float
    :param lora_use_rslora: Whether to use root-scaled LoRA.
    :type lora_use_rslora: bool
    :param use_qlora: Whether to quantize the frozen base weights to 4-bit (QLoRA).
    :type use_qlora: bool
    :param lora_quant_type: 4-bit quantization data type. Options: "nf4", "fp4".
    :type lora_quant_type: Literal["nf4", "fp4"]
    :param lora_compute_dtype: Dtype used for the dequantized forward computation.
    :type lora_compute_dtype: torch.dtype
    :param lora_compress_statistics: Whether to use double quantization (quantize the quantization constants).
    :type lora_compress_statistics: bool
    """
    lora_target_modules: Optional[Union[list[str], str]] = None
    lora_exclude_modules: Optional[Union[list[str], str]] = None
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_use_rslora: bool = True
    use_qlora: bool = False
    lora_quant_type: Literal["nf4", "fp4"] = "nf4"
    lora_compute_dtype: torch.dtype = torch.bfloat16
    lora_compress_statistics: bool = True

    @property
    def quant_kwargs(self):
        """
        Keyword arguments forwarded to ``bitsandbytes.nn.Linear4bit``.

        :return: Quantization keyword arguments.
        :rtype: dict
        """
        return {
            'quant_type': self.lora_quant_type,
            'compute_dtype': self.lora_compute_dtype,
            'compress_statistics': self.lora_compress_statistics,
        }


class LoRABaseLayer:
    """
    Base class for LoRA layers.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    """
    def __init__(self,
                 lora_config):
        self.rank = lora_config.lora_rank
        self.lora_alpha = lora_config.lora_alpha
        self.lora_dropout = nn.Dropout(lora_config.lora_dropout) if lora_config.lora_dropout > 0.0 else nn.Identity()
        self.scaling = lora_config.lora_alpha / (lora_config.lora_rank ** 0.5 if lora_config.lora_use_rslora else lora_config.lora_rank)

    def load_weights_to_lora_layer(self, state_dict):
        """
        Loads weights into the LoRA layer.

        :param state_dict: State dictionary containing weights and biases.
        :type state_dict: dict
        """
        self.weight.data = state_dict['weight']
        if 'bias' in state_dict.keys():
            self.bias.data = state_dict['bias']


class LoRALinearLayer(nn.Linear, LoRABaseLayer):
    """
    LoRA-enhanced Linear layer.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    :param input_dim: Number of input features.
    :type input_dim: int
    :param output_dim: Number of output features.
    :type output_dim: int
    :param bias: Whether to include a bias term.
    :type bias: bool
    """
    def __init__(self,
                 lora_config,
                 input_dim,
                 output_dim,
                 bias=True,
                 **kwargs):
        nn.Linear.__init__(self, in_features=input_dim, out_features=output_dim, bias=bias, **kwargs)
        LoRABaseLayer.__init__(self, lora_config=lora_config)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(size=(input_dim, self.rank)))
        self.lora_B = nn.Parameter(torch.zeros(size=(self.rank, output_dim)))

        torch.nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

    def _merge_weights(self):
        merged_weights = self.weight.data + (self.lora_A @ self.lora_B).T*self.scaling

        state_dict = {'weight': merged_weights}
        if self.bias is not None:
            state_dict['bias'] = self.bias

        merged_linear = nn.Linear(self.in_features, self.out_features, bias=True if self.bias is not None else False)
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):
        """
        Forward pass for the LoRA-enhanced Linear layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        linear_out = nn.Linear.forward(self,x)

        output = linear_out + (self.lora_dropout(x) @ self.lora_A) @ self.lora_B * self.scaling

        return output


class QLoRALinearLayer(bnn.Linear4bit, LoRABaseLayer):
    """
    QLoRA-enhanced Linear layer. A wrapper around ``bitsandbytes.nn.Linear4bit``
    that stores the frozen base weights in 4-bit precision and adds the trainable
    lora_A / lora_B adapters in full precision.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    :param input_dim: Number of input features.
    :type input_dim: int
    :param output_dim: Number of output features.
    :type output_dim: int
    :param bias: Whether to include a bias term.
    :type bias: bool
    """
    def __init__(self,
                 lora_config,
                 input_dim,
                 output_dim,
                 bias=True,
                 **kwargs):
        bnn.Linear4bit.__init__(self,
                                input_features=input_dim,
                                output_features=output_dim,
                                bias=bias,
                                **lora_config.quant_kwargs,
                                **kwargs)
        LoRABaseLayer.__init__(self, lora_config=lora_config)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(size=(input_dim, self.rank)))
        self.lora_B = nn.Parameter(torch.zeros(size=(self.rank, output_dim)))

        torch.nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

    def _merge_weights(self):
        # dequantize the 4-bit base weights back to full precision before merging
        if self.weight.quant_state is not None:
            weight = bnb.functional.dequantize_4bit(self.weight.data.clone(), self.weight.quant_state)
        else:
            weight = self.weight.data.clone()

        merged_weights = weight + (self.lora_A @ self.lora_B).T*self.scaling

        state_dict = {'weight': merged_weights}
        if self.bias is not None:
            state_dict['bias'] = self.bias

        merged_linear = nn.Linear(self.in_features, self.out_features, bias=True if self.bias is not None else False)
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):
        """
        Forward pass for the QLoRA-enhanced Linear layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        linear_out = bnn.Linear4bit.forward(self, x)

        output = linear_out + (self.lora_dropout(x) @ self.lora_A) @ self.lora_B * self.scaling

        return output


class LoRAEmbeddingLayer(nn.Embedding, LoRABaseLayer):
    """
    LoRA-enhanced Embedding layer. Shared by LoRA and QLoRA: bitsandbytes has no
    4-bit Embedding, so the base weights always stay in full precision here while
    the lora_A / lora_B adapters are trained as usual.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    :param num_embeddings: Number of embeddings.
    :type num_embeddings: int
    :param embedding_dim: Dimension of each embedding.
    :type embedding_dim: int
    """
    def __init__(self,
                 lora_config,
                 num_embeddings,
                 embedding_dim,
                 **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRABaseLayer.__init__(self, lora_config=lora_config)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(size=(num_embeddings, self.rank)))
        self.lora_B = nn.Parameter(torch.zeros(size=(self.rank, embedding_dim)))

        torch.nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

    def _merge_weights(self):
        merged_weights = self.weight.data + (self.lora_A @ self.lora_B)*self.scaling

        state_dict = {'weight': merged_weights}

        merged_embd = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_embd.load_state_dict(state_dict)

        return merged_embd

    def forward(self, x):
        """
        Forward pass for the LoRA-enhanced Embedding layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        org_embd = nn.Embedding.forward(self, x)
        lora_A_embd = F.embedding(input=x,
                                  weight=self.lora_A,
                                  padding_idx=self.padding_idx,
                                  max_norm=self.max_norm,
                                  norm_type=self.norm_type,
                                  scale_grad_by_freq=self.scale_grad_by_freq,
                                  sparse=self.sparse)

        out_embd = org_embd + (lora_A_embd @ self.lora_B)*self.scaling

        return out_embd


class LoRAConv2dLayer(nn.Conv2d, LoRABaseLayer):
    """
    LoRA-enhanced Conv2D layer.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the convolutional kernel.
    :type kernel_size: Union[int, tuple]
    :param stride: Stride of the convolution.
    :type stride: int
    :param padding: Padding added to all sides of the input.
    :type padding: int
    :param bias: Whether to include a bias term.
    :type bias: bool
    """
    def __init__(self,
                 lora_config,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)
        LoRABaseLayer.__init__(self, lora_config=lora_config)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(size=(self.rank, in_channels, *self.kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros(size=(self.rank, out_channels)))

        torch.nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

    def _merge_weights(self):
        # og_weights: (out_chan, in_chan, kernel_size, kernel_size)
        lora_A_flattened = self.lora_A.flatten(start_dim=1) # (rank, in_chan*kernel_size*kernel_size)
        lora_AB_flattened = (self.lora_B.T @ lora_A_flattened) * self.scaling # (out_chan, in_chan*kernel_s*kernel_s)
        lora_AB = lora_AB_flattened.reshape(self.out_channels, -1, *self.kernel_size)
        merged_weights = self.weight.data + lora_AB

        state_dict = {'weight': merged_weights}
        if self.bias is not None:
            state_dict['bias'] = self.bias

        merged_conv2d = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  bias=self.bias is not None)
        merged_conv2d.load_state_dict(state_dict)

        return merged_conv2d

    def forward(self, x):
        """
        Forward pass for the LoRA-enhanced Conv2D layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        linear_out = nn.Conv2d.forward(self,x) # (B,out_channels,H',W')

        lora_A_out = F.conv2d(input=x,
                              weight=self.lora_A,
                              padding=self.padding,
                              stride=self.stride,
                              bias=None)

        lora_A_out = lora_A_out.permute(0,2,3,1) # (B,H',W',rank)
        lora_AB = (self.lora_dropout(lora_A_out) @ self.lora_B) * self.scaling # (B,H',W',out_channels)
        lora_AB = lora_AB.permute(0,3,1,2) # (B,out_channels,H',W')
        lora_out = linear_out + lora_AB

        return lora_out


class QLoRAConv2dLayer(bnn.Linear4bit, LoRABaseLayer):
    """
    QLoRA-enhanced Conv2D layer. As bitsandbytes provides no 4-bit Conv2d, the
    convolution is expressed as a quantized Linear projection over Im2Col-unfolded
    patches, while the LoRA branch stays a regular (full-precision) convolution.

    :param lora_config: Configuration object for LoRA.
    :type lora_config: LoRAConfig
    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the convolutional kernel.
    :type kernel_size: Union[int, tuple]
    :param stride: Stride of the convolution.
    :type stride: int
    :param padding: Padding added to all sides of the input.
    :type padding: int
    :param bias: Whether to include a bias term.
    :type bias: bool
    """
    def __init__(self,
                 lora_config,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        bnn.Linear4bit.__init__(self,
                                input_features=self.in_channels * self.kernel_size[0] * self.kernel_size[1],
                                output_features=self.out_channels,
                                bias=bias,
                                **lora_config.quant_kwargs,
                                **kwargs)
        LoRABaseLayer.__init__(self, lora_config=lora_config)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(size=(self.rank, in_channels, *self.kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros(size=(self.rank, out_channels)))

        torch.nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

    def load_weights_to_lora_layer(self, state_dict):
        """
        Loads weights into the QLoRA Conv2d layer. The conv weight
        ([out, in, kh, kw]) is flattened to [out, in*kh*kw] to fit the underlying
        4-bit Linear buffer; quantization happens lazily on ``.to(device)``.

        :param state_dict: State dictionary containing weights and biases.
        :type state_dict: dict
        """
        self.weight.data = state_dict['weight'].flatten(1)
        if 'bias' in state_dict.keys():
            self.bias.data = state_dict['bias']

    def _merge_weights(self):
        # dequantize the 4-bit base weights and restore the conv weight shape
        if self.weight.quant_state is not None:
            weight = bnb.functional.dequantize_4bit(self.weight.data.clone(), self.weight.quant_state)
        else:
            weight = self.weight.data.clone()
        weight = weight.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        # og_weights: (out_chan, in_chan, kernel_size, kernel_size)
        lora_A_flattened = self.lora_A.flatten(start_dim=1) # (rank, in_chan*kernel_size*kernel_size)
        lora_AB_flattened = (self.lora_B.T @ lora_A_flattened) * self.scaling # (out_chan, in_chan*kernel_s*kernel_s)
        lora_AB = lora_AB_flattened.reshape(self.out_channels, -1, *self.kernel_size)
        merged_weights = weight + lora_AB

        state_dict = {'weight': merged_weights}
        if self.bias is not None:
            state_dict['bias'] = self.bias

        merged_conv2d = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  bias=self.bias is not None)
        merged_conv2d.load_state_dict(state_dict)

        return merged_conv2d

    def forward(self, x):
        """
        Forward pass for the QLoRA-enhanced Conv2D layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        # Im2Col: (B,C,H,W) -> (B, C*kh*kw, num_patches) -> (B, num_patches, C*kh*kw)
        unfolded_x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        unfolded_x = unfolded_x.transpose(1, 2)

        # quantized linear projection of each patch, back to (B, out_channels, num_patches)
        proj_x = bnn.Linear4bit.forward(self, unfolded_x).transpose(1, 2)

        h_out = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        linear_out = proj_x.reshape(-1, self.out_channels, h_out, w_out) # (B,out_channels,H',W')

        lora_A_out = F.conv2d(input=x,
                              weight=self.lora_A,
                              padding=self.padding,
                              stride=self.stride,
                              bias=None)

        lora_A_out = lora_A_out.permute(0,2,3,1) # (B,H',W',rank)
        lora_AB = (self.lora_dropout(lora_A_out) @ self.lora_B) * self.scaling # (B,H',W',out_channels)
        lora_AB = lora_AB.permute(0,3,1,2) # (B,out_channels,H',W')
        lora_out = linear_out + lora_AB

        return lora_out


class LoRAModel(nn.Module):
    """
    Wrapper class for applying LoRA / QLoRA to a model. The variant is selected by
    ``config.use_qlora``: when True, Linear/Conv2d layers are replaced with their
    4-bit quantized QLoRA counterparts.

    :param model: The base model to which LoRA will be applied.
    :type model: nn.Module
    :param config: Configuration object for LoRA.
    :type config: LoRAConfig
    """
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        if isinstance(config.lora_exclude_modules, str):
            config.lora_exclude_modules = [config.lora_exclude_modules]
        elif config.lora_exclude_modules is None:
            config.lora_exclude_modules = []
        elif not isinstance(config.lora_exclude_modules, list):
            raise Exception

        self.print_number_of_trainable_parameters()

        self.disable_gradients()

        self.apply_LoRA(self.model)

        self.enable_biases_grad()

        self.print_number_of_trainable_parameters()

    def forward(self, *input, **kwargs):
        """
        Forward pass for the LoRA-enhanced model.

        :param input: Input arguments for the base model.
        :param kwargs: Keyword arguments for the base model.
        :return: Output of the base model.
        """
        return self.model(*input, **kwargs)

    def _check_module_presence(self, module_name, array):
        if array is None or len(array) == 0:
            return False
        return any([name in module_name for name in array])

    def print_number_of_trainable_parameters(self):
        counter = 0
        for param in self.model.parameters():
            if param.requires_grad:
                counter += param.numel()

        print(f'No. of trainable parameters: {counter}.')

    def disable_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not self._check_module_presence(name, self.config.lora_exclude_modules):
                    param.requires_grad = False

    def apply_LoRA(self, module):
        # pick LoRA vs QLoRA layer variants based on the config
        linear_cls = QLoRALinearLayer if self.config.use_qlora else LoRALinearLayer
        conv2d_cls = QLoRAConv2dLayer if self.config.use_qlora else LoRAConv2dLayer

        for name, child in module.named_children():
            if self._check_module_presence(name, self.config.lora_target_modules):
                if isinstance(child, nn.Linear):
                    lora_layer = linear_cls(
                        lora_config=self.config,
                        input_dim=child.in_features,
                        output_dim=child.out_features,
                        bias=child.bias is not None
                    )
                    lora_layer.load_weights_to_lora_layer(child.state_dict())
                    setattr(module, name, lora_layer)
                elif isinstance(child, nn.Embedding):
                    lora_layer = LoRAEmbeddingLayer(
                        lora_config=self.config,
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim
                    )
                    lora_layer.load_weights_to_lora_layer(child.state_dict())
                    setattr(module, name, lora_layer)
                elif isinstance(child, nn.Conv2d):
                    lora_layer = conv2d_cls(
                        lora_config=self.config,
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        bias=child.bias is not None,
                        padding=child.padding,
                        stride=child.stride,
                        kernel_size=child.kernel_size
                    )
                    lora_layer.load_weights_to_lora_layer(child.state_dict())
                    setattr(module, name, lora_layer)

            if len(list(child.children())) > 0 and not self._check_module_presence(name, self.config.lora_exclude_modules):
                self.apply_LoRA(child)

    def enable_biases_grad(self):
        for name, param in self.model.named_parameters():
            if not self._check_module_presence(name, self.config.lora_exclude_modules):
                if '.bias' in name:
                    if self.config.lora_bias == 'none':
                        param.requires_grad = False
                    elif self.config.lora_bias == 'all':
                        param.requires_grad = True
                    elif self.config.lora_bias == 'lora_only' and self._check_module_presence(name, self.config.lora_target_modules):
                        param.requires_grad = True

    def _merge_all_weights(self, module):
        for name, child in module.named_children():
            if isinstance(child, (LoRALinearLayer, QLoRALinearLayer, LoRAEmbeddingLayer, LoRAConv2dLayer, QLoRAConv2dLayer)):
                merged_weights = child._merge_weights()
                setattr(module, name, merged_weights)

            if len(list(child.children())) > 0:
                self._merge_all_weights(child)

    def save_weights(self, path, merge_weights=True):
        if merge_weights:
            self._merge_all_weights(self.model)
            state_dict = {name.replace('lora_model.', ''):param.detach().cpu() for name, param in self.model.named_parameters()}
        else:
            state_dict = {name:param.detach().cpu() for name, param in self.model.named_parameters() if param.requires_grad}
        # print('\n'.join(state_dict.keys()))
        save_file(state_dict, path)


if __name__ == '__main__':
    import transformers
    LoRA_config = LoRAConfig(lora_exclude_modules=['pooler'],
                             lora_target_modules=['word_embeddings', 'query', 'key', 'value', 'dense'],
                             use_qlora=True)
    model = transformers.AutoModel.from_pretrained('FacebookAI/roberta-base')
    lora_model = LoRAModel(model, LoRA_config).to('cuda')
    print(lora_model)
