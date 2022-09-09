import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn import init

from torch import Tensor, Size
from typing import Union, List, Tuple

_shape_t = Union[int, List[int], Size]
class LayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'attribute_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    attribute_affine: bool

    def __init__(self, normalized_shape: _shape_t,  eps: float = 1e-5, elementwise_affine: bool = True, attribute_affine: bool = True) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.attribute_affine = attribute_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.attribute_affine:
            self.attr_weight = Parameter(torch.Tensor(*self.normalized_shape, *self.normalized_shape))
            self.attr_bias = Parameter(torch.Tensor(*self.normalized_shape, *self.normalized_shape))
        else:
            self.register_parameter('attr_weight', None)
            self.register_parameter('attr_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
        if self.attribute_affine:
            init.zeros_(self.weight)
            init.zeros_(self.bias)


    def forward(self, input: Tensor, attribute_mask: Tensor) -> Tensor:

        if attribute_mask is None:
            return torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            output = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)
            # attribute_mask: (bs, seq)
            attribute_input = torch.einsum("bs,bsr->br", attribute_mask, input) # (bs, dim)
            attribute_wight = self.weight.mul(1 + torch.matmul(attribute_input, self.attr_weight)) # (bs, dim)
            attribute_bias = self.bias.add(torch.matmul(attribute_input, self.attr_bias)) # (bs, dim)
            output = torch.mul(output, attribute_wight[:,None,:]).add(attribute_bias[:,None,:])

            return output

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class AttLayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'attribute_affine']
    normalized_shape: Tuple[int]
    eps: float
    elementwise_affine: bool
    attribute: Tuple[int, ...]


    def __init__(self, normalized_shape: int, attribute_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(AttLayerNorm, self).__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if isinstance(attribute_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            attribute_shape = (attribute_shape,)
        self.attribute_shape = tuple(attribute_shape)
        self.attribute_dim = sum(self.attribute_shape)

        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(normalized_shape))
            self.bias = Parameter(torch.Tensor(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.weight_liner = nn.Linear(self.attribute_dim, normalized_shape)
        self.bias_liner = nn.Linear(self.attribute_dim, normalized_shape)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor, attributes: Tuple[Tensor, ...]) -> Tensor:

        attribute = torch.cat(attributes, dim=-1)

        attribute_weight = self.weight_liner(attribute) # (bs, dim)
        attribute_bias = self.bias_liner(attribute) # (bs, dim)

        weight = self.weight.mul(attribute_weight.add(1.)) # (bs, dim)
        bias = self.bias.add(attribute_bias) # (bs, dim)
        # weight = self.weight # (dim)
        # bias = self.bias # (dim)

        output = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)

        output = torch.mul(output, weight[:, None, :]).add(bias[:, None, :])
        # output = torch.mul(output, weight).add(bias)

        return output

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class AttLayerNorm_(Module):
    def __init__(self, hidden_state_dim, attribute_dims, eps: float = 1e-5) -> None:
        super(AttLayerNorm_, self).__init__()
        self.LayerNorm = nn.LayerNorm(hidden_state_dim, eps=eps)
        self.attribute_shape = tuple(attribute_dims)
        self.attribute_dim = sum(self.attribute_shape)

        self.weight_liner = nn.Linear(self.attribute_dim, hidden_state_dim)
        self.bias_liner = nn.Linear(self.attribute_dim, hidden_state_dim)

    def forward(self, input: Tensor, attributes: Tuple[Tensor, ...]) -> Tensor:

        hidden_state = self.LayerNorm(input)
        attribute = torch.cat(attributes, dim=-1)

        attribute_weight = self.weight_liner(attribute)[:, None, :] # (bs, 1, dim)
        attribute_bias = self.bias_liner(attribute)[:, None, :] # (bs, 1, dim)
        # print(hidden_state.shape)
        # print(attribute_weight.shape)
        # print(attribute_bias.shape)

        return hidden_state.mul(attribute_weight.add(1.)).add(attribute_bias)







