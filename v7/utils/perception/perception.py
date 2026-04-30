

# from https://github.com/ml-jku/hopfield-layers/blob/master/hflayers/transformer.py



import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn.modules import Module
from typing import Optional, Tuple, Union

from . import Hopfield









class HopfieldEncoderLayer(Module):
    """
    Module with underlying Hopfield association to be used as an encoder in transformer-like architectures.
    """

    def __init__(self,
                 hopfield_association: Hopfield,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = r'relu'
                 ):
        """
        Initialise a new instance of a Hopfield association-based encoder module.

        :param hopfield_association: instance of Hopfield association module
        :param dim_feedforward: depth of the linear projections applied internally
        :param activation: activation to be applied on the result of the internal linear projections
        :param dropout: dropout probability to be applied internally
        """
        super(HopfieldEncoderLayer, self).__init__()

        self.hopfield_association = deepcopy(hopfield_association)

        self.linear_residual = nn.Linear(self.hopfield_association.state_pattern_dim, dim_feedforward)
        self.dropout_residual = nn.Dropout(dropout)
        self.linear_output = nn.Linear(dim_feedforward, self.hopfield_association.state_pattern_dim)

        self.norm_residual = nn.LayerNorm(self.hopfield_association.state_pattern_dim)
        self.norm_output = nn.LayerNorm(self.hopfield_association.state_pattern_dim)
        self.dropout_hopfield_association = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.activation_residual = getattr(torch, activation, None)
        assert self.activation_residual is not None, r'invalid activation function supplied.'
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Reset parameters, including Hopfield association.

        :return: None
        """
        for module in (self.hopfield_association, self.linear_residual,
                       self.linear_output, self.norm_residual, self.norm_output):
            if hasattr(module, r'reset_parameters'):
                module.reset_parameters()




    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield encoding on specified data.

        :param src: data to be processed by Hopfield encoder module
        :param src_mask: mask to be applied on association matrix
        :param src_key_padding_mask: mask to be applied on stored patterns
        :return: Hopfield-encoded input data
        """
        data_associated = self.hopfield_association(
            input=src, stored_pattern_padding_mask=src_key_padding_mask, association_mask=src_mask)
        src = src + self.dropout_hopfield_association(input=data_associated)
        src = self.norm_residual(input=src)

        result_residual_inner = self.activation_residual(input=self.linear_residual(input=src))
        data_associated = self.linear_output(input=self.dropout_residual(input=result_residual_inner))
        src = src + self.dropout_output(input=data_associated)

        return self.norm_output(input=src)














    def get_association_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        """
        Fetch Hopfield association matrix gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        return self.hopfield_association.get_association_matrix(input=input)




    @property
    def batch_first(self) -> int:
        return self.hopfield_association.batch_first

    @property
    def input_size(self) -> int:
        return self.hopfield_association.input_size

    @property
    def output_size(self) -> int:
        return self.linear_output.out_features























