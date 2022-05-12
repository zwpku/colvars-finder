r"""Neural Networks --- :mod:`colvarsfinder.nn`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements PyTorch neural network classes that are used in module :mod:`colvarsfinder.core`.

.. autofunction:: create_sequential_nn

.. autoclass:: AutoEncoder
    :members:

.. autoclass:: EigenFunctions
    :members:
"""

import torch
import re

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    r""" Construct a feedforward Pytorch neural network

    :param layer_dims: dimensions of layers 
    :type layer_dims: list of int
    :param activation: PyTorch non-linear activation function

    :raises AssertionError: if length of **layer_dims** is not larger than 1.

    Example
    -------

    .. code-block:: python

        from colvarsfinder.nn import create_sequential_nn
        import torch

        nn1 = create_sequential_nn([10, 5, 1])
        nn2 = create_sequential_nn([10, 2], activation=torch.nn.ReLU())
    """

    assert len(layer_dims) >= 2, 'Error: at least 2 layers are needed to define a neural network (length={})!'.format(len(layer_dims))

    layers = torch.nn.Sequential()

    for i in range(len(layer_dims)-2) :
        layers.add_module('%d' % (i+1), torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.add_module('activation %d' % (i+1), activation)
    layers.add_module('%d' % (len(layer_dims)-1), torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return layers

class AutoEncoder(torch.nn.Module):
    r"""Neural network representing an autoencoder

    Args:
        e_layer_dims (list of ints): dimensions of layers of encoder
        d_layer_dims (list of ints): dimensions of layers of decoder
        activation: PyTorch non-linear activation function

    Raise:
        AssertionError: if e_layer_dims[-1] != d_layer_dims[0].

    Attributes:
        encoder: feedforward PyTorch neural network representing encoder
        decoder: feedforward PyTorch neural network representing decoder
        encoded_dim (int): encoded dimension

    """

    def __init__(self, e_layer_dims, d_layer_dims, activation=torch.nn.Tanh()):
        super().__init__()

        assert e_layer_dims[-1] == d_layer_dims[0], "ouput dimension of encoder and input dimension of decoder do not match!"

        self.encoder = create_sequential_nn(e_layer_dims, activation)
        self.decoder = create_sequential_nn(d_layer_dims, activation)
        self.encoded_dim = e_layer_dims[-1]
        self._num_encoder_layer = len(e_layer_dims) - 1

    def get_params_of_cv(self, cv_idx):
        r"""
        Args:
            cv_idx (int): index of collective variables
        Return:
            list of pairs of name and parameters of all linear layers.
        """
        param_vec = []
        for name, param in self.encoder.named_parameters():
            layer_idx = int(re.search(r'\d+', name).group())
            if layer_idx < self._num_encoder_layer:
                param_vec.append([name, param])
            else :
                param_vec.append([name, param[cv_idx:(cv_idx+1), ...]])
        return param_vec

    def forward(self, inp):
        """
        Return: 
            value of autoencoder given the input tensor *inp*
        """
        return self.decoder(self.encoder(inp))

# eigenfunction class
class EigenFunctions(torch.nn.Module):
    r"""Feedforward neural network.

    Args:
        layer_dims (list of ints): dimensions of layers  
        k (int): number of eigenfunctions
        activation: PyTorch non-linear activation function

    Raises:
        AssertionError: if layer_dims[-1] != 1.

    The object of this class defines :math:`k` functions :math:`g_1, g_2,
    \dots, g_k` and corresponds to the :attr:`model` of the class
    :class:`EigenFunctionTask` that is to be trained. Each
    :math:`g_i:\mathbb{R}^{d_r}\rightarrow \mathbb{R}` is represented by a
    feedforward neural network of the same architecture specified by
    *layer_dims*. After training, it can be concatenated to the preprocessing layer to obtain eigenfunctions, or collective variables.  See :ref:`loss_eigenfunction` for details.

    Note: 
        The first item of *layer_dims* should equal to :math:`d_r`, i.e., the output dimension of the preprocessing layer, while the last item of *layer_dims* needs to be one.

    Attributes:
        eigen_funcs (:external+pytorch:class:`torch.nn.ModuleList`): PyTorch module list that contains :math:`k` PyTorch neural networks of the same architecture.
    """

    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        r"""
        """
        super().__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def get_params_of_cv(self, cv_idx):
        r"""
        Args:
            cv_idx (int): index of collective variables
        Return:
            list of pairs of name and parameters of all linear layers.
        """
        param_vec = []
        for name, param in self.eigen_funcs[cv_idx].named_parameters():
            param_vec.append([name, param])
        return param_vec

    def forward(self, inp):
        r"""
        Args:
            inp: PyTorch tensor, the output of preprocessing layer. Its shape is :math:`[l, d_r]`.
        Return: 
            PyTorch tensor of shape :math:`[l, k]`, values of the :math:`k` functions :math:`g_1, \cdots, g_k` given the input tensor. 
        """
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

