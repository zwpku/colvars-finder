r"""Neural Networks --- :mod:`colvarsfinder.nn`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements PyTorch neural network classes that represent autoencoders or eigenfunctions. These classes are used in the module :mod:`colvarsfinder.core`.

.. autofunction:: create_sequential_nn

.. autoclass:: AutoEncoder
    :members:

.. autoclass:: RegAutoEncoder 
    :members:

.. autoclass:: RegModel 
    :members:

.. autoclass:: EigenFunctions
    :members:
"""

import torch
import re
import numpy as np

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    r""" Construct a feedforward Pytorch neural network

    :param layer_dims: dimensions of layers 
    :type layer_dims: list of int
    :param activation: PyTorch (nonlinear) activation function

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
    r"""Autoencoder neural network 

    Args:
        e_layer_dims (list of ints): dimensions of encoder's layers
        d_layer_dims (list of ints): dimensions of decoder's layers 
        activation: PyTorch activation function

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
        assert 0 <= cv_idx < self.encoded_dim, f"index {cv_idx} exceeded the range [0, {self.encoded_dim-1}]!"

        param_vec = []
        for name, param in self.encoder.named_parameters():
            layer_idx = int(re.search(r'\d+', name).group())
            if layer_idx < self._num_encoder_layer:
                param_vec.append([name, param])
            else :
                param_vec.append([name, param[cv_idx:(cv_idx+1), ...]])
        return param_vec

    def forward(self, inp):
        r"""
        Args:
            input PyTorch tensor *inp*
        Return: 
            output of autoencoder given the input tensor *inp*
        """
        return self.decoder(self.encoder(inp))

class RegAutoEncoder(torch.nn.Module):
    r"""Neural network representing a regularized autoencoder

    Args:
        e_layer_dims (list of ints): dimensions of encoder's layers 
        d_layer_dims (list of ints): dimensions of decoder's decoder
        reg_layer_dims (list of ints): dimensions of regularizer's layers 
        K (int): number of regularizers
        activation: PyTorch nonlinear activation function

    Raise:
        AssertionError: if e_layer_dims[-1] != d_layer_dims[0] or e_layer_dims[-1] != reg_layer_dims[0] 

    Attributes:
        encoder: feedforward PyTorch neural network representing encoder
        decoder: feedforward PyTorch neural network representing decoder
        reg:     feedforward PyTorch neural network representing regularizers, or None if K=0
        encoded_dim (int): encoded dimension
        num_reg (int) : number of eigenfunctions used for regularization (K)
    """

    def __init__(self, e_layer_dims, d_layer_dims, reg_layer_dims, K, activation=torch.nn.Tanh()):
        super(RegAutoEncoder, self).__init__()

        assert e_layer_dims[-1] == d_layer_dims[0], "ouput dimension of encoder and input dimension of decoder do not match!"

        self.num_reg = K
        assert self.num_reg == 0 or e_layer_dims[-1] == reg_layer_dims[0], "ouput dimension of encoder and input dimension of regulator part do not match!"

        self.encoder = create_sequential_nn(e_layer_dims, activation)
        self.decoder = create_sequential_nn(d_layer_dims, activation)
        self.encoded_dim = e_layer_dims[-1]

        self._num_encoder_layer = len(e_layer_dims) - 1

        if self.num_reg > 0 :
            self.reg = torch.nn.ModuleList([create_sequential_nn(reg_layer_dims, activation) for idx in range(self.num_reg)])
        else :
            self.reg = None

    def get_params_of_cv(self, cv_idx):
        r"""
        Args:
            cv_idx (int): index of collective variables
        Return:
            list of pairs of name and parameters of all linear layers.
        """
        assert 0 <= cv_idx < self.encoded_dim, f"index {cv_idx} exceeded the range [0, {self.encoded_dim-1}]!"
        param_vec = []
        for name, param in self.encoder.named_parameters():
            layer_idx = int(re.search(r'\d+', name).group())
            if layer_idx < self._num_encoder_layer:
                param_vec.append([name, param])
            else :
                param_vec.append([name, param[cv_idx:(cv_idx+1), ...]])
        return param_vec

    def forward_ae(self, inp):
        r"""
        Args:
            inp: PyTorch tensor 
        Return: 
            value of autoencoder given the input tensor *inp* 
        """

        return self.decoder(self.encoder(inp))

    def forward_reg(self, inp):
        r"""
        Args:
            inp: PyTorch tensor 
        Return: 
            value of regularizers (e.g. eigenfunctions) given the input tensor *inp* 
        """

        assert self.num_reg > 0, 'number of regularizers is not positive.'

        encoded = self.encoder(inp)
        return torch.cat([nn(encoded) for nn in self.reg], dim=1)

    def forward(self, inp):
        r"""
        Return: 
            values of autodecoder and regularizers given the input tensor *inp* 
        """

        encoded = self.encoder(inp)
        return torch.cat((self.decoder(encoded), torch.cat([nn(encoded) for nn in self.reg], dim=1)), dim=1)

class RegModel(torch.nn.Module):
    r"""Neural network representing the eigenfunctions built from a :class:`RegAutoEncoder`.

    Args:
        reg_ae (:class:`RegAutoEncoder`): an object of class :class:`RegAutoEncoder`
        cvec (list of int): order of regularizers 
    Raise:
        AssertionError: *reg_ae* doesn't have a regularizer 

    Attributes:
        encoder: feedforward PyTorch neural network representing encoder
        reg:     feedforward PyTorch neural network representing regularizers 
        encoded_dim (int): encoded dimension
        num_reg (int) : number of eigenfunctions used for regularization 
        cvec (list of int): same as input
    """
    def __init__(self, reg_ae, cvec):
        super(RegModel, self).__init__()

        assert reg_ae.num_reg > 0, 'number of regularizers is not positive.'
        assert len(cvec) == reg_ae.num_reg, 'length of cvec doesn\'t equal to number of regularizers'
        assert (sorted(cvec) == np.arange(reg_ae.num_reg)).all(), f'cvec should be a permutation of 0,1,...,{len(cvec)-1}.'

        self.encoder = reg_ae.encoder
        self.reg = reg_ae.reg
        self.cvec = cvec
        self.encoded_dim = reg_ae.encoded_dim
        self.num_reg = reg_ae.num_reg

    def forward(self, inp):
        r"""
            return values of regularizers (reordered according to *cvec*) given input tensor *inp*
        """
        encoded = self.encoder(inp)
        return torch.cat([self.reg[idx](encoded) for idx in self.cvec], dim=1)

# eigenfunction class
class EigenFunctions(torch.nn.Module):
    r"""Feedforward neural network representing eigenfunctions.

    Args:
        layer_dims (list of ints): dimensions of layers for each eigenfunction
        k (int): number of eigenfunctions
        activation: PyTorch nonlinear activation function

    Raises:
        AssertionError: if layer_dims[-1] != 1 (since each eigenfunction is scalar-valued function).

    This class defines :math:`k` functions :math:`g_1, g_2,
    \dots, g_k` and corresponds to the :attr:`model` used as an input paramter to define the class :class:`EigenFunctionTask`. Each
    :math:`g_i:\mathbb{R}^{d_r}\rightarrow \mathbb{R}` is represented by a
    feedforward neural network of the same architecture specified by
    *layer_dims*. After training, it can be concatenated to the preprocessing layer to obtain eigenfunctions, or collective variables.  See :ref:`loss_eigenfunction` for details.

    Note: 
        The first item in the list *layer_dims* should equal to :math:`d_r`, i.e. the output dimension of the preprocessing layer, while the last item in *layer_dims* should be one.

    Attributes:
        eigen_funcs (:external+pytorch:class:`torch.nn.ModuleList`): PyTorch module list that contains :math:`k` PyTorch feedforward neural networks of the same architecture.
    """

    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        r"""
        """
        super().__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be scalar-valued"

        self.eigen_funcs = torch.nn.ModuleList([create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def get_params_of_cv(self, cv_idx):
        r"""
        Args:
            cv_idx (int): index of collective variables (i.e. eigenfunctions)
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
            inp: PyTorch tensor. Typically it is the output of preprocessing layer, with shape :math:`[l, d_r]`.
        Return: 
            PyTorch tensor of shape :math:`[l, k]`, values of the :math:`k` functions :math:`g_1, \cdots, g_k` given the input tensor. 
        """
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

