r"""Training Tasks --- :mod:`colvarsfinder.core`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements classes for learning collective variables (ColVars).  
Two training tasks, both of which are derived from the base class :class:`TrainingTask`, 
are implemented:

    #. :class:`AutoEncoderTask`, which finds collective variables by training autoencoder.
    #. :class:`EigenFunctionTask`, which finds collective variables by computing eigenfunctions.

See :ref:`math_backgrounds`.

Base class
----------

.. autoclass:: TrainingTask
    :members:

Learning ColVars by training autoencoder
-----------------------------------------

.. autoclass:: AutoEncoder
    :members:

.. autoclass:: AutoEncoderTask
    :members:
    :show-inheritance:

Learning ColVars by computing eigenfunctions
--------------------------------------------

.. autoclass:: EigenFunctions
    :members:

.. autoclass:: EigenFunctionTask
    :members:
    :show-inheritance:
"""

import molann.ann as ann
import molann.feature as feature

import itertools 
import numpy as np
import torch
from sklearn.model_selection import train_test_split 
import pandas as pd
from tensorboardX import SummaryWriter
import os
import copy
from abc import ABC, abstractmethod
from tqdm import tqdm

from openmm import unit

class TrainingTask(ABC):
    r"""Abstract base class of train tasks. A training task should be based on this class.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): An object that holds trajectory data and weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer. It corresponds to the function :math:`r` in :ref:`rep_colvars`
        model : neural network to be trained
        load_model_filename (str): filename of a trained model, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): directory to save training results
        learning_rate (float): learning rate
        k (int): number of collective variables to be learned
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used to train neural networks. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true

    Attributes:
        traj_obj: the same as the input parameter
        preprocessing_layer: the same as the input parameter pp_layer
        model: the same as the input parameter
        load_model_filename: the same as the input parameter
        save_model_every_step : the same as the input parameter
        model_path : the same as the input parameter
        learning_rate: the same as the input parameter
        k: the same as the input parameter
        batch_size: the same as the input parameter
        num_epochs: the same as the input parameter
        test_ratio: the same as the input parameter
        optimizer_name: the same as the input parameter
        optimizer: either :external+pytorch:class:`torch.optim.Adam` or :external+pytorch:class:`torch.optim.SGD`
        device: the same as the input parameter
        verbose (bool): print more information if true

        writer (`SummaryWriter`): TensorboardX writer
    """
    def __init__(self, 
                    traj_obj, 
                    pp_layer, 
                    model,
                    load_model_filename,
                    save_model_every_step, 
                    model_path, 
                    learning_rate, 
                    k,
                    batch_size, 
                    num_epochs,
                    test_ratio, 
                    optimizer_name, 
                    device,
                    verbose):

        self.traj_obj = traj_obj
        self.preprocessing_layer = pp_layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.num_epochs= num_epochs
        self.test_ratio = test_ratio
        self.k = k
        self.model = model
        self.load_model_filename = load_model_filename 
        self.save_model_every_step = save_model_every_step
        self.model_path = model_path
        self.optimizer_name = optimizer_name
        self.device = device
        self.verbose = verbose

        self.model_name = type(self).__name__

        if self.verbose: print ('\n[Info] Log directory: {}\n'.format(self.model_path), flush=True)

        self.writer = SummaryWriter(self.model_path)

    def init_model_and_optimizer(self):
        r"""Initialize :attr:`model` and :attr:`optimizer`.

        The previously saved model will be loaded for initialization, if :attr:`load_model_filename` points to an existing file.

        The attribute :attr:`optimizer` is set to
        :external+pytorch:class:`torch.optim.Adam`, if :attr:`optimizer_name` = 'Adam'; Otherwise, it is set to :external+pytorch:class:`torch.optim.SGD`.

        This function shall be called in the constructor of derived classes.
        """

        if self.load_model_filename and os.path.isfile(self.load_model_filename): 
            self.model.load_state_dict(torch.load(self.load_model_filename))
            if self.verbose: print (f'model parameters loaded from: {self.load_model_filename}')

        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, epoch):
        r"""Save model to file.

        Args:
            epoch (int): current epoch

        The state_dict of the trained :attr:`model` will be saved at `trained_model.pt` under the output directory. 

        The neural network representing collective variables corresponding to :attr:`model` is first constructed by calling :meth:`colvar_model`, then compiled to a :external+pytorch:class:`torch.jit.ScriptModule`, which is finally saved at `trained_cv_scripted.pt` under the output directory.

        This function is called by :meth:`train`.

        """

        if self.verbose: print (f"\n\nEpoch={epoch}:") 

        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.model.state_dict(), trained_model_filename)  

        if self.verbose: print (f'  trained model saved at:\n\t{trained_model_filename}')

        cv = self.colvar_model()

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        if self.verbose: print (f'  script model for CVs saved at:\n\t{trained_cv_script_filename}\n', flush=True)

    @abstractmethod
    def train(self):
        r"""Function to train the model.

        This function has to be implemented in derived class.
        """

        pass

    @abstractmethod
    def colvar_model(self):
        r"""
        Return:
            :external+pytorch:class:`torch.nn.Module`: neural network that represents collective variables given :attr:`preprocessing_layer` and :attr:`model`.
        This function is called by :meth:`save_model`.
        """
        pass


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

        self.eigen_funcs = torch.nn.ModuleList([ann.create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        r"""
        Args:
            inp: PyTorch tensor, the output of preprocessing layer. Its shape is :math:`[l, d_r]`.
        Return: 
            PyTorch tensor of shape :math:`[l, k]`, values of the :math:`k` functions :math:`g_1, \cdots, g_k` given the input tensor. 
        """
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

class EigenFunctionTask(TrainingTask):
    r"""The class for training eigenfunctions.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): An object that holds trajectory data and weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer. It corresponds to the function :math:`r` in :ref:`rep_colvars`
        model (:class:`EigenFunctions`): feedforward neural network to be trained. It corresponds to :math:`g_1, \dots, g_k` in :ref:`loss_eigenfunction`
        load_model_filename (str): filename of a trained model, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): directory to save training results
        beta (float): the value of :math:`(k_BT)^{-1}`
        diag_coeff (:external+pytorch:class:`torch.Tensor`): 1D PyTorch tensor of length :math:`3N`, which contains the diagonal entries of the matrix :math:`a` in the :ref:`loss_eigenfunction`
        alpha (float): penalty constant :math:`\alpha` in the loss function
        eig_weights (list of floats): :math:`k` weights :math:`\omega_1 > \omega_2 > \dots > \omega_k > 0` in the loss functions in :ref:`loss_eigenfunction`
        learning_rate (float): learning rate
        sort_eigvals_in_training (bool): whether or not to reorder the :math:`k` eigenfunctions according to estimation of eigenvalues
        k (int): number of eigenfunctions to be learned
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used to train neural networks. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true

    Attributes:
        model: the same as the input parameter

    """

    def __init__(self, traj_obj, 
                        pp_layer, 
                        model,
                        load_model_filename,
                        save_model_every_step, 
                        model_path, 
                        beta, 
                        diag_coeff,
                        alpha,
                        eig_weights, 
                        learning_rate, 
                        sort_eigvals_in_training=True, 
                        k=1,
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        verbose=True):

        super().__init__( traj_obj, pp_layer,  model, load_model_filename, save_model_every_step, model_path,  learning_rate, k, batch_size, num_epochs, test_ratio, optimizer_name, device, verbose)

        self.model = model

        self._beta = beta
        self._alpha = alpha
        self._sort_eigvals_in_training = sort_eigvals_in_training
        self._eig_w = eig_weights
        self._diag_coeff = diag_coeff

        # list of (i,j) pairs in the penalty term
        self._ij_list = list(itertools.combinations(range(self.k), 2))
        self._num_ij_pairs = len(self._ij_list)

        #--- prepare the data ---
        self._weights = torch.tensor(traj_obj.weights)

        if self.verbose: print ('\nEigenfunctions:\n', self.model, flush=True)

        self.init_model_and_optimizer()

        traj = torch.tensor(traj_obj.trajectory)
        tot_dim = traj.shape[1] * 3 

        if self.verbose: print ('\nPrecomputing gradients of features...')
        traj.requires_grad_()
        self._feature_traj = self.preprocessing_layer(traj)

        f_grad_vec = [torch.autograd.grad(outputs=self._feature_traj[:,idx].sum(), inputs=traj, retain_graph=True)[0] for idx in range(self.preprocessing_layer.output_dimension())]

        self._feature_grad_vec = torch.stack([f_grad.reshape((-1, tot_dim)) for f_grad in f_grad_vec], dim=2).detach().to(self.device)

        self._feature_traj = self._feature_traj.detach()

        if self.verbose:
            print ('  shape of feature_gradient vec:', self._feature_grad_vec.shape)
            print ('Done\n', flush=True)

    def get_reordered_eigenfunctions(self, model, cvec):
        r"""
            Args: 
                model (:class:`EigenFunctions`): model whose module list :func:`EigenFunctions.eigen_funcs` are to be reordered.
                cvec (list of int): a permutation of :math:`[0, 1, \dots, k-1]` 

            Return: 
                a new object of :class:`EigenFunctions` by deep-copy whose module list are reordered according to cvec.

            Functions in :attr:`model` may not be sorted according to the magnitude of eigenvalues. This function returns a sorted model that can then be saved to file.
        """

        copyed_model = copy.deepcopy(model)
        self.eigen_funcs = torch.nn.ModuleList([copy.deepcopy(model.eigen_funcs[idx]) for idx in cvec])
        return copyed_model

    def colvar_model(self):
        r"""
            Return:
                :external+molann:class:`molann.ann.MolANN`: neural network that represents :math:`\xi=(g_1\circ r, \dots, g_k\circ r)^T`,
                built from :attr:`preprocessing_layer` that represents
                :math:`r` and :attr:`model` that represents :math:`g_1, g_2,
                \cdots, g_k`. See :ref:`loss_eigenfunction`.
        """
        reordered_model = self.get_reordered_eigenfunctions(self.model, self._cvec)
        return ann.MolANN(self.preprocessing_layer, reordered_model)

    def loss_func(self, X, weight, f_grad):
        # Evaluate function value on data
        y = self.model(X)

        """
          Compute gradients with respect to features
          The flag create_graph=True is needed, because later we need to compute
          gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
        """
        y_grad_wrt_f_vec = torch.stack([torch.autograd.grad(outputs=y[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0] for idx in range(self.k)], dim=2)

        # use chain rule to get gradients wrt positions
        y_grad_vec = torch.bmm(f_grad, y_grad_wrt_f_vec)

        # Total weight, will be used for normalization 
        tot_weight = weight.sum()

        # Mean and variance evaluated on data
        mean_list = [(y[:,idx] * weight).sum() / tot_weight for idx in range(self.k)]
        var_list = [(y[:,idx]**2 * weight).sum() / tot_weight - mean_list[idx]**2 for idx in range(self.k)]

        # Compute Rayleigh quotients as eigenvalues
        eig_vals = torch.tensor([1.0 / (tot_weight * self._beta) * torch.sum((y_grad_vec[:,:,idx]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[idx] for idx in range(self.k)])

        cvec = range(self.k)
        if self._sort_eigvals_in_training :
            cvec = np.argsort(eig_vals)
            # Sort the eigenvalues 
            eig_vals = eig_vals[cvec]

        non_penalty_loss = 1.0 / (tot_weight * self._beta) * sum([self._eig_w[idx] * torch.sum((y_grad_vec[:,:,cvec[idx]]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[cvec[idx]] for idx in range(self.k)])

        penalty = torch.zeros(1, requires_grad=True)

        # Sum of squares of variance for each eigenfunction
        penalty = sum([(var_list[idx] - 1.0)**2 for idx in range(self.k)])

        for idx in range(self._num_ij_pairs):
          ij = self._ij_list[idx]
          # Sum of squares of covariance between two different eigenfunctions
          penalty += ((y[:, ij[0]] * y[:, ij[1]] * weight).sum() / tot_weight - mean_list[ij[0]] * mean_list[ij[1]])**2

        loss = 1.0 * non_penalty_loss + self._alpha * penalty 

        return loss, eig_vals, non_penalty_loss, penalty, cvec

    def train(self):
        r"""
        Function to train the model.
        """
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._feature_traj, self._weights, torch.arange(self._feature_traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  

        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, w_train, index_train),
                                                   batch_size=self.batch_size,
                                                   drop_last=True,
                                                   shuffle=False)
        test_loader  = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, w_test, index_test),
                                                   batch_size=self.batch_size,
                                                   drop_last=True,
                                                   shuffle=False)
        
        assert len(train_loader) > 0 and len(test_loader) > 0, 'DataLoader is empty, possibly because batch size is too large comparing to trajectory data!'

        # --- start the training over the required number of epochs ---
        self.loss_list = []

        print ("\nTraining starts.\n%d epochs in total, batch size: %d" % (self.num_epochs, self.batch_size)) 
        print ("\nTrain set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_train), len(train_loader), len(train_loader) * self.num_epochs), flush=True)
        print ("Test set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_test), len(test_loader), len(test_loader) * self.num_epochs), flush=True)

        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []
            for iteration, [X, weight, index] in enumerate(train_loader) :

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                # we will compute spatial gradients
                X.requires_grad_()
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad(set_to_none=True)

                f_grad = self._feature_grad_vec[index, :, :].to(self.device)

                # Evaluate loss
                loss, eig_vals, non_penalty_loss, penalty, self._cvec = self.loss_func(X, weight, f_grad)
                # Get gradient with respect to parameters of the model
                loss.backward(retain_graph=True)
                # Store loss
                train_loss.append(loss)
                # Updating parameters
                self.optimizer.step()

            # Evaluate the test loss on the test dataset
            test_loss = []
            test_eig_vals = []
            test_penalty = []
            for iteration, [X, weight, index] in enumerate(test_loader):

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                X.requires_grad_()
                f_grad = self._feature_grad_vec[index, :, :].to(self.device)
                loss, eig_vals, non_penalty_loss, penalty, cvec = self.loss_func(X, weight, f_grad)
                # Store loss
                test_loss.append(loss)
                test_eig_vals.append(eig_vals)
                test_penalty.append(penalty)

            self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            self.writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)), epoch)
            self.writer.add_scalar('Loss/test', torch.mean(torch.tensor(test_loss)), epoch)
            self.writer.add_scalar('penalty', torch.mean(torch.tensor(test_penalty)), epoch)

            for idx in range(self.k):
                self.writer.add_scalar(f'{idx}th eigenvalue', torch.mean(torch.stack(test_eig_vals)[:,idx]), epoch)

            if epoch % self.save_model_every_step == self.save_model_every_step - 1 :
                self.save_model(epoch)

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

        self.encoder = ann.create_sequential_nn(e_layer_dims, activation)
        self.decoder = ann.create_sequential_nn(d_layer_dims, activation)
        self.encoded_dim = e_layer_dims[-1]

    def forward(self, inp):
        """
        Return: 
            value of autoencoder given the input tensor *inp*
        """
        return self.decoder(self.encoder(inp))


# Task to solve autoencoder
class AutoEncoderTask(TrainingTask):
    r"""Training task for autoencoder.

    Args:
        traj_obj (:class:`colvarsfinder.trajectory.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer. It corresponds to the function :math:`r:\mathbb{R}^{d}\rightarrow \mathbb{R}^{d_r}` in :ref:`rep_colvars`
        model (:class:`AutoEncoder`): neural network to be trained
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): directory to save training results
        learning_rate (float): learning rate
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true
        
    This task trains autoencoder using the loss discussed in :ref:`loss_autoencoder`. The neural networks representing the encoder :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and the decoder :math:`f_{enc}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}` are stored in :attr:`model.encoder` and :attr:`model.decoder`, respectively.

    Attributes:
        model: the same as the input parameter
        preprocessing_layer: the same as the input parameter pp_layer

    """
    def __init__(self, traj_obj, 
                        pp_layer, 
                        model,
                        load_model_filename,
                        save_model_every_step, 
                        model_path, 
                        learning_rate, 
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        verbose=True):

        super().__init__( traj_obj, pp_layer,  model, load_model_filename, save_model_every_step, model_path, learning_rate, model.encoded_dim, batch_size, num_epochs, test_ratio, optimizer_name, device, verbose)

        self.init_model_and_optimizer()

        #--- prepare the data ---
        self.weights = torch.tensor(traj_obj.weights)
        self._feature_traj = self.preprocessing_layer(torch.tensor(traj_obj.trajectory))

        # print information of trajectory
        if self.verbose: print ( '\nShape of trajectory data array:\n {}'.format(self._feature_traj.shape), flush=True )

    def colvar_model(self):
        r"""
        Return:
            :external+molann:class:`molann.ann.MolANN`: neural network that represents collective variables :math:`\xi=f_{enc}\circ g`, given the :attr:`preprocessing_layer` that represents :math:`g` and the encoder :attr:`model.encoder` that represents :math:`f_{enc}`. 

        This function is called by :meth:`TrainingTask.save_model` in the base class.
        """
        return ann.MolANN(self.preprocessing_layer, self.model.encoder)

    def weighted_MSE_loss(self, X, weight):
        # Forward pass to get output
        out = self.model(X)
        # Evaluate loss
        return (weight * torch.sum((out-X)**2, dim=1)).sum() / weight.sum()

    def train(self):
        """Function to train the model
        """
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._feature_traj, self.weights, torch.arange(self._feature_traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  

        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(X_train, w_train, index_train),
                                                   batch_size=self.batch_size,
                                                   drop_last=True,
                                                   shuffle=False)
        test_loader  = torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(X_test, w_test, index_test),
                                                   batch_size=self.batch_size,
                                                   drop_last=True,
                                                   shuffle=False)

        # --- start the training over the required number of epochs ---
        self.loss_list = []

        print ("\nTraining starts.\n%d epochs in total, batch size: %d" % (self.num_epochs, self.batch_size)) 
        print ("\nTrain set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_train), len(train_loader), len(train_loader) * self.num_epochs), flush=True)
        print ("Test set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_test), len(test_loader), len(test_loader) * self.num_epochs), flush=True)

        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []
            for iteration, [X, weight, index] in enumerate(train_loader):

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad(set_to_none=True)
                # Evaluate loss
                loss = self.weighted_MSE_loss(X, weight)
                # Get gradient with respect to parameters of the model
                loss.backward()
                # Store loss
                train_loss.append(loss)
                # Updating parameters
                self.optimizer.step()
            # Evaluate the test loss on the test dataset
            self.model.eval()
            with torch.no_grad():
                # Evaluation of test loss
                test_loss = []
                for iteration, [X, weight, index] in enumerate(test_loader):

                    X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                    loss = self.weighted_MSE_loss(X, weight)
                    # Store loss
                    test_loss.append(loss)
                self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            self.writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)), epoch)
            self.writer.add_scalar('Loss/test', torch.mean(torch.tensor(test_loss)), epoch)

            if epoch % self.save_model_every_step == self.save_model_every_step - 1 :
                self.save_model(epoch)

