"""Training tasks --- :mod:`colvarsfinder.core`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements classes for learning collective variables.  

Classes
-------

.. autoclass:: TrainingTask
    :members:

.. autoclass:: AutoEncoder
    :members:

.. autoclass:: AutoEncoderTask
    :members:

.. autoclass:: EigenFunctions
    :members:

.. autoclass:: EigenFunctionTask
    :members:

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

from openmm import unit

class TrainingTask(object):
    r"""Base class of train tasks. A train task should be derived from this class.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer
        learning_rate (float): learning rate
        model : neural network to be trained
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): the directory to save training results
        k (int): number of collective variables
        batch_size (int): batch-size
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of states used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true

    Example:

    .. code-block:: python

        import torch

    Raises:
        AssertionError: if feature_list is empty.

    Returns:
        :external+pytorch:class:`torch.Tensor` that stores the aligned states

    """
    def __init__(self, 
                    traj_obj, 
                    pp_layer, 
                    learning_rate, 
                    model,
                    load_model_filename,
                    save_model_every_step, 
                    model_path, 
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

        if self.load_model_filename and os.path.isfile(self.load_model_filename): 
            self.model.load_state_dict(torch.load(self.load_model_filename))
            if self.verbose: print (f'model parameters loaded from: {self.load_model_filename}')

        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, epoch):

        if self.verbose: print (f"\n\nEpoch={epoch}:") 

        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.model.state_dict(), trained_model_filename)  

        if self.verbose: print (f'  trained model saved at:\n\t{trained_model_filename}')

        cv = self.colvar_model()

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        if self.verbose: print (f'  script model for CVs saved at:\n\t{trained_cv_script_filename}\n', flush=True)


# eigenfunction class
class EigenFunctions(torch.nn.Module):
    r"""Feedforward neural network that will be concatenated to the preprocessing layer to represent eigenfunctions.

    Args:
        layer_dims (list of ints): dimensions of layers, shared by each
            eigenfunction. 
        k (int): number of eigenfunctions.
        activation: PyTorch non-linear activation function.

    Raises:
        AssertionError: if layer_dims[-1] != 1.

    Note: 
        The first item of *layer_dims* should equal the output dimension of the preprocessing layer, while the last item of *layer_dims* needs to be one.

    Example
    -------
    """

    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        super(EigenFunctions, self).__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([ann.create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        r"""
        Args:
            inp: PyTorch tensor, the output of preprocessing layer.
        Return: 
            values of eigenfunctions given the input tensor.
        """
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

# eigenfunction class
class _ReorderedEigenFunctions(torch.nn.Module):
    r"""TBA

    Parameters
    ----------

    Attributes
    ----------

    Example
    -------
    """
    def __init__(self, eigenfunction_model, cvec):
        super(_ReorderedEigenFunctions, self).__init__()
        self.eigen_funcs = torch.nn.ModuleList([copy.deepcopy(eigenfunction_model.eigen_funcs[idx]) for idx in cvec])

    def forward(self, inp):
        r""" 
        """
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

class EigenFunctionTask(TrainingTask):
    r"""The class for training eigenfunctions.

    Args:
        traj_obj (:class:`colvarsfinder.trajectory.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer
        learning_rate (float): learning rate
        model (:class:`EigenFunctions`): feedforward neural network to be trained
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): the directory to save training results
        beta (float): the value of :math:`(k_BT)^{-1}`
        diag_coeff (:external+pytorch:class:`torch.Tensor`): one dimensional
        alpha (float): penalty constant
        eig_weights (list of floats): :math:`k` weights in the loss functions
        k (int): number of eigenfunctions to be learned
        batch_size (int): batch-size
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, the ratio of the amount of states used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true
    """

    def __init__(self, traj_obj, 
                        pp_layer, 
                        learning_rate, 
                        model,
                        load_model_filename,
                        save_model_every_step, 
                        model_path, 
                        beta, 
                        diag_coeff,
                        alpha,
                        eig_weights, 
                        sort_eigvals_in_training=True, 
                        k=1,
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        verbose=True):

        super(EigenFunctionTask, self).__init__( traj_obj, pp_layer, learning_rate, model, load_model_filename, 
                                save_model_every_step, model_path, k, batch_size, num_epochs, test_ratio, optimizer_name, device, verbose)

        self.beta = beta
        self.alpha = alpha
        self.sort_eigvals_in_training = sort_eigvals_in_training
        self.eig_w = eig_weights
        self.model = model
        self.diag_coeff = diag_coeff

        # list of (i,j) pairs in the penalty term
        self.ij_list = list(itertools.combinations(range(self.k), 2))
        self.num_ij_pairs = len(self.ij_list)

        #--- prepare the data ---
        self.weights = torch.tensor(traj_obj.weights)

        if self.verbose: print ('\nEigenfunctions:\n', self.model, flush=True)

        self.init_model_and_optimizer()

        traj = torch.tensor(traj_obj.trajectory)
        self.tot_dim = traj.shape[1] * 3 

        if self.verbose: print ('\nPrecomputing gradients of features...')
        traj.requires_grad_()
        self.feature_traj = self.preprocessing_layer(traj)

        f_grad_vec = [torch.autograd.grad(outputs=self.feature_traj[:,idx].sum(), inputs=traj, retain_graph=True)[0] for idx in range(self.preprocessing_layer.output_dimension())]

        self.feature_grad_vec = torch.stack([f_grad.reshape((-1, self.tot_dim)) for f_grad in f_grad_vec], dim=2).detach().to(self.device)

        self.feature_traj = self.feature_traj.detach()

        if self.verbose:
            print ('  shape of feature_gradient vec:', self.feature_grad_vec.shape)
            print ('Done\n', flush=True)

    def colvar_model(self):
        reordered_model = _ReorderedEigenFunctions(self.model, self.cvec)
        return ann.MolANN(self.preprocessing_layer, reordered_model)

    def cv_on_feature_data(self, X):
        return self.model(X)[:,self.cvec]

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
        eig_vals = torch.tensor([1.0 / (tot_weight * self.beta) * torch.sum((y_grad_vec[:,:,idx]**2 * self.diag_coeff).sum(dim=1) * weight) / var_list[idx] for idx in range(self.k)])

        cvec = range(self.k)
        if self.sort_eigvals_in_training :
            cvec = np.argsort(eig_vals)
            # Sort the eigenvalues 
            eig_vals = eig_vals[cvec]

        non_penalty_loss = 1.0 / (tot_weight * self.beta) * sum([self.eig_w[idx] * torch.sum((y_grad_vec[:,:,cvec[idx]]**2 * self.diag_coeff).sum(dim=1) * weight) / var_list[cvec[idx]] for idx in range(self.k)])

        penalty = torch.zeros(1, requires_grad=True)

        # Sum of squares of variance for each eigenfunction
        penalty = sum([(var_list[idx] - 1.0)**2 for idx in range(self.k)])

        for idx in range(self.num_ij_pairs):
          ij = self.ij_list[idx]
          # Sum of squares of covariance between two different eigenfunctions
          penalty += ((y[:, ij[0]] * y[:, ij[1]] * weight).sum() / tot_weight - mean_list[ij[0]] * mean_list[ij[1]])**2

        loss = 1.0 * non_penalty_loss + self.alpha * penalty 

        return loss, eig_vals, non_penalty_loss, penalty, cvec

    def train(self):
        """Function to train the model
        """
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self.feature_traj, self.weights, torch.arange(self.feature_traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  

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

                f_grad = self.feature_grad_vec[index, :, :].to(self.device)

                # Evaluate loss
                loss, eig_vals, non_penalty_loss, penalty, self.cvec = self.loss_func(X, weight, f_grad)
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
                f_grad = self.feature_grad_vec[index, :, :].to(self.device)
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

# autoencoder class 
class AutoEncoder(torch.nn.Module):
    r"""TBA

    Parameters
    ----------

    Attributes
    ----------

    Example
    -------
    """

    def __init__(self, e_layer_dims, d_layer_dims, activation=torch.nn.Tanh()):
        super(AutoEncoder, self).__init__()
        self.encoder = ann.create_sequential_nn(e_layer_dims, activation)
        self.decoder = ann.create_sequential_nn(d_layer_dims, activation)

    def forward(self, inp):
        """TBA
        """
        return self.decoder(self.encoder(inp))


# Task to solve autoencoder
class AutoEncoderTask(TrainingTask):
    r"""Training task for autoencoder.

    Args:
        traj_obj (:class:`colvarsfinder.trajectory.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+molann:class:`molann.ann.PreprocessingANN`): preprocessing layer
        learning_rate (float): learning rate
        model (:class:`AutoEncoder`): neural network to be trained
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training
        save_model_every_step (int): how often to save model
        model_path (str): the directory to save training results
        k (int): encoded dimension 
        batch_size (int): batch-size
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of states used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true

    """
    def __init__(self, traj_obj, 
                        pp_layer, 
                        learning_rate, 
                        model,
                        load_model_filename,
                        save_model_every_step, 
                        model_path, 
                        k=1,
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        verbose=True):

        super(AutoEncoderTask, self).__init__( traj_obj, pp_layer, learning_rate, model, load_model_filename, model_save_dir, 
                                save_model_every_step, model_path, k, batch_size, num_epochs, test_ratio, optimizer_name, device, verbose)

        self.init_model_and_optimizer()

        #--- prepare the data ---
        self.weights = torch.tensor(traj_obj.weights)
        self.feature_traj = self.preprocessing_layer(torch.tensor(traj_obj.trajectory))

        # print information of trajectory
        if self.verbose: print ( '\nShape of trajectory data array:\n {}'.format(self.feature_traj.shape), flush=True )

    def colvar_model(self):
        return ann.MolANN(self.preprocessing_layer, self.model.encoder)

    def weighted_MSE_loss(self, X, weight):
        # Forward pass to get output
        out = self.model(X)
        # Evaluate loss
        return (weight * torch.sum((out-X)**2, dim=1)).sum() / weight.sum()

    def cv_on_feature_data(self, X):
        return self.model.encoder(X)

    def train(self):
        """Function to train the model
        """
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self.feature_traj, self.weights, torch.arange(self.feature_traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  

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

