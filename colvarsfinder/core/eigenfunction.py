"""Eigenfunction Task --- :mod:`colvarsfinder.core.eigenfunction`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements a class that defines a feature of molecular system
(:class:`molann.feature.Feature`), and a class that constructs a list of
features from a feature file (:class:`molann.feature.FeatureFileReader`).

Classes
-------
.. autoclass:: EigenFunction
    :members:

.. autoclass:: EigenFunctionTask
    :members:

"""


import molann.ann as ann
import molann.feature as feature
import cv2 as cv
import itertools 
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pandas as pd
from tqdm import tqdm
import os
import copy

from colvarsfinder.core.base import TrainingTask

# eigenfunction class
class EigenFunction(torch.nn.Module):
    r"""TBA

    Parameters
    ----------

    Attributes
    ----------

    Example
    -------
    """

    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        super(EigenFunction, self).__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([ann.create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        """TBA"""
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

# eigenfunction class
class _ReorderedEigenFunction(torch.nn.Module):
    r"""TBA

    Parameters
    ----------

    Attributes
    ----------

    Example
    -------
    """
    def __init__(self, eigenfunction_model, cvec):
        super(_ReorderedEigenFunction, self).__init__()
        self.eigen_funcs = torch.nn.ModuleList([copy.deepcopy(eigenfunction_model.eigen_funcs[idx]) for idx in cvec])

    def forward(self, inp):
        """TBA"""
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

class EigenFunctionTask(TrainingTask):
    """Training task for eigenfunctions 
    """

    def __init__(self, traj_obj, 
                        pp_layer, 
                        learning_rate, 
                        model,
                        load_model_filename,
                        model_save_dir, 
                        save_model_every_step, 
                        model_path, 
                        beta, 
                        diag_coeff,
                        alpha_vec,
                        eig_weights, 
                        sort_eigvals_in_training=True, 
                        k=1,
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        verbose=True):

        super(EigenFunctionTask, self).__init__( traj_obj, pp_layer, learning_rate, model, load_model_filename, model_save_dir, 
                                save_model_every_step, model_path, k, batch_size, num_epochs, test_ratio, optimizer_name, device, verbose)

        self.beta = beta
        self.alpha = alpha_vec
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
        reordered_model = _ReorderedEigenFunction(self.model, self.cvec)
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
