"""Autoencoder Task --- :mod:`colvarsfinder.autoencoder`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements a class that defines a feature of molecular system
(:class:`molann.feature.Feature`), and a class that constructs a list of
features from a feature file (:class:`molann.feature.FeatureFileReader`).

Classes
-------

.. autoclass:: AutoEncoder
    :members:

.. autoclass:: AutoEncoderTask
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

from colvarsfinder.base import TrainingTask

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
    """Training task for autoencoder
    """
    def __init__(self, traj_obj, 
                        pp_layer, 
                        learning_rate, 
                        model,
                        load_model_filename,
                        model_save_dir, 
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

