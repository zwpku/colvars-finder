r"""Training Tasks --- :mod:`colvarsfinder.core`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements classes for learning collective variables (ColVars).  
The following training tasks derived from the base class :class:`TrainingTask` are implemented:

    #. :class:`AutoEncoderTask`, which finds collective variables by training autoencoder.
    #. :class:`RegAutoEncoderTask`, which finds collective variables by training a regularized autoencoder.
    #. :class:`EigenFunctionTask`, which finds collective variables by computing eigenfunctions.

See :ref:`math_backgrounds`.

Base class
----------

.. autoclass:: TrainingTask
    :members:

Learning ColVars by training autoencoder
-----------------------------------------

.. autoclass:: AutoEncoderTask
    :members:
    :show-inheritance:

Learning ColVars by training regularized autoencoder
-----------------------------------------

.. autoclass:: RegAutoEncoderTask
    :members:
    :show-inheritance:

Learning ColVars by computing eigenfunctions
--------------------------------------------

.. autoclass:: EigenFunctionTask
    :members:
    :show-inheritance:
"""

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

from colvarsfinder.nn import RegModel

from openmm import unit

class TrainingTask(ABC):
    r"""Abstract base class of train tasks. A training task should be based on this class.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): An object that holds trajectory data and weights
        pp_layer (:external+pytorch:class:`torch.nn.Module`): preprocessing layer. It corresponds to the function :math:`r` in :ref:`rep_colvars`
        model : neural network to be trained
        model_path (str): directory to save training results
        learning_rate (float): learning rate
        load_model_filename (str): filename of a trained model, used to restart from a previous training 
        save_model_every_step (int): how often to save model
        k (int): number of collective variables to be learned
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used to train neural networks. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.torch.device`): computing device, either CPU or GPU
        verbose (bool): print more information if true

    Attributes:
        traj_obj: the same as the input parameter
        preprocessing_layer: the same as the input parameter pp_layer
        model: the same as the input parameter
        model_path : the same as the input parameter
        learning_rate: the same as the input parameter
        load_model_filename: the same as the input parameter
        save_model_every_step : the same as the input parameter
        k: the same as the input parameter
        batch_size: the same as the input parameter
        num_epochs: the same as the input parameter
        test_ratio: the same as the input parameter
        optimizer_name: the same as the input parameter
        optimizer: either :external+pytorch:class:`torch.optim.Adam` or :external+pytorch:class:`torch.optim.SGD`
        device: the same as the input parameter
        plot_class: plot callback class
        plot_frequency: how often (epoch) to call plot function 
        verbose (bool): print more information if true
    """
    def __init__(self, 
                    traj_obj, 
                    pp_layer, 
                    model,
                    model_path, 
                    learning_rate, 
                    load_model_filename,
                    save_model_every_step, 
                    k,
                    batch_size, 
                    num_epochs,
                    test_ratio, 
                    optimizer_name, 
                    device,
                    plot_class,
                    plot_frequency, 
                    verbose,
                    debug_mode):

        self.traj_obj = traj_obj
        self.preprocessing_layer = pp_layer.to(device)
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
        self.plot_class = plot_class
        self.plot_frequency = plot_frequency
        self.verbose = verbose
        self.debug_mode = debug_mode

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

        if self.load_model_filename :
            if os.path.isfile(self.load_model_filename): 
                self.model.load_state_dict(torch.load(self.load_model_filename, map_location=self.device), strict=False)
                if self.verbose: print (f'model parameters loaded from: {self.load_model_filename}')
            else :
                if self.verbose: print (f'model file not found: {self.load_model_filename}')

        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, epoch, description="latest"):
        r"""Save model to file.

        Args:
            epoch (int): current epoch
            description (str): name of subdirectory to save files

        The state_dict of the trained :attr:`model` will be saved at `model.pt` under the subdirectory specified by *description* of the output directory. 
        The weights and biases of each layer are also saved in text files. 

        The neural network representing collective variables corresponding to :attr:`model` is first constructed by calling :meth:`colvar_model`, then compiled to a :external+pytorch:class:`torch.jit.ScriptModule`, which is finally saved under the output directory. If the device is GPU, both CPU and CUDA versions will be saved.

        This function is called by :meth:`train`.

        """

        if self.verbose: print (f"\n\nEpoch={epoch}:") 

        if self.debug_mode is True :
            model_save_dir = f'{self.model_path}/models'
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            #save the model
            model_filename = f'{model_save_dir}/model_{epoch}.pt'
            torch.save(self.model.state_dict(), model_filename)  

        model_save_dir = f'{self.model_path}/{description}'

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        #save the model
        model_filename = f'{model_save_dir}/model.pt'
        torch.save(self.model.state_dict(), model_filename)  

        for idx in range(self.k):
            param_vec = self.model.get_params_of_cv(idx) 
            for named_param in param_vec:
                name, param = named_param
                fname = '%s/%d_' % (model_save_dir, idx) + name.replace('.', '_') + '.txt'
                np.savetxt(fname, param.detach().numpy())

        if self.verbose: print (f'  trained model saved at:\n\t{model_filename}')

        cv = self.colvar_model()

        if self.device.type == 'cuda':
            scripted_cv_filename = f'{model_save_dir}/scripted_cv_gpu.pt'
            torch.jit.script(cv).save(scripted_cv_filename)
            if self.verbose: print (f'  script (GPU) model for CVs saved at:\n\t{scripted_cv_filename}\n', flush=True)

            cv.to('cpu') 
            scripted_cv_filename = f'{model_save_dir}/scripted_cv_cpu.pt'
            torch.jit.script(cv).save(scripted_cv_filename)
            if self.verbose: print (f'  script (CPU) model for CVs saved at:\n\t{scripted_cv_filename}\n', flush=True)
            cv.to(self.device)
        else :
            scripted_cv_filename = f'{model_save_dir}/scripted_cv_cpu.pt'
            torch.jit.script(cv).save(scripted_cv_filename)
            if self.verbose: print (f'  script (CPU) model for CVs saved at:\n\t{scripted_cv_filename}\n', flush=True)

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

    @abstractmethod
    def reg_model(self):
        pass

class EigenFunctionTask(TrainingTask):
    r"""The class for training eigenfunctions.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): An object that holds trajectory data and weights
        pp_layer (:external+pytorch:class:`torch.nn.Module`): preprocessing layer. It corresponds to the function :math:`r` in :ref:`rep_colvars`
        model (:class:`colvarsfinder.nn.EigenFunctions`): feedforward neural network to be trained. It corresponds to :math:`g_1, \dots, g_k` in :ref:`loss_eigenfunction` 
        model_path (str): directory to save training results
        beta (float): the value of :math:`(k_BT)^{-1}`
        lag_tau (float): 'lag time' (ps) in the loss function. Positive value corresponds to transfer operator, while 0 corresponds to generator.
        diag_coeff (:external+pytorch:class:`torch.Tensor`): 1D PyTorch tensor of length :math:`d`, which contains the diagonal entries of the matrix :math:`a` in the :ref:`loss_eigenfunction`
        alpha (float): penalty constant :math:`\alpha` in the loss function
        eig_weights (list of floats): :math:`k` weights :math:`\omega_1 > \omega_2 > \dots > \omega_k > 0` in the loss functions in :ref:`loss_eigenfunction`
        learning_rate (float): learning rate
        load_model_filename (str): filename of a trained model, used to restart from a previous training if provided
        save_model_every_step (int): how often to save model
        sort_eigvals_in_training (bool): whether or not to reorder the :math:`k` eigenfunctions according to estimation of eigenvalues
        k (int): number of eigenfunctions to be learned
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used to train neural networks. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.torch.device`): computing device, either CPU or GPU
        plot_class: plot callback class
        plot_frequency: how often (epoch) to call plot function 
        verbose (bool): print more information if true

    Attributes:
        model: the same as the input parameter
        loss_list: list of loss values on training data and test data during the training
    """

    def __init__(self, traj_obj, 
                        pp_layer, 
                        model,
                        model_path, 
                        alpha,
                        eig_weights, 
                        diag_coeff=None,
                        beta=1.0, 
                        lag_tau=0,
                        learning_rate=0.01, 
                        load_model_filename=None,
                        save_model_every_step=10, 
                        sort_eigvals_in_training=True, 
                        k=1,
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        plot_class=None,
                        plot_frequency=0, 
                        verbose=True,
                        debug_mode=True):

        super().__init__( traj_obj, pp_layer,  model,  model_path, learning_rate, load_model_filename, save_model_every_step, k, batch_size, num_epochs, test_ratio, optimizer_name, device, plot_class, plot_frequency, verbose, debug_mode )

        self.model = model

        self._alpha = alpha
        self._sort_eigvals_in_training = sort_eigvals_in_training
        self._eig_w = eig_weights
        self._cvec = None

        self.traj_dt = traj_obj.dt 
        lag_idx = lag_tau / self.traj_dt
        assert abs(lag_idx-int(lag_idx)) < 1e-6, f'lag-time ({lag_tau}) not divisable by the timestep {self.traj_dt} of the trajectory'
        self.lag_idx = int(lag_idx)

        # list of (i,j) pairs in the penalty term
        self._ij_list = list(itertools.combinations(range(self.k), 2))
        self._num_ij_pairs = len(self._ij_list)

        #--- prepare the data ---
        self._weights = torch.tensor(traj_obj.weights).to(dtype=torch.get_default_dtype())

        if self.verbose: print ('\nEigenfunctions:\n', self.model, flush=True)

        self.init_model_and_optimizer()

        self._traj = torch.tensor(traj_obj.trajectory).to(dtype=torch.get_default_dtype())

        self.tot_dim = traj_obj.trajectory[0,...].size 

        if self.lag_idx == 0 :
            self._beta = beta
            if diag_coeff is not None :
                assert diag_coeff.dim() == 1 and diag_coeff.size(dim=0) == self.tot_dim, f'diag_coeff should be a 1d tensor of length {self.tot_dim}, current shape: {diag_coeff}'
                self._diag_coeff = diag_coeff
            else :
                self._diag_coeff = torch.ones(self.tot_dim)

    def get_reordered_eigenfunctions(self, model, cvec):
        r"""
            Args: 
                model (:class:`EigenFunctions`): model whose module list :func:`EigenFunctions.eigen_funcs` are to be reordered.
                cvec (list of int): a permutation of :math:`[0, 1, \dots, k-1]` 

            Return: 
                a new object of :class:`EigenFunctions` by deep copy whose module list are reordered according to cvec.

            Functions in :attr:`model` may not be sorted according to the magnitude of eigenvalues. This function returns a sorted model that can then be saved to file.
        """

        copyed_model = copy.deepcopy(model)
        copyed_model.eigen_funcs = torch.nn.ModuleList([copy.deepcopy(model.eigen_funcs[idx]) for idx in cvec])
        return copyed_model

    def colvar_model(self):
        r"""
            Return:
                :external+pytorch:class:`torch.nn.Module`: neural network that represents :math:`\xi=(g_1\circ r, \dots, g_k\circ r)^T`,
                built from :attr:`preprocessing_layer` that represents :math:`r` and :attr:`model` that represents :math:`g_1, g_2,
                \cdots, g_k`. See :ref:`loss_eigenfunction`.
        """
        if self._cvec is None :
            self._cvec = torch.arange(self.k)

        reordered_model = self.get_reordered_eigenfunctions(self.model, self._cvec)
        return torch.nn.Sequential(self.preprocessing_layer, reordered_model)

    def reg_model(self):
        return None

    def loss_func(self, X, weight, X_lagged, weight_lagged):

        # Evaluate function value on data
        y = self.model(self.preprocessing_layer(X))

        # Total weight, will be used for normalization 
        tot_weight = weight.sum()

        # Mean and variance evaluated on data
        mean_list = [(y[:,idx] * weight).sum() / tot_weight for idx in range(self.k)]
        var_list = [(y[:,idx]**2 * weight).sum() / tot_weight - mean_list[idx]**2 for idx in range(self.k)]

        if self.lag_idx > 0 :
            tot_weight_lagged = weight_lagged.sum()
            y_lagged = self.model(self.preprocessing_layer(X_lagged))
            mean_list_lagged = [(y_lagged[:,idx] * weight_lagged).sum() / tot_weight_lagged for idx in range(self.k)]
            var_list_lagged = [(y_lagged[:,idx]**2 * weight_lagged).sum() / tot_weight_lagged - mean_list_lagged[idx]**2 for idx in range(self.k)]

        if self.lag_idx == 0 :
            """
              Compute gradients with respect to coordinates
              The flag create_graph=True is needed, because later we need to compute
              gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
            """
            y_grad_vec = torch.stack([torch.autograd.grad(outputs=y[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0].reshape((-1,self.tot_dim)) for idx in range(self.k)], dim=2)
            # Compute Rayleigh quotients as eigenvalues
            eig_vals = torch.tensor([1.0 / (tot_weight * self._beta) * torch.sum((y_grad_vec[:,:,idx]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[idx] for idx in range(self.k)]).to(dtype=torch.get_default_dtype())
        else :
            eig_vals = 1.0 / (1e-3 * self.traj_dt * self.lag_idx) * torch.tensor([1.0 / tot_weight * torch.sum(((y_lagged[:,idx] - y[:,idx])**2) * weight) / (var_list[idx] + var_list_lagged[idx]) for idx in range(self.k)])

        cvec = range(self.k)
        if self._sort_eigvals_in_training :
            cvec = np.argsort(eig_vals)
            # Sort the eigenvalues 
            eig_vals = eig_vals[cvec]

        if self.lag_idx == 0 :
            non_penalty_loss = 1.0 / (tot_weight * self._beta) * sum([self._eig_w[idx] * torch.sum((y_grad_vec[:,:,cvec[idx]]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[cvec[idx]] for idx in range(self.k)])
        else :
            non_penalty_loss = 1.0 / (1e-3 * self.traj_dt * self.lag_idx) * 1.0 / tot_weight * sum([self._eig_w[idx] * torch.sum((y_lagged[:,idx] - y[:,idx])**2 * weight) / (var_list[cvec[idx]] + var_list_lagged[cvec[idx]]) for idx in range(self.k)])

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
        ll = self._traj.shape[0] - self.lag_idx

        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._traj[:ll, :], self._weights[:ll], torch.arange(ll, dtype=torch.long), test_size=self.test_ratio)  

        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._traj[:ll,:], self._weights[:ll], torch.arange(ll, dtype=torch.long), test_size=self.test_ratio)  

        bs_train = min(self.batch_size, X_train.shape[0])
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, w_train, index_train),
                                                   batch_size=bs_train,
                                                   drop_last=True,
                                                   shuffle=False)

        bs_test = min(self.batch_size, X_test.shape[0])
        test_loader  = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, w_test, index_test),
                                                   batch_size=bs_test,
                                                   drop_last=True,
                                                   shuffle=False)
        
        # assert len(train_loader) > 0 and len(test_loader) > 0, 'DataLoader is empty, possibly because batch size is too large comparing to trajectory data!'

        # --- start the training over the required number of epochs ---
        self.loss_list = []
        min_loss = float("inf") 

        print ("\nTraining starts.\n%d epochs in total, batch sizes (train/test): %d/%d" % (self.num_epochs, bs_train, bs_test)) 
        print ("\nTrain set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_train), len(train_loader), len(train_loader) * self.num_epochs), flush=True)
        print ("Test set:\n\t%d data, %d iterations per epoch, %d iterations in total." % (len(index_test), len(test_loader), len(test_loader) * self.num_epochs), flush=True)

        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []

            for iteration, [X, weight, index] in enumerate(train_loader) :

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad(set_to_none=True)

                if self.lag_idx == 0 :
                    # we will compute spatial gradients
                    X.requires_grad_()
                    X_lagged = None
                else :
                    X_lagged = self._traj[index + self.lag_idx]
                    weight_lagged = self._weights[index + self.lag_idx]

                # Evaluate loss
                loss, eig_vals, non_penalty_loss, penalty, self._cvec = self.loss_func(X, weight, X_lagged, weight_lagged)
                # Get gradient with respect to parameters of the model
                loss.backward(retain_graph=True)
                # Store loss
                train_loss.append([loss, non_penalty_loss, penalty] + [eig_vals[i] for i in range(self.k)])

                # Updating parameters
                self.optimizer.step()

            if epoch % self.save_model_every_step == self.save_model_every_step - 1 :
                self.save_model(epoch)
                if loss < min_loss:
                    min_loss = loss
                    self.save_model(epoch, 'best')

            if self.plot_frequency > 0 and epoch % self.plot_frequency == self.plot_frequency - 1 :
                if self.plot_class is not None : 
                    self.plot_class.plot(self.colvar_model(), epoch=epoch)

            # Evaluate the test loss on the test dataset
            test_loss = []
            for iteration, [X, weight, index] in enumerate(test_loader):

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)

                if self.lag_idx == 0 :
                    # we will compute spatial gradients
                    X.requires_grad_()
                    X_lagged = None
                    weight_lagged = None
                else :
                    X_lagged = self._traj[index + self.lag_idx]
                    weight_lagged = self._weights[index + self.lag_idx]

                loss, eig_vals, non_penalty_loss, penalty, cvec = self.loss_func(X, weight, X_lagged, weight_lagged)
                # Store loss
                test_loss.append([loss, non_penalty_loss, penalty] + [eig_vals[i] for i in range(self.k)])

            self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            loss_names = ['loss', 'eigen_non_penalty', 'eigen_penalty'] + ['eig_%d' % (i+1) for i in range(self.k)] 

            mean_train_loss = torch.mean(torch.tensor(train_loss), 0)
            mean_test_loss = torch.mean(torch.tensor(test_loss), 0)
            for i, name in enumerate(loss_names):
                self.writer.add_scalar('%s/train' % name, mean_train_loss[i], epoch)
                self.writer.add_scalar('%s/test' % name, mean_test_loss[i], epoch)

# Task to solve autoencoder
class AutoEncoderTask(TrainingTask):
    r"""Training task for autoencoder.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+pytorch:class:`torch.nn.Module`): preprocessing layer. It corresponds to the function :math:`r:\mathbb{R}^{d}\rightarrow \mathbb{R}^{d_r}` in :ref:`rep_colvars`
        model (:class:`colvarsfinder.nn.AutoEncoder`): neural network to be trained
        model_path (str): directory to save training results
        learning_rate (float): learning rate
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training if provided
        save_model_every_step (int): how often to save model
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD'
        device (:external+pytorch:class:`torch.torch.device`): computing device, either CPU or GPU
        plot_class: plot callback class
        plot_frequency: how often (epoch) to call plot function 
        verbose (bool): print more information if true
        
    This task trains autoencoder using the loss discussed in :ref:`loss_autoencoder`. The neural networks representing the encoder :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and the decoder :math:`f_{enc}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}` are stored in :attr:`model.encoder` and :attr:`model.decoder`, respectively.

    Attributes:
        model: the same as the input parameter
        preprocessing_layer: the same as the input parameter pp_layer
        loss_list: list of loss values on training data and test data during the training

    """
    def __init__(self, traj_obj, 
                        pp_layer, 
                        model,
                        model_path, 
                        learning_rate=0.01, 
                        load_model_filename=None,
                        save_model_every_step=10, 
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        device= torch.device('cpu'),
                        plot_class=None,
                        plot_frequency=0, 
                        verbose=True,
                        debug_mode=True):

        super().__init__( traj_obj, pp_layer,  model, model_path, learning_rate, load_model_filename, save_model_every_step, model.encoded_dim, batch_size, num_epochs, test_ratio, optimizer_name, device, plot_class, plot_frequency, verbose, debug_mode)

        self.init_model_and_optimizer()

        #--- prepare the data ---
        self._weights = torch.tensor(traj_obj.weights).to(dtype=torch.get_default_dtype())
        self._feature_traj = self.preprocessing_layer(torch.tensor(traj_obj.trajectory).to(dtype=torch.get_default_dtype()))

        # print information of trajectory
        if self.verbose: print ( '\nShape of trajectory data array:\n {}'.format(self._feature_traj.shape), flush=True )

    def colvar_model(self):
        r"""
        Return:
            :external+pytorch:class:`torch.nn.Module`: neural network that represents collective variables :math:`\xi=f_{enc}\circ g`, given the :attr:`preprocessing_layer` that represents :math:`g` and the encoder :attr:`model.encoder` that represents :math:`f_{enc}`. 

        This function is called by :meth:`TrainingTask.save_model` in the base class.
        """
        return torch.nn.Sequential(self.preprocessing_layer, self.model.encoder)

    def reg_model(self):
        return None

    def weighted_MSE_loss(self, X, weight):
        # Forward pass to get output
        out = self.model(X)
        # Evaluate loss
        return (weight * torch.sum((out-X)**2, dim=1)).sum() / weight.sum()

    def train(self):
        """Function to train the model
        """
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._feature_traj, self._weights, torch.arange(self._feature_traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  

        bs_train = min(self.batch_size, X_train.shape[0])
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(X_train, w_train, index_train),
                                                   batch_size=bs_train,
                                                   drop_last=True,
                                                   shuffle=False)

        bs_test = min(self.batch_size, X_test.shape[0])
        test_loader  = torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(X_test, w_test, index_test),
                                                   batch_size=bs_test,
                                                   drop_last=True,
                                                   shuffle=False)

        # --- start the training over the required number of epochs ---
        self.loss_list = []
        min_loss = float("inf") 

        print ("\nTraining starts.\n%d epochs in total, batch sizes (train/test): %d/%d" % (self.num_epochs, bs_train, bs_test)) 
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

            if epoch % self.save_model_every_step == self.save_model_every_step - 1 :
                self.save_model(epoch)
                if loss < min_loss:
                    min_loss = loss
                    self.save_model(epoch, 'best')

            if self.plot_frequency > 0 and epoch % self.plot_frequency == self.plot_frequency - 1 :
                if self.plot_class is not None : 
                    self.plot_class.plot(self.colvar_model(), epoch=epoch)

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

# Task to train a regularized autoencoder
class RegAutoEncoderTask(TrainingTask):
    r"""Training task for regularized autoencoder.

    Args:
        traj_obj (:class:`colvarsfinder.utils.WeightedTrajectory`): trajectory data with weights
        pp_layer (:external+pytorch:class:`torch.nn.Module`): preprocessing layer. It corresponds to the function :math:`r:\mathbb{R}^{d}\rightarrow \mathbb{R}^{d_r}` in :ref:`rep_colvars`
        model (:class:`colvarsfinder.nn.AutoEncoder`): neural network to be trained
        model_path (str): directory to save training results
        eig_weights (list of floats): weights in the regularization part of the loss function involving eigenfunctions 
        learning_rate (float): learning rate
        load_model_filename (str): filename of a trained neural network, used to restart from a previous training if provided
        save_model_every_step (int): how often to save model
        batch_size (int): size of mini-batch 
        num_epochs (int): number of training epochs
        test_ratio: float in :math:`(0,1)`, ratio of the amount of data used as test data
        optimizer_name (str): name of optimizer used for training. either 'Adam' or 'SGD' 
        alpha (float): weight of the reconstruction loss
        gamma (list of floats length of 2): weights in the regularization loss involving eigenfunctions
        eta (list of floats, length of 3): weights in the regularization loss, related to constraints on the (squared, integrated) gradient norm, the norm, and the orthogonality of the encoders
        lag_tau_ae (float): 'lag time' (in ps) in the reconstruction loss. Positive number corresponds to time-lagged autoencoder, while 0 for standard autoencoder
        lag_tau_reg (float): 'lag time' (in ps) in the regularization loss involving eigenfunctions. Positive number corresponds to transfer operator, while 0 corresponds to generator
        beta (float): inverse of temperature, only relevant when the regularization loss corresponds to generator (i.e. lag_idx=0)
        device (:external+pytorch:class:`torch.torch.device`): computing device, either CPU or GPU
        plot_class: plot callback class
        plot_frequency: how often (epoch) to call plot function 
        verbose (bool): print more information if true
        
    This task trains a regularized autoencoder using the loss discussed in :ref:`loss_autoencoder`. The neural networks representing the encoder :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and the decoder :math:`f_{enc}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}` are stored in :attr:`model.encoder` and :attr:`model.decoder`, respectively.

    Attributes:
        model: the same as the input parameter
        preprocessing_layer: the same as the input parameter pp_layer
        loss_list: list of loss values on training data and test data during the training
    """
    def __init__(self, traj_obj, 
                        pp_layer, 
                        model,
                        model_path, 
                        eig_weights=[], 
                        learning_rate=0.01, 
                        load_model_filename=None,
                        save_model_every_step=10, 
                        batch_size=1000, 
                        num_epochs=10,
                        test_ratio=0.2, 
                        optimizer_name='Adam', 
                        alpha=1.0, 
                        gamma=[0.0, 0.0], 
                        eta=[0.0, 0.0, 0.0],
                        lag_tau_ae=0,
                        lag_tau_reg=0,
                        beta=1.0,
                        device= torch.device('cpu'),
                        plot_class=None,
                        plot_frequency=0, 
                        freeze_encoder=False,
                        verbose=True,
                        debug_mode=True):

        super().__init__( traj_obj, pp_layer,  model, model_path, learning_rate, load_model_filename, save_model_every_step, model.encoded_dim, batch_size, num_epochs, test_ratio, optimizer_name, device, plot_class, plot_frequency, verbose, debug_mode)

        self.init_model_and_optimizer()

        assert model.num_reg == len(eig_weights), 'number of weights does not match the number of eigenfunctions!'

        #--- prepare the data ---
        self._weights = torch.tensor(traj_obj.weights).to(dtype=torch.get_default_dtype())
        self._feature_traj = self.preprocessing_layer(torch.tensor(traj_obj.trajectory).to(dtype=torch.get_default_dtype()))

        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.num_reg = model.num_reg
        self.tot_dim = self._feature_traj[0,...].shape[0]
        self._eps = 1e-5
        self._eig_w = eig_weights
        self._cvec = None
        self.freeze_encoder = freeze_encoder

        self.traj_dt = traj_obj.dt 

        lag_ae_idx = lag_tau_ae / self.traj_dt
        lag_idx = lag_tau_reg / self.traj_dt

        assert abs(lag_ae_idx-int(lag_ae_idx)) < 1e-6 and abs(lag_idx-int(lag_idx)) < 1e-6, f'lag-times ({lag_tau_ae}, {lag_tau_reg}) not divisable by the timestep {self.traj_dt} of the trajectory'

        self.lag_ae_idx = int(lag_ae_idx)
        self.lag_idx = int(lag_idx)

        # list of (i,j) pairs in the penalty term
        if self.gamma[0] + self.gamma[1] > self._eps :
            assert self.num_reg > 0, 'number of eigenfunctions must be positive!'
            self._ij_list = list(itertools.combinations(range(self.num_reg), 2))
            self._num_ij_pairs = len(self._ij_list)
            if self.lag_idx == 0 :
                self._beta = beta
                self._diag_coeff = torch.ones(self.tot_dim)

        if self.eta[2] > self._eps : # orthogonality constraints
            self._enc_ij_list = list(itertools.combinations(range(self.k), 2))
            self._enc_num_ij_pairs = len(self._enc_ij_list)

        # print information of trajectory
        if self.verbose: print ( '\nShape of trajectory data array:\n {}'.format(self._feature_traj.shape), flush=True )

    def colvar_model(self):
        r"""
        Return:
            :external+pytorch:class:`torch.nn.Module`: neural network that represents collective variables :math:`\xi=f_{enc}\circ g`, given the :attr:`preprocessing_layer` that represents :math:`g` and the encoder :attr:`model.encoder` that represents :math:`f_{enc}`. 

        This function is called by :meth:`TrainingTask.save_model` in the base class.
        """
        return torch.nn.Sequential(self.preprocessing_layer, self.model.encoder)

    def reg_model(self):

        if self._cvec is None :
            self._cvec = torch.arange(self.model.num_reg)

        reg_reordered = RegModel(self.model, self._cvec) 
        #torch.nn.ModuleList([copy.deepcopy(self.model.reg[idx]) for idx in self._cvec])
        return torch.nn.Sequential(self.preprocessing_layer, reg_reordered)

    def weighted_MSE_loss(self, X, X_lagged, weight):
        # Forward pass to get output
        out = self.model.forward_ae(X)
        # Evaluate loss
        return (weight * torch.sum((out-X_lagged)**2, dim=1)).sum() / weight.sum()

    def reg_enc_grad_loss(self, X, weight):

        X.requires_grad_()

        tot_weight = weight.sum()

        enc = self.model.encoder(X) 
        enc_grad_vec = [torch.autograd.grad(outputs=enc[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0] for idx in range(self.k)]
        enc_grad_vec = [enc_grad.reshape((-1, self.tot_dim)) for enc_grad in enc_grad_vec]
        loss = sum([1.0 / tot_weight * torch.sum((enc_grad_vec[idx]**2).sum(dim=1) * weight) for idx in range(self.k)])

        return loss

    def reg_enc_norm_loss(self, X, weight):

        tot_weight = weight.sum()

        enc = self.model.encoder(X) 

        # Mean and variance evaluated on data
        mean_list = [(enc[:,idx] * weight).sum() / tot_weight for idx in range(self.k)]
        var_list = [(enc[:,idx]**2 * weight).sum() / tot_weight - mean_list[idx]**2 for idx in range(self.k)]

        # Sum of squares of variance for each cv component
        loss = sum([(var_list[idx] - 1.0)**2 for idx in range(self.k)])

        return loss

    def reg_enc_orthognal_loss(self, X, weight):

        tot_weight = weight.sum()

        enc = self.model.encoder(X) 

        # Mean and variance evaluated on data
        mean_list = [(enc[:,idx] * weight).sum() / tot_weight for idx in range(self.k)]
        var_list = [(enc[:,idx]**2 * weight).sum() / tot_weight - mean_list[idx]**2 for idx in range(self.k)]

        # Sum of squares of variance for each cv component
        loss = 0.0

        for idx in range(self._enc_num_ij_pairs):
            ij = self._enc_ij_list[idx]
          # Sum of squares of covariance between two different cv components
            loss += ((enc[:, ij[0]] * enc[:, ij[1]] * weight).sum() / tot_weight - mean_list[ij[0]] * mean_list[ij[1]])**2

        return loss

    def reg_eigen_loss(self, X, weight, X_lagged, weight_lagged):
        
        if self.lag_idx == 0 :
            X.requires_grad_()

        tot_weight = weight.sum()

        # Forward pass to get output
        y = self.model.forward_reg(X) 
        
        # Mean and variance evaluated on data
        mean_list = [(y[:,idx] * weight).sum() / tot_weight for idx in range(self.num_reg)]
        var_list = [(y[:,idx]**2 * weight).sum() / tot_weight - mean_list[idx]**2 for idx in range(self.num_reg)]

        if self.lag_idx > 0 :
            tot_weight_lagged = weight_lagged.sum()
            y_lagged = self.model.forward_reg(X_lagged)
            mean_list_lagged = [(y_lagged[:,idx] * weight_lagged).sum() / tot_weight_lagged for idx in range(self.num_reg)]
            var_list_lagged = [(y_lagged[:,idx]**2 * weight_lagged).sum() / tot_weight_lagged - mean_list_lagged[idx]**2 for idx in range(self.num_reg)]

        if self.lag_idx == 0:
            y_grad_vec = torch.stack([torch.autograd.grad(outputs=y[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0].reshape((-1,self.tot_dim)) for idx in range(self.num_reg)], dim=2)
            # Compute Rayleigh quotients as eigenvalues
            eig_vals = torch.tensor([1.0 / (tot_weight * self._beta) * torch.sum((y_grad_vec[:,:,idx]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[idx] for idx in range(self.num_reg)]).to(dtype=torch.get_default_dtype())
        else :
            eig_vals = 1.0 / (1e-3 * self.traj_dt * self.lag_idx) * torch.tensor([1.0 / tot_weight * torch.sum(((y_lagged[:,idx] - y[:,idx])**2) * weight) / (var_list_lagged[idx] + var_list[idx]) for idx in range(self.num_reg)])

        cvec = np.argsort(eig_vals)
        # Sort the eigenvalues 
        eig_vals = eig_vals[cvec]
                                            
        if self.lag_idx == 0 :
            non_penalty_loss = 1.0 / (tot_weight * self._beta) * sum([self._eig_w[idx] * torch.sum((y_grad_vec[:,:,cvec[idx]]**2 * self._diag_coeff).sum(dim=1) * weight) / var_list[cvec[idx]] for idx in range(self.num_reg)])
        else :
            non_penalty_loss = 1.0 / (1e-3 * self.traj_dt * self.lag_idx) * 1.0 / tot_weight * sum([self._eig_w[idx] * torch.sum((y_lagged[:,idx] - y[:,idx])**2 * weight) / (var_list_lagged[cvec[idx]] + var_list[cvec[idx]]) for idx in range(self.num_reg)])

        # Sum of squares of variance for each eigenfunction
        penalty = sum([(var_list[idx] - 1.0)**2 for idx in range(self.num_reg)])

        for idx in range(self._num_ij_pairs):
            ij = self._ij_list[idx]
          # Sum of squares of covariance between two different eigenfunctions
            penalty += ((y[:, ij[0]] * y[:, ij[1]] * weight).sum() / tot_weight - mean_list[ij[0]] * mean_list[ij[1]])**2

        return eig_vals, non_penalty_loss, penalty, cvec

    def train(self):
        """Function to train the model
        """

        ll = self._feature_traj.shape[0] - max(self.lag_idx, self.lag_ae_idx)

        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self._feature_traj[:ll, :], self._weights[:ll], torch.arange(ll, dtype=torch.long), test_size=self.test_ratio)  

        bs_train = min(self.batch_size, X_train.shape[0])
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(X_train, w_train, index_train),
                                                   batch_size=bs_train,
                                                   drop_last=True,
                                                   shuffle=False)

        bs_test = min(self.batch_size, X_test.shape[0])
        test_loader  = torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(X_test, w_test, index_test),
                                                   batch_size=bs_test,
                                                   drop_last=True,
                                                   shuffle=False)

        # --- start the training over the required number of epochs ---
        self.loss_list = []
        min_loss = float("inf") 

        print ("\nTraining starts.\n%d epochs in total, batch sizes (train/test): %d/%d" % (self.num_epochs, bs_train, bs_test)) 
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

                if self.freeze_encoder is True :
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False

                if self.alpha > self._eps :
                    if self.lag_ae_idx > 0 :
                        X_lagged = self._feature_traj[index+self.lag_ae_idx].to(self.device)
                    else :
                        X_lagged = X

                    ae_loss = self.weighted_MSE_loss(X, X_lagged, weight) 
                else :
                    ae_loss = 0.0

                if self.eta[0]  > self._eps :
                    reg_enc_loss_0 = self.reg_enc_grad_loss(X, weight)
                else :
                    reg_enc_loss_0 = 0.0

                if self.eta[1]  > self._eps :
                    reg_enc_loss_1 = self.reg_enc_norm_loss(X, weight)
                else :
                    reg_enc_loss_1 = 0.0

                if self.eta[2]  > self._eps :
                    reg_enc_loss_2 = self.reg_enc_orthognal_loss(X, weight)
                else :
                    reg_enc_loss_2 = 0.0

                if self.gamma[0] + self.gamma[1] > self._eps :


                    if self.lag_idx > 0 :
                        X_lagged = self._feature_traj[index+self.lag_idx]
                        weight_lagged = self._weights[index + self.lag_idx]
                    else :
                        X_lagged = None
                        weight_lagged = None

                    eig_vals, reg_eigen_loss_0, reg_eigen_loss_1, self._cvec = self.reg_eigen_loss(X, weight, X_lagged, weight_lagged)
                else :
                    reg_eigen_loss_0 = 0 
                    reg_eigen_loss_1 = 0
                    eig_vals = np.zeros(self.num_reg)

                loss = self.alpha * ae_loss + self.gamma[0] * reg_eigen_loss_0 + self.gamma[1] * reg_eigen_loss_1 \
                        + self.eta[0] * reg_enc_loss_0 + self.eta[1] * reg_enc_loss_1 + self.eta[2] * reg_enc_loss_2

                # Get gradient with respect to parameters of the model
                loss.backward()

                if self.freeze_encoder is True :
                    for param in self.model.encoder.parameters():
                        param.requires_grad = True

                # Store loss
                train_loss.append([loss, ae_loss, reg_eigen_loss_0, reg_eigen_loss_1] \
                            + [eig_vals[i] for i in range(self.num_reg)] + [reg_enc_loss_0, reg_enc_loss_1, reg_enc_loss_2])
                # Updating parameters
                self.optimizer.step()

            if epoch % self.save_model_every_step == self.save_model_every_step - 1 :
                self.save_model(epoch)
                if loss < min_loss:
                    min_loss = loss
                    self.save_model(epoch, 'best')

            if self.plot_frequency > 0 and epoch % self.plot_frequency == self.plot_frequency - 1 :
                if self.plot_class is not None : 
                    self.plot_class.plot(self.colvar_model(), self.reg_model(), epoch=epoch)

            # Evaluate the test loss on the test dataset
            test_loss = []
            for iteration, [X, weight, index] in enumerate(test_loader):

                X, weight, index = X.to(self.device), weight.to(self.device), index.to(self.device)
                # Evaluate loss

                if self.alpha > self._eps :
                    if self.lag_ae_idx > 0 :
                        X_lagged = self._feature_traj[index+self.lag_ae_idx].to(self.device)
                    else :
                        X_lagged = X

                    ae_loss = self.weighted_MSE_loss(X, X_lagged, weight) 
                else :
                    ae_loss = 0 

                if self.gamma[0] + self.gamma[1] > self._eps :
                    if self.lag_idx > 0 :
                        X_lagged = self._feature_traj[index+self.lag_idx]
                        weight_lagged = self._weights[index + self.lag_idx]
                    else :
                        X_lagged = None
                        weight_lagged = None

                    eig_vals, reg_eigen_loss_0, reg_eigen_loss_1, self._cvec = self.reg_eigen_loss(X, weight, X_lagged, weight_lagged)
                else :
                    reg_eigen_loss_0 = 0 
                    reg_eigen_loss_1 = 0
                    eig_vals = np.zeros(self.num_reg)

                if self.eta[0]  > self._eps :
                    reg_enc_loss_0 = self.reg_enc_grad_loss(X, weight)
                else :
                    reg_enc_loss_0 = 0.0

                if self.eta[1]  > self._eps :
                    reg_enc_loss_1 = self.reg_enc_norm_loss(X, weight)
                else :
                    reg_enc_loss_1 = 0.0

                if self.eta[2]  > self._eps :
                    reg_enc_loss_2 = self.reg_enc_orthognal_loss(X, weight)
                else :
                    reg_enc_loss_2 = 0.0

                loss = self.alpha * ae_loss + self.gamma[0] * reg_eigen_loss_0 + self.gamma[1] * reg_eigen_loss_1 \
                        + self.eta[0] * reg_enc_loss_0 + self.eta[1] * reg_enc_loss_1 + self.eta[2] * reg_enc_loss_2

                # Store loss
                test_loss.append([loss, ae_loss, reg_eigen_loss_0, reg_eigen_loss_1] \
                            + [eig_vals[i] for i in range(self.num_reg)] + [reg_enc_loss_0, reg_enc_loss_1, reg_enc_loss_2])

            self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])

            loss_names = ['loss', 'ae_loss', 'eigen_non_penalty', 'eigen_penalty'] \
                            + ['eig_%d' % i for i in range(self.num_reg)] \
                            + ['encoder_gradient', 'encoder_norm', 'encoder_orthogonality']

            mean_train_loss = torch.mean(torch.tensor(train_loss), 0)
            mean_test_loss = torch.mean(torch.tensor(test_loss), 0)
            for i, name in enumerate(loss_names):
                self.writer.add_scalar('%s/train' % name, mean_train_loss[i], epoch)
                self.writer.add_scalar('%s/test' % name, mean_test_loss[i], epoch)

