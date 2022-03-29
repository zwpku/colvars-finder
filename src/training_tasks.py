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
from tensorboardX import SummaryWriter
import os

# autoencoder class 
class AutoEncoder(torch.nn.Module):
    def __init__(self, e_layer_dims, d_layer_dims, activation=torch.nn.Tanh()):
        super(AutoEncoder, self).__init__()
        self.encoder = ann.create_sequential_nn(e_layer_dims, activation)
        self.decoder = ann.create_sequential_nn(d_layer_dims, activation)

    def forward(self, inp):
        """TBA
        """
        return self.decoder(self.encoder(inp))

# eigenfunction class
class EigenFunction(torch.nn.Module):
    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        super(EigenFunction, self).__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([ann.create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        """TBA"""
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

class TrainingTask(object):
    """class for a training task
    """
    def __init__(self, args, traj_obj, pp_layer, model_path, histogram_feature_mapper=None, output_feature_mapper=None):

        self.learning_rate = args.learning_rate
        self.num_epochs= args.num_epochs
        self.batch_size = args.batch_size 
        self.test_ratio = args.test_ratio
        self.save_model_every_step = args.save_model_every_step
        self.histogram_feature_mapper = histogram_feature_mapper
        self.output_feature_mapper = output_feature_mapper
        self.traj_obj = traj_obj
        self.args = args
        self.k = args.k
        self.model_path = model_path
        self.num_scatter_states = args.num_scatter_states
        self.device = args.device
        self.use_gpu = args.use_gpu

        self.beta = args.beta

        self.preprocessing_layer = pp_layer
        self.feature_dim = pp_layer.output_dimension()

        print ('\nLog directory: {}\n'.format(self.model_path), flush=True)
        self.writer = SummaryWriter(self.model_path)

        if self.histogram_feature_mapper is not None :
            histogram_feature = self.histogram_feature_mapper(traj_obj.trajectory).detach().numpy()
            feature_names = self.histogram_feature_mapper.feature_all_names()
            df = pd.DataFrame(data=histogram_feature, columns=feature_names) 

            fig, ax = plt.subplots()
            df.hist(ax=ax)
            fig_name = f'{self.model_path}/histogram_feature.png'
            fig.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'histogram features', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            df.plot(subplots=True) 
            plt.legend(loc='best')
            fig_name = f'{self.model_path}/feature_along_trajectory.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'feature along trajectory', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            print (f'Histogram and trajectory plots of features saved.', flush=True) 

        if self.output_feature_mapper is not None :
            self.output_features = self.output_feature_mapper(traj_obj.trajectory).detach().numpy()
        else :
            self.output_features = None

    def save_model(self, epoch=0):

        print (f"\n\nepoch={epoch}") 

        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.model.state_dict(), trained_model_filename)  
        print (f'\ntrained model saved at:\n\t{trained_model_filename}\n')

        cv = self.colvar_model()

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        print (f'script model for CVs saved at:\n\t{trained_cv_script_filename}\n', flush=True)

    def plot_scattered_cv_on_feature_space(self, epoch): 

        index = np.random.choice(np.arange(self.feature_traj.shape[0], dtype=int), self.num_scatter_states, replace=False)

        X = self.feature_traj[index,:].to(self.device)
        feature_data = self.output_features[index,:]
        cv_vals = self.cv_on_data(X).cpu()

        k = cv_vals.size(1)

        for idx in range(k) :
            fig, ax = plt.subplots()
            sc = ax.scatter(feature_data[:,0], feature_data[:,1], s=2.0, c=cv_vals[:,idx].detach().numpy(), cmap='jet')

            ax.set_title(f'{idx+1}th dimension', fontsize=27)
            ax.set_xlabel(r'{}'.format(self.output_feature_mapper.feature_name(0)), fontsize=25, labelpad=3, rotation=0)
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_ylabel(r'{}'.format(self.output_feature_mapper.feature_name(1)), fontsize=25, labelpad=-10, rotation=0)

            cax = fig.add_axes([0.92, 0.10, .02, 0.80])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.tick_params(labelsize=20)

            fig_name = f'{self.model_path}/scattered_{self.model_name}_{epoch}_{idx}.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()

            self.writer.add_image(f'scattered {self.model_name} {idx}', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), global_step=epoch, dataformats='HWC')

# Task to solve autoencoder
class AutoEncoderTask(TrainingTask):
    """Training task for autoencoder
    """

    def __init__(self, args, traj_obj, pp_layer, model_path, histogram_feature_mapper=None, output_feature_mapper=None):

        super(AutoEncoderTask, self).__init__(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)

        self.model_name = 'autoencoder'

        # sizes of feedforward neural networks
        e_layer_dims = [self.feature_dim] + args.e_layer_dims + [self.k]
        d_layer_dims = [self.k] + args.d_layer_dims + [feature_dim]

        # define autoencoder
        self.model = AutoEncoder(e_layer_dims, d_layer_dims, args.activation()).to(device=self.device)
        # print the model
        print ('\nAutoencoder: input dim: {}, encoded dim: {}\n'.format(feature_dim, self.k), self.model)

        if os.path.isfile(args.load_model_filename): 
            self.model.load_state_dict(torch.load(args.load_model_filename))
            print (f'model parameters loaded from: {self.args.load_model_filename}')

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        #--- prepare the data ---
        self.weights = torch.tensor(traj_obj.weights)
        self.feature_traj = self.preprocessing_layer(traj_obj.trajectory)

        # print information of trajectory
        print ( '\nshape of trajectory data array:\n {}'.format(self.feature_traj.shape), flush=True )

    def colvar_model(self):
        return ann.MolANN(self.preprocessing_layer, self.model.encoder)

    def weighted_MSE_loss(self, X, weight):
        # Forward pass to get output
        out = self.model(X)
        # Evaluate loss
        return (weight * torch.sum((out-X)**2, dim=1)).sum() / weight.sum()

    def cv_on_data(self, X):
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
        print ("\ntraining starts, %d epochs in total." % self.num_epochs) 
        print ("%d iterations per epoch, %d iterations in total." % (len(train_loader), len(train_loader) * self.num_epochs), flush=True)

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

            if self.output_features is not None :
                self.plot_scattered_cv_on_feature_space(epoch)

            if epoch % self.save_model_every_step == 0 :
                self.save_model(epoch=epoch)

        print ("\ntraining ends.\n") 

class EigenFunctionTask(TrainingTask):
    """Training task for eigenfunctions 
    """

    def __init__(self, args, traj_obj,  pp_layer, model_path, histogram_feature_mapper=None, output_feature_mapper=None):

        super(EigenFunctionTask, self).__init__(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)

        self.model_name = 'eigenfunction'

        self.alpha = args.alpha
        self.sort_eigvals_in_training = args.sort_eigvals_in_training
        self.eig_w = args.eig_w

        # list of (i,j) pairs in the penalty term
        self.ij_list = list(itertools.combinations(range(self.k), 2))
        self.num_ij_pairs = len(self.ij_list)

        #--- prepare the data ---
        self.weights = torch.tensor(traj_obj.weights)
        traj = traj_obj.trajectory

        # print information of trajectory
        print ( '\nshape of trajectory data array:\n {}'.format(traj.shape) )

        self.tot_dim = traj.shape[1] * 3 

        # diagnoal matrix 
        # the unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
        self.diag_coeff = torch.ones(self.tot_dim).double().to(self.device) * args.diffusion_coeff * 1e7 * self.beta

        layer_dims = [feature_dim] + args.layer_dims + [1]

        self.model = EigenFunction(layer_dims, self.k, self.args.activation()).to(self.device)

        print ('\nEigenfunctions:\n', self.model, flush=True)

        print ('\nPrecomputing gradients of features...')
        traj.requires_grad_()
        self.feature_traj = self.preprocessing_layer(traj)

        f_grad_vec = [torch.autograd.grad(outputs=self.feature_traj[:,idx].sum(), inputs=traj, retain_graph=True)[0] for idx in range(feature_dim)]
        self.feature_grad_vec = torch.stack([f_grad.reshape((-1, self.tot_dim)) for f_grad in f_grad_vec], dim=2).detach().to(self.device)

        self.feature_traj = self.feature_traj.detach()

        print ('  shape of feature_gradient vec:', self.feature_grad_vec.shape)

        print ('Done\n', flush=True)

        if os.path.isfile(args.load_model_filename): 
            self.model.load_state_dict(torch.load(args.load_model_filename))
            print (f'model parameters loaded from: {self.args.load_model_filename}')

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def colvar_model(self):
        return MolANN(self.preprocessing_layer, self.model)

    def cv_on_data(self, X):
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

        penalty = torch.zeros(1, requires_grad=True).double()

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
        
        # --- start the training over the required number of epochs ---
        self.loss_list = []
        print ("\ntraining starts, %d epochs in total." % self.num_epochs) 
        print ("%d iterations per epoch, %d iterations in total." % (len(train_loader), len(train_loader) * self.num_epochs), flush=True)

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

            if self.output_features is not None :
                self.plot_scattered_cv_on_feature_space(epoch)

            if epoch % self.save_model_every_step == 0 :
                self.save_model(epoch=epoch)

        print ("\ntraining ends.\n") 

