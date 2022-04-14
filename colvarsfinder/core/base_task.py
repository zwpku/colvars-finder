"""Training Task --- :mod:`colvarsfinder.core.base_task`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements a class that defines a feature of molecular system
(:class:`molann.feature.Feature`), and a class that constructs a list of
features from a feature file (:class:`molann.feature.FeatureFileReader`).

Classes
-------

.. autoclass:: TrainingArgs
    :members:

.. autoclass:: TrainingTask
    :members:

"""

import cv2 as cv
import itertools 
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
import os
import configparser

from openmm import unit

class TrainingArgs(object):
    r"""TBA

    Parameters
    ----------

    Attributes
    ----------

    Example
    -------
    """

    def __init__(self, config_filename='params.cfg'):

        config = configparser.ConfigParser()
        config.read(config_filename)

        self.sys_name = config['System'].get('sys_name')
        self.pdb_filename = config['System'].get('pdb_filename')
        self.temp = config['System'].getfloat('sys_temperature')

        sampling_path = config['Sampling'].get('sampling_path')
        self.traj_dcd_filename = config['Sampling'].get('traj_dcd_filename')
        self.traj_weight_filename = config['Sampling'].get('traj_weight_filename')

        # add path to filenames
        self.traj_dcd_filename = os.path.join(sampling_path, self.traj_dcd_filename)
        self.traj_weight_filename = os.path.join(sampling_path, self.traj_weight_filename)

         # unit: kJ/mol
        kT = self.temp * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA / unit.kilojoule_per_mole
        self.beta = 1.0 / kT
        
        #set training parameters
        self.cutoff_weight_min = config['Training'].getfloat('cutoff_weight_min')
        self.cutoff_weight_max = config['Training'].getfloat('cutoff_weight_max')

        self.use_gpu =config['Training'].getboolean('use_gpu')
        self.batch_size = config['Training'].getint('batch_size')
        self.num_epochs = config['Training'].getint('num_epochs')
        self.test_ratio = config['Training'].getfloat('test_ratio')
        self.learning_rate = config['Training'].getfloat('learning_rate')
        self.optimizer = config['Training'].get('optimizer') # 'Adam' or 'SGD'
        self.load_model_filename =  config['Training'].get('load_model_filename')
        self.model_save_dir = config['Training'].get('model_save_dir') 
        self.save_model_every_step = config['Training'].getint('save_model_every_step')
        self.task_type = config['Training'].get('task_type')

        if self.task_type == 'Autoencoder' :
            # encoded dimension
            self.k = config['AutoEncoder'].getint('encoded_dim')
            self.e_layer_dims = [int(x) for x in config['AutoEncoder'].get('encoder_hidden_layer_dims').split(',')]
            self.d_layer_dims = [int(x) for x in config['AutoEncoder'].get('decoder_hidden_layer_dims').split(',')]
            self.activation_name = config['AutoEncoder'].get('activation') 
        else :
            if self.task_type == 'Eigenfunction':
                self.k = config['EigenFunction'].getint('num_eigenfunction')
                self.layer_dims = [int(x) for x in config['EigenFunction'].get('hidden_layer_dims').split(',')]
                self.activation_name = config['EigenFunction'].get('activation') 
                self.alpha = config['EigenFunction'].getfloat('penalty_alpha')
                self.eig_w = [float(x) for x in config['EigenFunction'].get('eig_w').split(',')]
                self.diffusion_coeff = config['EigenFunction'].getfloat('diffusion_coeff')
                self.sort_eigvals_in_training = config['EigenFunction'].getboolean('sort_eigvals_in_training')
            else :
                raise ValueError(f'Task {self.task_type} not implemented!  Possible values: Autoencoder, Eigenfunction.')

        self.activation = getattr(torch.nn, self.activation_name) 

        self.align_selector = config['Training'].get('align_mda_selector')
        self.feature_file = config['Training'].get('feature_file')
        self.seed = config['Training'].getint('seed')
        self.num_scatter_states = config['Training'].getint('num_scatter_states')

        # CUDA support
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False

        print (f'\n[Info] Parameters loaded from: {config_filename}\n', flush=True)

class TrainingTask(object):
    """class for a training task
    """
    def __init__(self, args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper, verbose):

        self.learning_rate = args.learning_rate
        self.num_epochs= args.num_epochs
        self.batch_size = args.batch_size 
        self.test_ratio = args.test_ratio
        self.save_model_every_step = args.save_model_every_step
        self.histogram_feature_mapper = histogram_feature_mapper
        self.output_feature_mapper = output_feature_mapper
        self.traj_obj = traj_obj
        self.k = args.k
        self.model_path = model_path
        self.num_scatter_states = args.num_scatter_states
        self.device = args.device
        self.use_gpu = args.use_gpu
        self.optimizer_name = args.optimizer
        self.load_model_filename = args.load_model_filename 
        self.verbose = verbose

        self.beta = args.beta

        self.preprocessing_layer = pp_layer
        self.feature_dim = pp_layer.output_dimension()

        self.model_name = type(self).__name__

        if self.verbose: print ('\n[Info] Log directory: {}\n'.format(self.model_path), flush=True)

        self.writer = SummaryWriter(self.model_path)

        if self.histogram_feature_mapper is not None :
            histogram_feature = self.histogram_feature_mapper(torch.tensor(traj_obj.trajectory)).detach().numpy()
            feature_names = self.histogram_feature_mapper.get_feature_info()['name']
            df = pd.DataFrame(data=histogram_feature, columns=feature_names) 

            df.hist(figsize=(5,5))
            fig_name = f'{self.model_path}/feature_histograms.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'feature histograms', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            df.plot(figsize=(5,5), subplots=True) 
            plt.legend(loc='best')
            fig_name = f'{self.model_path}/features_along_trajectory.png'
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'features along trajectory', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            if self.verbose: print (f'Histogram and trajectory plots of features saved.', flush=True) 

        if self.output_feature_mapper is not None :
            self.output_features = self.output_feature_mapper(torch.tensor(traj_obj.trajectory)).detach().numpy()
        else :
            self.output_features = None

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

    def plot_scattered_cv_on_feature_space(self, epoch): 

        index = np.random.choice(np.arange(self.feature_traj.shape[0], dtype=int), self.num_scatter_states, replace=False)

        X = self.feature_traj[index,:].to(self.device)
        feature_data = self.output_features[index,:]
        cv_vals = self.cv_on_feature_data(X).cpu()

        k = cv_vals.size(1)

        for idx in range(k) :
            fig, ax = plt.subplots()
            sc = ax.scatter(feature_data[:,0], feature_data[:,1], s=2.0, c=cv_vals[:,idx].detach().numpy(), cmap='jet')

            ax.set_title(f'{idx+1}th CV', fontsize=27)
            ax.set_xlabel(r'{}'.format(self.output_feature_mapper.get_feature(0).get_name()), fontsize=25, labelpad=3, rotation=0)
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_ylabel(r'{}'.format(self.output_feature_mapper.get_feature(0).get_name()), fontsize=25, labelpad=-10, rotation=0)

            cax = fig.add_axes([0.92, 0.10, .02, 0.80])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.tick_params(labelsize=20)

            fig_name = f'{self.model_path}/cv_scattered_{self.model_name}_{epoch}_{idx}.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()

            self.writer.add_image(f'scattered {self.model_name} {idx}', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), global_step=epoch, dataformats='HWC')

