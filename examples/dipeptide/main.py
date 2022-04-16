#!/usr/bin/env python
# +
from molann.feature import FeatureFileReader
import molann.ann as ann 
import timeit
import torch
import random
import numpy as np
import os
import sys
import time
import MDAnalysis as mda
import pandas as pd

#sys.path.append('../colvarsfinder/core/')
sys.path.append('../')

from colvarsfinder.core.autoencoder import AutoEncoderTask
from colvarsfinder.core.eigenfunction import EigenFunctionTask 
from colvarsfinder.core.trajectory import WeightedTrajectory

# +
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

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

def main():

    # read configuration parameters
    args = TrainingArgs()

    print (f'===Computing Devices===')
    # CUDA support
    if args.use_gpu:
        print (f'Device name: {args.device}')
        print ('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices: ', torch.cuda.device_count())
        print ('CUDA name: ', torch.cuda.get_device_name(0))
    else:
        print (f'Device name: {args.device}')

    print (f'=======================\n')

    if args.seed:
        set_all_seeds(args.seed)

    universe = mda.Universe(args.pdb_filename)

    atoms_info = pd.DataFrame(
        np.array([universe.atoms.ids, universe.atoms.names,
            universe.atoms.types, universe.atoms.masses,
            universe.atoms.resids, universe.atoms.resnames]).T, 
        columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
        )

    print ('==========System Info=================\n', atoms_info)

    print ('\nSummary:\n', atoms_info['type'].value_counts().rename_axis('type').reset_index(name='counts'))

    # print information of trajectory
    print ('{} atoms, {} residues.'.format(universe.atoms.n_atoms, universe.residues.n_residues) )
    print ('==========End of System Info==========\n')

    print ('==============Features===================\n')
    print ('Features file: {}'.format(args.feature_file)) 

    # read features from file to define preprocessing
    feature_reader = FeatureFileReader(args.feature_file, 'Preprocessing', universe)
    feature_list = feature_reader.read()
    
    # define the map from positions to features 
    feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=False)

    print ('\nFeatures in preprocessing layer:')
    # display information of features used 
    print (feature_mapper.get_feature_info())

    print ('==============End of Features===================\n')

    print ('===================Alignment Info======================\n')

    if 'position' in [f.get_type() for f in feature_list] : # if atom positions are used, add alignment to preprocessing layer
        # define alignment using positions in pdb file
        align_atom_group = universe.select_atoms(args.align_selector)
        print ('\nAdd alignment to preprocessing layer.\naligning by atoms:')
        print (atoms_info.loc[atoms_info['id'].isin(align_atom_group.ids)][['id','name', 'type']], flush=True)
        align = ann.AlignmentLayer(align_atom_group)
        # align.show_info()
    else :
        print ('No aligment used.')
        align = None

    print ('==============End of Alignment Info===================\n')

    pp_layer = ann.PreprocessingANN(align, feature_mapper)

    universe = mda.Universe(args.pdb_filename, args.traj_dcd_filename)

    print ('====================Trajectory Info===================')

    # load the trajectory data from DCD file
    traj_obj = WeightedTrajectory(universe, args.traj_weight_filename, args.cutoff_weight_min, args.cutoff_weight_max)

    print ('================End of Trajectory Info=================\n')

    if args.task_type == 'Autoencoder' :
        # path to store log data
        prefix = f"{args.sys_name}-autoencoder-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

        # sizes of feedforward neural networks
        e_layer_dims = [self.feature_dim] + args.e_layer_dims + [self.k]
        d_layer_dims = [self.k] + args.d_layer_dims + [self.feature_dim]

        # define autoencoder
        self.model = AutoEncoder(e_layer_dims, d_layer_dims, args.activation()).to(device=self.device)
        # print the model
        if self.verbose: print ('\nAutoencoder: input dim: {}, encoded dim: {}\n'.format(self.feature_dim, self.k), self.model)

        # define training task
        train_obj = AutoEncoderTask(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)
    else : # task_type: Eigenfunction
        prefix = f"{args.sys_name}-eigenfunction-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        layer_dims = [self.feature_dim] + args.layer_dims + [1]
        self.model = EigenFunction(layer_dims, self.k, args.activation()).to(self.device)
        if self.verbose: print ('\nEigenfunctions:\n', self.model, flush=True)
        # diagnoal matrix 
        # the unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
        self.diag_coeff = torch.ones(self.tot_dim).to(self.device) * args.diffusion_coeff * 1e7 * self.beta
        train_obj = EigenFunctionTask(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper, True)

    if args.use_gpu :
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # train autoencoder
        train_obj.train()
        end.record()
        torch.cuda.synchronize()
        print ('Runtime : {:.2f}s\n'.format(start.elapsed_time(end) * 1e-3) )
    else :
        start = timeit.default_timer()
        # train autoencoder
        train_obj.train()
        print ('Runtime : {:.2f}s\n'.format(timeit.default_timer() - start) )

if __name__ == "__main__":
    main()

