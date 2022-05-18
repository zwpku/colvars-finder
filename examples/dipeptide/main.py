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
import configparser
from openmm import unit
import argparse

#sys.path.append('../colvarsfinder/core/')
sys.path.append('../')

from colvarsfinder.core import AutoEncoderTask, EigenFunctionTask
from colvarsfinder.nn import AutoEncoder, EigenFunctions 
from colvarsfinder.utils import WeightedTrajectory, integrate_md_langevin, calc_weights

# +
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

class Params(object):
    def __init__(self, task, config_filename='params.cfg'):

        config = configparser.ConfigParser()
        config.read(config_filename)

        if task == 'sampling':
            self.pdb_filename = config['System']['pdb_filename']
            self.psf_filename = config['System']['psf_filename']
            self.charmm_param_filename = config['System']['charmm_param_filename']
            self.n_steps = config['Sampling'].getint('n_steps')
            self.pre_steps = config['Sampling'].getint('pre_steps')
            self.step_size = config['Sampling'].getfloat('step_size') * unit.femtoseconds
            self.frictionCoeff = config['Sampling'].getfloat('friction') / unit.picosecond
            self.sampling_temp = config['Sampling'].getfloat('sampling_temperature') * unit.kelvin
            self.sampling_output_path = config['Sampling'].get('sampling_output_path')

            self.traj_dcd_filename = config['Sampling']['traj_dcd_filename']
            self.csv_filename = config['Sampling']['csv_filename']

            self.report_interval = config['Sampling'].getint('report_interval')
            self.report_interval_stdout = config['Sampling'].getint('report_interval_stdout')
            self.plumed_script = config['Sampling'].get('plumed_script')

        if task == 'calc_weights':
            self.sys_temp = config['System'].getfloat('sys_temperature') * unit.kelvin
            self.sampling_temp = config['Sampling'].getfloat('sampling_temperature') * unit.kelvin

            sampling_output_path = config['Sampling'].get('sampling_output_path')
            csv_filename = config['Sampling']['csv_filename']
            traj_weight_filename = config['Sampling']['traj_weight_filename']

            self.energy_col_idx_in_csv = config['Sampling'].getint('energy_col_idx_in_csv')
            assert self.energy_col_idx_in_csv , "Column idx in csv file not specified for calculating weights!"

            self.csv_filename=os.path.join(sampling_output_path, csv_filename)
            self.traj_weight_filename = os.path.join(sampling_output_path, traj_weight_filename)

        if task == 'training':
            self.sys_name = config['System'].get('sys_name')
            self.pdb_filename = config['System'].get('pdb_filename')
            self.temp = config['System'].getfloat('sys_temperature') * unit.kelvin
            sampling_output_path = config['Sampling'].get('sampling_output_path')
            self.traj_dcd_filename = config['Sampling'].get('traj_dcd_filename')
            self.traj_weight_filename = config['Sampling'].get('traj_weight_filename')

            # add path to filenames
            self.traj_dcd_filename = os.path.join(sampling_output_path, self.traj_dcd_filename)
            self.traj_weight_filename = os.path.join(sampling_output_path, self.traj_weight_filename)

             # unit: kJ/mol
            kT = self.temp * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA / unit.kilojoule_per_mole
            self.beta = 1.0 / kT
            
            #set training parameters
            self.cutoff_weight_min = config['Training'].getfloat('cutoff_weight_min')
            self.cutoff_weight_max = config['Training'].getfloat('cutoff_weight_max')

            self.use_gpu =config['Training'].getboolean('use_gpu')
            self.batch_size = config['Training'].getint('batch_size')
            self.num_epochs = config['Training'].getint('num_epochs')
            self.test_ratio = config['Training'].getfloat('test_ratio')
            self.learning_rate = config['Training'].getfloat('learning_rate')
            self.optimizer_name = config['Training'].get('optimizer') # 'Adam' or 'SGD'
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
            self.input_selector = config['Training'].get('input_mda_selector')
            self.feature_file = config['Training'].get('feature_file')
            self.seed = config['Training'].getint('seed')
            self.verbose = config['Training'].getboolean('verbose')
            if self.verbose is None :
                self.verbose = True 

            # CUDA support
            if torch.cuda.is_available() and self.use_gpu:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                self.use_gpu = False

        print (f'\n[Info] Parameters loaded from: {config_filename}\n', flush=True)

def train(args):

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
    input_ag = universe.select_atoms(args.input_selector)
    feature_mapper = ann.FeatureLayer(feature_list, input_ag, use_angle_value=False)

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
        align = ann.AlignmentLayer(align_atom_group, input_ag)
        # align.show_info()
    else :
        print ('No aligment used.')
        align = None

    print ('==============End of Alignment Info===================\n')

    pp_layer = ann.PreprocessingANN(align, feature_mapper)

    universe = mda.Universe(args.pdb_filename, args.traj_dcd_filename)

    print ('====================Trajectory Info===================')

    # load the trajectory data from DCD file
    traj_obj = WeightedTrajectory(universe, input_ag, None, args.traj_weight_filename, args.cutoff_weight_min, args.cutoff_weight_max)

    feature_dim = pp_layer.output_dimension()

    print ('================End of Trajectory Info=================\n')
    
    # path to store log data
    prefix = f"{args.sys_name}-{args.task_type}-" 
    model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    if args.task_type == 'Autoencoder' :

        # sizes of feedforward neural networks
        e_layer_dims = [feature_dim] + args.e_layer_dims + [args.k]
        d_layer_dims = [args.k] + args.d_layer_dims + [feature_dim]
        # define autoencoder
        model = AutoEncoder(e_layer_dims, d_layer_dims, args.activation()).to(device=args.device)

        # print the model
        if args.verbose: print ('\nAutoencoder: input dim: {}, encoded dim: {}\n'.format(feature_dim, args.k), model)

        # define training task
        train_obj = AutoEncoderTask(traj_obj, pp_layer,  model,  model_path, args.learning_rate, args.load_model_filename, args.save_model_every_step, args.batch_size, args.num_epochs, args.test_ratio, args.optimizer_name, args.device, args.verbose)

    else : # task_type: Eigenfunction

        layer_dims = [feature_dim] + args.layer_dims + [1]
        model = EigenFunctions(layer_dims, args.k, args.activation()).to(args.device)

        if args.verbose: print ('\nEigenfunctions:\n', model, flush=True)

        tot_dim = input_ag.n_atoms * 3
        # diagnoal matrix 
        # the unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
        diag_coeff = torch.ones(tot_dim).to(args.device) * args.diffusion_coeff * 1e7 * args.beta

        train_obj = EigenFunctionTask(traj_obj, pp_layer, model,  
                model_path, args.beta, diag_coeff, args.alpha, args.eig_w, args.learning_rate,
                args.load_model_filename, args.save_model_every_step,
                args.sort_eigvals_in_training, args.k, args.batch_size, args.num_epochs, args.test_ratio,
                args.optimizer_name, args.device, args.verbose)

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

    print ("\nTraining ends.\n") 

if __name__ == "__main__":

    if len(sys.argv) != 2 or sys.argv[1] not in ['sampling', 'calc_weights', 'training'] :
        print (f'Usage:\n' \
                '  1. To generate trajectory data: \n\t./main.py sampling\n' \
                '  2. To calculate weights of trajectory data: \n\t./main.py calc_weights\n' \
                '  3. To learn CVs: \n\t./main.py training')
    else :
        task = sys.argv[1]
        args = Params(task)
        if task == 'sampling' :
            if not os.path.exists(args.sampling_output_path):
                os.makedirs(args.sampling_output_path)
            integrate_md_langevin(args.pdb_filename, args.psf_filename, args.charmm_param_filename, args.n_steps, args.sampling_output_path, args.sampling_temp,  args.pre_steps, args.step_size, args.frictionCoeff, args.traj_dcd_filename, args.csv_filename, args.report_interval, args.report_interval_stdout)

        if task == 'calc_weights' :
            calc_weights(args.csv_filename, args.sampling_temp, args.sys_temp, args.traj_weight_filename, args.energy_col_idx_in_csv)

        if task == 'training' :
            train(args)

