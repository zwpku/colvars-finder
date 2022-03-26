#!/usr/bin/env python
# +
from molann.trajectory import Trajectory
from molann.feature import FeatureFileReader, FeatureMap
import configparser
import timeit
from cvfinder.training_tasks import AutoEncoderTask, EigenFunctionTask 
import torch
import random
import numpy as np
import os
import time

# -

# +
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

class MyArgs(object):

    def __init__(self, config_filename='params.cfg'):

        config = configparser.ConfigParser()
        config.read(config_filename)

        self.pdb_filename = config['System'].get('pdb_filename')
        self.traj_dcd_filename = config['System'].get('traj_dcd_filename')
        self.traj_weight_filename = config['System'].get('traj_weight_filename')
        self.sys_name = config['System'].get('sys_name')
        self.temp = config['System'].getfloat('temperature')

         # unit: kJ/mol
        kT = self.temp * 1.380649 * 6.02214076 * 1e-3
        self.beta = 1.0 / kT
         
        #set training parameters
        self.use_gpu =config['Training'].getboolean('use_gpu')
        self.batch_size = config['Training'].getint('batch_size')
        self.num_epochs = config['Training'].getint('num_epochs')
        self.test_ratio = config['Training'].getfloat('test_ratio')
        self.learning_rate = config['Training'].getfloat('learning_rate')
        self.optimizer = config['Training'].get('optimizer') # 'Adam' or 'SGD'
        self.load_model_filename =  config['Training'].get('load_model_filename')
        self.model_save_dir = config['Training'].get('model_save_dir') 
        self.save_model_every_step = config['Training'].getint('save_model_every_step')
        self.train_ae = config['Training'].getboolean('train_autoencoder')

        if self.train_ae :
            # encoded dimension
            self.k = config['AutoEncoder'].getint('encoded_dim')
            self.e_layer_dims = [int(x) for x in config['AutoEncoder'].get('encoder_hidden_layer_dims').split(',')]
            self.d_layer_dims = [int(x) for x in config['AutoEncoder'].get('decoder_hidden_layer_dims').split(',')]
            self.activation_name = config['AutoEncoder'].get('activation') 
        else :
            self.k = config['EigenFunction'].getint('num_eigenfunction')
            self.layer_dims = [int(x) for x in config['EigenFunction'].get('hidden_layer_dims').split(',')]
            self.activation_name = config['EigenFunction'].get('activation') 
            self.alpha = config['EigenFunction'].getfloat('penalty_alpha')
            self.eig_w = [float(x) for x in config['EigenFunction'].get('eig_w').split(',')]
            self.diffusion_coeff = config['EigenFunction'].getfloat('diffusion_coeff')
            self.sort_eigvals_in_training = config['EigenFunction'].getboolean('sort_eigvals_in_training')

        self.activation = getattr(torch.nn, self.activation_name) 

        self.align_selector = config['Training'].get('align_mda_selector')
        self.feature_file = config['Training'].get('feature_file')
        self.seed = config['Training'].getint('seed')
        self.num_scatter_states = config['Training'].getint('num_scatter_states')

        if self.seed:
            set_all_seeds(self.seed)

        # CUDA support
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda')
            print (f'device name: {self.device}')
            print ('Active CUDA Device: GPU', torch.cuda.current_device())
            print ('Available devices: ', torch.cuda.device_count())
            print ('CUDA name: ', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
            print (f'device name: {self.device}')

        print (f'Parameters loaded from: {config_filename}\n', flush=True)

def main():

    # read configuration parameters
    args = MyArgs()

    # read trajectory
    traj_obj = Trajectory(args.pdb_filename, args.traj_dcd_filename, args.beta, args.traj_weight_filename)

    # read features for histogram plot
    feature_reader = FeatureFileReader(args.feature_file, 'Histogram', traj_obj.u, ignore_position_feature=True)
    feature_list = feature_reader.read()

    histogram_feature_mapper = FeatureMap(feature_list, use_angle_value=True)
    histogram_feature_mapper.info('\nFeatures to plot histograms\n')

    # make sure each feature is one-dimensional
    assert histogram_feature_mapper.feature_total_dimension() == len(feature_list), "Feature map for histogram is incorrect" 

    # features to define a 2d space for output
    feature_reader = FeatureFileReader(args.feature_file, 'Output', traj_obj.u, ignore_position_feature=True) # positions are ignored
    feature_list= feature_reader.read()

    if len(feature_list) == 2 : # use it only if it is 2D
        output_feature_mapper = FeatureMap(feature_list, use_angle_value=True)
        output_feature_mapper.info('\n2d feature List for output:\n')
    else :
        print (f'\nOutput feature mapper set to None, since 2d feature required for output, but {len(feature_list)} are provided.')
        output_feature_mapper = None

    if args.train_ae :
        # path to store log data
        prefix = f"{args.sys_name}-autoencoder-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        # define training task
        train_obj = AutoEncoderTask(args, traj_obj, model_path, histogram_feature_mapper, output_feature_mapper)
    else :
        prefix = f"{args.sys_name}-eigenfunction-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        train_obj = EigenFunctionTask(args, traj_obj, model_path, histogram_feature_mapper, output_feature_mapper)

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


