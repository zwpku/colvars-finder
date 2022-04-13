import torch
import molann.ann as ann
import configparser
import os 

from openmm import unit

class Args(object):

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

        # CUDA support
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False

        print (f'\n[Info] Parameters loaded from: {config_filename}\n', flush=True)

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

