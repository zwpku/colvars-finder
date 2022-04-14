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
from colvarsfinder.core.base_task import TrainingArgs

# +
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

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

    print ('\nFeatures for histogram: \n')
    # read features for histogram plot
    feature_reader = FeatureFileReader(args.feature_file, 'Histogram', universe)
    feature_list = feature_reader.read()

    if len(feature_list) == 0 : 
        print ("No feature found for histogram! \n") 
        histogram_feature_mapper = None
    else :
        histogram_feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=True)
        print (histogram_feature_mapper.get_feature_info())
        # make sure each feature is one-dimensional
        assert histogram_feature_mapper.output_dimension() == len(feature_list), "Features for histogram are incorrect, output of each feature must be one-dimensional!" 

    print ('\nFeatures for ouptut: \n')
    # features to define a 2d space for output
    feature_reader = FeatureFileReader(args.feature_file, 'Output', universe) 
    feature_list= feature_reader.read()
    if len(feature_list) == 2 :
        output_feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=True)
    else:
        output_feature_mapper = None

    if output_feature_mapper is not None and output_feature_mapper.output_dimension() == 2 : # use it only if it is 2D
        print (output_feature_mapper.get_feature_info())
    else :
        print (f'\nOutput feature mapper set to None, since 2d feature required for output.')
        output_feature_mapper = None

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
        # define training task
        train_obj = AutoEncoderTask(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)
    else : # task_type: Eigenfunction
        prefix = f"{args.sys_name}-eigenfunction-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
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

