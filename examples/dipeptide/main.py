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

sys.path.append('../src/cvfinder/')
from training_tasks import AutoEncoderTask, EigenFunctionTask 
from trajectory import WeightedTrajectory
from utils import Args
# -

# +
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

def main():

    # read configuration parameters
    args = Args()

    if args.seed:
        set_all_seeds(args.seed)

    # load the trajectory data from DCD file
    universe = mda.Universe(args.pdb_filename, args.traj_dcd_filename)

    atoms_info = pd.DataFrame(
        np.array([universe.atoms.ids, universe.atoms.names,
            universe.atoms.types, universe.atoms.masses,
            universe.atoms.resids, universe.atoms.resnames]).T, 
        columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
        )

    print ('Atom information:\n', atoms_info)

    print ('\nSummary:\n', atoms_info['type'].value_counts().rename_axis('type').reset_index(name='counts'))

    # print information of trajectory
    print ('\nno. of atoms: {}\nno. of residues: {}\n'.format(universe.trajectory.n_atoms, universe.residues.n_residues) )

    # read features for histogram plot
    feature_reader = FeatureFileReader(args.feature_file, 'Histogram', universe)
    feature_list = feature_reader.read()

    histogram_feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=True)
    print (histogram_feature_mapper.get_feature_info())

    # make sure each feature is one-dimensional
    assert histogram_feature_mapper.output_dimension() == len(feature_list), "Features for histogram are incorrect, output of each feature must be one-dimensional!" 
    # features to define a 2d space for output
    feature_reader = FeatureFileReader(args.feature_file, 'Output', universe) 
    feature_list= feature_reader.read()
    output_feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=True)

    if output_feature_mapper.output_dimension() == 2 and len(feature_list) == 2 : # use it only if it is 2D
        print (output_feature_mapper.get_feature_info())
    else :
        print (f'\nOutput feature mapper set to None, since 2d feature required for output.')
        output_feature_mapper = None

    # read features from file to define preprocessing
    feature_reader = FeatureFileReader(args.feature_file, 'Preprocessing', universe)
    feature_list = feature_reader.read()
    
    # define the map from positions to features 
    feature_mapper = ann.FeatureLayer(feature_list, use_angle_value=False)

    # display information of features used 
    print ('\nFeatures in preprocessing layer:')
    print (feature_mapper.get_feature_info())

    if 'position' in [f.get_type() for f in feature_list] : # if atom positions are used, add alignment to preprocessing layer
        # define alignment using positions in pdb file
        ref_universe = mda.Universe(args.pdb_filename)
        align_atom_group = ref_universe.select_atoms(args.align_selector)
        print ('\nAdd alignment to preprocessing layer.\naligning by atoms:')
        print (atoms_info.loc[atoms_info['id'].isin(align_atom_group.ids)][['id','name', 'type']], flush=True)
        align = ann.AlignmentLayer(align_atom_group)
        align.show_info()
    else :
        align = None

    pp_layer = ann.PreprocessingANN(align, feature_mapper)

    # read trajectory
    traj_obj = WeightedTrajectory(universe, args.traj_weight_filename, args.cutoff_weight_min, args.cutoff_weight_max)

    if args.train_ae :
        # path to store log data
        prefix = f"{args.sys_name}-autoencoder-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        # define training task
        train_obj = AutoEncoderTask(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)
    else :
        prefix = f"{args.sys_name}-eigenfunction-" 
        model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        train_obj = EigenFunctionTask(args, traj_obj, pp_layer, model_path, histogram_feature_mapper, output_feature_mapper)

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

