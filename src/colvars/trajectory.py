import MDAnalysis as mda
import torch
import pandas as pd
import numpy as np

class WeightedTrajectory(object):
    r"""class that stores trajectory data.

    Parameters
    ----------
    pdb_filename : str
        name of pdf file
    traj_dcd_filename : str
        name of dcd file containing trajectory data

    Attributes
    ----------

    Example
    -------
    """
    def __init__(self, universe, weight_filename=None, min_w=0.0, max_w=1e10, verbose=True):

        if verbose: print ('\nloading trajectory to numpy array...', end='') 

        # load trajectory 
        self.trajectory = universe.trajectory.timeseries(order='fac')

        if verbose: print ('done.') 

        self.start_time = universe.trajectory.time
        self.dt = universe.trajectory.dt
        self.n_frames = universe.trajectory.n_frames

        # print information of trajectory
        if verbose: 
            print ('\nTrajectory Info:\n' \
                   '  no. of frames in trajectory data: {}\n' \
                   '  time of first frame: {:.1f}ps\n'\
                   '  time of last frame: {:.1f}ps\n'\
                   '  stepsize: {:.1f}ps\n' \
                   '  time length: {:.1f}ps\n'\
                   '  shape of trajectory data array: {}\n'.format(self.n_frames, 
                                                                   self.start_time, 
                                                                   universe.trajectory[-1].time,
                                                                   self.dt, 
                                                                   universe.trajectory.totaltime,
                                                                   self.trajectory.shape
                                                                )
                  )

        if weight_filename :

            time_weight_vec = pd.read_csv(weight_filename)
            # normalize
            time_weight_vec['weight'] /= time_weight_vec['weight'].mean()

            if verbose: 
                print ('\nloading weights from file: ', weight_filename)
                print ('\nWeights:\n', time_weight_vec['weight'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]))

            if self.start_time - time_weight_vec.iloc[0,0] > 0.01 or self.n_frames != len(time_weight_vec.index) :
                raise ValueError('Time in weight file does match the trajectory data!\n')
            else :
                if verbose: print ('\nCompatibility of weights and trajectory verified.\n')

            selected_idx = (time_weight_vec['weight'] > min_w) & (time_weight_vec['weight'] < max_w)
            weights = time_weight_vec[selected_idx].copy()

            self.trajectory = self.trajectory[selected_idx,:,:]

            weights['weight'] /= weights['weight'].mean()

            if verbose: 
                print ('\nAfter selecting states whose weights are in [{:.3e}, {:.3e}] and renormalization:\n' \
                       '\nShape of trajectory: {}'.format(min_w, max_w, self.trajectory.shape)
                      )
                print ('\nWeights:\n', weights['weight'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]))

            self.weights = weights['weight'].to_numpy()
        else :
            self.weights = np.ones(self.n_frames)
