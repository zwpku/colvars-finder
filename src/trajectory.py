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
    def __init__(self, universe, weight_filename=None):


        print ('\nloading trajectory to numpy array...', end='')

        # load trajectory to torch tensor 
        self.trajectory = universe.trajectory.timeseries(order='fac')

        print ('done.')

        self.start_time = universe.trajectory.time
        self.dt = universe.trajectory.dt
        self.n_frames = universe.trajectory.n_frames

        # print information of trajectory
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
            self.weights = self.load_weights(weight_filename)
        else :
            self.weights = np.ones(universe.n_frames)

    def load_weights(self, weight_filename):
        """
        TBA
        """
        print ('\nloading weights from file: ', weight_filename)

        time_weight_vec = pd.read_csv(weight_filename)
        # normalize
        time_weight_vec['weight'] /= time_weight_vec['weight'].mean()

        print ('\n', time_weight_vec.head(8))

        if self.start_time - time_weight_vec.iloc[0,0] > 0.01 or self.n_frames != len(time_weight_vec.index) :
            print ('Error: time in weight file does match the trajectory data!\n')
            exit(0)

        # weights are in the second column
        return time_weight_vec.iloc[:,1].to_numpy()

