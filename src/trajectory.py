import MDAnalysis as mda
import torch
import pandas as pd
import numpy as np

class Trajectory(object):
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
    def __init__(self, pdb_filename, traj_dcd_filename, beta=1.0, weight_filename=None):

        # load the trajectory data from DCD file
        self.u = mda.Universe(pdb_filename, traj_dcd_filename)

        # load the reference configuration from the PDB file
        self.ref = mda.Universe(pdb_filename) 

        self.atoms_info = pd.DataFrame(
            np.array([self.ref.atoms.ids, self.ref.atoms.names,
                self.ref.atoms.types, self.ref.atoms.masses,
                self.ref.atoms.resids, self.ref.atoms.resnames]).T, 
            columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
            )

        # print information of trajectory
        print ('\nMD system:\n\
        \tno. of atoms: {}\n\
        \tno. of residues: {}\n'.format(self.ref.trajectory.n_atoms, self.ref.residues.n_residues)
              )
        print ('Detailed atom information:\n', self.atoms_info)

        print ('\nSummary:\n', self.atoms_info['type'].value_counts().rename_axis('type').reset_index(name='counts'))

        self.load_traj()

        self.ref_pos = self.ref.atoms.positions

        if weight_filename :
            self.weights = self.load_weights(weight_filename)
            # normalize
            self.weights = self.weights / np.mean(self.weights)
        else :
            self.weights = np.ones(self.n_frames)

    def load_traj(self):

        print ('\nloading trajectory to numpy array...', end='')
        # load trajectory to torch tensor 
        self.trajectory = torch.from_numpy(self.u.trajectory.timeseries(order='fac')).double()

        print ('done.')

        self.start_time = self.u.trajectory.time
        self.dt = self.u.trajectory.dt
        self.n_frames = self.u.trajectory.n_frames

        # print information of trajectory
        print ('\nTrajectory Info:\n\
        \tno. of frames in trajectory data: {}\n\
        \tstepsize: {:.1f}ps\n\
        \ttime of first frame: {:.1f}ps\n\
        \ttime length: {:.1f}ps\n\
        \tshape of trajectory data array: {}\n'.format(self.n_frames, 
                                          self.dt, 
                                          self.start_time, 
                                          self.u.trajectory.totaltime,
                                          self.trajectory.shape
                                         )
              )


    def load_weights(self, weight_filename):
        print ('\nloading weights from file: ', weight_filename)
        time_weight_vec = pd.read_csv(weight_filename)
        time_weight_vec['weight'] /= time_weight_vec['weight'].mean()
        print ('\n', time_weight_vec.head(8))
        time_weight_vec = time_weight_vec.to_numpy()
        if self.start_time - time_weight_vec[0,0] > 0.01 or self.n_frames != time_weight_vec.shape[0] :
            print ('Error: time in weight file does match the trajectory data!\n')
            exit(0)
        # weights are in the second column
        return time_weight_vec[:,1]

