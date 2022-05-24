r"""Trajectory Data --- :mod:`colvarsfinder.utils`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements the function :meth:`integrate_md_langevin` and :meth:`integrate_sde_overdamped` for sampling trajectory data of (molecular) dynamical systems, the function :meth:`calc_weights` for calculating weights of the states, and the class :class:`WeightedTrajectory` for holding trajectory information.

.. rubric:: Typical usage

The main purpose of this module is to prepare training data for the training tasks in the module :mod:`colvarsfinder.core`.
Assume that the invariant distribution of the system is :math:`\mu` at temperature :math:`T`.
Direct simulation becomes inefficient, when there is metastability in system's dynamics.
To mitigate the difficulty due to metastability, one can use this module in the following two steps.

    #. Run :meth:`integrate_md_langevin` or :meth:`integrate_sde_overdamped` to sample states :math:`(x_l)_{1\le l \le n}` of the (biased) system at a slightly high temperature :math:`T_{sim} > T`; 

    #. Use :meth:`calc_weights` to generate a CSV file that contains the weights :math:`(w_l)_{1\le l \le n}` of the sampled states. 


The weights are calculated in a way such that one can approximate the distribution :math:`\mu` using the biased data :math:`(x_l)_{1\le l \le n}` by

.. math::
   \int_{\mathbb{R}^{d}} f(x) \mu(dx) \approx \frac{\sum_{l=1}^n v_l f(x_l)}{\sum_{l=1}^n v_l}\,,

for test functions :math:`f`, where :math:`d` is the dimension of the system.

Using the states and weights generated in the above two steps, one can construct the class :class:`WeightedTrajectory`, which can then be passed on to training task classes in the module :mod:`colvarsfinder.core`.

Classes
-------
.. autoclass:: WeightedTrajectory
    :members:

Functions
---------

.. autofunction:: integrate_md_langevin

.. autofunction:: integrate_sde_overdamped

.. autofunction:: calc_weights

"""


import math
from random import random, randint
import numpy as np
import time
import datetime
import os, sys
import warnings
from sys import stdout
import pandas as pd

from openmm import *
from openmm.app import *

# ### MD simulation

class WeightedTrajectory:
    r"""Class that stores trajectory data assoicated to an MDAnalysis Universe.

    Args:
        universe (:external+mdanalysis:class:`MDAnalysis.core.universe.Universe`): a MDAnalysis Universe that contains trajectory data
        input_ag (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): atom group used for input. All atoms are selected, if it is none. This argument is relevent only when *universe* is not None.
        traj_filename (str): filename of a text file that contains trajectory data 
        weight_filename (str): filename of a CSV file that contains weights of trajectory 
        min_w (float): minimal value of weights below which the corresponding states will be discarded
        max_w (float): maximal value of weights above which the corresponding states will be discarded
        verbose (bool): print more information if ture

    Note: 
        #. Trajectory data will be loaded from `universe` if it is not None. Otherwise, trajectory data will be loaded from the text file specified by `traj_filename`.
        #. When loading trajectory data from text file, each line of the file should contain :math:`d+1` floats, separated by a space. The first float is the current time, and the remaining :math:`d` floats are coordinates of the states.

    Note:
        #. Weights are loaded from a CSV file if a filename is provided. The weights are normalized such that its mean value is one. Then, states in the trajectory data whose weights are not within [min_w, max_w] will be discarded, and weights of the remaining states are normalized again to have mean value one.
        #. All weights will be set to one, if weight_filename is None.

    Raises:

        FileNotFoundError: if `univsere` is None and `traj_filename` does not point to an existing file.
        ValueError: if the lengths of trajectory data and weights are not the same.
      
    Attributes:
        trajectory (numpy array): states in the trajectory. When the trajectory is loaded from a `universe`, its shape is :math:`[n, N, 3]`, where :math:`n` =n_frames is the number of states, and :math:`N` is the number of atoms of the system. When the trajectory is loaded from a text file, its shape is :math:`[n, d]`, where :math:`d` is system's dimension.
            
        n_frames (int): number of states in the trajectory
        weights (1d numpy array): weights of states

    """
    def __init__(self, universe=None, input_ag=None, traj_filename=None, weight_filename=None, min_w=0.0, max_w=float("inf"), verbose=True):
        
        if universe is not None :

            if verbose: print ('\nloading trajectory to numpy array...', end='') 

            if input_ag is None :
                input_atom_indices = univser.atoms.ids - 1
            else :
                input_atom_indices = input_ag.ids - 1

            # load trajectory 
            self.trajectory = universe.trajectory.timeseries(order='fac')[:,input_atom_indices,:]

            if verbose: print ('done.') 

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
                                                                       universe.trajectory.time, 
                                                                       universe.trajectory[-1].time,
                                                                       universe.trajectory.dt, 
                                                                       universe.trajectory.totaltime,
                                                                       self.trajectory.shape
                                                                    )
                      )

        else : # otherwise, read trajectory data from text file
            if traj_filename is None or not os.path.exists(traj_filename):
                raise FileNotFoundError('trajectory file not found')
            data_block = np.loadtxt(traj_filename)
            self.n_frames = data_block.shape[0]
            self.trajectory = data_block[:,1:]

        if weight_filename :

            weight_vec = pd.read_csv(weight_filename, usecols=[0], header=None)

            # normalize
            weight_vec[0] /= weight_vec[0].mean()

            if verbose: 
                print ('\nloading weights from file: ', weight_filename)
                print ('\nWeights:\n', weight_vec[0].describe(percentiles=[0.2, 0.4, 0.6, 0.8]))

            if self.n_frames != len(weight_vec.index) :
                raise ValueError('length in weight file does match the trajectory data!\n')

            selected_idx = (weight_vec[0] > min_w) & (weight_vec[0] < max_w)
            weights = weight_vec[selected_idx].copy()

            self.trajectory = self.trajectory[selected_idx,...]

            weights[0] /= weights[0].mean()

            if verbose: 
                print ('\nAfter selecting states whose weights are in [{:.3e}, {:.3e}] and renormalization:\n' \
                       '\nShape of trajectory: {}'.format(min_w, max_w, self.trajectory.shape)
                      )
                print ('\nWeights:\n', weights[0].describe(percentiles=[0.2, 0.4, 0.6, 0.8]))

            self.weights = weights[0].to_numpy()
        else :
            self.weights = np.ones(self.n_frames)

# Generate MD trajectory data using OpenMM 
def integrate_md_langevin(pdb, system, integrator, n_steps, sampling_output_path,  pre_steps=0, traj_dcd_filename='traj.dcd', csv_filename='output.csv', report_interval=100, report_interval_stdout=100, plumed_script=None):
    r"""Generate trajectory data by integrating Langevin dynamics using OpenMM.

    Args:
        pdb (:external+openmm:class:`openmm.app.pdbfile.PDBFile`): PDB file
        system (:external+openmm:class:`openmm.openmm.System`): system to simulate
        integrator (:external+openmm:class:`openmm.openmm.Integrator`): integrator used to simulate the system
        n_steps (int): total number of steps to integrate
        sampling_output_path (str): directory to save results
        pre_steps (int): number of warm-up steps to run before integrating the system for n_steps
        traj_dcd_filename (str): filename of the DCD file to save trajectory 
        csv_filename (str): filename of the CSV file to store statistics
        report_interval (int): how often to write trajectory to DCD file and to write statistics to CSV file
        report_interval_stdout (int): how often to print to stdout
        plumed_script (str): script of PLUMED commands 

    This class encloses commands to simulate a Langevin dynamics using `OpenMM <https://openmm.org>`__ package.

    Note: 
       #. The first line of the CSV file contains titles of the output statistics. Each of the following lines records the time, potential energy, total energy, and temperature of the system, separated by a comma.  An example of the output CSV file is given below.
       #. Optionally, by taking advantage of the `OpenMM PLUMED plugin <https://www.github.com/openmm/openmm-plumed>`__, a `PLUMED <https://www.plumed.org>`__ script can be provided, either to record statistics or even to add biasing forces.

    Example:
        Below is an example of the CSV file containing statistics recorded during simulation.
     

    .. code-block:: text
        :caption: output.csv

        #"Time (ps)","Potential Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)"
        1.9999999999998905,-20.050460354856185,40.50556969779072,285.61632731254656
        2.9999999999997806,-15.15398066696433,84.48038169480243,469.9317413501974
        3.9999999999996705,15.302661169416211,121.54570823139409,501.1020187082061
        5.000000000000004,12.581352923170044,96.93859523113919,397.87624303095436
        6.000000000000338,-12.222791961491907,69.45248734168707,385.22659570846105
        7.000000000000672,-16.41391301364837,91.95546048082197,511.13097116410677
        8.000000000001005,12.60815636162124,79.3336199461453,314.71484888738695
        ...

    """

    print ( 'Directory to save trajectory ouptuts: %s' % sampling_output_path)

    # add path to filenames
    traj_dcd_filename = os.path.join(sampling_output_path, traj_dcd_filename)
    csv_filename=os.path.join(sampling_output_path, csv_filename)

    # prepare before simulation
    if plumed_script is not None :
        from openmmplumed import PlumedForce
        print ('plumed script: %s' % plumed_script)
        system.addForce(PlumedForce(plumed_script))

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    platform = simulation.context.getPlatform()
    print ("\nUsing OpenMM platform: %s\n" % platform.getName())

    print ('\nStep 1: Energy minimization...', end='')
    simulation.minimizeEnergy()
    print ('done.\n')
    print ('Step 2: Run {} steps before recording statistics...'.format(pre_steps), end='', flush=True)
    simulation.step(pre_steps)
    print ('done.\n')
    # registrate reporter for output
    simulation.reporters = []

    simulation.reporters.append(DCDReporter(traj_dcd_filename, report_interval))
    simulation.reporters.append(StateDataReporter(stdout, report_interval_stdout, step=True,
                                                  temperature=True, elapsedTime=True))
    simulation.reporters.append(StateDataReporter(csv_filename, report_interval, time=True,
                                                  potentialEnergy=True,
                                                  totalEnergy=True,
                                                  temperature=True))

    # run the simulation
    print ('Step 3: Simulation starts.', flush=True)
    start = time.time()
    simulation.step(n_steps)
    end = time.time()
    print ( 'Simulation ends, %d sec. elapsed.' % (end - start) )

    del simulation

def integrate_sde_overdamped(pot_obj, n_steps, sampling_output_path, pre_steps=0, step_size=0.01,
        traj_txt_filename='traj.txt', csv_filename='output.csv', report_interval=100, report_interval_stdout=100):
    r"""Generate trajectory data by integrating overdamped Langevin dynamics using Euler-Maruyama scheme.

    Args:
        pot_obj: class that specifies potential :math:`V`
        n_steps (int): total number of steps to integrate
        sampling_output_path (str): directory to save results
        pre_steps (int): number of warm-up steps to run before integrating the system for n_steps
        step_size (float): step-size to integrate SDE, unit: dimensionless
        traj_txt_filename (str): filename of the text file to save trajectory 
        csv_filename (str): filename of the CSV file to store statistics
        report_interval (int): how often to write trajectory to text file and to write statistics to CSV file
        report_interval_stdout (int): how often to print to stdout

    Note: 
       #. `pot_obj` is an object of a class which has attributes `dim` (int), `beta` (positive real), and memeber functions `V` and `gradV`.
       #. The first line of the CSV file contains titles of the output statistics. Each of the following lines records the time and energy of the system, separated by a comma.  An example of the output CSV file is given below.

    Example:
        Below is an example of `pot_obj`.

    .. code-block:: python

        class mypot(object):
            def __init__(self):
                self.dim = 2
                self.beta = 1.0

            def V(self, x):
                return 0.5 * x[0]**2 + 2.0 * x[1]**2

            def gradV(self, x): 
                return np.array([x[0], 4.0 * x[1]])

        pot_obj = mypot()

    Example:
        Below is an example of the CSV file containing statistics recorded during simulation.

    .. code-block:: text
        :caption: output.csv

        Time,Energy
        0.0,5.423187853452168
        1.0,6.17342106224266
        2.0,1.1311978694547915
        3.0,0.8239473412429524
        4.0,0.02854608581728689
        5.0,1.327977569243032
        6.0,5.452589054795339
        7.0,3.61438132406788
        8.0,2.297568687904894
        ...
    """

    dim = pot_obj.dim
    # beta is dimensionless
    sampling_beta = pot_obj.beta 

    print (f'\nSampling temperature: {sampling_temp}')
    print (f'Directory to save trajectory ouptuts: {sampling_output_path}')
    print (f'sampling beta={sampling_beta:.3f}, dt={step_size:.3f}\n')

    X0 = np.random.randn(dim)

    print (f'First, burning, total number of steps = {pre_steps}')
    for i in range(pre_steps):
        xi = np.random.randn(dim)
        X0 = X0 - pot_obj.gradV(X0) * step_size + np.sqrt(2 * step_size / sampling_beta) * xi

    # add path to filenames
    traj_txt_filename = os.path.join(sampling_output_path, traj_txt_filename)

    print (f'Next, run {n_steps} steps')

    csv_data_list = []

    with open(traj_txt_filename, 'w+') as f:
        for i in range(n_steps):
            xi = np.random.randn(dim)
            X0 = X0 - pot_obj.gradV(X0) * step_size + np.sqrt(2 * step_size / sampling_beta) * xi

            if i % report_interval == 0 :
                str_data = f"{i * step_size:.3f} " + ' '.join([f'{x:.6f}' for x in X0]) + '\n'
                f.write(str_data)
                energy = pot_obj.V(X0)
                csv_data_list.append([i * step_size, energy])

            if i % report_interval_stdout == 0:
                energy = pot_obj.V(X0)
                print (f'step={i}, time={i * step_size:.3f}, energy={energy:.3f}', flush=True)

    csv_data = pd.DataFrame(csv_data_list, columns=['Time', 'Energy'])
    csv_filename=os.path.join(sampling_output_path, csv_filename)
    csv_data.to_csv(csv_filename, index=False)

def calc_weights(csv_filename, sampling_beta, sys_beta, traj_weight_filename='weights.txt', energy_col_idx=1):
    r"""Calculate weights of the trajectory data.

    Args:
        csv_filename (str): filename of a CSV file generated by :meth:`integrate_md_langevin`
        sampling_beta (float): :math:`\beta_{sim}` 
        sys_beta (float): :math:`\beta_{sys}`
        traj_weight_filename (str): filename to output the weights of trajectory
        energy_col_idx (int): the index of the column of the CSV file, based on which the weights will be calculated.

    This function uses certain column (specified by energy_col_idx) of the CSV file (specified by csv_filename) as energy, and computes the weights 
    simply by 

    .. math::
        v_i = \frac{1}{Z} \mathrm{e}^{-(\beta_{sys}-\beta_{sim}) V_i}\,, 

    where :math:`i=1,\dots, l` is the index of states, :math:`V_i` is the energy value, and :math:`Z` is the normalizing constant such that the mean value of the weights are one.

    Example:

        Below is an example of the file generated by :meth:`calc_weights`: 

    .. code-block:: text
        :caption: weights.txt

        0.8979287896010564
        0.5424829921777613
        0.02360908666276056
        0.03124009081426178
        0.4012103169413903
        0.617590719346622
        0.031154032337133607
        0.299688734996611
        0.03230412279837258
        ...

    """

    # use potential energy in the csv file generated by OpenMM code
    print (f'\n=============Calculate Weights============')
    print (f'Reading potential from: {csv_filename}')
    vec = pd.read_csv(csv_filename)
    # modify the name of the first column
    vec.rename(columns={vec.columns[0]: 'Time (ps)'}, inplace=True)
    # show the data 
    print ('\nWhole data:\n', vec.head(8))

    # select the column containing energy used to calculate weights
    energy_col_name=vec.columns[energy_col_idx]
    print ('\nUse {:d}th column to reweight, name: {}'.format(energy_col_idx, energy_col_name) )

    energy_list = vec[energy_col_name]
    mean_energy = energy_list.mean()

    print (f'\nsampling beta={sampling_beta}, system beta={sys_beta}')

    # compute weights from potential energy
    nonnormalized_weights = [math.exp(-(sys_beta - sampling_beta) * (energy - mean_energy)) for energy in energy_list] 
    weights = pd.DataFrame(nonnormalized_weights / np.mean(nonnormalized_weights), columns=['weight'] )

    print ('\nWeight:\n', weights.head(8), '\n\nSummary of weights:\n', weights.describe())

    weights.to_csv(traj_weight_filename, header=False, index=False)
    print (f'weights saved to: {traj_weight_filename}')

