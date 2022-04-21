r"""Trajectory Data --- :mod:`colvarsfinder.utils`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements the function :meth:`integrate_langevin` for sampling
trajectory data, the function :meth:`calc_weights` for calculating weights of
the states, and the class :class:`WeightedTrajectory` for holding trajectory information.

.. rubric:: Typical usage

The main purpose of this module is to prepare training data for the training tasks in the module :mod:`colvarsfinder.core`.
Assume that the invariant distribution of the system is :math:`\mu` at temperature :math:`T`.
Direct simulation becomes inefficient, when there is metastability in system's dynamics.
To mitigate the difficulty due to metastability, one can use this module in the following two steps.

    #. Run :meth:`integrate_langevin` to sample states :math:`(x_l)_{1\le l \le n}` of the (biased) system at a slightly high temperature :math:`T_{sim} > T`; 

    #. Use :meth:`calc_weights` to generate a CSV file that contains the weights :math:`(w_l)_{1\le l \le n}` of the sampled states. 

The weights are calculated in a way such that one can approximate the distribution :math:`\mu` using the biased data :math:`(x_l)_{1\le l \le n}` by

.. math::
   \int_{\mathbb{R}^{d}} f(x) \mu(dx) \approx \frac{\sum_{l=1}^n v_l f(x_l)}{\sum_{l=1}^n v_l}\,,

for test functions :math:`f`, where :math:`d=3N` is the dimension of the system.

Using the states and weights generated in the above two steps, one can construct the class :class:`WeightedTrajectory`, which can then be passed on to training task classes in the module :mod:`colvarsfinder.core`.

Classes
-------
.. autoclass:: WeightedTrajectory
    :members:

Functions
---------

.. autofunction:: integrate_langevin

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
import MDAnalysis as mda
import torch

# import openmm
from openmm import *
from openmm.app import *

# ### MD simulation

class WeightedTrajectory:
    r"""Class that stores trajectory data assoicated to an MDAnalysis Universe.

    Args:
        universe (:external+mdanalysis:class:`MDAnalysis.core.universe.Universe`): a MDAnalysis Universe that contains trajectory data
        weight_filename (str): filename of a CSV file that contains weights of trajectory 
        min_w (float): minimal value of weights below which the corresponding states will be discarded
        max_w (float): maximal value of weights above which the corresponding states will be discarded
        verbose (bool): print more information if ture


    Note:
        #. Weights are loaded from a CSV file if a filename is provided. The weights are normalized such that its mean value is one. Then, states in the trajectory data whose weights are not within [min_w, max_w] will be discarded, and weights of the remaining states are normalized again to have mean value one.
        #. All weights will be set to one, if weight_filename is None.



    Raises:
        ValueError: if trajectory information in the CSV file does not match that in the universe.
      
    Attributes:
        trajectory (3d numpy array): states in the trajectory, shape:
            :math:`[n, N, 3]`, where :math:`n` =n_frames is the number of states, 
            and :math:`N` is the number of atoms of the system
        start_time (float): time of the first state in the trajectory, unit: ps
        dt (float): timestep of the trajectory data, unit: ps
        n_frames (int): number of states in the trajectory
        weights (1d numpy array): weights of states


    """
    def __init__(self, universe, weight_filename=None, min_w=0.0, max_w=float("inf"), verbose=True):

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

# Generate MD trajectory data using OpenMM 
def integrate_langevin(pdb_filename, n_steps, sampling_temp, sampling_output_path, pre_steps=0, step_size=1.0 * unit.femtoseconds,
        frictionCoeff=1.0 / unit.picosecond,  traj_dcd_filename='traj.dcd', csv_filename='output.csv', report_interval=100,
        report_interval_stdout=100, forcefield=None, plumed_script=None):
    r"""Generate trajectory data by integrating Langevin dynamics using OpenMM.

    Args:
        pdb_filename (str): filename of PDB file
        n_steps (int): total number of steps to integrate
        sampling_temp (:external+openmm:class:`openmm.unit.quantity.Quantity`): temperature
            used to sample states, unit: kelvin
        sampling_output_path (str): directory to save results
        pre_steps (int): number of warm-up steps to run before integrating the system for n_steps
        step_size (:external+openmm:class:`openmm.unit.quantity.Quantity`): step-size to integrate Langevin dynamics, unit: fs
        frictionCoeff (:external+openmm:class:`openmm.unit.quantity.Quantity`): friction
            coefficient in Langevin dynamics, unit: :math:`(\text{ps})^{-1}`
        traj_dcd_filename (str): filename of the DCD file to save trajectory 
        csv_filename (str): filename of the CSV file to store statistics
        report_interval (int): how often to write trajectory to DCD file and to write statistics to CSV file
        report_interval_stdout (int): how often to print to stdout
        forcefield (:external+openmm:class:`openmm.app.forcefield.ForceField`): OpenMM force field used to calculate force
        plumed_script (str): script of PLUMED commands 

    This class encloses commands to simulate a Langevin dynamics using `OpenMM <https://openmm.org>`__ package.

    Note: 
       #. OpenMM ForceField('amber14-all.xml') will be used, if forcefield=None
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

    print (f'\nSampling temperature: {sampling_temp}')
    print ( 'Directory to save trajectory ouptuts: %s' % sampling_output_path)

    # add path to filenames
    traj_dcd_filename = os.path.join(sampling_output_path, traj_dcd_filename)
    csv_filename=os.path.join(sampling_output_path, csv_filename)

    # prepare before simulation
    pdb = PDBFile(pdb_filename)

    if forcefield is None :
        forcefield = ForceField('amber14-all.xml')

    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)

    if plumed_script is not None :
        from openmmplumed import PlumedForce
        print ('plumed script: %s' % plumed_script)
        system.addForce(PlumedForce(plumed_script))

    integrator = LangevinIntegrator(sampling_temp, frictionCoeff, step_size)

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

def calc_weights(sys_temp, sampling_temp, csv_filename, traj_weight_filename, energy_col_idx=1):
    r"""Calculate weights of the trajectory data.

    Args:
        sys_temp (:external+openmm:class:`openmm.unit.quantity.Quantity`): system's true temperature, unit: kelvin
        sampling_temp (:external+openmm:class:`openmm.unit.quantity.Quantity`): temperature
            used to sample states, unit: kelvin
        csv_filename (str): filename of a CSV file generated by :meth:`integrate_langevin`
        traj_weight_filename (str): filename to output the weights of trajectory
        energy_col_idx (int): the index of the column of the CSV file, based on which the weights will be calculated.

    Let the system's true temperature and the sampling temperature be :math:`T` and :math:`T_{sim}`, respectively.
    Define :math:`\beta=(k_B T)^{-1}` and :math:`\beta_{sim}=(k_B T_{sim})^{-1}`.
    This function uses certain column (specified by energy_col_idx) of the CSV file (specified by csv_filename) as energy, and computes the weights 
    simply by 

    .. math::
        v_i = \frac{1}{Z} \mathrm{e}^{-(\beta_{sys}-\beta_{sim}) V_i}\,, 

    where :math:`i=1,\dots, l` is the index of states, :math:`V_i` is the energy value, and :math:`Z` is the normalizing constant such that the mean value of the weights are one.

    Example:

        Below is an example of the CSV file generated by :meth:`calc_weights`: 

    .. code-block:: text
        :caption: weights.csv

        Time (ps),weight
        1.9999999999998903,0.8979287896010564
        2.9999999999997806,0.5424829921777613
        3.9999999999996714,0.02360908666276056
        5.000000000000004,0.03124009081426178
        6.000000000000338,0.4012103169413903
        7.000000000000672,0.617590719346622
        8.000000000001005,0.031154032337133607
        9.000000000000451,0.299688734996611
        9.999999999999895,0.03230412279837258
        ...

    """

    # use potential energy in the csv file generated by OpenMM code
    print (f'\n=============Calculate Weights============')
    print (f'\nSampling temperature: {sampling_temp}, system temperature: {sys_temp}')
    print (f'Reading potential from: {csv_filename}')
    vec = pd.read_csv(csv_filename)
    # modify the name of the first column
    vec.rename(columns={vec.columns[0]: 'Time (ps)'}, inplace=True)
    # show the data 
    print ('\nWhole data:\n', vec.head(8))

    # select the column containing energy used to calculate weights
    energy_col_name=vec.columns[energy_col_idx]
    print ('\nUse {:d}th column to reweight, name: {}'.format(energy_col_idx, energy_col_name) )

    sampling_beta = 1.0 / (sampling_temp * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
    sys_beta = 1.0 / (sys_temp * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
    energy_list = vec[energy_col_name]
    mean_energy = energy_list.mean()

    print (f'\nsampling beta={sampling_beta}, system beta={sys_beta}')

    # compute weights from potential energy
    nonnormalized_weights = [math.exp(-(sys_beta - sampling_beta) * (energy - mean_energy) * unit.kilojoule_per_mole) for energy in energy_list] 
    weights = pd.DataFrame(nonnormalized_weights / np.mean(nonnormalized_weights), columns=['weight'] )

    # insert time info
    time_col_idx = 0
    time_col_name=vec.columns[time_col_idx]
    weights.insert(0, time_col_name, vec[time_col_name])
    print ('\nWeight:\n', weights.head(8), '\n\nSummary of weights:\n', weights.describe())

    weights.to_csv(traj_weight_filename, index=False)
    print (f'weights saved to: {traj_weight_filename}')

