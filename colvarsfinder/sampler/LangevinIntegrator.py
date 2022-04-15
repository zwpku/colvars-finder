#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Generate MD trajectory data using OpenMM 

# +
import math
from random import random, randint
import numpy as np
import time
import datetime
import os, sys
import warnings
from sys import stdout
import matplotlib.pyplot as plt
import configparser
import pandas as pd

# import openmm
from openmm import *
from openmm.app import *
from openmmplumed import PlumedForce
# -

# ### set parameters

config = configparser.ConfigParser()
config.read('params.cfg')
pdb_filename = config['System']['pdb_filename']
n_steps = config['Sampling'].getint('n_steps')
pre_steps = config['Sampling'].getint('pre_steps')
step_size = config['Sampling'].getfloat('step_size') * unit.femtoseconds
frictionCoeff = config['Sampling'].getfloat('friction') / unit.picosecond
sampling_temp = config['Sampling'].getfloat('sampling_temperature') * unit.kelvin

sampling_path = config['Sampling'].get('sampling_path')
if not os.path.exists(sampling_path):
    os.makedirs(sampling_path)

traj_dcd_filename = config['Sampling']['traj_dcd_filename']
csv_filename = config['Sampling']['csv_filename']

# add path to filenames
traj_dcd_filename = os.path.join(sampling_path, traj_dcd_filename)
csv_filename=os.path.join(sampling_path, csv_filename)

report_interval_dcd = config['Sampling'].getint('report_interval_dcd')
report_interval_stdout = config['Sampling'].getint('report_interval_stdout')
report_interval_csv = config['Sampling'].getint('report_interval_csv')
use_plumed_script = config['Sampling'].getboolean('use_plumed_script')
plumed_script = config['Sampling'].get('plumed_script')


# ### MD simulation

# +

print (f'\nSampling temperature: {sampling_temp}')
print ( 'Directory to save trajectory ouptuts: %s' % sampling_path)

# prepare before simulation
pdb = PDBFile(pdb_filename)
forcefield = ForceField('amber14-all.xml')

system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)

if use_plumed_script :
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

simulation.reporters.append(DCDReporter(traj_dcd_filename, report_interval_dcd))
simulation.reporters.append(StateDataReporter(stdout, report_interval_stdout, step=True,
                                              temperature=True, elapsedTime=True))
simulation.reporters.append(StateDataReporter(csv_filename, report_interval_csv, time=True,
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

# -

