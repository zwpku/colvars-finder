"""Training Task --- :mod:`colvarsfinder.core.base_task`
=================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements a class that defines a feature of molecular system
(:class:`molann.feature.Feature`), and a class that constructs a list of
features from a feature file (:class:`molann.feature.FeatureFileReader`).

Classes
-------

.. autoclass:: TrainingTask
    :members:

"""

import cv2 as cv
import itertools 
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
import os
import configparser

from openmm import unit

class TrainingTask(object):
    """class for a training task
    """
    def __init__(self, 
                    traj_obj, 
                    pp_layer, 
                    learning_rate, 
                    model,
                    load_model_filename,
                    model_save_dir, 
                    save_model_every_step, 
                    model_path, 
                    k,
                    batch_size, 
                    num_epochs,
                    test_ratio, 
                    optimizer_name, 
                    device,
                    verbose):

        self.traj_obj = traj_obj
        self.preprocessing_layer = pp_layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.num_epochs= num_epochs
        self.test_ratio = test_ratio
        self.k = k
        self.model = model
        self.load_model_filename = load_model_filename 
        self.model_save_dir = model_save_dir
        self.save_model_every_step = save_model_every_step
        self.model_path = model_path
        self.optimizer_name = optimizer_name
        self.device = device
        self.verbose = verbose

        self.model_name = type(self).__name__

        if self.verbose: print ('\n[Info] Log directory: {}\n'.format(self.model_path), flush=True)

        self.writer = SummaryWriter(self.model_path)

    def init_model_and_optimizer(self):

        if self.load_model_filename and os.path.isfile(self.load_model_filename): 
            self.model.load_state_dict(torch.load(self.load_model_filename))
            if self.verbose: print (f'model parameters loaded from: {self.load_model_filename}')

        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, epoch):

        if self.verbose: print (f"\n\nEpoch={epoch}:") 

        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.model.state_dict(), trained_model_filename)  

        if self.verbose: print (f'  trained model saved at:\n\t{trained_model_filename}')

        cv = self.colvar_model()

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        if self.verbose: print (f'  script model for CVs saved at:\n\t{trained_cv_script_filename}\n', flush=True)

