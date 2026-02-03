

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""



"""

Import python packages

"""

import os, sys
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current working directory: {}'.format( os.getcwd() ))

from cascade2p import checks
checks.check_packages()

import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
import torch

"""

Define function to load dF/F traces from disk

"""


def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    # replace this with your own code if necessary
    # traces = np.load(file_path)

    # # here numpy dictionary with key 'dff'
#    traces = np.load(file_path, allow_pickle=True).item()['dff']

    # # In case your data is in another format:
    # traces = traces.T        # transpose, if loaded matrix has shape (time, neurons)
    # traces = traces / 100    # normalize to fractions, in case df/f is in Percent

    # traces should be 2d array with shape (neurons, nr_timepoints)

    traces = sio.loadmat(file_path)['dF_traces']

    # PLEASE NOTE: If you use mat73 to load large *.mat-file, be aware of potential numerical errors, see issue #67 (https://github.com/HelmchenLabSoftware/Cascade/issues/67)

    return traces



"""

Load dF/F traces, define frame rate and plot example traces

"""


example_file = 'Example_datasets/Allen-Brain-Observatory-Visual-Coding-30Hz/Experiment_552195520_excerpt.mat'
frame_rate = 30 # in Hz

traces = load_neurons_x_time( example_file )
print('Number of neurons in dataset:', traces.shape[0])
print('Number of timepoints in dataset:', traces.shape[1])


noise_levels = plot_noise_level_distribution(traces,frame_rate)


#np.random.seed(3952)
neuron_indices = np.random.randint(traces.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate)


"""

Load list of available models

"""

# cascade.download_model( 'update_models',verbose = 1)

# yaml_file = open('Pretrained_models/available_models.yaml')
# X = yaml.load(yaml_file)
# list_of_models = list(X.keys())

# for model in list_of_models:
#   print(model)




"""

Select pretrained model and apply to dF/F data

"""

model_name = 'Global_30Hz_25ms'
# cascade.download_model( model_name,verbose = 1)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spike_prob = cascade.predict( model_name, traces, device=device )




neuron_indices = np.random.randint(traces.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)


"""

Save predictions to disk

"""


folder = os.path.dirname(example_file)
save_path = os.path.join(folder, 'full_prediction_'+os.path.basename(example_file))

# save as numpy file
#np.save(save_path, spike_prob)
sio.savemat(save_path+'.mat', {'spike_prob':spike_prob})

