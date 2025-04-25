# Script to run random search experiments on MNIST
# Totals 3,000 runs

import os, torch
from itertools import product
import numpy as np

from utils import run_experiment, build_args

# Constants
GROUP = 'mnist_rs'
SAVE_DIR = f'./results/{GROUP}'
os.makedirs(SAVE_DIR, exist_ok = True)

# Max random search budget
# NOTE: if adding more onto an existing group run, change the random seed!
N_RUNS_PER_OPT = 50 
np.random.seed(0)

# Base experiment settings (shared across all runs)
common_params = {

    'task': 'mnist',

    'model': 'mlp',
    'weight_decay': 0.0,
    'newtonschulz': torch.cuda.is_available(),
    
    'bn': True,
    'batch_size': 256,

    'epochs': 20,
    'patience': 3, # early stopping patience (-1 for no early stopping)

    'wandb': False, # adds about 2s to run time
    'log_optimizer_stats': True,
    'log_freq': 0.4, # log detailed stats every % of epoch
}

_lr = np.random.lognormal(-2.69, 1.42, size = N_RUNS_PER_OPT)
_muon_lr = np.random.lognormal(-2.69, 1.42, size = N_RUNS_PER_OPT)

_momentum = np.random.uniform(0.8, 0.99, size = N_RUNS_PER_OPT)
_nesterov = np.random.randint(0, 2, size = N_RUNS_PER_OPT).astype(bool)

changing_params = {

    'sgd_full':{
        'lr': _lr,
        'muon_momentum': _momentum,

        'optim': ['sgd'] * N_RUNS_PER_OPT,
    },

    'adam_full':{
        'lr': _lr,
        'betas': list(zip(
            1 - np.exp(np.random.uniform(-5, -1, size = N_RUNS_PER_OPT)),
            1 - np.exp(np.random.uniform(-5, -1, size = N_RUNS_PER_OPT)))),
        'eps': np.exp(np.random.uniform(-8, 0, size = N_RUNS_PER_OPT)),

        'optim': ['adam'] * N_RUNS_PER_OPT,
    },

    'muon_full':{
        'lr': _lr,
        'muon_lr': _muon_lr,
        'muon_momentum': _momentum,
        'muon_nesterov': _nesterov,

        'optim': ['muon'] * N_RUNS_PER_OPT,
    },

    'muon_no_momentum':{
        'lr': _lr,
        'muon_lr': _muon_lr,
        'muon_momentum': [0.0] * N_RUNS_PER_OPT,
        'muon_nesterov': _nesterov,

        'optim': ['muon'] * N_RUNS_PER_OPT,
    },

    'ortho_full':{
        'lr': _lr,
        'muon_lr': _muon_lr,
        'muon_momentum': _momentum,
        'muon_nesterov': _nesterov,

        'optim': ['ortho'] * N_RUNS_PER_OPT,
    },
}

# Build experiment list
experiments = []

for depth, hidden_dim, init_type in product([1, 2, 3], [64, 128], ['kaiming', 'orthogonal']):

    common_params['hidden_dim'] = hidden_dim
    common_params['depth'] = depth
    common_params['init_type'] = init_type

    for name, param_dict in changing_params.items():

        param_dict['group'] = [os.path.join(GROUP, name)] * N_RUNS_PER_OPT
        params = [dict(zip(param_dict.keys(), values)) for values in zip(*param_dict.values())]

        for param in params:
            args = build_args({**common_params, **param})
            experiments.append(args)


# Summary and execution
n_existing = len(os.listdir(SAVE_DIR))
print(f'Total experiments {len(experiments)} \tExisting runs: {n_existing}')

for run_id, args in enumerate(experiments, start = n_existing + 1):
    run_experiment(run_id, len(experiments), args, SAVE_DIR)
