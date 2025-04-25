# Script to run grid over learning rates on MNIST using MLPs
# Totals 360 runs

import os, torch
from itertools import product
import numpy as np

from utils import run_experiment, build_args

# Constants
GROUP = 'mnist_lr_hidden'
SAVE_DIR = f'./results/{GROUP}'
os.makedirs(SAVE_DIR, exist_ok = True)

# Max random search budget
# NOTE: if adding more onto an existing group run, change the random seed!
N_RUNS_PER_OPT = 15
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
    'log_optimizer_stats': False,
    'log_freq': 0.4, # log detailed stats every % of epoch
}

_lr = np.logspace(-4, 1, N_RUNS_PER_OPT)
print(_lr)

changing_params = {

    'sgd_lr':{
        'lr': _lr,

        'optim': ['sgd'] * N_RUNS_PER_OPT,
    },

    'adam_lr':{
        'lr': _lr,

        'optim': ['adam'] * N_RUNS_PER_OPT,
    },

    'muon_one_lr':{
        'lr': _lr,
        'muon_lr': _lr,

        'optim': ['muon'] * N_RUNS_PER_OPT,
    },

    'muon_no_momentum_one_lr':{
        'lr': _lr,
        'muon_lr': _lr,
        'muon_momentum': [0.0] * N_RUNS_PER_OPT,

        'optim': ['muon'] * N_RUNS_PER_OPT,
    },

    'ortho_one_lr':{
        'lr': _lr,
        'muon_lr': _lr,

        'optim': ['ortho'] * N_RUNS_PER_OPT,
    },
}

# Build experiment list
experiments = []

# TODO check if hidden dim makes sense
for depth, hidden_dim, init_type, seed in product([2], [64, 256, 1024], ['kaiming'], range(5)):

    common_params['depth'] = depth
    common_params['hidden_dim'] = hidden_dim
    common_params['init_type'] = init_type
    common_params['seed'] = seed

    for name, param_dict in changing_params.items():

        param_dict['group'] = [os.path.join(GROUP, name)] * N_RUNS_PER_OPT
        params = [dict(zip(param_dict.keys(), values)) for values in zip(*param_dict.values())]

        for param in params:
            args = build_args({**common_params, **param})
            experiments.append(args)


# Summary and execution
n_existing = len(os.listdir(SAVE_DIR))
print(f'Total experiments {len(experiments)} \tExisting runs: {n_existing}')

# for run_id, args in enumerate(experiments, start = n_existing + 1):
#     run_experiment(run_id, len(experiments), args, SAVE_DIR)
