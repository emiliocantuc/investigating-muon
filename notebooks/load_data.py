import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from itertools import product
from collections import defaultdict
import os, torch


def format_group_str(s):
    s = s.split('/')[-1]
    o = s.split('_')
    o[1] = "(" + o[1]
    o[-1] = o[-1] + ")"
    o = ' '.join(o)
    o = o.replace('_', ' ').title()
    o = o.replace('(Full)', '')
    return o.strip()

def load_rs_results(d):
    df = []
    for run_name in os.listdir(d):
        results = torch.load(os.path.join(d, run_name, 'results.p'), weights_only = False, map_location='cpu')
        tmp = vars(results['args'])

        for series, label in {'main/val-accuracy':'val-acc', 'main/train-loss':'train-loss'}.items():
            for ix in [None, 5, 10]:
                r = results['train_stats'][series]
                i = ix if ix else len(r)
                r = r[:i]
                at = '-at-' + str(ix) if ix else ''
                tmp[f'last-{label}' + at] = r[-1]
                tmp[f'mean-{label}' + at] = np.mean(r)
                tmp[f'best-{label}' + at] = np.max(r)

        df.append(tmp)

    df = pd.DataFrame(df)
    df['betas'] = df['betas'].apply(lambda x: str(x[0]) + '-' + str(x[1]))
    df['group'] = df['group'].apply(format_group_str)
    df['group'] = df['group'].replace({'Ortho': 'Orth-SGDM', 'Sgd': 'SGDM'})
    return df

def load_detailed_results(d):
    records = []

    for run_name in os.listdir(d):

        results = torch.load(os.path.join(d, run_name, 'results.p'), weights_only = False, map_location = 'cpu')
        args = vars(results['args'])
        args['group'] = format_group_str(args['group'])

        train_stats = results['train_stats']

        n = len(next(iter(train_stats.values())))
        for step in range(n):

            record = {
                'run_name': run_name, 'step': step, **args  # include all experiment parameters
            }

            for stat_name, values in train_stats.items():
                # Fill with None if early stopped
                record[stat_name] = values[step] if step < len(values) else None
                if isinstance(record[stat_name], torch.Tensor):
                    record[stat_name] = float(record[stat_name])

            records.append(record)

    df = pd.DataFrame(records)
    df['betas'] = df['betas'].apply(lambda x: str(x[0]) + '-' + str(x[1]))
    df['group'] = df['group'].replace({'Ortho': 'Orth-SGDM', 'Sgd': 'SGDM'})
    return df