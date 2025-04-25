import os, subprocess, time
import numpy as np

def run_experiment(run_id, total_runs, args, save_dir):
    save_path = os.path.join(save_dir, str(run_id))
    cmd = ['python3', 'train.py'] + args + ['--save', save_path]
    print(f'Starting run {run_id} / {total_runs}: {" ".join(cmd)}')
    start = time.time()
    subprocess.run(cmd, check = True)
    print(f'Run {run_id} completed in {time.time() - start:.2f}s')


def build_args(params):
    # Convert dictionaries into CLI arguments
    args = []
    for k, v in {**params}.items():

        if isinstance(v, bool) or isinstance(v, np.bool_):  # For boolean flags
            if v: args.append(f"--{k}")
        
        elif isinstance(v, tuple) and len(v) == 2:          # For e.g. betas (0.9, 0.999)
            args.extend([f"--{k}", str(v[0]), str(v[1])])

        else: args.extend([f"--{k}", str(v)])               # Everything else
    return args

