"""Two-dimensional local/global tolerance sweep at N_val = 2.

Local and global tolerances each take values {0.01, 0.1, 1, 2, 4}; every
dataset/property is run for all 25 combinations, 3 seeds each, with the
validation set fixed to 2 points.  Results:
``results/tolerance_grid/<name>/local<l>_global<g>/<seed>.npz``.

Usage:
    python experiments/tolerance_grid_sweep.py --all --workers 4
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_ROOT, results_dir
from saturatedmfml.experiment_params import get_hyperparams
from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager
from experiments.validation32_rerun import SERIES, load_dataset

LOCAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
GLOBAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
VAL_SIZE = 2
SEEDS = [0]   # single-seed grid; non-ANI uses seed 0
ROOT = RESULTS_ROOT / 'tolerance_grid'

SERIES_LIST = [
    'ANI_globalpass',
    'QeMFi_SCF_globalpass', 'QeMFi_EV_globalpass',
    'VIB5_CH3Cl_globalpass', 'VIB5_CH3F_globalpass',
]

BASE = {
    ('VIB5', 'CH3Cl'): dict(window=10, batch_size=[32, 8, 4, 2]),
    ('VIB5', 'CH3F'): dict(window=10, batch_size=[32, 8, 4, 2]),
    ('QeMFi', 'SCF'): dict(window=5, batch_size=[64, 16, 8, 2]),
    ('QeMFi', 'EV'): dict(window=5, batch_size=[32, 4, 2, 2]),
    ('ANI', 'ANI'): dict(window=5, batch_size=[32, 8, 4, 2]),
}


def tag(v):
    return f'{v:g}'


def run_combo(series, local_tol, global_tol, seed):
    family, name, index_file = SERIES[series]
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset(family, name)
    X_val, y_val = X_val[:VAL_SIZE], y_val[:VAL_SIZE]
    index = np.load(REPO_ROOT / index_file, allow_pickle=True)
    index = np.copy(index[0][:, 0])
    reg, sigma = get_hyperparams(family, name)
    base = BASE[(family, name)]
    config = MFMLConfig(
        reg=reg, sigma=sigma, scale=2, navg=1, window=base['window'],
        maxiter=100, max_passes=20,
        local_tol=[local_tol] * 4, global_tol=[global_tol] * 4,
        batch_size=base['batch_size'], initial_size=[32, 16, 8, 4],
        seed=seed, relative=(family == 'ANI'))
    manager = MFMLExperimentManager(
        X_tr, y_tr.T, index, X_val, y_val[:, -1], X_te, y_te[:, -1], config)
    mae, valmae, sizes, diffs = manager.global_multipass_MFML_wrapper()
    out = results_dir(ROOT / name / f'local{tag(local_tol)}_global{tag(global_tol)}')
    prefix = out / f'{seed}'
    np.save(f'{prefix}_mae.npy', np.asarray(mae))
    np.save(f'{prefix}_valmae.npy', np.asarray(valmae))
    np.save(f'{prefix}_trainsizes.npy', np.asarray(sizes))
    np.save(f'{prefix}_diffs.npy', np.asarray(diffs, dtype=float))
    return len(mae), float(np.asarray(mae)[-1])


def run_task(series, local_tol):
    _, name, _ = SERIES[series]
    for global_tol in GLOBAL_TOLS:
        d = ROOT / name / f'local{tag(local_tol)}_global{tag(global_tol)}'
        for seed in SEEDS:
            if (d / f'{seed}_mae.npy').exists():
                continue
            n, mae = run_combo(series, local_tol, global_tol, seed)
            print(f'{name} l={tag(local_tol)} g={tag(global_tol)} seed {seed}: '
                  f'{n} points, final {mae:.4g}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--run-task', nargs=2, metavar=('SERIES', 'LOCAL'))
    args = parser.parse_args()
    if args.run_task:
        run_task(args.run_task[0], float(args.run_task[1]))
        return
    if not args.all:
        parser.error('use --all or --run-task')
    tasks = [(s, l) for s in SERIES_LIST for l in LOCAL_TOLS]
    log_dir = results_dir(ROOT / 'logs')
    env = 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 '
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for series, l in tasks:
            cmd = (env + sys.executable + ' ' + str(Path(__file__).resolve())
                   + f' --run-task {series} {tag(l)}')
            log = open(log_dir / f'{series}_local{tag(l)}.log', 'w')
            futures.append((series, l, pool.submit(subprocess.run, cmd, shell=True,
                                                   stdout=log, stderr=subprocess.STDOUT), log))
        for series, l, fut, log in futures:
            fut.result(); log.close(); print('finished', series, l, flush=True)


if __name__ == '__main__':
    main()
