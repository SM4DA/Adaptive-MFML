"""Re-run the adaptive-MFML experiments with a 32-point validation set.

The validation set used for the stopping decision costs high-fidelity reference
labels.  This script repeats the main adaptive experiments with only the first
32 points of the shuffled 5 % validation split, keeping everything else (fixed
85:5:10 split, seeds, local/global tolerances, batch sizes) unchanged.

It also regenerates the QeMFi single-fidelity and basic-MFML curves, because
the QeMFi fidelity fix (fidelities 1-4, target def2-TZVP) changes their target
even though they do not use a validation set.  VIB5 and QeMFi report absolute
MAE; ANI reports relative MAE (RMAE), matching the manuscript figures.

Results are written to ``results/validation32`` using the same file naming as
the original averaged runs::

    {seed}_{name}_{strategy}_{mae,valmae,trainsizes,diffs}.npy

Usage:
    python experiments/validation32_rerun.py --series VIB5_CH3Cl_globalpass
    python experiments/validation32_rerun.py --all --workers 4
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from saturatedmfml.dataloaders import (
    partition_ANI_data, partition_QeMFi_data, partition_VIB5_data,
)
from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_VAL32, results_dir
from saturatedmfml.experiment_params import get_hyperparams
from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager

VAL_SIZE = 32
SEEDS = list(range(10))

# name -> (dataset family, property/molecule, index file)
SERIES = {
    'VIB5_CH3Cl_globalpass': ('VIB5', 'CH3Cl', 'CH3Cl_indexes.npy'),
    'VIB5_CH3Cl_cascading': ('VIB5', 'CH3Cl', 'CH3Cl_indexes.npy'),
    'VIB5_CH3F_globalpass': ('VIB5', 'CH3F', 'CH3F_indexes.npy'),
    'VIB5_CH3F_cascading': ('VIB5', 'CH3F', 'CH3F_indexes.npy'),
    'QeMFi_SCF_globalpass': ('QeMFi', 'SCF', 'SCF_indexes.npy'),
    'QeMFi_SCF_cascading': ('QeMFi', 'SCF', 'SCF_indexes.npy'),
    'QeMFi_EV_globalpass': ('QeMFi', 'EV', 'EV_indexes.npy'),
    'QeMFi_EV_cascading': ('QeMFi', 'EV', 'EV_indexes.npy'),
    'ANI_globalpass': ('ANI', 'ANI', 'ANI_indexes.npy'),
    'ANI_cascading': ('ANI', 'ANI', 'ANI_indexes.npy'),
}

# Local/global tolerances and sizes used in the main text (SI Table ST2).
CONFIGS = {
    ('VIB5', 'CH3Cl'): dict(local_tol=[1e-2] * 4, global_tol=[0.1] * 4,
                            window=10, batch_size=[32, 8, 4, 2]),
    ('VIB5', 'CH3F'): dict(local_tol=[1e-3] * 4, global_tol=[0.1] * 4,
                           window=10, batch_size=[32, 8, 4, 2]),
    ('QeMFi', 'SCF'): dict(local_tol=[1, 1, 2, 2], global_tol=[0.5] * 4,
                           window=5, batch_size=[64, 16, 8, 2]),
    ('QeMFi', 'EV'): dict(local_tol=[1e-1] * 4, global_tol=[0.1] * 4,
                          window=5, batch_size=[32, 4, 2, 2]),
    ('ANI', 'ANI'): dict(local_tol=[1, 1, 1, 1], global_tol=[1] * 4,
                         window=5, batch_size=[32, 8, 4, 2]),
}


def load_dataset(family, name):
    if family == 'VIB5':
        return partition_VIB5_data(molname=name, seed=42, center=True)
    if family == 'QeMFi':
        return partition_QeMFi_data(prop=name, seed=42, center=True)
    return partition_ANI_data(seed=42, center=True)


def make_config(family, name, seed):
    reg, sigma = get_hyperparams(family, name)
    settings = CONFIGS[(family, name)]
    return MFMLConfig(
        reg=reg, sigma=sigma, scale=2, navg=1,
        maxiter=100, max_passes=20, initial_size=[32, 16, 8, 4],
        seed=seed, relative=(family == 'ANI'), **settings)


def run_series(series, seeds=SEEDS, val_size=VAL_SIZE):
    family, name, index_file = SERIES[series]
    strategy = series.rsplit('_', 1)[-1]
    out_dir = results_dir(RESULTS_VAL32)

    todo = [s for s in seeds if not (out_dir / f'{s}_{name}_{strategy}_mae.npy').exists()]
    if not todo:
        print(f'[{series}] all seeds present, skipping')
        return

    print(f'[{series}] loading data')
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset(family, name)
    index = np.load(REPO_ROOT / index_file, allow_pickle=True)
    index = np.copy(index[0][:, 0])
    X_val, y_val = X_val[:val_size], y_val[:val_size]
    print(f'[{series}] X_train={X_tr.shape}, X_val={X_val.shape}', flush=True)

    for seed in todo:
        config = make_config(family, name, seed)
        manager = MFMLExperimentManager(
            X_tr, y_tr.T, index, X_val, y_val[:, -1], X_te, y_te[:, -1], config)
        if strategy == 'globalpass':
            mae, valmae, sizes, diffs = manager.global_multipass_MFML_wrapper()
        elif strategy == 'cascading':
            mae, valmae, sizes, diffs = manager.see_saw_saturation_MFML()
        else:
            raise ValueError(f'unknown strategy {strategy}')
        prefix = out_dir / f'{seed}_{name}_{strategy}'
        np.save(f'{prefix}_mae.npy', np.asarray(mae))
        np.save(f'{prefix}_valmae.npy', np.asarray(valmae))
        np.save(f'{prefix}_trainsizes.npy', np.asarray(sizes))
        np.save(f'{prefix}_diffs.npy', np.asarray(diffs, dtype=float))
        print(f'[{series}] seed {seed}: {len(mae)} points, '
              f'final MAE {np.asarray(mae)[-1]:.4f}', flush=True)


def run_qemfi_baselines(navg=10):
    """Regenerate QeMFi SF and basic-MFML curves for the corrected target."""
    out_dir = results_dir(RESULTS_VAL32)
    for prop in ['SCF', 'EV']:
        sf_path = out_dir / f'{prop}_sf_mae.npy'
        basic_path = out_dir / f'{prop}_2_basic_mfml.npy'
        if sf_path.exists() and basic_path.exists():
            print(f'[QeMFi {prop} baselines] present, skipping')
            continue
        X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset('QeMFi', prop)
        index = np.load(REPO_ROOT / f'{prop}_indexes.npy', allow_pickle=True)
        index = np.copy(index[0][:, 0])
        reg, sigma = get_hyperparams('QeMFi', prop)
        config = MFMLConfig(
            reg=reg, sigma=sigma, scale=2, navg=navg, window=5, maxiter=100,
            max_passes=20, nmax_sf=11, nmax_basic=9,
            local_tol=CONFIGS[('QeMFi', prop)]['local_tol'],
            global_tol=CONFIGS[('QeMFi', prop)]['global_tol'],
            batch_size=CONFIGS[('QeMFi', prop)]['batch_size'],
            initial_size=[32, 16, 8, 4], seed=0, relative=False)
        manager = MFMLExperimentManager(
            X_tr, y_tr.T, index, X_val[:VAL_SIZE], y_val[:VAL_SIZE, -1],
            X_te, y_te[:, -1], config)
        print(f'[QeMFi {prop}] single fidelity', flush=True)
        np.save(sf_path, manager.sf_LC())
        print(f'[QeMFi {prop}] basic MFML', flush=True)
        basic_mae, basic_sizes = manager.basic_MFML()
        np.save(basic_path, basic_mae)
        np.save(out_dir / f'{prop}_2_basic_mfml_sizes.npy', basic_sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', choices=sorted(SERIES))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--baselines', action='store_true',
                        help='also regenerate the QeMFi SF/basic curves')
    args = parser.parse_args()

    if args.all:
        log_dir = results_dir(RESULTS_VAL32 / 'logs')
        env_prefix = ('OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ')
        jobs = [['--series', s, '--seeds'] + [str(x) for x in args.seeds]
                for s in SERIES]
        if args.baselines:
            jobs.append(['--baselines'])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for job in jobs:
                cmd = (env_prefix + sys.executable + ' '
                       + str(Path(__file__).resolve()) + ' ' + ' '.join(job))
                tag = job[1] if job[0] == '--series' else 'qemfi_baselines'
                log = open(log_dir / f'{tag}.log', 'w')
                futures.append((tag, pool.submit(subprocess.run, cmd, shell=True,
                                                 stdout=log, stderr=subprocess.STDOUT), log))
            for tag, fut, log in futures:
                fut.result()
                log.close()
                print('finished', tag, flush=True)
        return

    if args.baselines:
        run_qemfi_baselines()
    if args.series:
        run_series(args.series, seeds=args.seeds)
    if not (args.series or args.baselines):
        parser.error('choose --series, --baselines or --all')


if __name__ == '__main__':
    main()
