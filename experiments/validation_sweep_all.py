"""Validation-set-size sweep (N_val = 2-32) for every dataset/property.

Runs the adaptive-MFML (globalpass) experiment with the first N_val points of
the shuffled 5 % validation split, 10 seeds per size, using the manuscript's
local/global tolerances (SI Table ST2).  Results:
``results/validation_sweep/<name>/val<N>/``.

N_val=32 results are copied from ``results/validation32``; the existing CH3Cl
2-16 results are moved into the per-dataset layout.

Usage:
    python experiments/validation_sweep_all.py --all --workers 4
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from saturatedmfml.dataset_paths import RESULTS_ROOT, RESULTS_VAL32, results_dir
from saturatedmfml.MFMLManager import MFMLExperimentManager
from experiments.validation32_rerun import SERIES, load_dataset, make_config

SWEEP_SERIES = [
    'ANI_globalpass',
    'QeMFi_SCF_globalpass', 'QeMFi_EV_globalpass',
    'VIB5_CH3Cl_globalpass', 'VIB5_CH3F_globalpass',
]
VAL_SIZES = [2, 4, 8, 16, 32]
SEEDS = list(range(10))
ROOT = RESULTS_ROOT / 'validation_sweep'


def out_dir(name, v):
    return results_dir(ROOT / name / f'val{v}')


def run_seed(series, v, seed):
    family, name, index_file = SERIES[series]
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_dataset(family, name)
    X_val, y_val = X_val[:v], y_val[:v]
    index = np.load(Path(__file__).resolve().parent.parent / index_file,
                    allow_pickle=True)
    index = np.copy(index[0][:, 0])
    config = make_config(family, name, seed)
    manager = MFMLExperimentManager(
        X_tr, y_tr.T, index, X_val, y_val[:, -1], X_te, y_te[:, -1], config)
    mae, valmae, sizes, diffs = manager.global_multipass_MFML_wrapper()
    prefix = out_dir(name, v) / f'{seed}_{name}_globalpass'
    np.save(f'{prefix}_mae.npy', np.asarray(mae))
    np.save(f'{prefix}_valmae.npy', np.asarray(valmae))
    np.save(f'{prefix}_trainsizes.npy', np.asarray(sizes))
    np.save(f'{prefix}_diffs.npy', np.asarray(diffs, dtype=float))
    return len(mae), float(np.asarray(mae)[-1])


def consolidate():
    """Move the old flat CH3Cl sweep into ROOT/CH3Cl/val<N>; copy val32."""
    for v in VAL_SIZES:
        src = ROOT / f'val{v}'
        if not src.is_dir():
            continue
        dst = out_dir('CH3Cl', v)
        for f in src.glob('*.npy'):
            shutil.move(str(f), dst / f.name)
        try:
            src.rmdir()
        except OSError:
            pass
    for series in SWEEP_SERIES:
        _, name, _ = SERIES[series]
        dst = out_dir(name, 32)
        for src in RESULTS_VAL32.glob(f'*_{name}_globalpass_*.npy'):
            target = dst / src.name
            if not target.exists():
                shutil.copy2(src, target)


def run_task(series, v):
    _, name, _ = SERIES[series]
    if v == 32:
        return
    for seed in SEEDS:
        if (ROOT / name / f'val{v}' / f'{seed}_{name}_globalpass_mae.npy').exists():
            continue
        n, mae = run_seed(series, v, seed)
        print(f'{name} val{v} seed {seed}: {n} points, final {mae:.4g}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--run-task', nargs=2, metavar=('SERIES', 'VAL'))
    args = parser.parse_args()
    if args.run_task:
        run_task(args.run_task[0], int(args.run_task[1]))
        return
    if not args.all:
        parser.error('use --all or --run-task')
    consolidate()
    tasks = [(s, v) for s in SWEEP_SERIES for v in VAL_SIZES if v != 32]
    # longest first (ANI), then QeMFi, then VIB5
    log_dir = results_dir(ROOT / 'logs')
    env = 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 '
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for series, v in tasks:
            cmd = (env + sys.executable + ' ' + str(Path(__file__).resolve())
                   + f' --run-task {series} {v}')
            log = open(log_dir / f'{series}_val{v}.log', 'w')
            futures.append((series, v, pool.submit(subprocess.run, cmd, shell=True,
                                                   stdout=log, stderr=subprocess.STDOUT), log))
        for series, v, fut, log in futures:
            fut.result(); log.close(); print('finished', series, v, flush=True)


if __name__ == '__main__':
    main()
