"""Local-tolerance / seed robustness sweep for the adaptive-MFML stopping rule.

Reviewer 2, second round (R2.7b): include the local tolerance used in the main
experiments (0.01) in the tolerance sweep, repeat across independent seeds and
representative datasets, and report the variability of the prediction error and
of the cost required to reach a fixed accuracy.

Each run is a full global multi-pass adaptive-MFML run. Results are stored per
(dataset, molecule, tolerance, seed) under ``results/tolerance_robustness`` and
the figure is written to ``Manuscript/plots/tolerance_robustness_*.pdf``.

Usage:
    python experiments/tolerance_robustness.py            # run + plot
    python experiments/tolerance_robustness.py --plot-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from saturatedmfml.dataloaders import partition_VIB5_data
from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_ROBUSTNESS, results_dir
from saturatedmfml.experiment_params import get_hyperparams
from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager

# Same per-fidelity time costs and plotting convention as the original
# threshold / validation-size figures (Plots.ipynb).
TIME_COST = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0  # hours

DATASETS = [('CH3Cl', 'CH3Cl_indexes.npy'), ('CH3F', 'CH3F_indexes.npy')]
LOCAL_TOLS = [1e-2, 1e-1, 1.0, 5.0]
SEEDS = [0, 1, 2]
TARGET_MAE = 2.0  # kcal/mol, the target used for the VIB5 curves in the paper


def run_one(molecule, index_file, tau, seed, maxiter=100, max_passes=20,
            val_size=None):
    X_tr, X_val, X_te, y_tr, y_val, y_te = partition_VIB5_data(
        molname=molecule, seed=42, center=True)
    if val_size is not None:
        X_val, y_val = X_val[:val_size], y_val[:val_size]
    index = np.load(REPO_ROOT / index_file, allow_pickle=True)
    index = np.copy(index[0][:, 0])

    reg, sigma = get_hyperparams('VIB5', molecule)
    config = MFMLConfig(
        reg=reg, sigma=sigma, scale=2, navg=1, window=10,
        local_tol=[tau] * 4, global_tol=[0.1] * 4,
        maxiter=maxiter, max_passes=max_passes,
        batch_size=[32, 8, 4, 2], initial_size=[32, 16, 8, 4],
        seed=seed, relative=False,
    )
    manager = MFMLExperimentManager(
        X_tr, y_tr.T, index, X_val, y_val[:, -1], X_te, y_te[:, -1], config)
    mae, valmae, sizes, diffs = manager.global_multipass_MFML_wrapper()
    return {
        'mae': np.asarray(mae), 'valmae': np.asarray(valmae),
        'trainsizes': np.asarray(sizes), 'diffs': np.asarray(diffs, dtype=float),
        'molecule': molecule, 'tau': tau, 'seed': seed,
    }


def save_run(out_dir, result):
    name = f"{result['molecule']}_tau{result['tau']:g}_seed{result['seed']}.npz"
    np.savez(out_dir / name, **{k: v for k, v in result.items()})
    return out_dir / name


def load_runs(out_dir):
    runs = {}
    for path in sorted(out_dir.glob('*.npz')):
        data = np.load(path, allow_pickle=True)
        mol, tau, seed = str(data['molecule']), float(data['tau']), int(data['seed'])
        runs.setdefault((mol, tau), []).append({
            'mae': data['mae'], 'trainsizes': data['trainsizes'], 'seed': seed})
    return runs


def costs(trainsizes):
    return np.asarray([TIME_COST @ sizes for sizes in trainsizes])


def cost_at_target(cost, mae, target=TARGET_MAE):
    """First cost at which the MAE curve drops below ``target`` (log interpolation)."""
    below = np.where(mae <= target)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return cost[0]
    x0, x1 = np.log10(cost[i - 1]), np.log10(cost[i])
    y0, y1 = mae[i - 1], mae[i]
    frac = (y0 - target) / (y0 - y1) if y0 != y1 else 1.0
    return 10 ** (x0 + frac * (x1 - x0))


def mean_curve(run_list, n_grid=100):
    """Interpolate the seed curves onto a common log-cost grid and average."""
    lo = max(costs(r['trainsizes']).min() for r in run_list)
    hi = min(costs(r['trainsizes']).max() for r in run_list)
    grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)
    curves = []
    for r in run_list:
        c = costs(r['trainsizes'])
        lg = np.log10(c)
        order = np.argsort(lg)
        curves.append(np.interp(np.log10(grid), lg[order], r['mae'][order]))
    arr = np.vstack(curves)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def make_figures(out_dir, plot_dir=REPO_ROOT / 'Manuscript' / 'plots', suffix=''):
    runs = load_runs(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(LOCAL_TOLS)))

    for molecule, _ in DATASETS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        fmae_mean, fmae_std, ctt_mean, ctt_std = [], [], [], []

        for color, tau in zip(colors, LOCAL_TOLS):
            run_list = runs.get((molecule, tau), [])
            if not run_list:
                continue
            grid, mean, std = mean_curve(run_list)
            ax1.loglog(grid, mean, color=color, label=fr'$\tau={tau:g}$')
            ax1.fill_between(grid, np.maximum(mean - std, 1e-6), mean + std,
                             color=color, alpha=0.2, linewidth=0)

            finals = np.asarray([r['mae'][-1] for r in run_list])
            ctt = np.asarray([cost_at_target(costs(r['trainsizes']), r['mae'])
                              for r in run_list])
            fmae_mean.append(finals.mean())
            fmae_std.append(finals.std())
            ctt_mean.append(np.nanmean(ctt))
            ctt_std.append(np.nanstd(ctt))

        ax1.axhline(TARGET_MAE, color='black', linestyle='--', linewidth=1.0)
        ax1.set_xlabel(r'$T_{\mathrm{train}}$ [hrs]', fontsize=12)
        ax1.set_ylabel('MAE (CCSD-T) [kcal/mol]', fontsize=12)
        ax1.set_title(f'{molecule}: learning curves (mean $\\pm$ std, {len(SEEDS)} seeds)',
                      fontsize=11)
        ax1.grid(True, which='both', ls='--', alpha=0.5)
        ax1.legend(fontsize=9, ncol=2)

        x = np.arange(len(LOCAL_TOLS))
        ax2.errorbar(x, fmae_mean, yerr=fmae_std, fmt='o-', color='tab:blue',
                     capsize=3, label='final MAE')
        ax2.axhline(TARGET_MAE, color='black', linestyle='--', linewidth=1.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels([fr'$10^{{{int(np.log10(t))}}}$' if t < 1 else fr'${t:g}$'
                             for t in LOCAL_TOLS])
        ax2.set_xlabel(r'local tolerance $\tau$', fontsize=12)
        ax2.set_ylabel('final test MAE [kcal/mol]', fontsize=12, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.grid(True, which='major', ls='--', alpha=0.5)

        ax3 = ax2.twinx()
        ax3.errorbar(x, ctt_mean, yerr=ctt_std, fmt='s--', color='tab:red',
                     capsize=3, label=f'cost to MAE={TARGET_MAE:g}')
        ax3.set_ylabel(f'cost to reach {TARGET_MAE:g} kcal/mol [hrs]',
                       fontsize=12, color='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:red')

        fig.suptitle(f'Robustness of adaptive-MFML to the local tolerance ({molecule})',
                     fontsize=12)
        fig.tight_layout()
        out_pdf = plot_dir / f'tolerance_robustness{suffix}_{molecule}.pdf'
        out_png = out_dir / f'tolerance_robustness{suffix}_{molecule}.png'
        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print('wrote', out_pdf, 'and', out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--taus', type=float, nargs='+', default=LOCAL_TOLS)
    parser.add_argument('--molecules', nargs='+', default=[m for m, _ in DATASETS])
    parser.add_argument('--val-size', type=int, default=None,
                        help='use only the first N validation points (default: full 5 %%)')
    args = parser.parse_args()

    suffix = '' if args.val_size is None else f'_val{args.val_size}'
    out_dir = results_dir(RESULTS_ROBUSTNESS / suffix.lstrip('_')
                          if suffix else RESULTS_ROBUSTNESS)
    if not args.plot_only:
        index_files = dict(DATASETS)
        for molecule in args.molecules:
            for tau in args.taus:
                for seed in args.seeds:
                    path = out_dir / f'{molecule}_tau{tau:g}_seed{seed}.npz'
                    if path.exists():
                        print('skip existing', path.name)
                        continue
                    print(f'--- {molecule} tau={tau:g} seed={seed}')
                    result = run_one(molecule, index_files[molecule], tau, seed,
                                     val_size=args.val_size)
                    save_run(out_dir, result)
                    print(f"    final MAE {result['mae'][-1]:.4f} kcal/mol, "
                          f"{len(result['mae'])} epochs")
    make_figures(out_dir, suffix=suffix)


if __name__ == '__main__':
    main()
