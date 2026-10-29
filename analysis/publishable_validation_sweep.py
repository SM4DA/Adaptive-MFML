"""Cosmetic versions of the validation-size sweep figures.

No new experiments: re-plots the existing results with the labels/sizes
requested for the SI.  Writes to Manuscript/publishable_plots/ and does not
touch the existing report PDFs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from saturatedmfml.dataset_paths import (
    REPO_ROOT, RESULTS_ARCHIVE, RESULTS_AVERAGED, RESULTS_SINGLE_PASS, RESULTS_VAL32,
)
from analysis.validation_sweep_all import (
    ANI_COST, QEMFI5, QEMFI_COST, VIB5_COST, VAL_SIZES, mean_curve, panel,
)

# Legacy runs already used the full 5 % validation set; only their time axis
# needs the validation-label cost added.
FULL_SPECS = {
    'CH3Cl': (RESULTS_ARCHIVE / 'avg_prereview', 2241),
    'CH3F': (RESULTS_ARCHIVE / 'avg_prereview', 4133),
    'SCF': (RESULTS_ARCHIVE / 'avg_prereview', 6750),
    'EV': (RESULTS_ARCHIVE / 'avg_prereview', 6750),
    'ANI': (RESULTS_AVERAGED, 2500),
}

OUT = REPO_ROOT / 'Manuscript' / 'publishable_plots'

GROUPS = [
    ('VIB5', [
        ('CH3Cl', VIB5_COST, VIB5_COST[-1], 'MAE (TBE) [kcal/mol]', 2.0,
         RESULTS_SINGLE_PASS),
        ('CH3F', VIB5_COST, VIB5_COST[-1], 'MAE (TBE) [kcal/mol]', 2.0,
         RESULTS_SINGLE_PASS),
    ]),
    ('QeMFi', [
        ('SCF', QEMFI_COST, QEMFI5[4], 'MAE [kcal/mol]', None, RESULTS_VAL32),
        ('EV', QEMFI_COST, QEMFI5[4], 'MAE [kcal/mol]', None, RESULTS_VAL32),
    ]),
    ('ANI', [
        ('ANI', ANI_COST, ANI_COST[-1], r'RMAE (DLPNO-CCSD(T)$^{*}$/CBS)',
         4.5e-3, RESULTS_SINGLE_PASS),
    ]),
]


def full_runs(name, cost_vector, val_cost):
    root, n_val = FULL_SPECS[name]
    runs = []
    for seed in range(10):
        p = root / f'{seed}_{name}_globalpass'
        mae_p = p.with_name(p.name + '_mae.npy')
        size_p = p.with_name(p.name + '_trainsizes.npy')
        if not (mae_p.exists() and size_p.exists()):
            continue
        mae = np.load(mae_p, allow_pickle=True).astype(float)
        sizes = np.load(size_p, allow_pickle=True).astype(int)
        cost = np.asarray([cost_vector @ s for s in sizes]) + n_val * val_cost
        runs.append((cost, mae))
    return runs, n_val


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for group, entries in GROUPS:
        n = len(entries)
        figsize = (5, 4) if n == 1 else (10, 4)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        for ax, (name, cost_vector, target_cost, ylabel, target, base) in zip(
                axes[0], entries):
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(VAL_SIZES)))
            panel(ax, name, cost_vector, target_cost, ylabel, target, base, colors)
            runs, n_val = full_runs(name, cost_vector, target_cost)
            if runs:
                g, m, sd = mean_curve(runs)
                ax.loglog(g, m, color='crimson', ls='--', lw=1.8,
                          label=f'Adaptive-MFML, full 5% ($N_{{\\rm val}}$={n_val})')
                ax.fill_between(g, np.maximum(m - sd, m * 0.05), m + sd,
                                color='crimson', alpha=0.12, lw=0)
                ax.legend(fontsize=6.5, ncol=2)
        fig.tight_layout()
        pdf = OUT / f'validation_sweep_{group}.pdf'
        png = OUT / f'validation_sweep_{group}.png'
        fig.savefig(pdf, bbox_inches='tight')
        fig.savefig(png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('wrote', pdf, 'and', png)


if __name__ == '__main__':
    main()
