"""Combined plots: SF, basic MFML, and adaptive-MFML for N_val = 2..32.

One PDF, three pages: VIB5 (CH3Cl/CH3F), QeMFi (SCF/EV), ANI.  Cost includes
the validation labels: T = T_train + N_val * c_target.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, NullFormatter

from saturatedmfml.dataset_paths import (
    REPO_ROOT, RESULTS_SINGLE_PASS, RESULTS_VAL32,
)

SWEEP = REPO_ROOT / 'results' / 'validation_sweep'
VAL_SIZES = [2, 4, 8, 16, 32]
SEEDS = list(range(10))
OUT_PDF = REPO_ROOT / 'Manuscript' / 'plots' / 'validation_sweep_all.pdf'

VIB5_COST = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
ANI_COST = np.asarray([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0
QEMFI5 = np.load(RESULTS_SINGLE_PASS / 'QeMFi_times.npy').mean(axis=0) / 3600.0
QEMFI_COST = QEMFI5[1:]          # new fidelity set: 3-21G .. def2-TZVP

GROUPS = [
    ('VIB5', [
        ('CH3Cl', VIB5_COST, VIB5_COST[-1], 'MAE (CCSD-T) [kcal/mol]', 2.0, RESULTS_SINGLE_PASS),
        ('CH3F', VIB5_COST, VIB5_COST[-1], 'MAE (CCSD-T) [kcal/mol]', 2.0, RESULTS_SINGLE_PASS),
    ]),
    ('QeMFi', [
        ('SCF', QEMFI_COST, QEMFI5[4], 'MAE [kcal/mol]', None, RESULTS_VAL32),
        ('EV', QEMFI_COST, QEMFI5[4], 'MAE [kcal/mol]', None, RESULTS_VAL32),
    ]),
    ('ANI', [
        ('ANI', ANI_COST, ANI_COST[-1], 'RMAE (CCSD-T)', 4.5e-3, RESULTS_SINGLE_PASS),
    ]),
]


def sf_curve(name, base):
    sf = np.load(base / f'{name}_sf_mae.npy', allow_pickle=True).astype(float)
    sizes = 2.0 ** np.arange(1, sf.shape[0] + 1)
    return sizes, sf.mean(axis=1)


def basic_curve(name, base, cost_vector):
    b = np.load(base / f'{name}_2_basic_mfml.npy', allow_pickle=True).astype(float)
    t = np.asarray([np.asarray([8, 4, 2, 1]) * 2 ** i
                    for i in range(1, b.shape[1] + 1)]) @ cost_vector
    return t, b.mean(axis=0)


def adaptive_curves(name, cost_vector, val_cost):
    out = {}
    for v in VAL_SIZES:
        runs = []
        for seed in SEEDS:
            p = SWEEP / name / f'val{v}' / f'{seed}_{name}_globalpass'
            mae_p, size_p = p.with_name(p.name + '_mae.npy'), p.with_name(p.name + '_trainsizes.npy')
            if not (mae_p.exists() and size_p.exists()):
                continue
            mae = np.load(mae_p, allow_pickle=True).astype(float)
            sizes = np.load(size_p, allow_pickle=True).astype(int)
            c = np.asarray([cost_vector @ s for s in sizes]) + v * val_cost
            runs.append((c, mae))
        if runs:
            out[v] = runs
    return out


def mean_curve(runs, n=200):
    lo = max(c.min() for c, _ in runs)
    hi = min(c.max() for c, _ in runs)
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    vals = np.vstack([np.interp(np.log10(grid), np.log10(c), m) for c, m in runs])
    return grid, vals.mean(axis=0), vals.std(axis=0)


def cost_at_target(c, m, target):
    if target is None:
        return np.nan
    o = np.argsort(c); c, m = np.asarray(c)[o], np.asarray(m)[o]
    for i in range(1, len(m)):
        if m[i] <= target and m[i - 1] > target:
            f = (m[i - 1] - target) / (m[i - 1] - m[i]) if m[i - 1] != m[i] else 0
            return 10 ** (np.log10(c[i - 1]) + f * (np.log10(c[i]) - np.log10(c[i - 1])))
    return np.nan


def panel(ax, name, cost_vector, target_cost, ylabel, target, base, colors):
    t_sf, m_sf = sf_curve(name, base)
    t_b, m_b = basic_curve(name, base, cost_vector)
    ax.loglog(t_sf * target_cost, m_sf, color='black', lw=1.5, label='Single fidelity KRR')
    ax.loglog(t_b, m_b, color='tab:orange', lw=1.8, label='Basic MFML')

    rows = []
    curves = adaptive_curves(name, cost_vector, target_cost)
    for color, v in zip(colors, VAL_SIZES):
        runs = curves.get(v, [])
        if not runs:
            continue
        g, mean, std = mean_curve(runs)
        ax.loglog(g, mean, color=color, lw=1.8, label=f'$N_{{\\rm val}}$ = {v}')
        ax.fill_between(g, np.maximum(mean - std, mean * 0.05), mean + std,
                        color=color, alpha=0.15, lw=0)
        finals = np.asarray([m[-1] for _, m in runs])
        if target is None:
            ctt = np.full(len(runs), np.nan)
        else:
            ctt = np.asarray([cost_at_target(c, m, target) for c, m in runs])
        rows.append((name, v, finals.mean(), finals.std(),
                     np.nanmean(ctt), np.nanstd(ctt)))
    if target is not None:
        ax.axhline(target, color='black', ls='--', lw=1.0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.set_xlabel(r'$T_{\mathrm{train}} + T_{\mathrm{val}}$ [hrs]')
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    ax.legend(fontsize=7.5, ncol=2)
    return rows


def main():
    all_rows = []
    with PdfPages(OUT_PDF) as pdf:
        for group, entries in GROUPS:
            n = len(entries)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.6), squeeze=False)
            for ax, (name, cost_vector, target_cost, ylabel, target, base) in zip(
                    axes[0], entries):
                colors = plt.cm.viridis(np.linspace(0, 0.9, len(VAL_SIZES)))
                all_rows += panel(ax, name, cost_vector, target_cost, ylabel,
                                  target, base, colors)
            fig.suptitle(f'{group}: single fidelity, basic MFML, and adaptive-MFML '
                         f'for $N_{{\\rm val}}$ = 2-32 (total cost)',
                         fontsize=13 if n > 1 else 10)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            fig.savefig(OUT_PDF.with_name(f'validation_sweep_all_{group}.pdf'),
                        bbox_inches='tight')
            fig.savefig(OUT_PDF.with_name(f'validation_sweep_all_{group}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
    print('wrote', OUT_PDF)
    print(f'{"series":>6} {"N_val":>6} {"final":>16} {"cost-to-target [h]":>24}')
    for name, v, m, s, c, cs in all_rows:
        print(f'{name:>6} {v:6d} {m:9.4g} ± {s:7.3g} {c:15.4g} ± {cs:6.4g}')


if __name__ == '__main__':
    main()
