"""Fresh main-text figures with N_val = 2 and validation cost included.

Style follows the legacy figures: single fidelity KRR (gray dashed, circles),
basic MFML (orange, squares), adaptive-MFML (green, with individual runs in
gray).  The dashed budget guides and their text labels are recomputed from the
N_val=2 curves (their old values change).

Writes to Manuscript/main_text_plots/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from saturatedmfml.dataset_paths import (
    REPO_ROOT, RESULTS_ROOT, RESULTS_SINGLE_PASS, RESULTS_VAL32,
)

OUT = REPO_ROOT / 'Manuscript' / 'main_text_plots'
VIB5 = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
ANI = np.asarray([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0
Q5 = np.load(RESULTS_SINGLE_PASS / 'QeMFi_times.npy').mean(axis=0) / 3600.0

COL_SF = 'gray'
COL_MFML = 'orange'
COL_AD = 'green'


def mean_curve(costs, maes, n=200):
    lo = max(c.min() for c in costs)
    hi = max(c.max() for c in costs)
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    vals = np.vstack([np.interp(np.log10(grid), np.log10(c), m)
                      for c, m in zip(costs, maes)])
    return grid, vals.mean(axis=0), vals.std(axis=0)


def adaptive(name, cost_vector, val_cost):
    costs, maes = [], []
    for seed in range(10):
        p = RESULTS_ROOT / 'validation_sweep' / name / 'val2' / f'{seed}_{name}_globalpass'
        mae = np.load(p.with_name(p.name + '_mae.npy'), allow_pickle=True).astype(float)
        sizes = np.load(p.with_name(p.name + '_trainsizes.npy'), allow_pickle=True).astype(int)
        costs.append(np.asarray([cost_vector @ s for s in sizes]) + val_cost)
        maes.append(mae)
    return costs, maes


def baselines(name, base, cost_vector, target_cost):
    sf = np.load(base / f'{name}_sf_mae.npy', allow_pickle=True).astype(float)
    basic = np.load(base / f'{name}_2_basic_mfml.npy', allow_pickle=True).astype(float)
    t_sf = 2.0 ** np.arange(1, sf.shape[0] + 1) * target_cost
    t_b = np.asarray([np.asarray([8, 4, 2, 1]) * 2 ** i
                      for i in range(1, basic.shape[1] + 1)]) @ cost_vector
    return t_sf, sf.mean(axis=1), t_b, basic.mean(axis=0)


def cross_log(x, y, target):
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y)[o]
    for i in range(1, len(y)):
        if y[i] <= target <= y[i - 1]:
            f = (y[i - 1] - target) / (y[i - 1] - y[i]) if y[i - 1] != y[i] else 0
            return 10 ** (np.log10(x[i - 1]) + f * (np.log10(x[i]) - np.log10(x[i - 1])))
    return np.nan


def y_at_x(x, y, xq):
    o = np.argsort(x)
    return 10 ** np.interp(np.log10(xq), np.log10(np.asarray(x)[o]),
                           np.log10(np.asarray(y)[o]))


def style_axis(ax, ylabel, title):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.set_xlabel(r'$T_{\mathrm{train}} + T_{\mathrm{val}}$ [hrs]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_baselines_adaptive(ax, name, base, cost_vector, target_cost, val_cost):
    t_sf, m_sf, t_b, m_b = baselines(name, base, cost_vector, target_cost)
    costs, maes = adaptive(name, cost_vector, val_cost)
    for c, m in zip(costs, maes):
        ax.loglog(c, m, color='gray', alpha=0.35, lw=0.8)
    g, m, s = mean_curve(costs, maes)
    ax.loglog(t_sf, m_sf, color=COL_SF, ls='--', marker='o', ms=5, lw=1.3,
              label='Single Fidelity KRR')
    ax.loglog(t_b, m_b, color=COL_MFML, ls='-', marker='s', ms=5, lw=1.6,
              label='MFML')
    ax.loglog(g, m, color=COL_AD, lw=2, label='Adaptive-MFML')
    ax.fill_between(g, np.maximum(m - s, m * 0.05), m + s, color=COL_AD, alpha=0.15, lw=0)
    return (t_sf, m_sf), (t_b, m_b), (g, m)


def annotate_target(ax, target, xs, label_y):
    ax.axhline(target, color='black', ls='--', lw=1.0)
    offsets = [0.85, 1.0, 0.85]   # stagger labels vertically
    for x, col, off in zip(xs, [COL_SF, COL_MFML, COL_AD], offsets):
        if np.isfinite(x):
            ax.axvline(x, color=col, ls='--', lw=1.0)
            ax.text(x, label_y * off, f'{x:.0f}', color=col, ha='center', va='top',
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))


def fig_vib5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    target = 2.0
    for ax, name in zip(axes, ['CH3Cl', 'CH3F']):
        sf, b, ad = plot_baselines_adaptive(
            ax, name, RESULTS_SINGLE_PASS, VIB5, VIB5[-1], 2 * VIB5[-1])
        style_axis(ax, 'MAE (TBE) [kcal/mol]', name)
        ax.legend(fontsize=8.5)
        if name == 'CH3Cl':
            x_sf = cross_log(*sf, target)
            x_b = cross_log(*b, target)
            x_ad = cross_log(*ad, target)
            annotate_target(ax, target, (x_sf, x_b, x_ad), target * 0.72)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'main_VIB5.{ext}', bbox_inches='tight',
                    dpi=150 if ext == 'png' else None)
    plt.close(fig)


def fig_qemfi():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    budget = 100.0
    for ax, name in zip(axes, ['SCF', 'EV']):
        sf, b, ad = plot_baselines_adaptive(
            ax, name, RESULTS_VAL32, Q5[1:], Q5[4], 2 * Q5[4])
        style_axis(ax, 'MAE [kcal/mol]', name)
        ax.legend(fontsize=8.5)
        if name == 'SCF':
            y_sf = y_at_x(*sf, budget)
            y_b = y_at_x(*b, budget)
            _, raw_maes = adaptive(name, Q5[1:], 2 * Q5[4])
            y_ad = float(np.mean([m[-1] for m in raw_maes]))  # all stop before 100 h
            ax.axvline(budget, color='black', ls='--', lw=1.0)
            labels = [(y_sf, COL_SF, 1.10), (y_b, COL_MFML, 0.88),
                      (y_ad, COL_AD, 1.22)]
            for y, col, mult in labels:
                ax.axhline(y, xmin=0.03, xmax=0.92, color=col, ls='--', lw=1.0)
                ax.text(0.05, y * mult, f'{y:.1f}', color=col, fontsize=8.5,
                        fontweight='bold', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
            ax.text(budget, ax.get_ylim()[0] * 1.2, '100', color='black',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'main_QeMFi.{ext}', bbox_inches='tight',
                    dpi=150 if ext == 'png' else None)
    plt.close(fig)


def fig_ani():
    fig, ax = plt.subplots(figsize=(5, 4))
    target = 4.5e-3
    sf, b, ad = plot_baselines_adaptive(
        ax, 'ANI', RESULTS_SINGLE_PASS, ANI, ANI[-1], 2 * ANI[-1])
    style_axis(ax, r'RMAE (DLPNO-CCSD(T)$^{*}$/CBS)', 'ANI')
    ax.legend(fontsize=8.5)
    annotate_target(ax, target,
                    (cross_log(*sf, target), cross_log(*b, target),
                     cross_log(*ad, target)), target * 0.55)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'main_ANI.{ext}', bbox_inches='tight',
                    dpi=150 if ext == 'png' else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_vib5(); fig_qemfi(); fig_ani()
    print('wrote main-text figures to', OUT)


if __name__ == '__main__':
    main()
