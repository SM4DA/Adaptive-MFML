"""SI with-ceiling figures with the N_val=2 validation cost added.

Uses the existing (legacy) adaptive and ceiling-adaptive trajectories; no new
runs.  Only the cost axis changes: two validation labels are added to each
adaptive curve.  Colours follow the legacy withSS figures: gray dashed circles
(SF), orange squares (MFML), green (adaptive), blue (ceiling-adaptive).
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
    REPO_ROOT, RESULTS_ARCHIVE, RESULTS_AVERAGED, RESULTS_SINGLE_PASS,
)

OUT = REPO_ROOT / 'Manuscript' / 'publishable_plots'
VIB5 = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
ANI = np.asarray([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0
Q5 = np.load(RESULTS_SINGLE_PASS / 'QeMFi_times.npy').mean(axis=0) / 3600.0
VAL_SIZE = 2

COL_SF, COL_MFML, COL_AD, COL_CEIL = 'gray', 'orange', 'green', 'blue'

SPECS = {
    'CH3Cl': dict(dir=RESULTS_ARCHIVE / 'avg_prereview', cost=VIB5,
                  target_cost=VIB5[-1], label_file='CH3Cl', base=RESULTS_SINGLE_PASS),
    'CH3F': dict(dir=RESULTS_ARCHIVE / 'avg_prereview', cost=VIB5,
                 target_cost=VIB5[-1], label_file='CH3F', base=RESULTS_SINGLE_PASS),
    'SCF': dict(dir=RESULTS_ARCHIVE / 'avg_prereview', cost=Q5[:4],
                target_cost=Q5[3], label_file='SCF', base=RESULTS_SINGLE_PASS),
    'EV': dict(dir=RESULTS_ARCHIVE / 'avg_prereview', cost=Q5[:4],
               target_cost=Q5[3], label_file='EV', base=RESULTS_SINGLE_PASS),
    'ANI': dict(dir=RESULTS_AVERAGED, cost=ANI, target_cost=ANI[-1],
                label_file='ANI', base=RESULTS_SINGLE_PASS),
}


def runs(spec, strategy):
    out = []
    for seed in range(10):
        p = spec['dir'] / f'{seed}_{spec["label_file"]}_{strategy}'
        mae_p = p.with_name(p.name + '_mae.npy')
        size_p = p.with_name(p.name + '_trainsizes.npy')
        if not (mae_p.exists() and size_p.exists()):
            continue
        mae = np.load(mae_p, allow_pickle=True).astype(float)
        sizes = np.load(size_p, allow_pickle=True).astype(int)
        c = np.asarray([spec['cost'] @ s for s in sizes])
        c = c + VAL_SIZE * spec['target_cost']
        out.append((c, mae))
    return out


def mean_curve(rs, n=200):
    lo = max(c.min() for c, _ in rs)
    hi = max(c.max() for c, _ in rs)
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    vals = np.vstack([np.interp(np.log10(grid), np.log10(c), m) for c, m in rs])
    return grid, vals.mean(axis=0), vals.std(axis=0)


def baselines(spec):
    name = spec['label_file']
    sf = np.load(spec['base'] / f'{name}_sf_mae.npy', allow_pickle=True).astype(float)
    basic = np.load(spec['base'] / f'{name}_2_basic_mfml.npy', allow_pickle=True).astype(float)
    t_sf = 2.0 ** np.arange(1, sf.shape[0] + 1) * spec['target_cost']
    t_b = np.asarray([np.asarray([8, 4, 2, 1]) * 2 ** i
                      for i in range(1, basic.shape[1] + 1)]) @ spec['cost']
    return t_sf, sf.mean(axis=1), t_b, basic.mean(axis=0)


def draw(ax, name, spec, legend=True):
    t_sf, m_sf, t_b, m_b = baselines(spec)
    ax.loglog(t_sf, m_sf, color=COL_SF, ls='--', marker='o', ms=5, lw=1.3,
              label='Single Fidelity KRR')
    ax.loglog(t_b, m_b, color=COL_MFML, ls='-', marker='s', ms=5, lw=1.6,
              label='MFML')
    for strategy, color, label in [('globalpass', COL_AD, 'Adaptive-MFML'),
                                   ('cascading', COL_CEIL, 'Ceiling-Adaptive-MFML')]:
        rs = runs(spec, strategy)
        if not rs:
            continue
        for c, m in rs:
            ax.loglog(c, m, color='gray', alpha=0.30, lw=0.6)
        g, m, s = mean_curve(rs)
        ax.loglog(g, m, color=color, lw=2, label=label)
        ax.fill_between(g, np.maximum(m - s, m * 0.05), m + s, color=color,
                        alpha=0.15, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.set_xlabel(r'$T_{\mathrm{train}} + T_{\mathrm{val}}$ [hrs]')
    ax.set_title(name)
    if legend:
        ax.legend(fontsize=8)


def make(group, entries, ylabel, yname):
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(10, 4) if n == 2 else (5, 4),
                             squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[0], entries)):
        draw(ax, name, SPECS[name], legend=(i == n - 1))
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'withSS_{yname}.{ext}', bbox_inches='tight',
                    dpi=150 if ext == 'png' else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make('VIB5', ['CH3Cl', 'CH3F'], 'MAE (TBE) [kcal/mol]', 'VIB5')
    make('QeMFi', ['SCF', 'EV'], 'MAE [kcal/mol]', 'QeMFi')
    make('ANI', ['ANI'], r'RMAE (DLPNO-CCSD(T)$^{*}$/CBS)', 'ANI')
    print('wrote withSS figures to', OUT)


if __name__ == '__main__':
    main()
