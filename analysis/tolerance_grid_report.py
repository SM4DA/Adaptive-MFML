"""Heatmaps of the local/global tolerance grid at N_val = 2.

For every dataset/property: cost to reach the target accuracy and final test
MAE, as functions of the local (y) and global (x) tolerance.  Each cell is the
mean over 3 seeds; cost includes the validation labels (2 x c_target).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_SINGLE_PASS

GRID = REPO_ROOT / 'results' / 'tolerance_grid'
LOCAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
GLOBAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
SEEDS = [0]   # single-seed grid; non-ANI uses seed 0
VAL_SIZE = 2

VIB5_COST = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
ANI_COST = np.asarray([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0
QEMFI5 = np.load(RESULTS_SINGLE_PASS / 'QeMFi_times.npy').mean(axis=0) / 3600.0
QEMFI_COST = QEMFI5[1:]

DATASETS = [
    ('CH3Cl', VIB5_COST, VIB5_COST[-1], 2.0, 'MAE [kcal/mol]'),
    ('CH3F', VIB5_COST, VIB5_COST[-1], 2.0, 'MAE [kcal/mol]'),
    ('SCF', QEMFI_COST, QEMFI5[4], 12.0, 'MAE [kcal/mol]'),
    ('EV', QEMFI_COST, QEMFI5[4], 4.0, 'MAE [kcal/mol]'),
    ('ANI', ANI_COST, ANI_COST[-1], 4.5e-3, 'RMAE'),
]
GROUPS = [('VIB5', ['CH3Cl', 'CH3F']),
          ('QeMFi', ['SCF', 'EV']),
          ('ANI', ['ANI'])]
OUT_PDF = REPO_ROOT / 'Manuscript' / 'plots' / 'tolerance_grid_all.pdf'


def tag(v):
    return f'{v:g}'


def load_cell(name, lt, gt, cost_vector, val_cost, target):
    finals, costs = [], []
    for seed in SEEDS:
        d = GRID / name / f'local{tag(lt)}_global{tag(gt)}'
        mae_p, size_p = d / f'{seed}_mae.npy', d / f'{seed}_trainsizes.npy'
        if not (mae_p.exists() and size_p.exists()):
            continue
        mae = np.load(mae_p, allow_pickle=True).astype(float)
        sizes = np.load(size_p, allow_pickle=True).astype(int)
        c = np.asarray([cost_vector @ s for s in sizes]) + VAL_SIZE * val_cost
        finals.append(mae[-1])
        costs.append(cost_at_target(c, mae, target))
    if not finals:
        return np.nan, np.nan
    return float(np.mean(finals)), float(np.nanmean(costs))


def cost_at_target(c, m, target):
    o = np.argsort(c); c, m = np.asarray(c)[o], np.asarray(m)[o]
    if len(m) and m[0] <= target:
        return c[0]
    for i in range(1, len(m)):
        if m[i] <= target and m[i - 1] > target:
            f = (m[i - 1] - target) / (m[i - 1] - m[i]) if m[i - 1] != m[i] else 0
            return 10 ** (np.log10(c[i - 1]) + f * (np.log10(c[i]) - np.log10(c[i - 1])))
    return np.nan


def matrices(name, cost_vector, val_cost, target):
    fin = np.full((len(LOCAL_TOLS), len(GLOBAL_TOLS)), np.nan)
    ctt = np.full_like(fin, np.nan)
    for i, lt in enumerate(LOCAL_TOLS):
        for j, gt in enumerate(GLOBAL_TOLS):
            fin[i, j], ctt[i, j] = load_cell(name, lt, gt, cost_vector, val_cost, target)
    return fin, ctt


def annotate(ax, data, fmt):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='0.4')
            else:
                ax.text(j, i, fmt(v), ha='center', va='center', fontsize=7.0,
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=0.9))


def panel(ax, data, title, cbar_label, log=False):
    cmap = plt.cm.viridis.copy(); cmap.set_bad('0.85')
    if log:
        vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 1
        im = ax.imshow(np.ma.masked_invalid(data), origin='lower', cmap=cmap,
                       norm=LogNorm(vmin=vmin, vmax=np.nanmax(data)), aspect='auto')
    else:
        im = ax.imshow(np.ma.masked_invalid(data), origin='lower', cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(GLOBAL_TOLS)))
    ax.set_xticklabels([tag(v) for v in GLOBAL_TOLS])
    ax.set_yticks(range(len(LOCAL_TOLS)))
    ax.set_yticklabels([tag(v) for v in LOCAL_TOLS])
    ax.set_xlabel('global tolerance')
    ax.set_ylabel('local tolerance')
    ax.set_title(title, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


def summary_page(pdf, results):
    fig = plt.figure(figsize=(8.27, 11.69))
    lines = [
        'Local / global tolerance grid',
        'N_val = 2, seed 0, 5 x 5 grid',
        '',
        'Protocol',
        '  local tau, global tau in {0.01, 0.1, 1, 2, 4}',
        '  one adaptive-MFML globalpass run per cell (seed 0)',
        '  cost = training + 2 validation labels',
        '  targets: CH3Cl/CH3F 2 kcal/mol; QeMFi SCF 12; EV 4;',
        '           ANI 4.5e-3 RMAE',
        '  blank cell = target not reached (loose tolerance stops early)',
        '',
        'Best cell (lowest cost to target)',
        f'{"dataset":>8} {"cost [h]":>10} {"(tau, tau_g)":>18} {"final":>10}',
    ]
    for name, cv, vc, tgt, _ in DATASETS:
        fin, ctt = results[name]
        i, j = np.unravel_index(np.nanargmin(ctt), ctt.shape)
        lines.append(f'{name:>8} {ctt[i, j]:10.4g} '
                     f'{str((LOCAL_TOLS[i], GLOBAL_TOLS[j])):>18} {fin[i, j]:10.4g}')
    lines += ['', 'Global tolerance has little effect on this grid; the local',
              'tolerance controls whether the target is reached at all.']
    fig.text(0.08, 0.95, '\n'.join(lines), va='top', family='monospace', fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def main():
    results = {name: matrices(name, cv, vc, tgt) for name, cv, vc, tgt, _ in DATASETS}
    with PdfPages(OUT_PDF) as pdf:
        summary_page(pdf, results)
        for group, names in GROUPS:
            ncol = len(names)
            fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol, 8.0), squeeze=False)
            for col, name in enumerate(names):
                entry = next(d for d in DATASETS if d[0] == name)
                fin, ctt = results[name]
                im1 = axes[0, col].imshow(np.ma.masked_invalid(ctt), origin='lower',
                                          cmap=plt.cm.viridis, aspect='auto',
                                          norm=LogNorm(vmin=np.nanmin(ctt[ctt > 0]),
                                                       vmax=np.nanmax(ctt)))
                axes[0, col].set_title(f'{name}: cost to target', fontsize=10)
                axes[1, col].imshow(np.ma.masked_invalid(fin), origin='lower',
                                    cmap=plt.cm.viridis, aspect='auto')
                axes[1, col].set_title(f'{name}: final {entry[4]}', fontsize=10)
                for row, data in [(0, ctt), (1, fin)]:
                    ax = axes[row, col]
                    ax.set_xticks(range(len(GLOBAL_TOLS)))
                    ax.set_xticklabels([tag(v) for v in GLOBAL_TOLS])
                    ax.set_yticks(range(len(LOCAL_TOLS)))
                    ax.set_yticklabels([tag(v) for v in LOCAL_TOLS])
                    ax.set_xlabel('global tolerance'); ax.set_ylabel('local tolerance')
                    annotate(ax, data, lambda v: f'{v:.3g}')
                fig.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04,
                             label='cost [h]')
            fig.suptitle(f'{group}: local/global tolerance grid (N_val = 2, seed 0)',
                         fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(OUT_PDF.with_name(f'tolerance_grid_{group}.pdf'), bbox_inches='tight')
            fig.savefig(OUT_PDF.with_name(f'tolerance_grid_{group}.png'), dpi=150,
                        bbox_inches='tight')
            pdf.savefig(fig); plt.close(fig)
    print('wrote', OUT_PDF)
    print(f'{"dataset":>7} {"best cost [h]":>14} {"at (l,g)":>14} {"final MAE":>12}')
    for name, cv, vc, tgt, _ in DATASETS:
        fin, ctt = results[name]
        i, j = np.unravel_index(np.nanargmin(ctt), ctt.shape)
        print(f'{name:>7} {ctt[i, j]:14.4g} {str((LOCAL_TOLS[i], GLOBAL_TOLS[j])):>14} '
              f'{fin[i, j]:12.4g}')


if __name__ == '__main__':
    main()
