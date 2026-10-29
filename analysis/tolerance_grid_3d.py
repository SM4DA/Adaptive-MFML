"""3D local/global tolerance plots (final error on z), legacy style.

For each global tolerance a coloured line runs over the local tolerances;
colour follows the global tolerance (viridis), as in the legacy tolerance
study.  x and y are plotted on log10 positions with the actual values as tick
labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from analysis.tolerance_grid_report import DATASETS, LOCAL_TOLS, GLOBAL_TOLS, matrices

OUT = Path(__file__).resolve().parent.parent / 'Manuscript' / 'publishable_plots'
GROUPS = [('VIB5', ['CH3Cl', 'CH3F']),
          ('QeMFi', ['SCF', 'EV']),
          ('ANI', ['ANI'])]


def draw(ax, name, fin, zlabel):
    lx = np.arange(len(GLOBAL_TOLS))
    ly = np.arange(len(LOCAL_TOLS))
    logvals = np.log10(GLOBAL_TOLS)
    norm = Normalize(vmin=logvals.min(), vmax=logvals.max())
    cmap = plt.cm.viridis
    for j, g in enumerate(GLOBAL_TOLS):
        z = fin[:, j]
        ax.plot(lx[j] * np.ones_like(ly), ly, z, color=cmap(norm(logvals[j])),
                marker='o', ms=4, lw=1.6)
    ax.set_xticks(lx)
    ax.set_xticklabels([f'{v:g}' for v in GLOBAL_TOLS], fontsize=7.5)
    ax.set_yticks(ly)
    ax.set_yticklabels([f'{v:g}' for v in LOCAL_TOLS], fontsize=7.5)
    ax.set_xlabel(r'global $\tau$', fontsize=9, labelpad=4)
    ax.set_ylabel(r'local $\tau$', fontsize=9, labelpad=4)
    ax.set_zlabel(zlabel, fontsize=9, labelpad=8)
    ax.set_title(name, fontsize=10)
    ax.view_init(elev=24, azim=-62)
    ax.grid(True, alpha=0.35)
    return ScalarMappable(norm=norm, cmap=cmap)


def page(pdf, group, names, lookup):
    n = len(names)
    fig = plt.figure(figsize=(12.0, 5.0) if n == 2 else (6.0, 5.0))
    fig.subplots_adjust(left=0.03, right=0.88 if n == 2 else 0.84,
                        bottom=0.10, top=0.93, wspace=0.28)
    for i, name in enumerate(names):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        fin, _ = lookup[name]
        z = fin * 1e3 if name == 'ANI' else fin
        zlabel = r'RMAE $\times 10^{3}$' if name == 'ANI' else 'final MAE [kcal/mol]'
        sm = draw(ax, name, z, zlabel)
    fig.savefig(OUT / f'tolerance_grid_3d_{group}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'tolerance_grid_3d_{group}.png', dpi=150, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = {}
    for name, cv, vc, tgt, _ in DATASETS:
        fin, _ = matrices(name, cv, vc, tgt)
        lookup[name] = (fin, None)
    with PdfPages(OUT / 'tolerance_grid_3d_all.pdf') as pdf:
        for group, names in GROUPS:
            page(pdf, group, names, lookup)
    print('wrote 3D tolerance plots to', OUT)


if __name__ == '__main__':
    main()
