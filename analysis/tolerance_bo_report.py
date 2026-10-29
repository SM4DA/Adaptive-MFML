"""Human-readable report for the tied cost-aware BO of the adaptive tolerances.

Reads ``results/tolerance_bo/CH3Cl/evaluations.csv`` and writes
``Manuscript/tolerance_bo_CH3Cl_report.md`` plus
``Manuscript/plots/tolerance_bo_CH3Cl.pdf``.
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

from saturatedmfml.dataset_paths import REPO_ROOT
from experiments.tolerance_bo_ch3cl import (
    CSV_PATH, EPS, LAMBDA, LOG_HIGH, LOG_LOW, VAL_SIZE,
    cost_at_target, fit_gp, load_rows, point_stats,
)

OUT_MD = REPO_ROOT / 'Manuscript' / 'tolerance_bo_CH3Cl_report.md'
OUT_PDF = REPO_ROOT / 'Manuscript' / 'plots' / 'tolerance_bo_CH3Cl.pdf'
DEFAULT = (0.01, 0.1)


def fmt_setting(key):
    return f'({key[0]:g}, {key[1]:g})'


def table_rows(pts):
    rows = []
    for key, v in sorted(pts.items(), key=lambda kv: kv[1]['J']):
        rows.append((key, v))
    return rows


def write_markdown(rows, pts, best, path=OUT_MD):
    n_runs = len(rows)
    n_points = len(pts)
    best_key, best_v = best
    default_v = pts.get((round(DEFAULT[0], 10), round(DEFAULT[1], 10)))
    finalized = any(r['stage'] == 'final' for r in rows)
    status = 'final' if finalized else 'interim (initial design / BO rounds)'
    lines = [
        '# Cost-aware Bayesian optimisation of the adaptive-MFML tolerances',
        '',
        f'**Status: {status} — smoke test / offline benchmark tuning**  ',
        f'Dataset: VIB5 CH$_3$Cl; tied tolerances (one local $\\tau$ for all fidelities); '
        f'$N_{{\\rm val}}={VAL_SIZE}$; {n_points} settings, {n_runs} adaptive runs.',
        '',
        '## Protocol',
        '',
        'This is an *offline benchmark* experiment. Evaluating a tolerance setting '
        'requires running the adaptive loop and generating its reference labels; '
        'the method is therefore not an on-the-fly hyperparameter optimisation '
        '(see `hyperopt_propose.md`). It is used here to study whether a '
        'cost-aware surrogate can find better defaults than the grid/manual choice.',
        '',
        f'- Objective (penalised cost): $J = T_{{\\rm cross}} + \\lambda\\,'
        f'\\max(0, M-\\epsilon)/\\epsilon$, with $\\epsilon={EPS:g}$ kcal/mol, '
        f'$\\lambda={LAMBDA:g}$ h.',
        f'- $T_{{\\rm cross}}$: total cost (training + {VAL_SIZE} validation labels) at the '
        'first crossing of $\\epsilon$; if never reached, the final cost of the run.',
        '- Cost vector: VIB5 TBE/fidelity costs from SI Table ST3; validation label '
        'cost 26.7067 h each.',
        '- Search space: $\\log_{10}\\tau$ and $\\log_{10}\\tau_{\\rm global} \\in '
        f'[\\log_{{10}}10^{{{LOG_LOW:.0f}}}, \\log_{{10}}10^{{{LOG_HIGH:.2f}}}]$ '
        '(i.e. 0.01-5).',
        '- Initial design: default $(0.01,0.1)$, corners $(0.01,0.01)$, $(5,5)$, '
        'and 5 Latin-hypercube points; 2 seeds each.',
        '- Surrogate: GP with Matérn-5/2 ARD kernel on log-tolerances and '
        'heteroscedastic seed noise; acquisition: q-Expected Improvement with '
        'Kriging-Believer batch selection (q=3).',
        '- 3 BO rounds x 3 candidates x 2 seeds; the top 3 settings are then '
        're-evaluated with 5 seeds.',
        '',
        '## Results',
        '',
        '| tau_local | tau_global | seeds | mean J [h] | std J | mean T_cross [h] | mean final MAE |',
        '|---|---|---|---|---|---|---|',
    ]
    for key, v in table_rows(pts):
        mark = ' **best**' if key == best_key else ''
        lines.append(f'| {key[0]:.4g} | {key[1]:.4g} | {v["n"]} | {v["J"]:.0f} | '
                     f'{v["Jstd"]:.0f} | {v["T"]:.0f} | {v["M"]:.4f} |{mark}')
    lines += [
        '',
        f'**Best setting:** $\\tau={best_key[0]:g}$, '
        f'$\\tau_{{\\rm global}}={best_key[1]:g}$: mean J = {best_v["J"]:.1f} h, '
        f'final MAE = {best_v["M"]:.4f} kcal/mol over {best_v["n"]} seeds.',
        ''
    ]
    if default_v:
        delta = default_v['J'] - best_v['J']
        pooled = float(np.sqrt(best_v['Jstd'] ** 2 + default_v['Jstd'] ** 2))
        within = delta < pooled
        lines += [
            f'Manuscript default $(0.01,0.1)$: mean J = {default_v["J"]:.1f} h, '
            f'final MAE = {default_v["M"]:.4f} ({default_v["n"]} seeds).',
            '',
            f'Difference of the BO best from the default: {delta:.0f} h '
            f'({100 * delta / default_v["J"]:.1f} %), compared with a pooled '
            f'seed std of {pooled:.0f} h — '
            + ('within one standard deviation, i.e. **not statistically '
               'significant**.' if within else 'larger than the seed noise.'),
            ''
        ]
        if within:
            lines += [
                'The tied two-dimensional tolerance surface is flat around the '
                'default: the BO rounds did not improve on the initial design, and '
                'the best setting is indistinguishable from the manuscript value. '
                'This is consistent with the insensitivity observed in SF1 and the '
                '5x5 grid, and it argues against spending reference labels on '
                'tolerance optimisation for this property; a self-calibrating '
                'relative criterion (see `hyperopt_propose.md`) is the more '
                'appropriate long-term answer.',
                ''
            ]
    # optional comparison with the grid
    grid = REPO_ROOT / 'results' / 'tolerance_grid' / 'CH3Cl'
    if grid.exists():
        lines += ['A full 5x5 local/global grid at $N_{\\rm val}=2$ is being '
                  'generated separately (`results/tolerance_grid/CH3Cl`); the BO '
                  'best should be compared against the grid best once complete.',
                  '']
    lines += [
        '## Figures',
        '',
        f'- `Manuscript/plots/tolerance_bo_CH3Cl.pdf`: convergence and the '
        'GP-predicted objective surface.',
        '',
        '## Caveats',
        '',
        '- Smoke test on one dataset; the tied variant ignores possible '
        'per-fidelity differences.',
        '- BO evaluations use 2 seeds, so the surrogate sees noisy J; the final '
        'top-3 are re-evaluated with 5 seeds.',
        '- $\\lambda$ is a modelling choice (accuracy/cost exchange rate); '
        'different values change the ranking of inaccurate-but-cheap settings.',
        '- This is offline tuning; it does not make the adaptive run '
        'hyperparameter-free (see `hyperopt_propose.md`).',
        '',
        '## Next steps',
        '',
        '- Compare with the complete 5x5 grid best.',
        '- Increase seeds for the top settings (10, manuscript protocol).',
        '- If successful, repeat for CH$_3$F and the free (per-fidelity) variant.',
    ]
    Path(path).write_text('\n'.join(lines) + '\n')
    print('wrote', path)


def convergence_panel(ax, rows, pts):
    order = {'init': 0, 'bo1': 1, 'bo2': 2, 'bo3': 3, 'final': 4}
    # cumulative evaluation index in CSV order, collapse by setting
    seen = {}
    x = []
    y = []
    stage_colors = {'init': 'tab:grey', 'bo1': 'tab:blue', 'bo2': 'tab:cyan',
                    'bo3': 'tab:green', 'final': 'tab:red'}
    best = np.inf
    for i, r in enumerate(rows, start=1):
        key = (round(r['local'], 10), round(r['global'], 10))
        seen.setdefault(key, []).append(r['J'])
        mj = float(np.mean(seen[key]))
        best = min(best, mj)
        x.append(i); y.append(best)
        last_stage = r['stage']
    from matplotlib.lines import Line2D
    ax.plot(x, y, color='black', lw=1.5, label='best mean J so far')
    used = set()
    stages_seen = []
    for i, r in enumerate(rows, start=1):
        key = (round(r['local'], 10), round(r['global'], 10))
        if key in used:
            continue
        used.add(key)
        if r['stage'] not in stages_seen:
            stages_seen.append(r['stage'])
        ax.scatter(i, pts[key]['J'], s=26, color=stage_colors.get(r['stage'], 'k'),
                   zorder=3)
    handles = [Line2D([], [], marker='o', ls='', color=stage_colors.get(st, 'k'),
                      label=st) for st in stage_colors if st in stages_seen]
    handles.append(Line2D([], [], color='black', lw=1.5, label='best so far'))
    ax.set_xlabel('evaluation (CSV order)')
    ax.set_ylabel(r'penalised cost $J$ [h]')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_title('BO convergence', fontsize=10)


def surface_panel(ax, rows, pts, best):
    gp, X, y = fit_gp(rows)
    lo, hi = LOG_LOW, LOG_HIGH
    gx = np.linspace(lo, hi, 120)
    gy = np.linspace(lo, hi, 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = gp.predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    zmin = max(np.min(y) * 0.7, 1.0)
    zmax = np.max(y) * 1.3
    Z = np.clip(Z, zmin, zmax)
    im = ax.pcolormesh(10 ** GX, 10 ** GY, Z, cmap='viridis', shading='auto',
                       vmin=zmin, vmax=zmax)
    for key, v in pts.items():
        ax.scatter(key[0], key[1], c='white', edgecolors='black', s=28, zorder=3)
    ax.scatter(best[0][0], best[0][1], marker='*', s=220, c='red',
               edgecolors='black', zorder=4, label='best')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'global tolerance $\tau_{\rm global}$')
    ax.set_ylabel(r'local tolerance $\tau$')
    ax.set_title('GP-predicted penalised cost', fontsize=10)
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='J [h]')


def text_page(pdf, pts, best, n_rows, finalized):
    best_key, best_v = best
    fig = plt.figure(figsize=(8.27, 11.69)); fig.subplots_adjust(0.08, 0.06, 0.95, 0.95)
    txt = [
        'Cost-aware Bayesian optimisation of adaptive-MFML tolerances',
        'Smoke test: VIB5 CH3Cl, tied local tolerance, N_val = 2',
        '',
        'Protocol',
        '  objective  J = T_cross + lambda * max(0, MAE_final - eps) / eps',
        f'  eps = {EPS:g} kcal/mol, lambda = {LAMBDA:g} h',
        '  search     log10 tau_local, log10 tau_global in [log10(0.01), log10(5)]',
        '  surrogate  GP Matern-5/2 ARD on log-tolerances, heteroscedastic seed noise',
        '  acquisition q-EI, Kriging-Believer batch (q=3)',
        '  budget     8 initial settings x 2 seeds; 3 rounds x 3 candidates x 2 seeds;',
        '             top-3 re-evaluated with 5 seeds',
        '',
        f'Status: {"final" if finalized else "interim"}',
        f'Settings evaluated: {len(pts)}; adaptive runs: {n_rows}',
        '',
        f'Best: tau = {best_key[0]:g}, tau_global = {best_key[1]:g}',
        f'      J = {best_v["J"]:.1f} h, final MAE = {best_v["M"]:.4f} kcal/mol',
        f'      ({best_v["n"]} seeds, T_cross = {best_v["T"]:.0f} h)',
        '',
        'Top settings (mean over seeds)',
        f'{"tau":>10} {"tau_g":>10} {"n":>4} {"J [h]":>10} {"MAE":>8} {"T [h]":>9}',
    ]
    for key, v in table_rows(pts)[:10]:
        txt.append(f'{key[0]:10.4g} {key[1]:10.4g} {v["n"]:4d} {v["J"]:10.0f} '
                   f'{v["M"]:8.3f} {v["T"]:9.0f}')
    txt += ['', 'This is offline benchmark tuning; see tolerance_bo_CH3Cl_report.md']
    fig.text(0.05, 0.96, '\n'.join(txt), va='top', family='monospace', fontsize=8.5)
    pdf.savefig(fig); plt.close(fig)


def main():
    rows = load_rows()
    if not rows:
        print('no evaluations yet'); return
    pts = point_stats(rows)
    best = min(pts.items(), key=lambda kv: kv[1]['J'])
    write_markdown(rows, pts, best)
    with PdfPages(OUT_PDF) as pdf:
        text_page(pdf, pts, best, len(rows), any(r['stage'] == 'final' for r in rows))
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6))
        convergence_panel(axes[0], rows, pts)
        surface_panel(axes[1], rows, pts, best)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('wrote', OUT_PDF)


if __name__ == '__main__':
    main()
