"""Cost-aware Bayesian optimisation of the tied adaptive-MFML tolerances.

Smoke test on VIB5 CH3Cl only.  Tied variant: one scalar local tolerance shared
by all fidelities plus one global tolerance.  Objective (penalised cost):

    J = T_cross + lambda * max(0, MAE_final - eps) / eps

where T_cross is the total cost (training + 2 validation labels) at which the
test MAE first reaches eps, or the final cost if it never does.  This is an
*offline benchmark tuning* experiment; it does not change the manuscript's
fixed default tolerances or any existing results.

Protocol:
  * search space: log10 tau_local, log10 tau_global in [log10(0.01), log10(5)]
  * initial design: 8 points (default + corners + LHS), 2 seeds each
  * 3 BO rounds x 3 candidates (q-EI, Kriging-Believer), 2 seeds each
  * final: top 3 settings re-evaluated with 5 seeds
  * GP: Matern-5/2 ARD on log-tolerances, heteroscedastic seed noise

Usage:
    python experiments/tolerance_bo_ch3cl.py
"""
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import norm, qmc

from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_ROOT, results_dir
from saturatedmfml.dataloaders import partition_VIB5_data
from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager

EPS = 2.0            # kcal/mol
LAMBDA = 1000.0      # hours per unit relative overshoot
VAL_SIZE = 2
COST = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
VAL_COST = VAL_SIZE * COST[-1]
LOG_LOW, LOG_HIGH = np.log10(0.01), np.log10(5.0)
SEEDS_BO = [0, 1]
SEEDS_FINAL = [0, 1, 2, 3, 4]
N_INIT = 8
N_ROUNDS = 3
Q = 3
OUT = RESULTS_ROOT / 'tolerance_bo' / 'CH3Cl'
CSV_PATH = OUT / 'evaluations.csv'
FIELDS = ['stage', 'local', 'global', 'seed', 'final_mae', 't_cross', 'J', 'n_points']


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def cost_at_target(c, m, target):
    order = np.argsort(c)
    c, m = np.asarray(c)[order], np.asarray(m)[order]
    for i in range(1, len(m)):
        if m[i] <= target and m[i - 1] > target:
            f = (m[i - 1] - target) / (m[i - 1] - m[i]) if m[i - 1] != m[i] else 0
            return 10 ** (np.log10(c[i - 1]) + f * (np.log10(c[i]) - np.log10(c[i - 1])))
    return np.nan


def _eval_task(args):
    local, glob, seed, stage = args
    return evaluate(local, glob, seed, stage)


def evaluate(local, glob, seed, stage='bo'):
    X_tr, X_val, X_te, y_tr, y_val, y_te = partition_VIB5_data(
        molname='CH3Cl', seed=42, center=True)
    X_val, y_val = X_val[:VAL_SIZE], y_val[:VAL_SIZE]
    index = np.load(REPO_ROOT / 'CH3Cl_indexes.npy', allow_pickle=True)
    index = np.copy(index[0][:, 0])
    config = MFMLConfig(
        reg=6e-10, sigma=170.0, scale=2, navg=1, window=10, maxiter=100,
        max_passes=20, local_tol=[local] * 4, global_tol=[glob] * 4,
        batch_size=[32, 8, 4, 2], initial_size=[32, 16, 8, 4],
        seed=seed, relative=False)
    manager = MFMLExperimentManager(
        X_tr, y_tr.T, index, X_val, y_val[:, -1], X_te, y_te[:, -1], config)
    mae, _, sizes, _ = manager.global_multipass_MFML_wrapper()
    mae = np.asarray(mae)
    c = np.asarray([COST @ s for s in sizes]) + VAL_COST
    t_cross = cost_at_target(c, mae, EPS)
    if np.isnan(t_cross):
        t_cross = float(c[-1])
    final = float(mae[-1])
    J = float(t_cross + LAMBDA * max(0.0, final - EPS) / EPS)
    return {'stage': stage, 'local': float(local), 'global': float(glob),
            'seed': int(seed), 'final_mae': final,
            't_cross': float(t_cross), 'J': J, 'n_points': len(mae)}


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
def load_rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: (float(v) if k not in ('stage', 'seed', 'n_points') else
                        (int(v) if k in ('seed', 'n_points') else v)) for k, v in r.items()})
    return out


def append_rows(rows):
    results_dir(OUT)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})


def evaluated_keys(rows):
    return {(round(r['local'], 10), round(r['global'], 10), r['seed']) for r in rows}


def point_stats(rows):
    pts = {}
    for r in rows:
        key = (round(r['local'], 10), round(r['global'], 10))
        pts.setdefault(key, []).append(r)
    out = {}
    for key, rs in pts.items():
        js = np.asarray([r['J'] for r in rs])
        out[key] = dict(n=len(rs), J=float(js.mean()), Jstd=float(js.std(ddof=1)) if len(js) > 1 else 0.0,
                        M=float(np.mean([r['final_mae'] for r in rs])),
                        T=float(np.mean([r['t_cross'] for r in rs])))
    return out


def evaluate_batch(points, stage, executor, rows):
    keys = evaluated_keys(rows)
    tasks = [(lo, gl, s) for (lo, gl) in points for s in SEEDS_BO
             if (round(lo, 10), round(gl, 10), s) not in keys]
    if not tasks:
        return []
    results = list(executor.map(_eval_task, [(lo, gl, s, stage) for lo, gl, s in tasks]))
    append_rows(results)
    for r in results:
        print(f"[{stage}] l={r['local']:.4g} g={r['global']:.4g} s={r['seed']} "
              f"J={r['J']:.1f} M={r['final_mae']:.3f} T={r['t_cross']:.0f}", flush=True)
    return results


# --------------------------------------------------------------------------
# surrogate
# --------------------------------------------------------------------------
def fit_gp(rows):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    pts = point_stats(rows)
    X = np.asarray([[np.log10(lo), np.log10(gl)] for (lo, gl) in pts])
    y = np.asarray([v['J'] for v in pts.values()])
    noise = np.asarray([max(v['Jstd'] / np.sqrt(max(v['n'], 1)), 1.0) ** 2
                        for v in pts.values()])
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 10.0), nu=2.5)
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e4)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  alpha=noise + 1e-8, n_restarts_optimizer=5,
                                  random_state=0)
    gp.fit(X, y)
    return gp, X, y


def propose_q(gp, X, y, q=Q, n_pool=4096, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    pool = sampler.random(n_pool)
    pool = qmc.scale(pool, [LOG_LOW, LOG_LOW], [LOG_HIGH, LOG_HIGH])
    chosen = []
    Xt, yt = X.copy(), y.copy()
    kernel = gp.kernel_
    for _ in range(q):
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         optimizer=None, random_state=0)
        model.fit(Xt, yt)
        mu, std = model.predict(pool, return_std=True)
        std = np.maximum(std, 1e-9)
        best = np.min(yt)
        z = (best - mu) / std
        ei = (best - mu) * norm.cdf(z) + std * norm.pdf(z)
        idx = int(np.argmax(ei))
        chosen.append(pool[idx])
        Xt = np.vstack([Xt, pool[idx]])
        yt = np.append(yt, mu[idx])          # Kriging Believer
        pool = np.delete(pool, idx, axis=0)
    return 10 ** np.asarray(chosen)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    results_dir(OUT)
    t0 = time.time()
    rows = load_rows()
    print(f'loaded {len(rows)} existing evaluations')

    rng = np.random.default_rng(0)
    lhs = qmc.scale(qmc.LatinHypercube(d=2, seed=0).random(N_INIT - 3),
                    [LOG_LOW, LOG_LOW], [LOG_HIGH, LOG_HIGH])
    init_log = np.vstack([[[np.log10(0.01), np.log10(0.1)],   # manuscript default
                           [LOG_LOW, LOG_LOW], [LOG_HIGH, LOG_HIGH]], lhs])
    init_pts = 10 ** init_log

    with ProcessPoolExecutor(max_workers=4) as ex:
        rows += evaluate_batch(init_pts, 'init', ex, rows)
        for rnd in range(1, N_ROUNDS + 1):
            gp, X, y = fit_gp(rows)
            cand = propose_q(gp, X, y, seed=rnd)
            rows += evaluate_batch(cand, f'bo{rnd}', ex, rows)
            pts = point_stats(rows)
            best = min(pts.items(), key=lambda kv: kv[1]['J'])
            print(f'round {rnd}: best J={best[1]["J"]:.1f} at '
                  f'local={best[0][0]:.4g}, global={best[0][1]:.4g}', flush=True)

        # final validation of the top three settings with 5 seeds
        pts = point_stats(rows)
        top = sorted(pts.items(), key=lambda kv: kv[1]['J'])[:3]
        keys = evaluated_keys(rows)
        tasks = [(lo, gl, s) for (lo, gl), _ in top for s in SEEDS_FINAL
                 if (round(lo, 10), round(gl, 10), s) not in keys]
        results = list(ex.map(_eval_task, [(lo, gl, s, 'final') for lo, gl, s in tasks]))
        append_rows(results)
        for r in results:
            print(f"[final] l={r['local']:.4g} g={r['global']:.4g} s={r['seed']} "
                  f"J={r['J']:.1f} M={r['final_mae']:.3f}", flush=True)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
