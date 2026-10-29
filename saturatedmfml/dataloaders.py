"""Single source of truth for loading and splitting the MFML datasets.

Every experiment script imports its partitioning function from here so that
the fidelity ordering, the train/validation/test splits and the centering are
defined in exactly one place.  QeMFi fidelities 1-4 are used everywhere: the
lowest fidelity (STO-3G) is omitted and the target is always def2-TZVP.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .dataset_paths import ANI_DIR, QEMFI_DIR, QEMFI_REPS_DIR, VIB5_DIR, VIB5_RAWDATA_DIR

VIB5_FIDELITIES = ['CCSD-T', 'HF-QZ', 'HF-TZ', 'MP2']

# Increasing fidelity: HF/STO-3G, HF/3-21G, ...
ANI_FIDELITIES = np.asarray([
    'ccsd(t)_cbs.energy',
    'wb97x_tz.energy', 'wb97x_dz.energy',
    'hf_dz.energy',
])[::-1]

QEMFI_MOLECULES = ['urea', 'acrolein', 'alanine', 'sma', 'nitrophenol',
                   'urocanic', 'dmabn', 'thymine', 'o-hbdi']
QEMFI_N_CONF_PER_MOL = 15000


def _split(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.85 / 0.9, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _center_fidelities(y_train, y_val, y_test):
    """Subtract each fidelity's training mean."""
    for i in range(y_train.shape[1]):
        fidelity_mean = np.mean(y_train[:, i])
        y_train[:, i] -= fidelity_mean
        y_test[:, i] -= fidelity_mean
        y_val[:, i] -= fidelity_mean


def generate_indexes(nfids=4, size=1000):
    """Patched index arrays (sample, sample) expected by ModelMFML."""
    indexes = np.zeros((nfids), dtype=object)
    ordered_ind = np.arange(0, size)
    patched_ind = np.vstack([ordered_ind, ordered_ind]).T
    for i in range(nfids):
        indexes[i] = np.copy(patched_ind)
    return indexes


def load_ANI_arrays(seed=42):
    """Load the unsplit ANI1x features, energies and geometries."""
    X = np.load(f'{ANI_DIR}/ANI1x_50k_SLATM_features.npy')
    npzfile = np.load(f'{ANI_DIR}/ANI1x_multifidelity_50k_allfids.npz', allow_pickle=True)
    R = npzfile['X']
    energies = np.zeros((X.shape[0], ANI_FIDELITIES.shape[0]), dtype=float)
    for i, fids in enumerate(ANI_FIDELITIES):
        energies[:, i] = npzfile[fids].astype(float) * 630  # kcal/mol
    indexes = np.arange(0, energies[:, 0].shape[0])
    return indexes, energies, R, X


def partition_ANI_data(seed=42, center=True):
    indexes, energies, R, X = load_ANI_arrays(seed=seed)
    del indexes, R
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, energies, seed)
    del energies, X
    if center:
        _center_fidelities(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def partition_VIB5_data(molname='CH3Cl', seed=42, center=True):
    X = np.load(f'{VIB5_DIR}/{molname}_CM.npy')
    energies = np.zeros((X.shape[0], len(VIB5_FIDELITIES)), dtype=float)
    for i, fid in enumerate(VIB5_FIDELITIES):
        energies[:, i] = np.loadtxt(f'{VIB5_RAWDATA_DIR}/{molname}_{fid}.dat') * 630

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, energies, seed)
    del energies, X
    if center:
        _center_fidelities(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def partition_QeMFi_data(prop='EV', seed=42, center=True, molname=None):
    """QeMFi energies/features, concatenated over molecules or for one molecule.

    With ``molname`` set, returns that molecule's 15000 configurations (used
    by the per-molecule sweep); otherwise returns all nine molecules
    concatenated and shuffled (used by the combined benchmark).
    """
    if molname is not None:
        X = np.load(f'{QEMFI_REPS_DIR}/{molname}_CM.npy')
        energies = _load_qemfi_property(prop, molname, np.arange(X.shape[0]))
        X_train, X_val, X_test, y_train, y_val, y_test = _split(X, energies, seed)
        del energies, X
        if center:
            _center_fidelities(y_train, y_val, y_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    idx = shuffle(np.arange(0, QEMFI_N_CONF_PER_MOL), random_state=seed)
    X = np.load(f'{QEMFI_REPS_DIR}/o-hbdi_CM.npy')
    # largest SLATM is for o-hbdi with 6438 features; the rest are padded.
    X = np.zeros((QEMFI_N_CONF_PER_MOL * len(QEMFI_MOLECULES), X.shape[1]), dtype=float)
    energies = np.zeros((QEMFI_N_CONF_PER_MOL * len(QEMFI_MOLECULES), 4), dtype=float)

    start, end = 0, QEMFI_N_CONF_PER_MOL
    for m in QEMFI_MOLECULES:
        temp_X = np.load(f'{QEMFI_REPS_DIR}/{m}_CM.npy')
        X[start:end, :temp_X.shape[-1]] = temp_X[idx, :]
        energies[start:end, :] = _load_qemfi_property(prop, m, idx)
        start += QEMFI_N_CONF_PER_MOL
        end += QEMFI_N_CONF_PER_MOL

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, energies, seed)
    del energies, X
    if center:
        _center_fidelities(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _load_qemfi_property(prop, molname, idx):
    """Fidelities 1-4 for one molecule, indexed by ``idx`` (def2-TZVP last)."""
    data = np.load(f'{QEMFI_DIR}/QeMFi_{molname}.npz')
    if prop == 'EV':
        return data['EV'][:, :, 0][idx, 1:] * 0.0029  # kcal/mol
    elif prop == 'SCF':
        return data['SCF'][idx, 1:] * 630
    raise ValueError(f"Unknown QeMFi property: {prop}")


def get_dataset(source: str, name: str = None, seed: int = 42, center: bool = True):
    """Factory function to route data loading requests."""
    source = source.upper()
    if source == 'VIB5':
        return partition_VIB5_data(molname=name, seed=seed, center=center)
    elif source == 'QEMFI':
        return partition_QeMFi_data(prop=name, seed=seed, center=center)
    elif source == 'ANI':
        return partition_ANI_data(seed=seed, center=center)
    else:
        raise ValueError(f"Unknown data source: {source}")
