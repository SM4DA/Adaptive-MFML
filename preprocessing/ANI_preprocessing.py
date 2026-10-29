import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from sklearn.utils import shuffle
import qml
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm

from saturatedmfml.dataset_paths import ANI_DIR


target_keys = [
    'wb97x_dz.energy', 'wb97x_tz.energy', 'ccsd(t)_cbs.energy',
    'hf_dz.energy'
]

def iter_data_buckets(h5filename, keys):
    """Iterate over buckets and mask out configurations with NaNs in any requested key."""
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')

    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            if 'coordinates' not in grp: continue

            nc = grp['coordinates'].shape[0]
            mask = np.ones(nc, dtype=bool)
            data_cache = {}

            # Check for existence and NaNs across all keys
            valid_grp = True
            for k in keys:
                if k in grp:
                    val = grp[k][()]
                    data_cache[k] = val
                    v_reshaped = val.reshape(nc, -1)
                    mask = mask & ~np.isnan(v_reshaped).any(axis=1)
                else:
                    valid_grp = False # Skip bucket if a requested fidelity is missing
                    break

            if not valid_grp or not np.any(mask):
                continue

            d = {k: data_cache[k][mask] for k in keys}
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d

def build_multifidelity_npz(path_to_h5file, output_filename, num_samples=50000, seed=42):
    """Sample `num_samples` geometries with every requested fidelity present and save them."""
    flat_X = []
    flat_Z = []
    flat_energies = {k: [] for k in target_keys}

    for data in iter_data_buckets(path_to_h5file, keys=target_keys):
        nc = data['coordinates'].shape[0]

        flat_X.extend(data['coordinates'])
        z_repeated = np.tile(data['atomic_numbers'], (nc, 1))
        flat_Z.extend(z_repeated)

        for k in target_keys:
            flat_energies[k].extend(data[k])

    X = np.array(flat_X, dtype=object)
    Z = np.array(flat_Z, dtype=object)
    E_dict = {k: np.array(v, dtype=float) for k, v in flat_energies.items()}

    print(f"Total aligned geometries found: {len(X)}")
    if len(X) < num_samples:
        print(f"Warning: Only {len(X)} samples found meeting all criteria.")
        n_to_sample = len(X)
    else:
        n_to_sample = num_samples

    shuffled_indices = shuffle(np.arange(len(X)), random_state=seed)
    target_indices = shuffled_indices[:n_to_sample]

    X_final = X[target_indices]
    Z_final = Z[target_indices]
    E_final = {k: v[target_indices] for k, v in E_dict.items()}

    np.savez_compressed(output_filename, X=X_final, Z=Z_final, **E_final)

def generate_slatm_ani(input_filename, output_filename):
    """Generate global SLATM representations for the sampled geometries."""
    data = np.load(input_filename, allow_pickle=True)
    X = data['X']
    Z = data['Z']

    compounds = []

    for i in tqdm(range(len(X)), desc='Initializing Compounds'):
        mol = qml.Compound()
        mol.coordinates = X[i]
        mol.nuclear_charges = Z[i].astype(int)
        compounds.append(mol)

    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in compounds], dtype=object))
    #optionally save mbtypes array
    # np.save('ANI1x_50k_mbtypes.npy', np.asarray(mbtypes,dtype=object), allow_pickle=True)

    for mol in tqdm(compounds, desc='Generating SLATM Representations'):
        mol.generate_slatm(mbtypes, local=False)

    X_slatm = np.array([mol.representation for mol in compounds])
    np.save(output_filename, X_slatm)

def main():
    np.int = int  # qml 0.4 still refers to np.int, removed in NumPy 1.24

    build_multifidelity_npz(
        path_to_h5file=f'{ANI_DIR}/ANI-1x-release.h5',
        output_filename=f'{ANI_DIR}/ANI1x_multifidelity_50k_allfids.npz',
        num_samples=50000,
        seed=42,
    )
    generate_slatm_ani(
        input_filename=f'{ANI_DIR}/ANI1x_multifidelity_50k.npz',
        output_filename=f'{ANI_DIR}/ANI1x_50k_SLATM_features.npy',
    )

if __name__ == '__main__':
    main()
