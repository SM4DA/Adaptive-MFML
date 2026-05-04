import h5py
import numpy as np
from sklearn.utils import shuffle

path_to_h5file = '/home/vvinod/2025/BigDatasets/ANI1x/ANI-1x-release.h5'
output_filename = 'ANI1x_multifidelity_50k_allfids.npz'
num_samples = 50000
seed = 42

target_keys = [
    'wb97x_dz.energy', 'wb97x_tz.energy', 'ccsd(t)_cbs.energy',
    'hf_dz.energy', 
    #'hf_tz.energy', 'hf_qz.energy',
    #'mp2_dz.corr_energy', 'mp2_tz.corr_energy', 'mp2_qz.corr_energy',
    #'npno_ccsd(t)_dz.corr_energy', 'npno_ccsd(t)_tz.corr_energy', 'npno_ccsd(t)_qz.corr_energy'
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