import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def partition_ANI_data(seed=42,center=True):
    X = np.load('ANI1x_50k_SLATM_features.npy')
    target_keys = [
        'ccsd(t)_cbs.energy',
        'wb97x_tz.energy', 'wb97x_dz.energy', 
        'hf_dz.energy']
    target_keys = np.asarray(target_keys)[::-1]
    energies = np.zeros((X.shape[0],target_keys.shape[0]),dtype=float)
    npzfile = np.load('ANI1x_multifidelity_50k_allfids.npz',allow_pickle=True)
    count = 0
    for fids in target_keys:
        energies[:,count] = npzfile[fids].astype(float)*630 #kcal/mol
        count += 1
    X_train, X_test, y_train, y_test = train_test_split(X, energies, train_size=0.9, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.85/0.9, random_state=seed)
    
    del energies, X, npzfile
    if center:
        for i in range(4):
            fidelity_mean = np.mean(y_train[:, i])
            y_train[:, i] -= fidelity_mean
            y_test[:, i] -= fidelity_mean
            y_val[:, i] -= fidelity_mean
    return X_train, X_val, X_test, y_train, y_val, y_test

def partition_VIB5_data(molname='CH3Cl', seed=42, center=True):
    X = np.load(f'VIB5/{molname}_CM.npy')
    energies = np.zeros((X.shape[0], 4), dtype=float)
    fids = ['CCSD-T','HF-QZ','HF-TZ','MP2']
    for i in range(4):
        energies[:,i] = np.loadtxt(f'VIB5/{molname}_{fids[i]}.dat')*630

    X_train, X_test, y_train, y_test = train_test_split(X, energies, train_size=0.9, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.85/0.9, random_state=seed)
    
    del energies, X
    if center:
        for i in range(4):
            fidelity_mean = np.mean(y_train[:, i])
            y_train[:, i] -= fidelity_mean
            y_test[:, i] -= fidelity_mean
            y_val[:, i] -= fidelity_mean
            
    return X_train, X_val, X_test, y_train, y_val, y_test


def partition_QeMFi_data(prop='EV', seed=42, center=True):
    molnames = ['urea','acrolein','alanine','sma','nitrophenol',
                'urocanic','dmabn','thymine','o-hbdi']
    idx = np.arange(0, 15000)
    idx = shuffle(idx, random_state=seed)
    X = np.load('/home/vvinod/2024/QeMFi/MFML/Reps/o-hbdi_CM.npy')
    # largest SLATM is for o-hbdi with 6438 features.
    X = np.zeros((135000, X.shape[1]), dtype=float)  
    # Rest will be padded to this size.
    energies = np.zeros((135000, 4), dtype=float)
    
    start = 0
    end = 15000
    idx_names = np.zeros((135000), dtype=float)
    for i, m in enumerate(molnames):
        names = np.full(15000, i)
        idx_names[start:end] = np.copy(names)
        temp_X = np.load(f'QeMFi/{m}_CM.npy')
        X[start:end, :temp_X.shape[-1]] = temp_X[idx, :]
        if prop == 'EV':
            temp_data = np.load(f'QeMFi/QeMFi_{m}.npz')['EV'][:, :, 0] * 0.0029
        elif prop == 'SCF':
            temp_data = np.load(f'QeMFi/QeMFi_{m}.npz')['SCF'] * 630
        energies[start:end, :] = temp_data[idx, 1:]
        
        # increment for next molecule
        start += 15000
        end += 15000
    
    ####
    X_train, X_test, y_train, y_test = train_test_split(X, energies, 
                                                        train_size=0.9, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=0.85/0.9, random_state=seed)

    del energies, X
    if center == True:
        for i in range(4):
            hello_mean_here = np.mean(y_train[:, i])
            y_train[:, i] = y_train[:, i] - hello_mean_here
            y_test[:, i] = y_test[:, i] - hello_mean_here
            y_val[:, i] = y_val[:, i] - hello_mean_here
            
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset(source: str, name: str=None, seed: int = 42, center: bool = True):
    """Factory function to route data loading requests."""
    source = source.upper()
    if source == 'VIB5':
        return partition_VIB5_data(molname=name, seed=seed, center=center)
    elif source == 'QEMFI':
        return partition_QeMFi_data(prop=name, seed=seed, center=center)
    elif source=='ANI':
        return partition_ANI_data(seed=seed,center=center)
    else:
        raise ValueError(f"Unknown data source: {source}")