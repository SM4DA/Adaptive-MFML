import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager
from saturatedmfml.dataloaders import generate_indexes, get_dataset
from saturatedmfml.dataset_paths import RESULTS_SINGLE_PASS, results_dir
from saturatedmfml.experiment_params import get_hyperparams

def main(source='VIB5', name='CH3Cl', reg=None, sigma=None,
         global_tol=[1e-3], window=5, 
         local_tol=[1e-4, 1e-4, 1e-3, 1e-2], 
         initial_size=[64, 32, 16, 8], batch_size=[128,8,4,2], 
         seed:int = 42):
    if reg is None or sigma is None:
        reg, sigma = get_hyperparams(source.upper() if source.upper() != 'QEMFI' else 'QeMFi', name)
    out = results_dir(RESULTS_SINGLE_PASS)

    # Initial Config
    config = MFMLConfig(
        reg=reg,
        sigma=sigma,
        scale=2,
        navg=10,  
        window=window, 
        global_tol=np.asarray(global_tol),
        local_tol=np.asarray(local_tol),
        max_passes=20,
        maxiter=100, 
        batch_size=batch_size,#[128, 8, 4, 2], 
        initial_size=initial_size,#[16, 8, 4, 2], 
        nmax_basic=9, 
        nmax_sf=11,
        seed=seed,
        relative=True
    )

    #Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(source, name, seed=42, center=True)
    # print(y_train.shape)
    #Load indices
    try:
        indexes = np.load(f'{name}_indexes.npy', allow_pickle=True)
        indexes = np.copy(indexes[0][:, 0])
    except FileNotFoundError:
        print(f"Indexes file '{name}_indexes.npy' not found! Generating.")
        indexes = generate_indexes(nfids=y_train.shape[1],size=y_train.shape[0])
        indexes = np.copy(indexes[0][:,0])
        #return

    # init manager
    manager = MFMLExperimentManager(
        X_train=X_train, 
        y_trains=y_train.T, 
        index=indexes, 
        X_val=X_val, 
        y_val_target=y_val[:, -1], 
        X_test=X_test, 
        y_test_target=y_test[:, -1], 
        config=config
    )

    # #single fidelity
    sf_mae = manager.sf_LC()
    np.save(f'{out}/{name}_sf_mae.npy', sf_mae)
    
    # #Basic MFML
    basic_mae, basic_sizes = manager.basic_MFML()
    np.save(f'{out}/{name}_{config.scale}_basic_mfml.npy', basic_mae)
    np.save(f'{out}/{name}_{config.scale}_basic_mfml_sizes.npy', basic_sizes)
    

if __name__=='__main__':
    main(source='ANI', name='ANI')