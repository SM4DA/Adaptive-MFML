import os
import numpy as np

from MFMLManager import MFMLConfig, MFMLExperimentManager
from dataloaders import get_dataset

def generate_indexes(nfids=4,size=1000):
    
    indexes = np.zeros((nfids),dtype=object)
    
    ordered_ind = np.arange(0,size)
    patched_ind = np.vstack([ordered_ind,ordered_ind]).T
    for i in range(nfids):
        indexes[i] = np.copy(patched_ind)
    return indexes

def main(source='VIB5', name='CH3Cl', reg=1e-1, 
         sigma=170.0, global_tol=[1e-3], window=5, 
         local_tol=[1e-4, 1e-4, 1e-3, 1e-2], 
         initial_size=[64, 32, 16, 8], batch_size=[128,8,4,2], 
         seed:int = 42):
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
        seed=seed
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
    np.save(f'outputs/{name}_sf_mae.npy', sf_mae)
    
    # #Basic MFML
    basic_mae = manager.basic_MFML()
    np.save(f'outputs/{name}_{config.scale}_basic_mfml.npy', basic_mae)
    

if __name__=='__main__':
    os.makedirs('outputs', exist_ok=True)
    main(source='VIB5', name='CH3Cl', reg=6e-10, sigma=170.0)
    main(source='VIB5', name='CH3F', reg=5e-11, sigma=50.0)
    main(source='QEMFI', name='SCF', reg=6e-10, sigma=70.0)
    main(source='QEMFI', name='EV', reg=1e-14, sigma=1600.0)
    main(source='ANI',name='ANI', reg=2e-12, sigma=8e4)