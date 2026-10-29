import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm.auto import tqdm
from saturatedmfml.MFML_Model import ModelMFML

from saturatedmfml.dataloaders import partition_VIB5_data
from saturatedmfml.dataset_paths import RESULTS_SINGLE_PASS, results_dir
from saturatedmfml.experiment_params import get_hyperparams


def ordered_MFML(X_train, y_trains, index,
                 X_test, y_test, start_size, seed=42,
                 reg=1e-10, sigma=160.0):
    np.random.seed(seed)
    nfids = y_trains.shape[0]
    
    maes = []
    selected_inds = []
    s = start_size[0]
    random_select = np.random.choice(index,size=s,replace=False)
    selected_inds.extend(random_select)
    
    X_train = X_train[selected_inds]
    # print(X_train.shape)
    energies = np.zeros((nfids),dtype=object)
    for i in range(nfids):
        energies[i] = np.copy(y_trains[i,selected_inds])

    MFML_indexes =  generate_indexes(nfids=4,size=start_size[0])

    model = ModelMFML(reg=reg, kernel='matern', 
                      sigma=sigma, order=1, metric='l2', 
                      gammas=None, p_bar=False)
    model.train(X_train_parent = X_train, fidelities=None, 
                y_trains=energies, indexes=MFML_indexes, 
                shuffle=False, 
                n_trains=start_size, seed=42)
    _ = model.predict(X_test = X_test, y_test = y_test, 
                      X_val = None, y_val = None, 
                      optimiser='default')

    #DO NOT DELETE, for later
    # _ = model.predict(X_test = X_test, y_test = y_test, 
    #                   X_val = X_val, y_val = y_val, 
    #                   optimiser='OLS')
    # omfml_mae[i-1]+= np.copy(model.mae)
    # _ = model.predict(X_test=X_test,y_test=y_test,
    #                   X_val=X_val, y_val=y_val, optimiser='KRR', 
    #                   sigma=250.0, reg=1e-10, kernel_type='gaussian', 
    #                   order=1,metric='l2')
    # krr_mae[i-1] += np.copy(model.mae)
    # _ = model.predict(X_test=X_test,y_test=y_test,
    #                   X_val=X_val, y_val=y_val, optimiser='CompKRR', 
    #                   sigma=250.0, reg=1e-10, kernel_type='gaussian', order=1,metric='l2')
    # compkrr_mae[i-1] +=np.copy(model.mae)
    return model.mae

def global_multipass_MFML(X_train, y_trains, index, X_test, y_test,
                          reg:float=1e-10, sigma:float=20.0, 
                          X_val=None, y_val=None, 
                          navg:int=10, scale:int=2, window:int=1,
                          local_tol:list=[1e-5,1e-10,1e-10,1e-10], 
                          global_tol:float=1e-5, max_passes:int=10,
                          maxiter:int=1000, batch_size:list=[256,8,4,2],
                          initial_size:list=[13072, 128, 64, 32]):
    
    all_fidelity_maes = np.zeros((4,navg),dtype=object)
    all_fidelity_valmaes = np.zeros((4,navg),dtype=object)
    all_fidelity_diffs = np.zeros((4,navg),dtype=object)
    all_fidelity_trainsize = np.zeros((4,navg),dtype=object)

    for n in tqdm(range(navg), desc='Average loop for Global Multi-Pass'):
        start_size = np.copy(initial_size)
        
        # Tracking dictionaries for this specific run
        maes = {f: [] for f in range(4)}
        val_maes = {f: [] for f in range(4)}
        diffs = {f: [] for f in range(4)}
        training_sizes = {f: [] for f in range(4)}
        
        prev_global_val_mae = float('inf')
        
        for pass_num in range(max_passes):
            for f in range(4):
                local_val_history = []
                
                # Baseline validation eval for the local moving average history
                v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                local_val_history.append(v_mae)
                
                for i in range(maxiter):
                    # Constraint check (Gamma scaling constraint)
                    if f > 0 and (start_size[f] + batch_size[f]) >= start_size[f-1] * scale:
                        break
                        
                    # Add samples
                    start_size[f] += batch_size[f]
                    
                    # Evaluate
                    v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                    t_mae = ordered_MFML(X_train, y_trains, index, X_test, y_test, start_size, seed=n, reg=reg, sigma=sigma)
                    
                    # Record tracking metrics
                    maes[f].append(t_mae)
                    val_maes[f].append(v_mae)
                    training_sizes[f].append(np.copy(start_size))
                    
                    # Check local tolerance using moving average
                    if len(local_val_history) >= window:
                        baseline = np.mean(local_val_history[-window:])
                        improvement = baseline - v_mae
                        diffs[f].append(improvement)
                        if improvement < local_tol[f]:
                            break
                    else:
                        diffs[f].append(0.0) 
                        
                    local_val_history.append(v_mae)
            
            # End of Pass: Check overall global progress
            curr_global_val_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
            global_improvement = prev_global_val_mae - curr_global_val_mae
            
            if global_improvement < global_tol:
                # Early stop: The full pass didn't yield enough global improvement
                break
                
            prev_global_val_mae = curr_global_val_mae
            
        # Store histories into standard return arrays
        for f in range(4):
            all_fidelity_maes[f, n] = np.array(maes[f])
            all_fidelity_valmaes[f, n] = np.array(val_maes[f])
            all_fidelity_diffs[f, n] = np.array(diffs[f])
            all_fidelity_trainsize[f, n] = np.array(training_sizes[f])
            
    return all_fidelity_maes, all_fidelity_valmaes, all_fidelity_trainsize, all_fidelity_diffs


def cascading_saturation_MFML(X_train, y_trains, index, X_test, y_test,
                              reg:float=1e-10, sigma:float=20.0, 
                              X_val=None, y_val=None, 
                              navg:int=10, scale:int=2, window:int=1,
                              local_tol:list=[1e-5,1e-10,1e-10,1e-10], 
                              global_tol:float=1e-5, 
                              maxiter:int=1000, batch_size:list=[256,8,4,2],
                              initial_size:list=[13072, 128, 64, 32]):
    
    all_fidelity_maes = np.zeros((4,navg),dtype=object)
    all_fidelity_valmaes = np.zeros((4,navg),dtype=object)
    all_fidelity_diffs = np.zeros((4,navg),dtype=object)
    all_fidelity_trainsize = np.zeros((4,navg),dtype=object)

    for n in tqdm(range(navg), desc='Average loop for Cascading Saturation'):
        start_size = np.copy(initial_size)
        
        maes = {f: [] for f in range(4)}
        val_maes = {f: [] for f in range(4)}
        diffs = {f: [] for f in range(4)}
        training_sizes = {f: [] for f in range(4)}
        
        for active_ceiling in range(4):
            while True:
                # Baseline validation to check global progress after this saturation round
                pass_start_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                
                for f in range(active_ceiling + 1):
                    local_val_history = []
                    v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                    local_val_history.append(v_mae)
                    
                    for i in range(maxiter):
                        # Constraint check (Gamma scaling constraint)
                        if f > 0 and (start_size[f] + batch_size[f]) >= start_size[f-1] * scale:
                            break
                            
                        # Add samples
                        start_size[f] += batch_size[f]
                        
                        # Evaluate
                        v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                        t_mae = ordered_MFML(X_train, y_trains, index, X_test, y_test, start_size, seed=n, reg=reg, sigma=sigma)
                        
                        maes[f].append(t_mae)
                        val_maes[f].append(v_mae)
                        training_sizes[f].append(np.copy(start_size))
                        
                        # Check local tolerance using moving average
                        if len(local_val_history) >= window:
                            baseline = np.mean(local_val_history[-window:])
                            improvement = baseline - v_mae
                            diffs[f].append(improvement)
                            if improvement < local_tol[f]:
                                break
                        else:
                            diffs[f].append(0.0)
                            
                        local_val_history.append(v_mae)
                        
                # Finished saturating fidelities up to the active ceiling. Check global improvement.
                pass_end_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, start_size, seed=n, reg=reg, sigma=sigma)
                global_improvement = pass_start_mae - pass_end_mae
                
                if global_improvement < global_tol:
                    # Fully saturated up to this ceiling, break the inner while loop to raise ceiling
                    break
                    
        # Store histories into standard return arrays
        for f in range(4):
            all_fidelity_maes[f, n] = np.array(maes[f])
            all_fidelity_valmaes[f, n] = np.array(val_maes[f])
            all_fidelity_diffs[f, n] = np.array(diffs[f])
            all_fidelity_trainsize[f, n] = np.array(training_sizes[f])
            
    return all_fidelity_maes, all_fidelity_valmaes, all_fidelity_trainsize, all_fidelity_diffs

def main(molname='CH3Cl'):
    out = results_dir(RESULTS_SINGLE_PASS)
    reg, sigma = get_hyperparams('VIB5', molname)

    # Screen out data
    print(f"Loading data and partitioning for {molname}...")
    X_train, X_val, X_test, y_train, y_val, y_test = partition_VIB5_data(molname=molname, seed=42, center=True)
    
    # Load indexes
    try:
        indexes = np.load(f'{molname}_indexes.npy', allow_pickle=True)
        indexes = np.copy(indexes[0][:,0])
    except FileNotFoundError:
        print(f"Indexes file '{molname}_indexes.npy' not found! Make sure it is generated first.")
        return

    # -------- Run Idea 1: Global Multi-Pass --------
    print("\n--- Starting Global Multipass MFML (Idea 1) ---")
    m_glob, v_glob, n_glob, a_glob = global_multipass_MFML(
        X_train, y_train.T, indexes, X_test, y_test[:,-1],
        reg=reg, sigma=sigma,
        X_val=X_val, y_val=y_val[:,-1],
        navg=1, scale=2, window=1,
        local_tol=[1e-5,1e-10,1e-10,1e-10], global_tol=1e-5, max_passes=5,
        maxiter=50, batch_size=[256,8,4,2], initial_size=[13072, 128, 64, 32]
    )
    
    # Save the output files
    np.save(f'{out}/5{molname}_globalpass_mae.npy', m_glob)
    np.save(f'{out}/5{molname}_globalpass_valmae.npy', v_glob)
    np.save(f'{out}/5{molname}_globalpass_trainsizes.npy', n_glob)
    np.save(f'{out}/5{molname}_globalpass_diffs.npy', a_glob)


    # -------- Run Idea 2: Cascading Saturation --------
    print("\n--- Starting Cascading Saturation MFML (Idea 2) ---")
    m_casc, v_casc, n_casc, a_casc = cascading_saturation_MFML(
        X_train, y_train.T, indexes, X_test, y_test[:,-1],
        reg=reg, sigma=sigma,
        X_val=X_val, y_val=y_val[:,-1],
        navg=1, scale=2, window=1,
        local_tol=[1e-5,1e-10,1e-10,1e-10], global_tol=1e-5,
        maxiter=50, batch_size=[256,8,4,2], initial_size=[13072, 128, 64, 32]
    )

    # Save the output files
    np.save(f'{out}/5{molname}_cascading_mae.npy', m_casc)
    np.save(f'{out}/5{molname}_cascading_valmae.npy', v_casc)
    np.save(f'{out}/5{molname}_cascading_trainsizes.npy', n_casc)
    np.save(f'{out}/5{molname}_cascading_diffs.npy', a_casc)
    
    print("\nProcess completed successfully!")

if __name__=='__main__':
    main('CH3Cl')