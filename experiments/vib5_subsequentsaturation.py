import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
from saturatedmfml.MFML_Model import ModelMFML

from saturatedmfml.dataloaders import generate_indexes, partition_VIB5_data
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

def further_saturated_MFML(X_train, y_trains, index, X_test, y_test,
                           reg:float=1e-10, sigma:float=20.0, 
                           X_val=None, y_val=None, 
                           navg:int=10, nstart:int=5, scale:int=2, window:int=10,
                           tol:float=1e-4, maxiter:int=1000, batch_size:int=32):
    # mfml_mae = np.zeros((navg,nmax),dtype=float)
    all_fidelity_maes = np.zeros((4,navg),dtype=object)
    all_fidelity_valmaes = np.zeros((4,navg),dtype=object)
    all_fidelity_diffs = np.zeros((4,navg),dtype=object)
    all_fidelity_trainsize = np.zeros((4,navg),dtype=object)
    # start_size = np.asarray([scale**3,scale**2,scale,1])*2**nstart
    # print('Starting Multifidelity Training Set Size: ',start_size)
    for n in tqdm(range(navg),leave=True,desc='Average loop for saturating MFML'):
        start_size = np.asarray([scale**3,scale**2,scale,1])*2**nstart
        # start_size = np.asarray([13072,   128,    64,    32])
        maes = []
        val_maes = []
        training_sizes = []
        training_sizes.append(np.copy(start_size))
        temp_mae = ordered_MFML(X_train, y_trains, index, X_test, y_test, 
                                start_size, seed=n,reg=reg,sigma=sigma)
        v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, 
                             start_size, seed=n,reg=reg,sigma=sigma)
        
        maes.append(np.copy(temp_mae))
        val_maes.append(np.copy(v_mae))
        mae_diff = []
        mae_diff.append(np.copy(v_mae))
        addition = np.zeros((4),dtype=int)
        
        #start by adding training samples at the fidelity f
        fidelity_count = 0
        for i in tqdm(range(maxiter),desc=f'Recursive Saturating',leave=False):
            addition[fidelity_count] = batch_size[fidelity_count]
            start_size+=addition
            
            ####break conditions for size consistency of lower fidelities
            if fidelity_count>0 and start_size[fidelity_count]>=start_size[fidelity_count-1]*scale:
                #if N_f approaches gamma*N_{f-1} then we are back to MFML regular.
                #this condition will avoid it
                print('terminated for size inconsistency')
                break
            
            training_sizes.append(np.copy(start_size))
            v_mae = ordered_MFML(X_train,y_trains,index,X_val,y_val,start_size,seed=n,reg=reg,sigma=sigma)
            temp_mae = ordered_MFML(X_train,y_trains,index,X_test,y_test,start_size,seed=n,reg=reg,sigma=sigma)
            maes.append(np.copy(temp_mae))
            val_maes.append(np.copy(v_mae))
            
            if i>window:
                moving_avg = np.mean(val_maes[-window:])
                mae_diff.append(np.copy(moving_avg))
                value = np.abs(mae_diff[i-window]-mae_diff[i-1-window])
                if value<tol[f]:
                    break
        all_fidelity_maes[f,n] = np.copy(maes)
        all_fidelity_valmaes[f,n] = np.copy(val_maes)
        all_fidelity_diffs[f,n] = np.copy(mae_diff)
        all_fidelity_trainsize[f,n] = np.copy(training_sizes)
    
    return all_fidelity_maes, all_fidelity_valmaes, all_fidelity_trainsize, all_fidelity_diffs    

def main(molname='CH3Cl'):
    out = results_dir(RESULTS_SINGLE_PASS)
    reg, sigma = get_hyperparams('VIB5', molname)

    #screen out data
    X_train, X_val, X_test, y_train, y_val, y_test = partition_VIB5_data(molname=molname, seed=42, center=True)
    indexes = np.load(f'{molname}_indexes.npy',allow_pickle=True)
    indexes=np.copy(indexes[0][:,0])

    ########saturated MFML
    m,v,n,a = further_saturated_MFML(X_train, y_train.T, indexes, X_test, y_test[:,-1],
                                   reg=reg, sigma=sigma,
                                   X_val=X_val, y_val=y_val[:,-1],
                                   navg=1, nstart=1, scale=2,
                                   tol=[1e-5,1e-10,1e-10,1e-10],
                                   maxiter=50, window=1, batch_size=[256,8,4,2])
    np.save(f'{out}/further_{molname}_satMFML_mae.npy',m)
    np.save(f'{out}/further_{molname}_satMFML_valmae.npy',v)
    np.save(f'{out}/further_{molname}_satMFML_trainsizes.npy',n)
    np.save(f'{out}/further_{molname}_satMFML_movingavg.npy',a)

if __name__=='__main__':
    main('CH3Cl')