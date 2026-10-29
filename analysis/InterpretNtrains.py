import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qml.kernels import matern_kernel
from qml.math import cho_solve
from saturatedmfml.MFML_Model import ModelMFML
from saturatedmfml.dataloaders import get_dataset
from saturatedmfml.dataset_paths import RESULTS_SINGLE_PASS, results_dir

TIME_COSTS = {
    'ANI': np.array([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0,
    'EV': np.array([49.82109751, 81.16041267, 116.57696209, 363.15718502]) / 3600.0,
    'SCF': np.array([49.82109751, 81.16041267, 116.57696209, 363.15718502]) / 3600.0,
    'CH3Cl': np.array([296.85, 298.24, 2224.71, 96144.24]) / 3600.0,
    'CH3F': np.array([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
}

HYPERPARAMS = {
    'ANI':   {'sigma': 8e4,    'reg': 2e-12},
    'EV': {'sigma': 1600.0, 'reg': 1e-14}, 
    'SCF': {'sigma': 70.0, 'reg': 6e-10}, 
    'CH3Cl': {'sigma': 170.0,  'reg': 6e-10},
    'CH3F':  {'sigma': 50.0,   'reg': 5e-11}
}



def train_sfml_qml(N_train, X_train, y_train_target, X_test, sigma, reg, seed=42):
    """Trains a Single Fidelity model using QML matern_kernel and cho_solve."""
    np.random.seed(seed)
    
    # Randomly select subset
    idx = np.random.choice(len(X_train), N_train, replace=False)
    X_sub, y_sub = X_train[idx], y_train_target[idx]
    
    # Train Kernel
    K_train = matern_kernel(X_sub, X_sub, sigma=sigma, order=1, metric='l2')
    K_train[np.diag_indices_from(K_train)] += reg
    alphas = cho_solve(K_train, y_sub)
    
    # Test Predictions
    K_test = matern_kernel(X_sub, X_test, sigma=sigma, order=1, metric='l2')
    predictions = np.dot(alphas, K_test)
    
    return predictions

def train_basic_mfml_qml(sizes, indexes, 
                         X_train, y_train_all, 
                         X_test, sigma, reg, seed=42):
    """Trains a Basic MFML model enforcing nested subsets."""
    np.random.seed(seed)
    s = sizes[0]
    nfids = y_train_all.shape[0]

    random_select = np.random.choice(indexes, size=s, replace=False)
    
    X_train_sub = X_train[random_select]
    energies = np.zeros((nfids), dtype=object)
    for i in range(nfids):
        energies[i] = np.copy(y_train_all[i, random_select])

    mfml_indexes = generate_mfml_indexes(nfids, s)

    model = ModelMFML(reg=reg, kernel='matern', 
                      sigma=sigma, order=1, metric='l2', gammas=None, p_bar=True)
                      
    model.train(X_train_parent=X_train_sub, fidelities=None, 
                y_trains=energies, indexes=mfml_indexes, 
                shuffle=True, n_trains=final_sizes, seed=seed)

def get_final_sizes(trainsizes_file):
    """
    Helper function to extract the absolute final training sizes 
    reached at the end of the adaptive sampling run.
    """
    try:
        # Load the arrays for the first average run [:, 0]
        n_data = np.load(trainsizes_file, allow_pickle=True)[-1][0][-1]
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {trainsizes_file}. Check your path!")
        n_data=np.asarray([32,16,8,4])
            
    return n_data

def generate_mfml_indexes(nfids, size):
    """Generates the patched indices required by ModelMFML."""
    indexes = np.zeros((nfids), dtype=object)
    ordered_ind = np.arange(0, size)
    patched_ind = np.vstack([ordered_ind, ordered_ind]).T
    for i in range(nfids):
        indexes[i] = np.copy(patched_ind)
    return indexes


def evaluate_dataset_endpoints(dataset, prop='EV'):
    print(f"Processing {dataset}...")
    
    costs = TIME_COSTS[prop]
    params = HYPERPARAMS[prop]
    sigma, reg = params['sigma'], params['reg']
    
    # Load cascading array using your specific helper
    trainsizes_file = f'{RESULTS_SINGLE_PASS}/{prop}_globalpass_trainsizes.npy'
    n_data = get_final_sizes(trainsizes_file)
    
    if n_data is None:
        print('got nothing')
        return None

    M_lowest = int(n_data[0])
    N_top = int(n_data[-1])
    
    # Calculate target sizes based on exact scaling requirements
    basic_mfml_sizes = np.array([N_top*8, N_top*4, N_top*2, N_top])
    theoretical_sizes = np.array([M_lowest, M_lowest//2, M_lowest//4, M_lowest//8])
    
    # Cost Projections
    sfml_cost = N_top * costs[3]
    basic_mfml_cost = np.sum(basic_mfml_sizes * costs)
    theoretical_cost = np.sum(theoretical_sizes * costs)
    
    # Data Loading (Using your exact params)
    X_train, _, X_test, y_train, _, y_test = get_dataset(dataset, prop, seed=42, center=True)
    y_test_target = y_test[:, -1]
    
    # Model Training
    preds_sfml = train_sfml_qml(N_top, X_train, y_train[:, -1], X_test, sigma, reg)
    mae_sfml = np.mean(np.abs(preds_sfml - y_test_target))
    
    preds_basic = train_basic_mfml_qml(basic_mfml_sizes, X_train, y_train, X_test, sigma, reg)
    mae_basic = np.mean(np.abs(preds_basic - y_test_target))
    
    # Return structured dict for CSV
    return {
        'Dataset': dataset,
        'N_top_fid': N_top,
        'M_lowest_fid': M_lowest,
        'Cost_SFML_hrs': round(sfml_cost, 2),
        'Cost_Basic_MFML_hrs': round(basic_mfml_cost, 2),
        'Cost_Theoretical_MFML_hrs': round(theoretical_cost, 2),
        'MAE_SFML': round(mae_sfml, 6),
        'MAE_Basic_MFML': round(mae_basic, 6),
        'MAE_Theoretical_MFML': 0.0  # As requested, left as 0
    }

if __name__ == '__main__':
    # Add or modify dataset/property combinations here
    datasets_to_run = [
        ('ANI', 'ANI'),     # Adjust second arg if you use ('ANI', 'ANI')
        ('QeMFi', 'EV'),
        ('QeMFi', 'SCF'),
        ('VIB5', 'CH3F'), 
        ('VIB5', 'CH3Cl')
    ]
    
    results = []
    for ds, prop in datasets_to_run:
        res = evaluate_dataset_endpoints(ds, prop)
        if res is not None:
            results.append(res)
            
    if results:
        csv_filename = f'{RESULTS_SINGLE_PASS}/endpoint_analysis_results.csv'
        results_dir(RESULTS_SINGLE_PASS)
        
        # Extract column headers
        fieldnames = results[0].keys()
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"All results successfully saved to {csv_filename}")