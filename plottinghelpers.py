import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D
from rdkit.Chem import Draw
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline

###########################
def average_plot(time_cost, label='globalpass',dataset='CH3Cl',ax=None, std=False, 
                  color='red',legend='Multi-Pass', inter_grid=50):
    ax = ax or plt.gca()
    x_arrays = []
    y_arrays = []
    sh = []
    for i in range(10): # 5 different lines
        x = np.load(f'outputs_avg/{i}_{dataset}_{label}_trainsizes.npy')
        y = np.load(f'outputs_avg/{i}_{dataset}_{label}_mae.npy')
        sh.append(y.shape[0])
        
        x = np.asarray([time_cost @ sizes for sizes in x])
        x_arrays.append(x)
        y_arrays.append(y)
    ####create grid corresponding to extremum of x values###
    x_min_common = max([x.min() for x in x_arrays])
    x_max_common = max([x.max() for x in x_arrays])    
    x_common = np.linspace(x_min_common, x_max_common, inter_grid) 
    y_interpolated_list = []
    for x, y in zip(x_arrays, y_arrays):
        y_interp = np.interp(x_common, x, y)
        y_interpolated_list.append(y_interp)
            
    y_interpolated_matrix = np.array(y_interpolated_list)
    y_mean = np.mean(y_interpolated_matrix, axis=0)
    for x, y in zip(x_arrays, y_arrays):
        ax.loglog(x, y, color='gray', alpha=0.4, linewidth=1)
    ax.loglog(x_common, y_mean, color=color, linewidth=1.5, label=legend)

    if std:
        y_std = np.std(y_interpolated_matrix, axis=0)
        ax.fill_between(x_common, np.abs(y_mean - y_std), y_mean + y_std, color=color, 
                         alpha=0.2)

###learning curves
def ANI_mfml_time_curve(scale=2, ax=None, saturated=True):
    # Base costs in hours
    time_cost = np.asarray([  9.64615,  16.8199 ,  40.34815, 157.2737 ])/3600.0
    ax = ax or plt.gca()
    
    t_sf = 2**np.arange(1, 12) * time_cost[-1]
    
    try:
        sf_mae = np.load('outputs/ANI_sf_mae.npy')
        ax.loglog(t_sf[:sf_mae.shape[0]], sf_mae.mean(axis=1), 
                  marker='o', label='Single Fidelity KRR', color='gray', linestyle='--')
    except FileNotFoundError:
        print(f"SF data not found for ANI. Skipping.")

    # Reconstruct Basic MFML Time Costs
    try:
        basic_mae = np.load(f'outputs/ANI_{scale}_basic_mfml.npy')
        basic_t = []
        for i in range(1, basic_mae.shape[1] + 1):
            sizes = np.array([scale**3, scale**2, scale, 1]) * (2**i)
            basic_t.append(time_cost @ sizes)
            
        ax.loglog(basic_t, basic_mae.mean(axis=0), 
                  marker='s', label='MFML', color='orange')
    except FileNotFoundError:
        print(f"Basic MFML data not found for ANI. Skipping.")


    def load_active_learning_curve(strategy_prefix):
        try:
            m_data = np.load(f'outputs/ANI_{strategy_prefix}_mae.npy')
            n_data = np.load(f'outputs/ANI_{strategy_prefix}_trainsizes.npy')
            
            
            all_t = np.asarray([time_cost @ sizes for sizes in n_data])

            sort_idx = np.argsort(all_t)
            all_t = np.copy(all_t[sort_idx])
            m_data = np.copy(m_data[sort_idx])
            
            return all_t, m_data
            
        except FileNotFoundError:
            print(f"{strategy_prefix} data not found for {molname}. Skipping.")
            return None, None

    if saturated:
        # Load the three active learning strategies
        t_glob, m_glob = load_active_learning_curve('globalpass')
        t_casc, m_casc = load_active_learning_curve('cascading')
    
        # Plot them with distinct styles
        if t_glob is not None:
            ax.loglog(t_glob, m_glob, label='Multi-Pass', color='blue')
            
        if t_casc is not None:
            ax.loglog(t_casc, m_casc, label='See-Saw', color='purple')

    # ==========================================
    ani_ref_mae = 321.57846043362053
    # Time cost is 2^11 multiplied by the highest fidelity cost (index -1)
    ani_ref_time = (2**11) * time_cost[-1] 
    
    # Draw the horizontal target error line
    ax.axhline(ani_ref_mae, color='black', linestyle='--', alpha=0.7, zorder=1)
    
    # Mark the specific time cost on that line
    ax.scatter(ani_ref_time, ani_ref_mae, color='black', marker='*', s=150, zorder=10, 
               label='ANI Baseline ($2^{11}$ CCSD(T))')
    ax.text(
        x=65, 
        y=200,              
        s=str(int(ani_ref_time)),           
        ha='center',        
        va='bottom', 
        fontweight='bold',
        # fontsize=10,        
        color='k'
    )
    
    ax.set_ylabel('MAE (CCSD-T) [kcal/mol]', fontsize=12)
    ax.set_xlabel('$T_{\mathrm{train}}$ [hrs]', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=10)

def vib5_mfml_time_curve(molname='CH3Cl', scale=2, ax=None, legend=True, saturated=True):
    # Base costs in hours
    time_cost = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600 
    
    # fig, ax = plt.subplots(figsize=(6, 5))
    ax = ax or plt.gca()
    
    
    t_sf = 2**np.arange(1, 12) * time_cost[-1]
    
    try:
        sf_mae = np.load(f'outputs/{molname}_sf_mae.npy')
        ax.loglog(t_sf[:sf_mae.shape[0]], sf_mae.mean(axis=1), 
                  marker='o', label='Single Fidelity KRR', color='gray', linestyle='--')
    except FileNotFoundError:
        print(f"SF data not found for {molname}. Skipping.")

    # Reconstruct Basic MFML Time Costs
    try:
        basic_mae = np.load(f'outputs/{molname}_{scale}_basic_mfml.npy')
        basic_t = []
        for i in range(1, basic_mae.shape[1] + 1):
            sizes = np.array([scale**3, scale**2, scale, 1]) * (2**i)
            basic_t.append(time_cost @ sizes)
            
        ax.loglog(basic_t, basic_mae.mean(axis=0), 
                  marker='s', label='MFML', color='orange')
    except FileNotFoundError:
        print(f"Basic MFML data not found for {molname}. Skipping.")


    def load_active_learning_curve(strategy_prefix):
        try:
            m_data = np.load(f'outputs/{molname}_{strategy_prefix}_mae.npy')
            n_data = np.load(f'outputs/{molname}_{strategy_prefix}_trainsizes.npy')
            
            
            all_t = np.asarray([time_cost @ sizes for sizes in n_data])
            
            sort_idx = np.argsort(all_t)
            all_t = np.copy(all_t[sort_idx])
            m_data = np.copy(m_data[sort_idx])
            
            return all_t, m_data
            
        except FileNotFoundError:
            print(f"{strategy_prefix} data not found for {molname}. Skipping.")
            return None, None

    if saturated:
        t_glob, m_glob = load_active_learning_curve('globalpass')
        t_casc, m_casc = load_active_learning_curve('cascading')
    
        if t_glob is not None:
            ax.loglog(t_glob, m_glob, label='Multi-Pass', color='blue')
            
        if t_casc is not None:
            ax.loglog(t_casc, m_casc, label='See-Saw', color='purple')
    
    ax.set_title(f'{molname}')
    ax.set_ylabel('MAE (CCSD-T) [kcal/mol]', fontsize=12)
    ax.set_xlabel('$T_{\mathrm{train}}$ [hrs]', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if legend:
        ax.legend(fontsize=10)

def get_qemfi_time_costs(seed=42):
    """
    Reconstructs the time costs for the QeMFi training set, applying the 
    exact same shuffling and double train_test_split as the data loader.
    """
    idx = np.arange(0, 15000)
    idx = shuffle(idx, random_state=seed)
    
    start = 0
    end = 15000
    idx_names = np.zeros((135000), dtype=float)
    
    for i in range(9):
        names = np.full(15000, i)
        idx_names[start:end] = np.copy(names)
        start += 15000
        end += 15000
        
    idx_names = shuffle(idx_names, random_state=seed)
    permolcost = np.load('QeMFi_times.npy')
    
    qemfi_time_costs = np.zeros((135000, 5), dtype=float)
    for i in range(135000):
        qemfi_time_costs[i, :] = np.copy(permolcost[int(idx_names[i]), :])

    costs_train, costs_test = train_test_split(qemfi_time_costs, train_size=0.9, random_state=seed)
    costs_train, costs_val = train_test_split(costs_train, train_size=0.85/0.9, random_state=seed)
    
    return costs_train[:, 1:] / 3600.0

def qemfi_mfml_time_curve(qemfi_time_cost, prop='EV', scale=2, ax=None, legend=True, saturated=True):
    """
    Plots the learning curves for the QeMFi dataset.
    """
    ax = ax or plt.gca()
    
    def calc_cost(sizes):
        total_time = 0.0
        for f in range(len(sizes)):
            total_time += qemfi_time_cost[:sizes[f], f].sum()
        return total_time

    sf_mae_file = f'outputs/{prop}_sf_mae.npy'
    if os.path.exists(sf_mae_file):
        sf_mae = np.load(sf_mae_file)
        n_sf = 2**np.arange(1, sf_mae.shape[0] + 1)
        t_sf = n_sf*qemfi_time_cost[-1] #np.array([qemfi_time_cost[:size, -1].sum() for size in n_sf])
        ax.loglog(t_sf, sf_mae.mean(axis=1), marker='o', color='gray', label='Single Fidelity KRR', linestyle='--')
    else:
        print(f"File not found: {sf_mae_file}. Skipping SF.")

    basic_mae_file = f'outputs/{prop}_{scale}_basic_mfml.npy'
    if os.path.exists(basic_mae_file):
        basic_mae = np.load(basic_mae_file)
        basic_t = []
        for i in range(1, basic_mae.shape[1] + 1):
            sizes = np.array([scale**3, scale**2, scale, 1]) * (2**i)
            basic_t.append(qemfi_time_cost @ sizes)
        ax.loglog(basic_t, basic_mae.mean(axis=0), marker='s', label=f'MFML', color='orange')
    else:
        print(f"File not found: {basic_mae_file}. Skipping Basic MFML.")

    
    def load_active_learning_curve(strategy_prefix):
        mae_file = f'outputs/{prop}_{strategy_prefix}_mae.npy'
        sizes_file = f'outputs/{prop}_{strategy_prefix}_trainsizes.npy'
        
        if not (os.path.exists(mae_file) and os.path.exists(sizes_file)):
            print(f"Files not found for {strategy_prefix}. Skipping.")
            return None, None
            
        try:
            m_data = np.load(mae_file)
            n_data = np.load(sizes_file)
            all_t = []
            for i in range(m_data.shape[0]):
                all_t.append(calc_cost(n_data[i,:]))
            
            sort_idx = np.argsort(all_t)
            all_t = np.asarray(all_t)
            m_data = np.asarray(m_data)
            all_t = np.copy(all_t[sort_idx])
            m_data = np.copy(m_data[sort_idx])
            return np.array(all_t), np.array(m_data)
            
        except Exception as e:
            print(f"Error loading data for {strategy_prefix}: {e}")
            return None, None

    if saturated:
        strategies = [
            ('globalpass', 'Multi-Pass', 'blue', 'x'),
            ('cascading', 'See-Saw', 'purple', '^')
        ]
    
        for prefix, label, color, marker in strategies:
            t, m = load_active_learning_curve(prefix)
            if t is not None and len(t) > 0:
                ax.loglog(t, m, label=label, color=color)#, marker=marker)

    ax.set_title(f'{prop}', fontsize=14)
    ax.set_ylabel('MAE (TZVP) [kcal/mol]', fontsize=12)
    ax.set_xlabel('$T_{\mathrm{train}}$ [hrs]', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Only show legend if there are actually lines plotted
    if legend:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=10)
    
    plt.tight_layout()

##################################################

def ANI_index_returns(seed=42,center=True):
    X = np.load('ANI1x_50k_SLATM_features.npy')
    target_keys = [
        'ccsd(t)_cbs.energy',
        'wb97x_tz.energy', 'wb97x_dz.energy', 
        'hf_dz.energy']
    target_keys = np.asarray(target_keys)[::-1]
    energies = np.zeros((X.shape[0],target_keys.shape[0]),dtype=float)
    # del X
    npzfile = np.load('ANI1x_multifidelity_50k_allfids.npz',allow_pickle=True)
    R = npzfile['X']
    count = 0
    for fids in target_keys:
        energies[:,count] = npzfile[fids].astype(float)*630 #kcal/mol
        count += 1
    indexes = np.arange(0,energies[:,0].shape[0])
    del npzfile
    ind_train, ind_test, y_train, y_test,R_train,R_test,X_train,X_test = train_test_split(indexes, energies, R, X,train_size=0.9, random_state=seed)
    del X
    ind_train, ind_val, y_train, y_val,X_train,_,R_train,R_val = train_test_split(ind_train, y_train, X_train, R_train,train_size=0.85/0.9, random_state=seed)

    del energies, R#, X, npzfile
    if center:
        for i in range(4):
            fidelity_mean = np.mean(y_train[:, i])
            y_train[:, i] -= fidelity_mean
            y_test[:, i] -= fidelity_mean
            y_val[:, i] -= fidelity_mean
    
    return ind_train, ind_test, ind_val, y_train,y_test, y_val, fidelity_mean, R_train,R_test, X_train,X_test

def get_qemfi_time_costs(seed=42):
    idx = np.arange(0, 15000)
    idx = shuffle(idx, random_state=seed)
    
    start = 0
    end = 15000
    idx_names = np.zeros((135000), dtype=float)
    
    for i in range(9):
        names = np.full(15000, i)
        idx_names[start:end] = np.copy(names)
        start += 15000
        end += 15000
        
    idx_names = shuffle(idx_names, random_state=seed)
    permolcost = np.load('QeMFi_times.npy')
    
    qemfi_time_costs = np.zeros((135000, 5), dtype=float)
    for i in range(135000):
        qemfi_time_costs[i, :] = np.copy(permolcost[int(idx_names[i]), :])

    costs_train, costs_test = train_test_split(qemfi_time_costs, train_size=0.9, random_state=seed)
    costs_train, costs_val = train_test_split(costs_train, train_size=0.85/0.9, random_state=seed)
    
    return costs_train[:, 1:] / 3600.0

###########################################

def r_z_to_2d_image(R, Z, mae_mfml, mae_sfml, mae_ANI, save_path=None, net_charge=0):
    R = np.asarray(R)
    Z = np.asarray(Z).flatten()
    num_atoms = len(Z)
    
    rw_mol = Chem.RWMol()
    
    for z in Z:
        rw_mol.AddAtom(Chem.Atom(int(z)))
        
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        x, y, z = float(R[i, 0]), float(R[i, 1]), float(R[i, 2])
        conf.SetAtomPosition(i, Point3D(x, y, z))
        
    rw_mol.AddConformer(conf)
    
    try:
        rdDetermineBonds.DetermineBonds(rw_mol, charge=net_charge)
        mol = rw_mol.GetMol()
        
        mol = Chem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(mol)
        
        draw_options = Draw.MolDrawOptions()
        draw_options.minFontSize = 12 
        draw_options.maxFontSize = 24
        draw_options.bondLineWidth = 2 
        
        legend_str = f"Adaptive Error: {mae_mfml:.2f} kcal/mol\nSF Error: {mae_sfml:.2f} kcal/mol\nANI Error: {mae_ANI:.2f} kcal/mol"
        
        img = Draw.MolToImage(
            mol, 
            size=(500, 500), 
            options=draw_options,
            legend=legend_str,
            fitImage=True
        )
        
        if save_path:
            img.save(save_path)
            print(f"Saved to {save_path}")
            
        return img
        
    except Exception as e:
        print(f"Failed to process molecule: {e}")
        return None


def plot_ani_performance(y_true, y_pred, n_atoms, save_path=None, zoomlim=[50, 100]):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_atoms = np.asarray(n_atoms)
    
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax1 = axes[0]
    scatter = ax1.scatter(y_true, y_pred, c=n_atoms, cmap='viridis', 
                          alpha=0.8, edgecolor='w', linewidth=0.3, s=40)
    
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('$N_{\mathrm{atoms}}$', rotation=270, labelpad=15)
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    buffer = (max_val - min_val) * 0.05
    ax1.set_xlim(min_val - buffer, max_val + buffer)
    ax1.set_ylim(min_val - buffer, max_val + buffer)
    
    ax1.plot([min_val - buffer, max_val + buffer], 
             [min_val - buffer, max_val + buffer], 
             color='black', linestyle='--', linewidth=1.5, zorder=0)

    #inset
    axins = ax1.inset_axes([0.05, 0.55, 0.35, 0.35])
    axins.scatter(y_true, y_pred, c=n_atoms, cmap='viridis', 
                  alpha=0.8, edgecolor='w', linewidth=0.3, s=15)
    
    axins.plot(zoomlim, zoomlim, color='black', linestyle=':', linewidth=1.5, zorder=0)
    
    #zoom window limits
    axins.set_xlim(zoomlim[0], zoomlim[1])
    axins.set_ylim(zoomlim[0], zoomlim[1])
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.set_xticks([])
    axins.set_yticks([])
    ax1.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel("Reference Energy [kcal/mol]")
    ax1.set_ylabel("Predicted Energy [kcal/mol]")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax2 = axes[1]
    
    sns.histplot(errors, kde=True, ax=ax2, color='#1f77b4', 
                 edgecolor='black', linewidth=0.5, alpha=0.6)
    
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
    
    stats_text = f"MAE: {mae:.4f}"
    ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel("Error (Predicted - Reference) [kcal/mol]")
    ax2.set_ylabel("Density / Count")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(-100, 100)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_error_by_size(y_true, y_pred, n_atoms, save_path=None):
    """
    Plots the error distributions and MAE grouped by the number of atoms.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth reference energies.
    y_pred : array-like
        Model predicted energies.
    n_atoms : array-like
        1D array containing the total number of atoms for each molecule.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    # Ensure inputs are 1D numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n_atoms = np.asarray(n_atoms).flatten()
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Pack into a DataFrame for easy grouping and seaborn plotting
    df = pd.DataFrame({
        'Number of Atoms': n_atoms,
        'Error': errors,
        'Absolute Error': abs_errors
    })
    
    mae_df = df.groupby('Number of Atoms')['Absolute Error'].mean().reset_index()
    mae_df.rename(columns={'Absolute Error': 'MAE'}, inplace=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1 = axes[0]
    
    sns.boxplot(data=df, x='Number of Atoms', y='Error', ax=ax1, 
                palette='viridis', hue='Number of Atoms', legend=False, showfliers=False, linewidth=1.2)
    
    # Add a bold dashed line at zero error for reference
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=3)
    
    ax1.set_title("(a) Error Distribution by Molecule Size", fontweight='bold')
    ax1.set_xlabel("Number of Atoms")
    ax1.set_ylabel("Error (Predicted - Reference)")
    
    ax2 = axes[1]
    
    sns.barplot(data=mae_df, x='Number of Atoms', y='MAE', ax=ax2, 
                palette='viridis', hue='Number of Atoms', legend=False,edgecolor=None, linewidth=0.8)
    
    target_sizes = ['14', '27', '40']
    
    # Zip the patches and the tick labels together to check the size
    for p, label in zip(ax2.patches, ax2.get_xticklabels()):
        if label.get_text() in target_sizes:
            # 1. Keep the target bars fully opaque (alpha=1.0)
            p.set_alpha(1.0)
            
            # 2. Add the bold annotation
            ax2.annotate(f"{p.get_height():.2f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=9, xytext=(0, 5), 
                         textcoords='offset points', rotation=0,
                         color='red')
        else:
            # Dim all the other bars to push them into the background
            p.set_alpha(0.5)
    
    ax2.set_title("(b) MAE by Molecule Size", fontweight='bold')
    ax2.set_xlabel("Number of Atoms")
    ax2.set_ylabel("Mean Absolute Error")
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=0)
    # Define the specific labels we want to keep visible
    desired_labels = [str(i) for i in range(5, 45, 5)] 
    
    for ax in axes:
        # Get the current categorical labels automatically generated by seaborn
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        
        # Keep the label if it's a multiple of 5, otherwise make it blank
        new_labels = [label if label in desired_labels else "" for label in current_labels]
        
        # Apply the filtered labels
        ax.set_xticklabels(new_labels, rotation=0)
        
    plt.tight_layout()
    
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
        
    return fig


