import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds
from rdkit.Geometry import Point3D

def analyze_chemical_motifs(indices, R_array, Z_array):
    """
    Analyzes a subset of molecules for specific chemical features, 
    with a deep dive into bond types (double, triple, aromatic).
    """
    results = {
        'avg_C': 0, 'avg_N': 0, 'avg_O': 0,
        'avg_rings': 0,
        'avg_double_bonds': 0,
        'avg_triple_bonds': 0,
        'avg_aromatic_bonds': 0,
        'has_non_single': 0, # Percentage of molecules with at least one pi/aromatic bond
        'has_amide': 0, 'has_nitro': 0
    }
    
    valid_mols = 0
    
    amide_pattern = Chem.MolFromSmarts('C(=O)N')
    nitro_pattern = Chem.MolFromSmarts('[NX3](=O)=O')
    
    for idx in indices:
        Z = Z_array[idx]
        R = R_array[idx]
        
        # Build RDKit Mol
        rw_mol = Chem.RWMol()
        for z in Z: 
            rw_mol.AddAtom(Chem.Atom(int(z)))
            
        conf = Chem.Conformer(len(Z))
        for i, (x, y, z) in enumerate(R): 
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            
        rw_mol.AddConformer(conf)
        
        try:
            rdDetermineBonds.DetermineBonds(rw_mol)
            mol = rw_mol.GetMol()
            
            # 1. Count Elements & Rings
            results['avg_C'] += np.sum(Z == 6)
            results['avg_N'] += np.sum(Z == 7)
            results['avg_O'] += np.sum(Z == 8)
            results['avg_rings'] += Descriptors.RingCount(mol)
            
            if mol.HasSubstructMatch(amide_pattern): results['has_amide'] += 1
            if mol.HasSubstructMatch(nitro_pattern): results['has_nitro'] += 1
            
            # ==========================================
            # 2. NEW: Analyze Bond Types
            # ==========================================
            double_count = 0
            triple_count = 0
            aromatic_count = 0
            
            for bond in mol.GetBonds():
                b_type = bond.GetBondType()
                
                if b_type == Chem.rdchem.BondType.DOUBLE:
                    double_count += 1
                elif b_type == Chem.rdchem.BondType.TRIPLE:
                    triple_count += 1
                elif b_type == Chem.rdchem.BondType.AROMATIC:
                    aromatic_count += 1
                    
            # Add to running totals
            results['avg_double_bonds'] += double_count
            results['avg_triple_bonds'] += triple_count
            results['avg_aromatic_bonds'] += aromatic_count
            
            # Check if there is any non-single bond in the molecule
            if (double_count + triple_count + aromatic_count) > 0:
                results['has_non_single'] += 1
                
            valid_mols += 1
        except Exception:
            # Skip molecules where RDKit fails to infer valid chemistry
            continue
            
    # Average out the results
    if valid_mols > 0:
        for k in results:
            if 'avg' in k: 
                results[k] /= valid_mols
            elif 'has' in k: 
                results[k] = (results[k] / valid_mols) * 100 # Convert to percentage
                
    return results



def analyze_out_of_plane_distortion(indices, R_array, Z_array):
    """
    Calculates how 'flat' the heavy-atom backbone is, and exactly 
    how far the Hydrogen atoms are twisted out of that plane.
    """
    results = {
        'heavy_atom_thickness': 0.0,
        'hydrogen_out_of_plane_dist': 0.0
    }
    
    valid_mols = 0
    
    for idx in indices:
        R = R_array[idx]
        Z = Z_array[idx]
        
        # Isolate heavy atoms (Z > 1) and hydrogens (Z == 1)
        heavy_mask = Z > 1
        h_mask = Z == 1
        
        # We need at least 3 heavy atoms to define a plane
        if np.sum(heavy_mask) < 3 or np.sum(h_mask) == 0:
            continue
            
        R_heavy = R[heavy_mask]
        R_h = R[h_mask]
        
        # ==========================================
        # 1. Define the Heavy-Atom Best-Fit Plane
        # ==========================================
        # Center the heavy atoms at the origin
        center = np.mean(R_heavy, axis=0)
        centered_heavy = R_heavy - center
        
        # Calculate the covariance matrix and its eigenvalues
        cov = np.cov(centered_heavy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # The eigenvector with the SMALLEST eigenvalue is the "normal vector" 
        # (the arrow pointing perfectly straight up out of the plane)
        normal_vector = eigenvectors[:, 0]
        
        # The square root of the smallest eigenvalue is the RMS "thickness" 
        # of the heavy atoms. If the backbone is perfectly flat, this is ~0.
        heavy_thickness = np.sqrt(np.abs(eigenvalues[0]))
        
        # ==========================================
        # 2. Measure Hydrogen Out-Of-Plane Distances
        # ==========================================
        centered_h = R_h - center
        
        # The dot product of the H-coordinates with the plane's normal vector 
        # gives the exact perpendicular distance of the H atoms from the plane!
        h_distances = np.abs(np.dot(centered_h, normal_vector))
        avg_h_oop = np.mean(h_distances)
        
        # Add to running totals
        results['heavy_atom_thickness'] += heavy_thickness
        results['hydrogen_out_of_plane_dist'] += avg_h_oop
        valid_mols += 1
        
    # Average the results
    if valid_mols > 0:
        results['heavy_atom_thickness'] /= valid_mols
        results['hydrogen_out_of_plane_dist'] /= valid_mols
        
    return results

# ==========================================
# Example Usage:
# ==========================================
# good_indices = size_14[error[size_14] <= error_sf[size_14]]
# bad_indices = size_14[error[size_14] > error_sf[size_14]]
#
print("GOOD MFML MOLS:", analyze_out_of_plane_distortion(good_indices, R_test, Ztest))
print("BAD MFML MOLS: ", analyze_out_of_plane_distortion(bad_indices, R_test, Ztest))