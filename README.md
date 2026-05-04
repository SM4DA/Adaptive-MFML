# Adaptive-MFML

Scripts for Adaptive sampling for MFML. Includes scripts for plotting and hyper-parameter optimization. Datasets used are open sourced and can be accessed at their respective repositories.  
This code repository accompanies the preprint at [TBA].

The files are as follows:

1. `ANI_preprocessing.py` extracts ANI data, samples 50k random geometries along with their multifidelity energies. It also generates the SLATM descriptor for this dataset.
2. `MFMLmanager.py` contains the main bulk of the sampling strategies. It performs training and prediction with the MFML model.
3. `MFML_Model.py` is the script that runs the MFML scheme given a multifidelity training dataset.
4. `basic_experiments.py` contains scripts to run single fidelity and basic MFML learning curves.
5. `avg_experiments.py` runs the collection of different experiments from the manuscript. The appropriate line should be uncommented.
6. `plottinghelpers.py` contains functions to assist with generating the plots.
7. `Plots.ipynb` is the jupyter notebook with the plots.
8. `requirements.txt` is the list of python packages used.
9. `hyperopt.py` carries out hyper-parameter optimization for kernel parameters using a GP-based Bayesian minimizer.

The datasets used are all open sources and should be downloaded before running these scripts. 

