#!/bin/bash

# Check if both XYZ file and fidelity (dz/tz/qz) were provided
if [ -z "$2" ]; then
    echo "Usage: ./run_ani_hf_parallel.sh <geometry.xyz> <dz|tz|qz>"
    exit 1
fi

XYZ_FILE=$1
ZETA=$2
BASENAME=$(basename "$XYZ_FILE" .xyz)

# Assign the correct Dunning cc-pVXZ basis set based on user input
if [ "$ZETA" == "dz" ]; then
    BASIS="cc-pVDZ"
elif [ "$ZETA" == "tz" ]; then
    BASIS="cc-pVTZ"
elif [ "$ZETA" == "qz" ]; then
    BASIS="cc-pVQZ"
else
    echo "Error: Second argument must be 'dz', 'tz', or 'qz'."
    exit 1
fi

echo "--- Starting Parallel (16 Cores) ANI HF/$BASIS workflow for $BASENAME ---"

# ==========================================
# STEP 1: Pre-convergence (Fast Guess)
# ==========================================
echo "1. Generating Step 1 ($ZETA) input..."

# Note: We use 'HF' instead of 'wb97x'. 
# ORCA will automatically generate the required auxiliary basis for rijcosx.
cat <<EOF > ${BASENAME}_hf${ZETA}_step1.inp
! HF $BASIS soscf rijcosx PAL16
%maxcore 500
* xyzfile 0 1 $XYZ_FILE
EOF

echo "Running Step 1 (pre-convergence on 16 cores)..."
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_hf${ZETA}_step1.inp > ${BASENAME}_hf${ZETA}_step1.log

# Check if Step 1 succeeded
if [ ! -f "${BASENAME}_hf${ZETA}_step1.gbw" ]; then
    echo "Error: Step 1 failed to generate the .gbw file. Check ${BASENAME}_hf${ZETA}_step1.log"
    exit 1
fi

# ==========================================
# STEP 2: Final Production Run (Energies & Forces)
# ==========================================
echo "2. Generating Step 2 ($ZETA) input..."

# The %method block for grids is entirely removed since HF evaluates exact exchange analytically.
cat <<EOF > ${BASENAME}_hf${ZETA}_step2.inp
! HF $BASIS scfconv6 soscf easyconv SCFConvForced engrad MOREAD PAL16
%maxcore 500
%moinp "${BASENAME}_hf${ZETA}_step1.gbw"
* xyzfile 0 1 $XYZ_FILE
EOF

echo "Running Step 2 (Final evaluation on 16 cores)..."
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_hf${ZETA}_step2.inp > ${BASENAME}_hf${ZETA}_step2.log

echo "--- Workflow Complete ---"
echo "Total Time and final energies can be found at the bottom of ${BASENAME}_hf${ZETA}_step2.log"