#!/bin/bash

# Check if an XYZ file was provided
if [ -z "$1" ]; then
    echo "Usage: ./run_ani_wb97xtz_parallel.sh <geometry.xyz>"
    exit 1
fi

XYZ_FILE=$1
BASENAME=$(basename "$XYZ_FILE" .xyz)

echo "--- Starting Parallel (16 Cores) ANI wb97x/TZ workflow for $BASENAME ---"

# ==========================================
# STEP 1: Pre-convergence (Fast Guess)
# ==========================================
echo "1. Generating Step 1 (TZ) input..."

# Changed 6-31G to def2-TZVPP. def2/j is the correct auxiliary basis for def2-TZVPP.
cat <<EOF > ${BASENAME}_tz_step1.inp
! wb97x def2-TZVPP def2/j soscf rijcosx PAL16
%maxcore 500
* xyzfile 0 1 $XYZ_FILE
EOF

echo "Running Step 1 (pre-convergence on 16 cores)..."
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_tz_step1.inp > ${BASENAME}_tz_step1.log

# Check if Step 1 succeeded
if [ ! -f "${BASENAME}_tz_step1.gbw" ]; then
    echo "Error: Step 1 failed to generate the .gbw file. Check ${BASENAME}_tz_step1.log"
    exit 1
fi

# ==========================================
# STEP 2: Final Production Run (Energies & Forces)
# ==========================================
echo "2. Generating Step 2 (TZ) input..."

cat <<EOF > ${BASENAME}_tz_step2.inp
! wb97x def2-TZVPP scfconv6 soscf easyconv SCFConvForced engrad MOREAD PAL16
%maxcore 500
%moinp "${BASENAME}_tz_step1.gbw"
%method
  RadialGrid EulerMcLaurin
  AngularGrid 4
  HGridReduced false
  RadialGrid_fin EulerMcLaurin
  AngularGrid_fin 6
  IntAcc_fin 7.0
  HGridReduced_fin false
end
* xyzfile 0 1 $XYZ_FILE
EOF

echo "Running Step 2 (Final evaluation on 16 cores)..."
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_tz_step2.inp > ${BASENAME}_tz_step2.log

echo "--- Workflow Complete ---"
echo "Total Time and final energies can be found at the bottom of ${BASENAME}_tz_step2.log"