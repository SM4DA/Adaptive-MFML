#!/bin/bash

# Check if an XYZ file was provided
if [ -z "$1" ]; then
    echo "Usage: ./run_ani_wb97xdz_parallel.sh <geometry.xyz>"
    exit 1
fi

XYZ_FILE=$1
BASENAME=$(basename "$XYZ_FILE" .xyz)


# ==========================================
# STEP 1: Pre-convergence (Fast Guess)
# ==========================================
echo "1. Generating Step 1 input..."

cat <<EOF > ${BASENAME}_step1.inp
! wb97x 6-31G def2/j soscf rijcosx PAL16
%maxcore 500
%basis
  addgto C
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto N
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto O
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto S
    S 1
      1 0.65 1.0
    D 1
      1 0.65 1.0
  end
  addgto F
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto Cl
    S 1
      1 0.75 1.0
    D 1
      1 0.75 1.0
  end
end
* xyzfile 0 1 $XYZ_FILE
EOF

echo "Running Step 1 (pre-convergence on 16 cores)..."
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_step1.inp > ${BASENAME}_step1.log

# Check if Step 1 succeeded
if [ ! -f "${BASENAME}_step1.gbw" ]; then
    echo "Error: Step 1 failed to generate the .gbw file. Check ${BASENAME}_step1.out"
    exit 1
fi

# ==========================================
# STEP 2: Final Production Run (Energies & Forces)
# ==========================================
echo "2. Generating Step 2 input..."

cat <<EOF > ${BASENAME}_step2.inp
! wb97x 6-31G scfconv6 soscf easyconv SCFConvForced engrad MOREAD PAL16
%maxcore 500
%moinp "${BASENAME}_step1.gbw"
%basis
  addgto C
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto N
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto O
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto S
    S 1
      1 0.65 1.0
    D 1
      1 0.65 1.0
  end
  addgto F
    S 1
      1 0.8 1.0
    D 1
      1 0.8 1.0
  end
  addgto Cl
    S 1
      1 0.75 1.0
    D 1
      1 0.75 1.0
  end
end
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
~/../../opt/orca_5_0_3_linux_x86-64_openmpi411/orca ${BASENAME}_step2.inp > ${BASENAME}_step2.log

echo "--- Workflow Complete ---"