#!/bin/bash

# Set the base directory to the directory where the script is located
cd "$(dirname "$0")"

# Create build folder
mkdir build && cd build 

# Check if an argument was provided
if [ -n "$1" ]; then
  # Run cmake with the provided path as the MUI directory
  cmake -DCMAKE_PREFIX_PATH=$1 -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../ ..
else
  # Run cmake with the default MUI directory
  cmake -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../ ..
fi

# Run make to build the executable
make 2>&1 | tee make.log && cd .. 

# Python path to FEniCSx src
export PYTHONPATH=${PWD}/ParaSiF_FEniCS_Solver:$PYTHONPATH

domainFluidAlone=${PWD}
domainStructureAlone=${PWD}/structureDomainAlone

domainFluid=${PWD}
#domainStructure=${PWD}
domainStructure=${PWD}/structureDomain

# Ranks set to each domain
numProcsFluid=1
numProcsStructure=1

solverFluidAlone=./dummy_SPH_Standalone.x
solverStructureAlone=structureDomainRun.py

solverFluid=./dummy_SPH_MUI.x
#solverStructure=dummy_FEM_MUI.py
solverStructure=structureDomainRun.py

START=$SECONDS

# Run SPH standalone
#mpirun -np ${numProcsFluid} -wdir ${domainFluidAlone} ${solverFluidAlone}

# Run FEM standalone
#mpirun -np ${numProcsStructure} -wdir ${domainStructureAlone} python3 -m mpi4py ${solverStructureAlone}

# Run SPH-FEM coupling
# NOTE: By using '-m mpi4py' mpi4py module runs the script handling properly
#      aborts. If a python process exit with failure all processes in MPI_COMM_WORLD       
#      will be aborted so hangs are avoided.

mpirun -np ${numProcsFluid} -wdir ${domainFluid} ${solverFluid} :\
       -np ${numProcsStructure} -wdir ${domainStructure} python3 -m mpi4py ${solverStructure} 2>&1 | tee output.log

DURATION=$(( SECONDS - START ))

if (( $DURATION > 3600 )) ; then
    let "hours=DURATION/3600"
    let "minutes=(DURATION%3600)/60"
    let "seconds=(DURATION%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $DURATION > 60 )) ; then
    let "minutes=(DURATION%3600)/60"
    let "seconds=(DURATION%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
elif (( $DURATION < 1 )) ; then
    echo "Completed in less than a second"
else
    echo "Completed in $DURATION seconds"
fi