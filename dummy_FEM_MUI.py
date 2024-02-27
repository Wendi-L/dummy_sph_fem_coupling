"""
#
# @file: dummy_FEM_MUI.py
# @brief: dummy FEM Python code for the coupling between SPH and FEM solvers.
#         A 2-D flexible beam, which is 5m higt and 2m thick, clamped at
#         the bottom. SPH code supposed to calculate fluid forces acting
#         on the beam, while the FEM code supposed to calculate the beam
#         deflections. MUI is used to pass forces of SPH interface particles
#         from SPH code to FEM code, and pass deflections of FEM interface
#         grid points from FEM code to SPH code.
#         Note: internal particles/points are omitted for simplicity.
#
#                                 |
#                                 |
#      (0,5,0)         (2,5,0)    |      (0,5,0)         (2,5,0)
# SPH:        Q Q Q Q Q           | FEM:        +-+-+-+-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#             Q o o o Q           |             +-*-*-*-+
#     (0,0,0) Q o o o Q (2,0,0)   |     (0,0,0) +-*-*-*-+ (2,0,0)
#    ---------------------------  |    ---------------------------
#    //////////////////////////   |    //////////////////////////
#                                 |
#                                 |
# 
# Q: SPH interface particles
# o: SPH internal particles
# +: FEM interface grid points
# *: FEM internal grid points
# 
# USAGE: mpirun -np 1 ./dummy_SPH_MUI.x : -np 1 python3 -m mpi4py dummy_FEM_MUI.py
# 
#/
"""

from mpi4py import MPI
import mpi4py
import datetime
import numpy as np
import time
import os

# Include MUI header file and configure file
import mui4py

# MUI parameters
# Common world claims 
MUI_COMM_WORLD = mui4py.mpi_split_by_app()
# Declare MPI ranks
rank = MUI_COMM_WORLD.Get_rank()
# Declare MPI size
size = MUI_COMM_WORLD.Get_size()
# Define MUI dimension
dimensionMUI = 3
# Define the name of push/fetch values
name_pushX = "deflectionX"
name_pushY = "deflectionY"
name_pushZ = "deflectionZ"
name_fetchX = "forceX"
name_fetchY = "forceY"
name_fetchZ = "forceZ"
# Define MUI push/fetch data types
data_types = {name_pushX: mui4py.FLOAT64,
                name_pushY: mui4py.FLOAT64,
                name_pushZ: mui4py.FLOAT64,
                name_fetchX: mui4py.FLOAT64,
                name_fetchY: mui4py.FLOAT64,
                name_fetchZ: mui4py.FLOAT64}
# MUI interface creation
domain = "femDomain"
config3d = mui4py.Config(dimensionMUI, mui4py.FLOAT64)
iface = ["couplingInterface"]
MUI_Interfaces = mui4py.create_unifaces(domain, iface, config3d)
MUI_Interfaces["couplingInterface"].set_data_types(data_types)

#Define parameters of the RBF sampler
rSampler = 0.6                                  # Define the search radius of the RBF sampler
basisFunc = 1                                   # Specify RBF basis function 0-Gaussian; 1-WendlandC0; 2-WendlandC2; 3-WendlandC4; 4-WendlandC6
conservative = True                             # Enable conservative OR consistent RBF form
cutOff = 1e-9                                   # Cut-off value for Gaussian RBF basis function
smoothFunc = False                              # Enable/disable RBF smoothing function during matrix creation
generateMatrix = True                           # Enable/disable writing of the matrix (if not reading)
cgSolveTol = 1e-6;                              # Conjugate Gradient solver tolerance
cgMaxIter = 500;                                # Conjugate Gradient solver maximum iterations (-1 = value determined by tolerance)
preconditioner = 1;                             # Preconditioner of Conjugate Gradient solver
pouSize = 50;                                   # RBF Partition of Unity patch size
rbfMatrixFolderX = "rbfMatrixX"                 # Output folder for the RBF matrix files
rbfMatrixFolderY = "rbfMatrixY"                 # Output folder for the RBF matrix files
rbfMatrixFolderZ = "rbfMatrixZ"                 # Output folder for the RBF matrix files

steps = 10                # number of time steps
iterations = 1            # number of iterations per step
r = 0.6                    # search radius
Nx = int(11)
Ny = int(((Nx-1)*5/2)+1)
Nz = int(1)
Npoints = Nx*Ny*Nz

local_x0 = 0
local_y0 = 0
local_z0 = 0
local_x1 = 2
local_y1 = 5
local_z1 = 0

# define push and fetch points and evaluation
interface_Point = np.zeros((Npoints, dimensionMUI))
c_0 = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                interface_Point[c_0] = [(local_x0 +((local_x1-local_x0)/(Nx-1))*i), (local_y0+((local_y1-local_y0)/(Ny-1))*j), 0.0]
            c_0 += 1

deflX = np.zeros(Npoints)
deflY = np.zeros(Npoints)
deflZ = np.zeros(Npoints)
forceX = np.zeros(Npoints)
forceY = np.zeros(Npoints)
forceZ = np.zeros(Npoints)
for i in range(Npoints):
    deflX[i] = 44.444
    deflY[i] = 55.555
    deflZ[i] = 66.666
    forceX[i] = 0.0
    forceY[i] = 0.0
    forceZ[i] = 0.0

# Declare list to store mui::point3d
point3dList = []

# Store mui::point3d that located in the fetch interface
c_0 = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                point_fetch = MUI_Interfaces["couplingInterface"].Point([interface_Point[c_0][0], interface_Point[c_0][1], interface_Point[c_0][2]])
                point3dList.append(point_fetch)
            c_0 += 1



# Define and announce MUI send/receive span
send_span = mui4py.geometry.Box([local_x0, local_y0, local_z0], [local_x1, local_y1, local_z1])
recv_span = mui4py.geometry.Box([local_x0, local_y0, local_z0], [local_x1, local_y1, local_z1])
MUI_Interfaces["couplingInterface"].announce_recv_span(0, steps, recv_span, False)
MUI_Interfaces["couplingInterface"].announce_send_span(0, steps, send_span, False)

# Spatial/temporal samplers
t_sampler = mui4py.TemporalSamplerExact()
s_samplerX = mui4py.SamplerRbf(rSampler, point3dList, basisFunc, conservative, smoothFunc, generateMatrix, rbfMatrixFolderX, cutOff, cgSolveTol, cgMaxIter, pouSize, preconditioner, MUI_COMM_WORLD)
s_samplerY = mui4py.SamplerRbf(rSampler, point3dList, basisFunc, conservative, smoothFunc, generateMatrix, rbfMatrixFolderY, cutOff, cgSolveTol, cgMaxIter, pouSize, preconditioner, MUI_COMM_WORLD)
s_samplerZ = mui4py.SamplerRbf(rSampler, point3dList, basisFunc, conservative, smoothFunc, generateMatrix, rbfMatrixFolderZ, cutOff, cgSolveTol, cgMaxIter, pouSize, preconditioner, MUI_COMM_WORLD)

# commit ZERO step
MUI_Interfaces["couplingInterface"].barrier(0)

for n in range(1, steps):
    for iter in range(iterations):

        if rank == 0:
            print("\n{dummy_FEM} Step ", n, "Iteration ", iter, flush=True)

        # MUI Fetch fluid forces at boundary points from SPH solver
        c_0 = 0
        force_integrationX = 0
        force_integrationY = 0
        force_integrationZ = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                        forceX[c_0]  = MUI_Interfaces["couplingInterface"].fetch("forceX", interface_Point[c_0], n, s_samplerX, t_sampler)
                        forceY[c_0]  = MUI_Interfaces["couplingInterface"].fetch("forceY", interface_Point[c_0], n, s_samplerY, t_sampler)
                        forceZ[c_0]  = MUI_Interfaces["couplingInterface"].fetch("forceZ", interface_Point[c_0], n, s_samplerZ, t_sampler)
                        force_integrationX += forceX[c_0]
                        force_integrationY += forceY[c_0]
                        force_integrationZ += forceZ[c_0]
                    c_0 += 1

        # FEM code update structure domain (omitted here)

        # MUI Push beam deflections at boundary points and commit current steps to SPH solver
        c_0 = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                        MUI_Interfaces["couplingInterface"].push("deflectionX", interface_Point[c_0], deflX[c_0])
                        MUI_Interfaces["couplingInterface"].push("deflectionY", interface_Point[c_0], deflY[c_0])
                        MUI_Interfaces["couplingInterface"].push("deflectionZ", interface_Point[c_0], deflZ[c_0])
                    c_0 += 1

        commit_return = MUI_Interfaces["couplingInterface"].commit(n)
        if (rank == 0):
            print ("{dummy_FEM} commit_return: ", commit_return)

        # MUI forget function
        if (n > 2):
            MUI_Interfaces["couplingInterface"].forget(n-2)

        # Print fetched values
        if (rank == 0):
            print ("{dummy_FEM} force integration fetched: ", force_integrationX, ", ", force_integrationY, ", ", force_integrationZ, " at timestep ", n )