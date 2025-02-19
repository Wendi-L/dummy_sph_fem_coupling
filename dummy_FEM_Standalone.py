"""
#
# @file: dummy_FEM_Standalone.py
# @brief: dummy FEM Python Standalone code as a reference for the coupling 
#         between SPH and FEM solvers.
#         A 3-D flexible beam, which is 5m height, 5m deep and 2m thick, clamped at
#         the bottom. FEM code supposed to calculate the beam deflections.
#         Note: internal particles/points are omitted for simplicity.
#
# FEM:     (0,5,5)  +-+-+-+-+ (2,5,5)
#                 +-+-+-+-+ +
#               +-+-+-+-+   +
#     (0,5,0) +-+-+-+-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     +
#             +-*-*-*-+     + (2,0,5)
#             +-*-*-*-+   +
#             +-*-*-*-+ +
#     (0,0,0) +-*-*-*-+ (2,0,0)
#    ---------------------------
#    //////////////////////////
#
#
# +: FEM interface grid points
# *: FEM internal grid points
# 
# USAGE: mpirun -np 1 python3 -m mpi4py dummy_FEM_Standalone.py
# 
#/
"""

from mpi4py import MPI
import mpi4py
import datetime
import numpy as np
import time
import os

steps = 100                # number of time steps
iterations = 1            # number of iterations per step
r = 0.6                    # search radius
Nx = int(11)
Ny = int(((Nx-1)*5/2)+1)
Nz = int(((Nx-1)*5/2)+1)
Npoints = Nx*Ny*Nz

local_x0 = 0
local_y0 = 0
local_z0 = 0
local_x1 = 2
local_y1 = 5
local_z1 = 5

# define interface points and evaluation
interface_Point = np.zeros((Npoints, 3))
c_0 = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                interface_Point[c_0] = [(local_x0 +((local_x1-local_x0)/(Nx-1))*i), (local_y0+((local_y1-local_y0)/(Ny-1))*j), (local_z0+((local_z1-local_z0)/(Nz-1))*k)]
            c_0 += 1

deflX = np.zeros(Npoints)
deflY = np.zeros(Npoints)
deflZ = np.zeros(Npoints)
forceX = np.zeros(Npoints)
forceY = np.zeros(Npoints)
forceZ = np.zeros(Npoints)

for n in range(1, steps):
    for iter in range(iterations):

        print("\n{dummy_FEM} Step ", n, "Iteration ", iter, flush=True)

        # FEM code update structure domain
        force_integrationX = 0
        force_integrationY = 0
        force_integrationZ = 0

        c_0 = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if ((i==0) or (i==(Nx-1)) or (j==(Ny-1))) :
                        forceX[i] = 11.111
                        forceY[i] = 22.222
                        forceZ[i] = 33.333

                        force_integrationX += forceX[c_0]
                        force_integrationY += forceY[c_0]
                        force_integrationZ += forceZ[c_0]

                    c_0 += 1

        for i in range(Npoints):
            deflX[i] = 44.444
            deflY[i] = 55.555
            deflZ[i] = 66.666

        # Print values
        print ("{dummy_FEM} force integration fetched: ", force_integrationX, ", ", force_integrationY, ", ", force_integrationZ, " at timestep ", n )