/**
 * @file: dummy_SPH_Standalone.cpp
 * @brief: dummy SPH C++ code standalone as a reference for the coupling 
 *         between SPH and FEM solvers.
 *         A 2-D flexible beam, which is 5m higt and 2m thick, clamped at
 *         the bottom. SPH code supposed to calculate fluid forces acting
 *         on the beam.
 *         Note: internal particles/points are omitted for simplicity.
 *
 *                                 
 *                                 
 *      (0,5,0)         (2,5,0)    
 * SPH:        Q Q Q Q Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *             Q o o o Q           
 *     (0,0,0) Q o o o Q (2,0,0)   
 *    ---------------------------  
 *    //////////////////////////   
 *                                 
 *                                 
 * 
 * Q: SPH interface particles
 * o: SPH internal particles
 * 
 * USAGE: mpirun -np 1 ./dummy_SPH_Standalone.x
 * 
 */

#include <iostream>

int main(int argc, char ** argv) {

	// setup parameters
    constexpr static int    Nx        = 5;              // number of particles in x axis
    constexpr static int    Ny        = ((Nx-1)*5/2)+1; // number of particles in y axis
    constexpr static int    Nz        = 1;              // number of particles in z axis
	const char* name_pushX = "forceX";		
	const char* name_pushY = "forceY";		
	const char* name_pushZ = "forceZ";		
	const char* name_fetchX = "deflectionX";
	const char* name_fetchY = "deflectionY";
	const char* name_fetchZ = "deflectionZ";
    double r    = 1;                      // search radius	
    int Nt = Nx * Ny * Nz; // total time steps	
    int steps = 10; // total time steps
	double local_x0 = 0; // local origin
    double local_y0 = 0;
	double local_z0 = 0;
    double local_x1 = 2;
    double local_y1 = 5;
	double local_z1 = 0;
    double interfacePoint[Nx][Ny][Nz][3], forceX[Nx][Ny][Nz], forceY[Nx][Ny][Nz], forceZ[Nx][Ny][Nz], deflX[Nx][Ny][Nz], deflY[Nx][Ny][Nz], deflZ[Nx][Ny][Nz];

	// interface points generation and evaluation
	for ( int i = 0; i < Nx; ++i ) {
        for ( int j = 0; j < Ny; ++j ) {
			for ( int k = 0; k < Nz; ++k ) {
				if ((i==0) || (i==(Nx-1)) || (j==(Ny-1))) {
					interfacePoint[i][j][k][0] = local_x0 + ((local_x1-local_x0) / (Nx - 1)) * i;
					interfacePoint[i][j][k][1] = local_y0 + ((local_y1-local_y0) / (Ny - 1)) * j;
					interfacePoint[i][j][k][2] = 0;

					forceX[i][j][k] = 0.0;
					forceY[i][j][k] = 0.0;
					forceZ[i][j][k] = 0.0;

					deflX[i][j][k] = 0.0;
					deflY[i][j][k] = 0.0;
					deflZ[i][j][k] = 0.0;
				}
			}
        }
	}

	// Begin time loops
    for ( int n = 1; n < steps; ++n ) {

		std::cout << std::endl;
		std::cout << "{dummy_SPH} " << n << " Step" << std::endl;
		std::cout << std::endl;

		// SPH fluid domain update
		double force_integrationX = 0.0;
		double force_integrationY = 0.0;
		double force_integrationZ = 0.0;

		for ( int i = 0; i < Nx; ++i ) {
	        for ( int j = 0; j < Ny; ++j ) {
				for ( int k = 0; k < Nz; ++k ) {
					if ((i==0) || (i==(Nx-1)) || (j==(Ny-1))) {
						forceX[i][j][k] = 11.111;
						forceY[i][j][k] = 22.222;
						forceZ[i][j][k] = 33.333;

						force_integrationX += forceX[i][j][k];
						force_integrationY += forceY[i][j][k];
						force_integrationZ += forceZ[i][j][k];

						deflX[i][j][k] = 44.444;
						deflY[i][j][k] = 55.555;
						deflZ[i][j][k] = 66.666;
					}
				}
	        }
		}

		// print the fetched beam deflextions
		std::cout << "{dummy_SPH} deflection [0][0][0]: " << deflX[0][0][0] << ", " << deflY[0][0][0] << ", " << deflZ[0][0][0] << " at timestep " << n << std::endl;
		std::cout << "{dummy_SPH} force integration pushed: " << force_integrationX << ", "<< force_integrationY << ", " << force_integrationZ << " at timestep " << n << std::endl;
	}

    return 0;
}
