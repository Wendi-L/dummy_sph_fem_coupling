/**
 * @file: dummy_SPH_MUI.cpp
 * @brief: dummy SPH C++ code for the coupling between SPH and FEM solvers.
 *         A 2-D flexible beam, which is 5m higt and 2m thick, clamped at
 *         the bottom. SPH code supposed to calculate fluid forces acting
 *         on the beam, while the FEM code supposed to calculate the beam
 *         deflections. MUI is used to pass forces of SPH interface particles
 *         from SPH code to FEM code, and pass deflections of FEM interface
 *         grid points from FEM code to SPH code.
 *         Note: internal particles/points are omitted for simplicity.
 *
 *                                 |
 *                                 |
 *      (0,5,0)         (2,5,0)    |      (0,5,0)         (2,5,0)
 * SPH:        Q Q Q Q Q           | FEM:        +-+-+-+-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *             Q o o o Q           |             +-*-*-*-+
 *     (0,0,0) Q o o o Q (2,0,0)   |     (0,0,0) +-*-*-*-+ (2,0,0)
 *    ---------------------------  |    ---------------------------
 *    //////////////////////////   |    //////////////////////////
 *                                 |
 *                                 |
 * 
 * Q: SPH interface particles
 * o: SPH internal particles
 * +: FEM interface grid points
 * *: FEM internal grid points
 * 
 * USAGE: mpirun -np 1 ./dummy_SPH_MUI.x : -np 1 python3 -m mpi4py dummy_FEM_MUI.py
 * 
 */

#include "mui.h"
#include "mui_config.h"

int main(int argc, char ** argv) {

	using namespace mui;

	/// Declare MPI common world with the scope of MUI
	MPI_Comm  world = mui::mpi_split_by_app();

	/// Declare MPI ranks and rank size
	int rank, size;
	MPI_Comm_rank( world, &rank );
	MPI_Comm_size( world, &size );

	if (size > 2) {
	  std::cout << "MPI Size larger than 2 does not supported in this demo case yet." << std::endl;
			  exit(EXIT_FAILURE);
	}

	/// Define the name of MUI domain
	std::string domain = "sphDomain";

	/// Define the name of MUI interfaces
	std::vector<std::string> interfaces;
	interfaces.emplace_back( "couplingInterface" );

	/// Declare MUI objects using MUI configure file
	auto ifs = mui::create_uniface<mui_config>( domain, interfaces );

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

	// Push & Fetch points generation and evaluation
	for ( int i = 0; i < Nx; ++i ) {
        for ( int j = 0; j < Ny; ++j ) {
			for ( int k = 0; k < Nz; ++k ) {
				if ((i==0) || (i==(Nx-1)) || (j==(Ny-1))) {
					interfacePoint[i][j][k][0] = local_x0 + ((local_x1-local_x0) / (Nx - 1)) * i;
					interfacePoint[i][j][k][1] = local_y0 + ((local_y1-local_y0) / (Ny - 1)) * j;
					interfacePoint[i][j][k][2] = 0;

					forceX[i][j][k] = 11.111;
					forceY[i][j][k] = 22.222;
					forceZ[i][j][k] = 33.333;

					deflX[i][j][k] = 0.0;
					deflY[i][j][k] = 0.0;
					deflZ[i][j][k] = 0.0;
				}
			}
        }
	}

    // annouce send span
    geometry::box<mui_config> send_region( {local_x0, local_y0, local_z0}, {local_x1, local_y1, local_z1} );
    geometry::box<mui_config> recv_region( {local_x0, local_y0, local_z0}, {local_x1, local_y1, local_z1} );
    printf( "{dummy_SPH} send region for rank %d: %lf %lf %lf - %lf %lf %lf\n", rank, local_x0, local_y0, local_z0, local_x1, local_y1, local_z1 );
    ifs[0]->announce_send_span( 0, steps, send_region );
    ifs[0]->announce_recv_span( 0, steps, recv_region );

	// define spatial and temporal samplers
    sampler_pseudo_nearest_neighbor<mui_config> s1(r);
	temporal_sampler_exact<mui_config>  s2;

	// commit ZERO step
	ifs[0]->commit(0);

	// Begin time loops
    for ( int n = 1; n < steps; ++n ) {

		printf("\n");
		printf("{dummy_SPH} %d Step", n );
		printf("\n");

		// SPH fluid domain update (omitted here)

		// push fluid forces to the FEM solver
		double force_integrationX = 0.0;
		double force_integrationY = 0.0;
		double force_integrationZ = 0.0;

		for ( int i = 0; i < Nx; ++i ) {
			for ( int j = 0; j < Ny; ++j ) {
				for ( int k = 0; k < Nz; ++k ) {
					if ((i==0) || (i==(Nx-1)) || (j==(Ny-1))) {
						point3d locp( interfacePoint[i][j][k][0], interfacePoint[i][j][k][1], interfacePoint[i][j][k][2] );
						ifs[0]->push( name_pushX, locp, forceX[i][j][k] );
						ifs[0]->push( name_pushY, locp, forceY[i][j][k] );
						ifs[0]->push( name_pushZ, locp, forceZ[i][j][k] );
						force_integrationX += forceX[i][j][k];
						force_integrationY += forceY[i][j][k];
						force_integrationZ += forceZ[i][j][k];
					}
				}
			}
		}
		// commit at time step 'n'
		int sent = ifs[0]->commit( n );

		// fetch beam deflections from the FEM solver
		for ( int i = 0; i < Nx; ++i ) {
			for ( int j = 0; j < Ny; ++j ) {
				for ( int k = 0; k < Nz; ++k ) {
					if ((i==0) || (i==(Nx-1)) || (j==(Ny-1))) {
						point3d locf( interfacePoint[i][j][k][0], interfacePoint[i][j][k][1], interfacePoint[i][j][k][2] );
						deflX[i][j][k] = ifs[0]->fetch( name_fetchX, locf, n, s1, s2 );
						deflY[i][j][k] = ifs[0]->fetch( name_fetchY, locf, n, s1, s2 );
						deflZ[i][j][k] = ifs[0]->fetch( name_fetchZ, locf, n, s1, s2 );
					}
				}
			}
		}

		// print the fetched beam deflextions
		printf( "{dummy_SPH} deflection [0][0][0]: %lf, %lf, %lf at timestep %d\n", deflX[0][0][0], deflY[0][0][0], deflZ[0][0][0], n );
		printf( "{dummy_SPH} force integration pushed: %lf, %lf, %lf at timestep %d\n", force_integrationX, force_integrationY, force_integrationZ, n );

	}


    return 0;
}
