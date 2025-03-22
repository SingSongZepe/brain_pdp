// Parallelised via geometric decomposition using blocking communications

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

// Boundary value at the LHS of the bar
#define LEFT_VALUE 1.0
// Boundary value at the RHS of the bar
#define RIGHT_VALUE 10.0
// The maximum number of iterations
#define MAX_ITERATIONS 100000
// How often to report the norm
#define REPORT_NORM_PERIOD 100

void initialise(double*, double*, int, int);

int main(int argc, char * argv[]) {
	int size, myrank, nx, ny, max_its;
	double convergence_accuracy;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if (argc != 5) {
		if (myrank==0) {
			printf("You should provide four command line arguments, the global size in X, the global size in Y, convergence accuracy and max number iterations\n");		
			printf("In the absence of this defaulting to x=128, y=1024, convergence=3e-3, no max number of iterations\n");
		}
		nx=128;
		ny=1024;
		convergence_accuracy=3e-3;
		max_its=0;
	} else {
		nx=atoi(argv[1]);
		ny=atoi(argv[2]);
		convergence_accuracy=atof(argv[3]);
		max_its=atoi(argv[4]);
	}

#ifdef INSTRUMENTED
	if (max_its < 1 || max_its > 100) {
		max_its=100;
		if (myrank==0) printf("Limiting the instrumented run to 100 iterations to keep file size small, you can change this in the code if you really want\n");
	}
#endif

	if (myrank==0) {
    		printf("Number Processes in X=%d\n", size);
    		printf("Global size in X=%d, Global size in Y=%d\n\n", nx, ny);
  	}

	int local_nx=nx/size;
  	if (local_nx * size < nx) {
    		if (myrank < nx - local_nx * size) local_nx++;
  	}

	int mem_size_x=local_nx+2;
	int mem_size_y=ny+2;

	double * u_k = malloc(sizeof(double) * mem_size_x * mem_size_y);
	double * u_kp1 = malloc(sizeof(double) * mem_size_x * mem_size_y);
	double * temp;
	double start_time;

	initialise(u_k, u_kp1, local_nx, ny);

	double rnorm=0.0, bnorm=0.0, norm, tmpnorm=0.0;

	int i,j,k;
	// Calculate the initial residual norm
	for (j=1;j<=local_nx;j++) {
		for (i=1;i<=ny;i++) {
			tmpnorm=tmpnorm+pow(u_k[i+(j*mem_size_y)]*4-u_k[(i-1)+(j*mem_size_y)]-
					u_k[(i+1)+(j*mem_size_y)]-u_k[i+((j-1)*mem_size_y)]-u_k[i+((j+1)*mem_size_y)], 2);
		}
	}
	MPI_Allreduce(&tmpnorm, &bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	bnorm=sqrt(bnorm);

	start_time=MPI_Wtime();
	for (k=0;k<MAX_ITERATIONS;k++) {
		if (myrank > 0) {
      			MPI_Sendrecv(&u_k[1+mem_size_y], ny, MPI_DOUBLE, myrank-1, 0, &u_k[1], ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    		}
    		if (myrank < size-1) {
      			MPI_Sendrecv(&u_k[(local_nx*mem_size_y)+1], ny, MPI_DOUBLE, myrank+1, 0, &u_k[((local_nx+1)*mem_size_y)+1], ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		 }

    		tmpnorm=0.0;
		// Calculates the current residual norm
		for (j=1;j<=local_nx;j++) {
			for (i=1;i<=ny;i++) {
				tmpnorm=tmpnorm+pow(u_k[i+(j*mem_size_y)]*4-u_k[(i-1)+(j*mem_size_y)]-
						u_k[(i+1)+(j*mem_size_y)]-u_k[i+((j-1)*mem_size_y)]-u_k[i+((j+1)*mem_size_y)], 2);
			}
		}
		MPI_Allreduce(&tmpnorm, &rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norm=sqrt(rnorm)/bnorm;

		if (norm < convergence_accuracy) break;
		if (max_its > 0 && k >= max_its) break;

		// Do the Jacobi iteration
		for (j=1;j<=local_nx;j++) {
			for (i=1;i<=ny;i++) {
				u_kp1[i+(j*mem_size_y)]=0.25 * (u_k[(i-1)+(j*mem_size_y)]+u_k[(i+1)+(j*mem_size_y)]+
						u_k[i+((j-1)*mem_size_y)]+u_k[i+((j+1)*mem_size_y)]);
			}
		}
		// Swap data structures round for the next iteration
		temp=u_kp1; u_kp1=u_k; u_k=temp;

		if (k % REPORT_NORM_PERIOD == 0 && myrank == 0) printf("Iteration= %d Relative Norm=%e\n", k, norm);
	}
	if (myrank == 0) printf("\nTerminated on %d iterations, Relative Norm=%e, Total time=%e seconds\n", k, norm,
			MPI_Wtime() - start_time);
	free(u_k);
	free(u_kp1);
	MPI_Finalize();
	return 0;
}

/**
 * Initialises the arrays, such that u_k contains the boundary conditions at the start and end points and all other
 * points are zero. u_kp1 is set to equal u_k
 */
void initialise(double * u_k, double * u_kp1, int nx, int ny) {
	int i,j;
	// We are setting the boundary (left and right) values here, in the parallel version this should be exactly the same and no changed required
	for (i=0;i<nx+1;i++) {
		u_k[i*(ny+2)]=LEFT_VALUE;
		u_k[(ny+1)+(i*(ny+2))]=RIGHT_VALUE;
	}
	for (j=0;j<=nx+1;j++) {
		for (i=1;i<=ny;i++) {
			u_k[i+(j*(ny+2))]=0.0;
		}
	}
	for (j=0;j<=nx+1;j++) {
		for (i=0;i<=ny+1;i++) {
			u_kp1[i+(j*(ny+2))]=u_k[i+(j*(ny+2))];
		}
	}
}
