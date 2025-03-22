! Parallelised via geometric decomposition using non-blocking communications and overlapping where possible
module problem_mod	
	implicit none
	include "mpif.h"

	! Boundary value at the LHS of the pipe
	real(kind=8), parameter :: LEFT_VALUE = 1.0
	! Boundary value at the RHS of the pipe
	real(kind=8), parameter :: RIGHT_VALUE = 10.0
	! The maximum number of iterations
	integer, parameter :: MAX_ITERATIONS = 100000
	! How often to report the norm
	integer, parameter :: REPORT_NORM_PERIOD = 100

contains

	subroutine run_solver()
		real(kind=8), dimension(:,:), allocatable :: u_k, u_kp1, temp
		real(kind=8) :: bnorm, rnorm, norm, tmpnorm, convergence_accuracy
		integer :: i, j, k, ierr, size, myrank, nx, ny, max_its, local_nx, requests(4), norm_req
		character(len=32) :: arg
		double precision :: start_time

		call mpi_init(ierr)  ! We just have this so we can use MPI Wtime in the serial code for timing
		call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
		call mpi_comm_rank(MPI_COMM_WORLD, myrank, ierr)

		if (command_argument_count() /= 4) then
			if (myrank == 0) then
				print *, &
					"You should provide four command line arguments, the global size in X,", &
					"the global size in Y, convergence accuracy and max number iterations"
				print *, "In the absence of this defaulting to x=128, y=1024, convergence=3e-3, no max number of iterations"
			end if
			nx=128
			ny=1024
			convergence_accuracy=3e-3
			max_its=0
		else
			call get_command_argument(1, arg)
			read(arg,*) nx
			call get_command_argument(2, arg)
			read(arg,*) ny
			call get_command_argument(3, arg)
			read(arg,*) convergence_accuracy
			call get_command_argument(4, arg)
			read(arg,*) max_its
		end if

#ifdef INSTRUMENTED
		! Later in the exercise we will be tracing the execution, for large numbers of iterations this can result in large file sizes - hence we have a safety 
		! check here and limit the number of iterations in this case (which still illustrates the parallel behaviour we are interested in.)
		if (max_its .lt. 1 .or. max_its .gt. 100) then
			max_its=100
			print *, "Limiting the instrumented run to 100 iterations to keep file size small, ", &
					"you can change this in the code if you really want"
		end if
#endif

		if (myrank == 0) then
			print *, "Number of processes in X=", size
			print *, "Global size in X=", nx, "Global size in Y=", ny
		end if

		local_nx=nx/size
		if (local_nx * size .lt. nx) then
			if (myrank .lt. nx - local_nx * size) local_nx=local_nx+1
		end if

		allocate(u_k(0:ny+1, 0:local_nx+1), u_kp1(0:ny+1, 0:local_nx+1))

		bnorm=0.0
		tmpnorm=0.0
		rnorm=0.0

		requests=MPI_REQUEST_NULL
    
		call initialise_values(u_k, u_kp1, local_nx, ny)		

		! Calculate the initial residual
		do i=1, local_nx
			do j=1, ny
				tmpnorm=tmpnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
			end do
		end do
		call MPI_Allreduce(tmpnorm, bnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
		bnorm=sqrt(bnorm)

		start_time=MPI_Wtime()
		do k=0, MAX_ITERATIONS      
			if (myrank .gt. 0) then
				call MPI_Isend(u_k(1,1), ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(1), ierr)
				call MPI_Irecv(u_k(1,0), ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(2), ierr)
			end if
			if (myrank .lt. size-1) then
				call MPI_Isend(u_k(1,local_nx), ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(3), ierr)
				call MPI_Irecv(u_k(1,local_nx+1), ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(4), ierr)
			end if			

			tmpnorm=0.0
			! Calculates the current residual norm
			do i=2, local_nx-1
				do j=1, ny
					tmpnorm=tmpnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
				end do
			end do
			call MPI_Waitall(4, requests, MPI_STATUSES_IGNORE, ierr)

			i=1
			do j=1, ny
				tmpnorm=tmpnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
			end do
			i=local_nx
			do j=1, ny
				tmpnorm=tmpnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
			end do

			call MPI_Iallreduce(tmpnorm, rnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, norm_req, ierr)
			if (max_its .gt. 0 .and. k .ge. max_its) then
				call MPI_Wait(norm_req, MPI_STATUS_IGNORE, ierr)
				exit
			end if			

			! Do the Jacobi iteration
			do i=1, local_nx
				do j=1, ny
					u_kp1(j,i)=0.25 * (u_k(j-1,i) + u_k(j+1,i) + u_k(j,i-1) + u_k(j,i+1))
				end do
			end do

			call MPI_Wait(norm_req, MPI_STATUS_IGNORE, ierr)
			norm=sqrt(rnorm)/bnorm
			if (norm .lt. convergence_accuracy) exit			

			! Swap data structures round for the next iteration
			call move_alloc(u_kp1, temp)
            		call move_alloc(u_k, u_kp1)
            		call move_alloc(temp, u_k)

			if (mod(k, REPORT_NORM_PERIOD)==0 .and. myrank == 0) print *, "Iteration=",k," Relative Norm=",norm
		end do
		if (myrank == 0) then
			print *, "Terminated on ",k," iterations, Relative Norm=", norm
			print *, "Total runtime is ", MPI_Wtime()-start_time, " seconds"
		end if
		deallocate(u_k, u_kp1)
		call mpi_finalize(ierr)
	end subroutine run_solver

	! Initialises the arrays, such that u_k contains the boundary conditions at the start and end points and all other
	! points are zero. u_kp1 is set to equal u_k
	subroutine initialise_values(u_k, u_kp1, nx, ny)
		real(kind=8), intent(inout) :: u_k(0:ny+1, 0:nx+1), u_kp1(0:ny+1, 0:nx+1)
		integer, intent(in) :: nx, ny

		! We are setting the boundary (left and right) values here, in the parallel version this should be exactly the same and no changed required

		u_k(0,:)=LEFT_VALUE
		u_k(ny+1,:)=RIGHT_VALUE
		u_k(1:ny,:)=0.0_8
		u_kp1=u_k
	end subroutine initialise_values  
end module problem_mod

program diffusion
	use problem_mod
	implicit none

	call run_solver()
end program diffusion
