#include "global.h"

int main(int argc, char** argv)
{
	phello();

#if DEBUG_MAIN
	printf("Debug mode turned on\n");
#endif
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#if 1
	printf("[rank %d] register the signal type and node info type for passing data to other ranks\n", world_rank);
#endif
	register_mpi_signal_type();
	register_mpi_node_info_type();

	if (argc != 3)
	{
		printf("you haven't pass the topological graph file and the number of nanoseconds to simulate\n");
		printf("we set file as \"small\" and 10 ns");
		argc = 3;
		argv[1] = "small";
		argv[2] = "10";
	}

	time_t t;
	time_t seed = time(&t);
	// Seed the random number generator
	srand((unsigned) seed);
#if DEBUG_MAIN
	printf("[rank %d] set the seed of random number generator to %ld\n", world_rank, seed);
#endif

	// Load brain map configuration from the file
#if DEBUG_MAIN
	printf("[rank %d] loading topological maps\n", world_rank);
#endif
	loadBrainGraph(argv[1]);

#if DEBUG_MAIN
	printf("[rank %d] Loaded brain graph file '%s'\n", world_rank, argv[1]);
#endif
	// Link the neurons to the edges in the data structure
	// every process load the file so that we don't need to pass complex struct to other ranks
	linkNodesToEdges();

	// apply the node to the current rank
	nodes_per_proc = num_brain_nodes / world_size;
	start_node = world_rank * nodes_per_proc;
	end_node = (world_rank == world_size - 1) ? num_brain_nodes : start_node + nodes_per_proc;

	// Initialise time
	time_t seconds = 0;
	time_t start_seconds = getCurrentSeconds();

	int num_ns_to_simulate = atoi(argv[2]);

	// Tracks performance data (number of iterations per ns)
	int total_iterations = 0, current_ns_iterations = 0, max_iteration_per_ns = -1, min_iteration_per_ns = -1;

	// main simulation loop
#if DEBUG_MAIN
	if (world_rank == 0)
	{
		printf("[rank %d] start simulation\n", world_rank);
	}
#endif

	while (elapsed_ns < num_ns_to_simulate)
	{
#if DEBUG_ITERATION_INFO
		printf("[rank %d] current elapsed nanoseconds: %d, current iteration: %d\n", world_rank, elapsed_ns, current_ns_iterations);
#endif

		time_t current_seconds = getCurrentSeconds();
		// First checks whether the time (in nanoseconds) needs to be updated
		if (current_seconds != seconds)
		{
			seconds = current_seconds;
			if (seconds - start_seconds > 0)
			{
				if ((seconds - start_seconds) % MIN_LENGTH_NS == 0)
				{
					if (elapsed_ns == 0)
					{
						max_iteration_per_ns = min_iteration_per_ns = current_ns_iterations;
					}
					else
					{
						if (current_ns_iterations > max_iteration_per_ns)
							max_iteration_per_ns = current_ns_iterations;
						if (current_ns_iterations < min_iteration_per_ns)
							min_iteration_per_ns = current_ns_iterations;
					}
					elapsed_ns++;
					current_ns_iterations = 0;
					for (int i = 0; i < num_brain_nodes; i++)
					{
						brain_nodes[i].signals_last_ns = brain_nodes[i].signals_this_ns;
						brain_nodes[i].signals_this_ns = 0;
					}
				}
			}
		}

#if DEBUG_MAIN
		printf("current elapsed nanoseconds: %d\n", elapsed_ns);
#endif

		MPI_Status status;
		int flag, recv_signals = 0;
#if DEBUG_MAIN
		printf("[rank %d] trying to recv signal\n", world_rank);
#endif
		do 
		{
			MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
			if (flag) 
			{
#if DEBUG_MPI_PROB
				printf("[rank %d] signal detected\n", world_rank);
#endif
				struct SignalStruct incoming;
				MPI_Recv(&incoming, 1, MPI_SignalType, status.MPI_SOURCE,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if DEBUG_MPI_PROB
				printf("[rank %d] recved signal\n", world_rank);

				print_signal(world_rank, &incoming);
#endif
				int found = 0;
				for (int i = start_node; i < end_node; ++i) 
				{
					if (brain_nodes[i].id == incoming.target_id) 
					{
						if (brain_nodes[i].num_outstanding_signals < SIGNAL_INBOX_SIZE) 
						{
							brain_nodes[i].signalInbox[brain_nodes[i].num_outstanding_signals++] = incoming;
						}
						found = 1;
						break;
					}
				}
				if (!found) 
				{
					fprintf(stderr, "Rank %d: Received signal for non-local node %d\n",
						world_rank, incoming.target_id);
				}
				recv_signals++;
			}
		} while (flag);

		for (int i = start_node; i < end_node; ++i)
		{
			updateNodes(i);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		current_ns_iterations++;
		total_iterations++;
	}
#if DEBUG_MAIN
	printf("[rank %d] simulation done\n", world_rank);

	if (world_rank == 0)
	{
		printf("[rank %d] start to generate report\n", world_rank);
	}
#endif
	
#if DEBUG_MAIN
	if (world_rank == 0)
	{
		printf("[rank %d] start to gather node infos from other ranks\n", world_rank);
	}
#endif

	// After calculating the local_node_info  
	int nodes_for_this_rank = end_node - start_node;
	struct NodeInfo* local_node_info = (struct NodeInfo*)malloc(nodes_for_this_rank * sizeof(struct NodeInfo));

	// Gather information for each node allocated to this rank  
	for (int i = 0; i < nodes_for_this_rank; ++i) {
		int global_index = start_node + i; 
		local_node_info[i].id = brain_nodes[global_index].id;
		local_node_info[i].node_type = brain_nodes[global_index].node_type;
		local_node_info[i].total_signal_recved = brain_nodes[global_index].total_signals_recieved;

		// Count the nerve inputs and outputs for each signal type  
		for (int j = 0; j < NUM_SIGNAL_TYPES; ++j) {
			local_node_info[i].num_nerve_inputs[j] = brain_nodes[global_index].num_nerve_inputs[j];
			local_node_info[i].num_nerve_outputs[j] = brain_nodes[global_index].num_nerve_outputs[j];
		}
	}

	// Prepare to receive the gathered node info in the root rank  
	struct NodeInfo* gathered_node_info = NULL;
	int* recv_counts = NULL;
	int* displs = NULL;

#if DEBUG_MAIN
	{
		printf("[rank %d] print some node info\n", world_rank);
		int cnt = 10;
		for (int i = 0; i < cnt; i++)
		{
			int idx = getRandomInteger(0, nodes_for_this_rank);
			print_node_info(world_rank, &local_node_info[idx]);
		}
	}
#endif

	if (world_rank == 0) {
		gathered_node_info = (struct NodeInfo*)malloc(num_brain_nodes * sizeof(struct NodeInfo));
		recv_counts = (int*)malloc(world_size * sizeof(int));
		displs = (int*)malloc(world_size * sizeof(int));

		// Calculate the sizes to receive  
		for (int i = 0; i < world_size; ++i) {
			int start = i * (num_brain_nodes / world_size);
			int end = (i == world_size - 1) ? num_brain_nodes : (i + 1) * (num_brain_nodes / world_size);
			recv_counts[i] = end - start; 
			displs[i] = (i == 0) ? 0 : (displs[i - 1] + recv_counts[i - 1]); 
		}
	}

	// Use MPI_Gatherv to collect the node information since nodes can vary per rank  
	MPI_Gatherv(local_node_info, nodes_for_this_rank, MPI_NodeInfoType,
		gathered_node_info, recv_counts, displs, MPI_NodeInfoType,
		0, MPI_COMM_WORLD);

#if DEBUG_MAIN
	{
		if (gathered_node_info != NULL)
		{
			printf("[rank %d] print some node info\n", world_rank);
			int cnt = 10;
			for (int i = 0; i < cnt; i++)
			{
				int idx = getRandomInteger(0, nodes_for_this_rank);
				print_node_info(world_rank, &gathered_node_info[idx]);
			}
		}
	}
#endif

	// Perform the report generation only for rank 0  
	if (world_rank == 0) {
		// Generate the report with the gathered information  
		generateReport(OUTPUT_REPORT_FILENAME, gathered_node_info);
		// Clean up the gathered memory after use  
		free(gathered_node_info);
		free(recv_counts);
		free(displs);
	}

#if OUTPUT_INFO 1
	// Highlight this has finished and report performance
	printf("Finished after %d ns, full report written to `%s` file\n", elapsed_ns, OUTPUT_REPORT_FILENAME);
	printf("Performance data: %d total iterations, maximum %d iterations per nanosecond and minimum %d iterations per nanosecond\n",
		total_iterations, max_iteration_per_ns, min_iteration_per_ns);
#endif

	free(local_node_info);

	mpi_finalize();
	return 0;
}