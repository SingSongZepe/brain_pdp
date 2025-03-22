#include "global.h"

// for debugging
//#define DEBUG
#define DEBUG_MPI_PROB 0
#define DEBUG_GENERATE_REPORT 1
#define DEBUG_PRINT_NODE_MORE 0


int main(int argc, char* argv[]) {
    argc = 3;
    argv[1] = "small";
    argv[2] = "10";
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    register_mpi_signal_type();

    register_mpi_neuron_nerve_type();

#ifdef DEBUG
    printf("[rank: %d] Hello World from %d\n", world_rank, world_rank);
#endif

    if (argc != 3) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s <brain_graph_file> <num_ns>\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

#ifdef DEBUG
    printf("[rank: %d] load the topological graph\n", world_rank);
#endif 
    loadBrainGraph(argv[1]);
    mpi_finalize();
    return 0;
    linkNodesToEdges();

    mpi_finalize();
    return 0;

#ifdef DEBUG
    printf("[rank: %d] load the graph done\n", world_rank);
#endif 

    nodes_per_proc = num_brain_nodes / world_size;
    start_node = world_rank * nodes_per_proc;
    end_node = (world_rank == world_size - 1)? num_brain_nodes : start_node + nodes_per_proc;
#ifdef DEBUG
    printf("[rank: %d]: start_node: %d, end_node: %d\n", world_rank, start_node, end_node);
#endif

    // Initialise time
    time_t seconds = 0;
    time_t start_seconds = getCurrentSeconds();

    int num_ns_to_simulate = atoi(argv[2]);

    // Tracks performance data (number of iterations per ns)
    int total_iterations = 0, current_ns_iterations = 0, max_iteration_per_ns = -1, min_iteration_per_ns = -1;

#ifdef DEBUG
    printf("[rank: %d] start to iterate\n", world_rank);
#endif

    while (elapsed_ns < num_ns_to_simulate) {
#ifdef DEBUG
        printf("[rank: %d] total iteration: %d\n", world_rank, total_iterations);
#endif

        //time_t current_seconds = getCurrentSeconds();
        //// First checks whether the time (in nanoseconds) needs to be updated
        //if (current_seconds != seconds)
        //{
        //    seconds = current_seconds;
        //    if (seconds - start_seconds > 0)
        //    {
        //        if ((seconds - start_seconds) % MIN_LENGTH_NS == 0)
        //        {
        //            if (elapsed_ns == 0)
        //            {
        //                max_iteration_per_ns = min_iteration_per_ns = current_ns_iterations;
        //            }
        //            else
        //            {
        //                if (current_ns_iterations > max_iteration_per_ns)
        //                    max_iteration_per_ns = current_ns_iterations;
        //                if (current_ns_iterations < min_iteration_per_ns)
        //                    min_iteration_per_ns = current_ns_iterations;
        //            }
        //            elapsed_ns++;
        //            current_ns_iterations = 0;
        //            for (int i = 0; i < num_brain_nodes; i++)
        //            {
        //                brain_nodes[i].signals_last_ns = brain_nodes[i].signals_this_ns;
        //                brain_nodes[i].signals_this_ns = 0;
        //            }
        //        }
        //    }
        //}

        MPI_Status status;
        int flag, received_signals = 0;

#if DEBUG_MPI_PROB
        printf("[rank: %d] trying to recv signal\n", world_rank);
#endif
        do {
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
#if DEBUG_MPI_PROB
                printf("[rank: %d] signal detected\n", world_rank);
#endif
                struct SignalStruct incoming;
                MPI_Recv(&incoming, 1, MPI_SignalType, status.MPI_SOURCE,
                    0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if DEBUG_MPI_PROB
                printf("[rank: %d] recved signal\n", world_rank);

                print_signal(&incoming);
#endif
                int found = 0;
                for (int i = start_node; i < end_node; ++i) {
                    if (brain_nodes[i].id == incoming.target_id) {
                        if (brain_nodes[i].num_outstanding_signals < SIGNAL_INBOX_SIZE) {
                            brain_nodes[i].signalInbox[brain_nodes[i].num_outstanding_signals++] = incoming;
                        }
                        else {
                            brain_nodes[i].signals_dropped++;
                        }
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    fprintf(stderr, "Rank %d: Received signal for non-local node %d\n",
                        world_rank, incoming.target_id);
                }
                received_signals++;
            }
        } while (flag);
#if DEBUG_MPI_PROB
        printf("[rank: %d] recv signal done\n", world_rank);
#endif
#ifdef DEBUG
        printf("[rank: %d] start to update nodes\n", world_rank);
#endif

        for (int i = start_node; i < end_node; ++i) {
            updateNodes(i);
        }
#ifdef DEBUG
        printf("update nodes done\n");
#endif
        MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
        printf("sync all processes done\n");
#endif 

        current_ns_iterations++;
        total_iterations++;
    }
    mpi_finalize();
    return 0;

#if 1
    printf("[rank: %d] simulation done\n", world_rank);

    printf("start to write file\n");
#endif


    struct NeuronNerveStruct* global_brain_nodes = NULL;
    if (world_rank == 0) {
        global_brain_nodes = malloc(num_brain_nodes * sizeof(struct NeuronNerveStruct));
#if 1
        printf("[rank %d] initialized the global_brain_nodes\n", world_rank);
#endif
    }

    int* recvcounts = NULL;
    int* displs = NULL;

    // 在main函数中：
    if (world_rank == 0) {
        recvcounts = malloc(world_size * sizeof(int));
        displs = malloc(world_size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            int start = i * nodes_per_proc;
            int end = (i == world_size - 1) ? num_brain_nodes : start + nodes_per_proc;
            recvcounts[i] = end - start;  // 元素个数，非字节数
            displs[i] = offset;
            offset += recvcounts[i];
        }
#if 0
        printf("[rank %d] recvcounts, displs initialized\n", world_rank);
#endif
    }

#if 1
    printf("[rank %d] gathering global brain nodes\n", world_rank);
#endif

    MPI_Gatherv(
        &brain_nodes[start_node],  // 发送起始地址
        end_node - start_node,     // 发送元素个数
        MPI_NeuronNerveType,       // 自定义数据类型
        global_brain_nodes,        // 接收缓冲区
        recvcounts,                // 每个进程接收的元素个数
        displs,                    // 位移（元素个数）
        MPI_NeuronNerveType,       // 接收数据类型
        0, MPI_COMM_WORLD
    );


#if 1
    printf("[rank %d] gather global brian nodes done\n", world_rank);
#endif


#if 1 // random print some node
    if (world_rank == 0 && global_brain_nodes == NULL)
    {
        printf("global_brain_nodes is NULL\n");
    }
#endif

#if 0 // random print some node
    int kk;
    for (kk = 0; kk < 30; kk++)
    {
        int idx = getRandomInteger(0, num_brain_nodes);
        print_brain_node(&global_brain_nodes[idx]);
    }
#endif

#if 0 // check the global brain nodes
    int ll = 0;
    for (ll = 0; ll < 30; ll++) {
        int idx = getRandomInteger(0, num_brain_nodes);
        //print_brain_node(&global_brain_nodes[idx]);
        
        int rank = idx / nodes_per_proc;
        if (world_rank == rank)
        {
            print_brain_node(&brain_nodes[idx]);
        }
    }
#endif

    if (world_rank == 0) {
        generateReport(OUTPUT_REPORT_FILENAME, brain_nodes);
#if 1
        printf("generating report\n");
#endif
        free(global_brain_nodes);
        free(recvcounts);
        free(displs);
    }    

    mpi_finalize();
    return 0;
}


static void fireSignal(int node_idx, float signal, int signal_type) {
#ifdef DEBUG
    printf("[rank: %d] fire signal: node: %d, signal strength %f, signal_type: %d\n", world_rank, node_idx, signal, signal_type);
#endif
    while (signal >= 0.001) {
        if (brain_nodes[node_idx].num_edges == 0) 
            break;

        int edge_to_use = getRandomInteger(0, brain_nodes[node_idx].num_edges);
        int edge_idx = brain_nodes[node_idx].edges[edge_to_use];

        int tgt_id = (edges[edge_idx].from == brain_nodes[node_idx].id) ?
            edges[edge_idx].to : edges[edge_idx].from;

        int target_rank = tgt_id / nodes_per_proc;
        if (target_rank >= world_size) 
            target_rank = world_size - 1;

        float signal_to_send = signal;
        // Check if the signal exceeds the capacity of the edge, if so will need to be sent in multiple chunks
        if (signal_to_send > edges[edge_idx].max_value)
            signal_to_send = edges[edge_idx].max_value;
        signal -= signal_to_send;

        float type_weight = edges[edge_idx].messageTypeWeightings[signal_type];
        signal_to_send *= type_weight;

        int is_local = 0;
        for (int i = start_node; i < end_node; ++i) {
            if (brain_nodes[i].id == tgt_id) {
                if (brain_nodes[i].num_outstanding_signals < SIGNAL_INBOX_SIZE) {
                    brain_nodes[i].signalInbox[brain_nodes[i].num_outstanding_signals].type = signal_type;
                    brain_nodes[i].signalInbox[brain_nodes[i].num_outstanding_signals].value = signal_to_send;
                    brain_nodes[i].num_outstanding_signals++;
                }
                else {
                    brain_nodes[i].signals_dropped++;
                }
                is_local = 1;
                break;
            }
        }

        if (!is_local) {
            struct SignalStruct remote_sig = { signal_type, signal, tgt_id };
            MPI_Send(&remote_sig, 1, MPI_SignalType, target_rank, 0, MPI_COMM_WORLD);
        }
    }
}

static void updateNodes(int node_idx) {
#ifdef DEBUG
    printf("[rank: %d] update nodes: start\n", world_rank);
#endif

    if (brain_nodes[node_idx].node_type == NERVE) 
    {
#ifdef DEBUG
        printf("[rank: %d] update nodes: is nerve\n", world_rank);
#endif
        int num_signals = getRandomInteger(0, MAX_RANDOM_NERVE_SIGNALS_TO_FIRE);
        for (int i = 0; i < num_signals; ++i) 
        {
            // 0 - 1000.0
            float val = generateDecimalRandomNumber(MAX_SIGNAL_VALUE);
            // 0 - 9 inclusive
            int type = getRandomInteger(0, NUM_SIGNAL_TYPES);

#ifdef DEBUG
            printf("[rank: %d] update nodes: fire signal\n", world_rank);
#endif

            fireSignal(node_idx, val, type);
        }
    }

#ifdef DEBUG
    printf("[rank: %d] update nodes: handle signal\n", world_rank);
#endif
    for (int i = 0; i < brain_nodes[node_idx].num_outstanding_signals; ++i) {
        //printf("%d\n", i);
        handleSignal(node_idx,
            brain_nodes[node_idx].signalInbox[i].value,
            brain_nodes[node_idx].signalInbox[i].type);
    }
#ifdef DEBUG
    printf("[rank: %d] update nodes: handle signal done\n", world_rank);
#endif
    brain_nodes[node_idx].total_signals_recieved += brain_nodes[node_idx].num_outstanding_signals;
    brain_nodes[node_idx].num_outstanding_signals = 0;
#ifdef DEBUG
    printf("[rank: %d] update nodes: done\n", world_rank);
#endif
}

static void handleSignal(int node_idx, float signal, int signal_type) {
#ifdef DEBUG
    printf("[rank: %d] handle signal: start\n", world_rank);
#endif
    if (brain_nodes[node_idx].node_type == NERVE) 
    {
#ifdef DEBUG
        printf("[rank: %d] handle signal: is nerve\n", world_rank);
#endif
        brain_nodes[node_idx].num_nerve_outputs[signal_type]++;
    }
    else 
    {
#ifdef DEBUG
        printf("[rank: %d] handle signal: is neuron\n", world_rank);
#endif
        float weight = NEURON_TYPE_SIGNAL_WEIGHTS[neuronTypeToIndex(brain_nodes[node_idx].neuron_type)];
        signal *= weight;
        int recentSignals = brain_nodes[node_idx].signals_last_ns + brain_nodes[node_idx].signals_this_ns;
        if (recentSignals > 500)
        {
#ifdef DEBUG
            printf("[rank %d] handle signal: recent signals > 500 (drop or reduce it)\n", world_rank);
#endif
            // If there have been lots of recent signals then the neuron is becomming overwhelmed, might drop a signal or reduce it
            if (getRandomInteger(0, 2) == 1)
                signal /= 2.0;
            if (getRandomInteger(0, 3) == 1)
                return;
        }
        fireSignal(node_idx, signal, signal_type);
    }
}
