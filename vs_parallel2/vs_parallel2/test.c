#include "global.h"

int main1(int argc, char** argv) 
{
    int world_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    register_mpi_signal_type();

    if (world_rank == 0)
    {
        struct SignalStruct signal = { 2, 3.14, 5 };
        MPI_Send(&signal, 1, MPI_SignalType, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 1)
    {
        struct SignalStruct received_signal;
        MPI_Recv(&received_signal, 1, MPI_SignalType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received: type=%d, value=%.2f, target_id=%d\n",
            received_signal.type, received_signal.value, received_signal.target_id);
    }

    MPI_Type_free(&MPI_SignalType);
    MPI_Finalize();
    return 0;
}

// MPI_gather
int main2(int argc, char** argv) {
    int world_rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    register_mpi_node_info_type();

    struct NodeInfo local_node_info;
    local_node_info.id = world_rank; // Just an example, each rank has a unique id  
    local_node_info.node_type = (world_rank % 2 == 0) ? NEURON : NERVE; // Alternate between NEURON and NERVE  
    local_node_info.total_signal_recved = world_rank * 10; // Example total_signal_recved  

    // Initialize num_nerve_inputs and num_nerve_outputs  
    for (int i = 0; i < NUM_SIGNAL_TYPES; ++i) {
        local_node_info.num_nerve_inputs[i] = world_rank + i; // Just for example  
        local_node_info.num_nerve_outputs[i] = world_rank + i + 1; // Just for example  
    }

    // Prepare to gather NodeInfo from all ranks at rank 0  
    struct NodeInfo* gathered_node_info = NULL;
    if (world_rank == 0) {
        gathered_node_info = (struct NodeInfo*)malloc(world_size * sizeof(struct NodeInfo)); // Allocate memory to gather info  
    }

    // Gather NodeInfo from all ranks to rank 0  
    MPI_Gather(&local_node_info, 1, MPI_NodeInfoType, // Send data  
        gathered_node_info, 1, MPI_NodeInfoType, // Receive at root  
        0, MPI_COMM_WORLD);

    // Rank 0 prints the gathered data  
    if (world_rank == 0) {
        printf("Gathered NodeInfo from all ranks:\n");
        for (int i = 0; i < world_size; ++i) {
            printf("Rank %d: ID = %d, Type = %d, Total Signals Received = %d\n",
                i,
                gathered_node_info[i].id,
                gathered_node_info[i].node_type,
                gathered_node_info[i].total_signal_recved);
            for (int j = 0; j < NUM_SIGNAL_TYPES; j++) {
                printf("  Inputs[%d] = %d, Outputs[%d] = %d\n",
                    j,
                    gathered_node_info[i].num_nerve_inputs[j],
                    j,
                    gathered_node_info[i].num_nerve_outputs[j]);
            }
        }
        free(gathered_node_info); // Ensure to free the allocated memory  
    }
    
    MPI_Type_free(&MPI_NodeInfoType);
    MPI_Finalize();
    return 0;
}

// MPI_gatherv
int main3(int argc, char** argv) {  
    int world_rank, world_size;  

    MPI_Init(&argc, &argv);  
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  

    // Register the custom MPI data type  
    register_mpi_node_info_type();  

    // Each rank creates a number of NodeInfo instances  
    int num_node_infos = world_rank + 1; // for example, rank 0 has 1, rank 1 has 2, etc.  
    struct NodeInfo *node_infos = malloc(num_node_infos * sizeof(struct NodeInfo));  

    // Populate the NodeInfo structures  
    for (int i = 0; i < num_node_infos; ++i) {  
        node_infos[i].id = i;  
        node_infos[i].node_type = (enum NodeType)(world_rank % 2); // simple alternating type  
        for (int j = 0; j < NUM_SIGNAL_TYPES; ++j) {  
            node_infos[i].num_nerve_inputs[j] = world_rank + j;  
            node_infos[i].num_nerve_outputs[j] = world_rank + j + 1;  
        }  
        node_infos[i].total_signal_recved = world_rank * 10 + i; // just an example value  
    }  

    // Gather the sizes of node_infos from each rank  
    int *recv_counts = NULL;  
    if (world_rank == 0) {  
        recv_counts = malloc(world_size * sizeof(int));  
    }  
    
    MPI_Gather(&num_node_infos, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);  

    // Calculate total number of NodeInfo structures to be gathered at rank 0  
    int total_node_infos = 0;
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            total_node_infos += recv_counts[i];
        }
    }

    // Prepare a buffer for all NodeInfos at rank 0  
    struct NodeInfo* all_node_infos = NULL;
    if (world_rank == 0) {
        all_node_infos = malloc(total_node_infos * sizeof(struct NodeInfo));
    }

    // Prepare the displacements array for rank 0  
    int* displacements = NULL;
    if (world_rank == 0) {
        displacements = (int*) malloc(world_size * sizeof(int));
        displacements[0] = 0; // Start at 0 for rank 0  
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1]; // Calculate displacement for each rank  
        }
    }

    // Gather the actual NodeInfo data to rank 0  
    MPI_Gatherv(node_infos, num_node_infos, MPI_NodeInfoType,
        all_node_infos, recv_counts, displacements,
        MPI_NodeInfoType, 0, MPI_COMM_WORLD);

    // Rank 0 prints the gathered NodeInfo data  
    if (world_rank == 0) {
        printf("Gathered NodeInfo Data:\n");
        for (int i = 0; i < total_node_infos; ++i) {
            printf("ID: %d, Type: %d, Total Signals Received: %d\n",
                all_node_infos[i].id,
                all_node_infos[i].node_type,
                all_node_infos[i].total_signal_recved);
        }
        free(recv_counts);
        free(all_node_infos);
        free(displacements); // Don't forget to free displacements  
    }

    free(node_infos);  
    MPI_Type_free(&MPI_NodeInfoType);  
    MPI_Finalize();  
    return 0;  
}  