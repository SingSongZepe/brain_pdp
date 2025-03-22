#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>

#define MAX_LINE_LEN 100
#define NUM_SIGNAL_TYPES 10
#define MIN_LENGTH_NS 2
#define SIGNAL_INBOX_SIZE 200
#define MAX_RANDOM_NERVE_SIGNALS_TO_FIRE 20
#define MAX_SIGNAL_VALUE 1000
#define OUTPUT_REPORT_FILENAME "summary_report"

enum ReadMode { NONE, NEURON_NERVE, EDGE };
enum NeuronType { SENSORY, MOTOR, UNIPOLAR, PSEUDOUNIPOLAR, BIPOLAR, MULTIPOLAR };
enum NodeType { NEURON, NERVE };
enum EdgeDirection { BIDIRECTIONAL, UNIDIRECTIONAL };

struct SignalStruct {
    int type;
    float value;
    int target_id;
};

#define MAX_EDGES 100
struct NeuronNerveStruct {
    int id, num_edges, num_outstanding_signals;
    int total_signals_recieved;
    int num_nerve_outputs[NUM_SIGNAL_TYPES]; 
    int num_nerve_inputs[NUM_SIGNAL_TYPES];
    int signals_this_ns, signals_last_ns, signals_dropped;
    float x, y, z;
    enum NodeType node_type;
    enum NeuronType neuron_type;
    int edges[MAX_EDGES];                     
    struct SignalStruct signalInbox[SIGNAL_INBOX_SIZE];
};

#define NUM_SIGNAL_TYPES 100  

struct EdgeStruct {
    int from, to;
    enum EdgeDirection direction;
    float messageTypeWeightings[NUM_SIGNAL_TYPES]; 
    float max_value;
};

const float NEURON_TYPE_SIGNAL_WEIGHTS[6] = { 0.8, 1.2, 1.1, 2.6, 0.3, 1.8 };

struct NeuronNerveStruct* brain_nodes = NULL;
struct EdgeStruct* edges = NULL;

int num_neurons = 0, num_nerves = 0, num_edges = 0, num_brain_nodes = 0;
int elapsed_ns = 0;
int world_size, world_rank;
int nodes_per_proc, start_node, end_node;
MPI_Datatype MPI_SignalType;
MPI_Datatype MPI_NeuronNerveType;
MPI_Datatype MPI_EdgeType;

static void generateReport(const char*, struct NeuronNerveStruct* brain_nodes);
static void linkNodesToEdges();
static int getNumberOfEdgesForNode(int);
static void updateNodes(int);
static void handleSignal(int, float, int);
static void fireSignal(int, float, int);
static int neuronTypeToIndex(enum NeuronType);
static void loadBrainGraph(char*);
static void freeMemory();
static int getRandomInteger(int, int);
static float generateDecimalRandomNumber(int);
static time_t getCurrentSeconds();
static void register_mpi_signal_type();
static void register_mpi_neuron_nerve_type();
static void register_mpi_edge_type();
static void mpi_finalize();
static void print_brain_node(struct NeuronNerveStruct* brain_node);
static void print_signal(struct SignalStruct* signal);

/**
 * Generates a random integer between two values, including the from value up to the to value minus
 * one, i.e. from=0, to=100 will generate a random integer between 0 and 99 inclusive
 **/
static int getRandomInteger(int from, int to)
{
    return (rand() % (to - from)) + from;
}

void mpi_finalize() {
    MPI_Type_free(&MPI_SignalType);
    MPI_Type_free(&MPI_NeuronNerveType);
    MPI_Type_free(&MPI_EdgeType);
    freeMemory();
    MPI_Finalize();
}

/**
 * Edges are read from the input file, but are not connected up. This function will associate, for each neuron or nerve,
 * the edges that go out of it (e.g. will be used to send signals).
 */
static void linkNodesToEdges()
{
    for (int i = 0; i < num_brain_nodes; i++)
    {
        int neuron_id = brain_nodes[i].id;
        int number_edges = getNumberOfEdgesForNode(neuron_id);
        brain_nodes[i].num_edges = number_edges;
        //brain_nodes[i].edges = (int*)malloc(sizeof(int) * number_edges);

        int edge_idx = 0;
        for (int j = 0; j < num_edges; j++)
        {
            // Check if we can add more edges  
            if (edge_idx < number_edges &&
                (edges[j].from == neuron_id ||
                    (edges[j].to == neuron_id && edges[j].direction == BIDIRECTIONAL)))
            {
                brain_nodes[i].edges[edge_idx] = j;
                edge_idx++;
            }
        }
    }
}

/**
 * Retrieved the number of edges associated in the out direction for a neuron or nerve
 **/
static int getNumberOfEdgesForNode(int node_id)
{
    int counted_edges = 0;
    for (int i = 0; i < num_edges; i++)
    {
        if (edges[i].from == node_id)
        {
            counted_edges++;
        }
        else if (edges[i].to == node_id && edges[i].direction == BIDIRECTIONAL)
        {
            counted_edges++;
        }
    }
    return counted_edges;
}

/**
 * Maps the enumeration type to integer value for neuron type
 **/
static int neuronTypeToIndex(enum NeuronType neuron_type)
{
    if (neuron_type == SENSORY)
        return 0;
    if (neuron_type == MOTOR)
        return 1;
    if (neuron_type == UNIPOLAR)
        return 2;
    if (neuron_type == PSEUDOUNIPOLAR)
        return 3;
    if (neuron_type == BIPOLAR)
        return 4;
    if (neuron_type == MULTIPOLAR)
        return 5;
    assert(0);
}

/**
 * Generates a floating point random number up to a specific integer value, e.g. providing 100 will generate
 * from 0.0 to 100.0
 **/
static float generateDecimalRandomNumber(int to)
{
    return (((float)rand()) / RAND_MAX) * to;
}

/**
 * Retrieves the current time in seconds
 **/
static time_t getCurrentSeconds()
{
    // Returns current time as seconds since the Epoch  
    return time(NULL);
}

/**
 * Parses the provided brain map file and uses this to build information
 * about each neuron, nerve and edge that connects them together
 **/
static void loadBrainGraph(char* filename)
{
    if (world_rank == 0) {

        const char whitespace[] = " \f\n\r\t\v";
        enum ReadMode currentMode = NONE;
        int currentNeuronIdx = 0, currentEdgeIdx = 0;
        char buffer[MAX_LINE_LEN];
        FILE* f;
        fopen_s(&f, filename, "r");
        if (f == NULL) {
            fprintf(stderr, "Error opening file '%s'\n", filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        while (fgets(buffer, MAX_LINE_LEN, f))
        {
            if (buffer[0] == '%')
                continue;
            char* line_contents = buffer + strspn(buffer, whitespace);
            if (strncmp("<num_neurons>", line_contents, 13) == 0)
            {
                char* s = strstr(line_contents, ">");
                num_neurons = atoi(&s[1]);
            }
            if (strncmp("<num_nerves>", line_contents, 12) == 0)
            {
                char* s = strstr(line_contents, ">");
                num_nerves = atoi(&s[1]);
            }
            if (brain_nodes == NULL && num_neurons > 0 && num_nerves > 0)
            {
                num_brain_nodes = num_neurons + num_nerves;
                brain_nodes = (struct NeuronNerveStruct*)malloc(sizeof(struct NeuronNerveStruct) * num_brain_nodes);
            }
            if (strncmp("<num_edges>", line_contents, 11) == 0)
            {
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                num_edges = atoi(&s[1]);
                edges = (struct EdgeStruct*)malloc(sizeof(struct EdgeStruct) * num_edges);
            }
            if (strncmp("<neuron>", line_contents, 8) == 0 || strncmp("<nerve>", line_contents, 7) == 0)
            {
                currentMode = NEURON_NERVE;
                brain_nodes[currentNeuronIdx].num_edges = 0;
                brain_nodes[currentNeuronIdx].num_outstanding_signals = 0;
                brain_nodes[currentNeuronIdx].signals_this_ns = 0;
                brain_nodes[currentNeuronIdx].signals_last_ns = 0;
                brain_nodes[currentNeuronIdx].total_signals_recieved = 0;
                //brain_nodes[currentNeuronIdx].signalInbox = (struct SignalStruct*)malloc(sizeof(struct SignalStruct) * SIGNAL_INBOX_SIZE);

                //brain_nodes[currentNeuronIdx].num_nerve_outputs = (int*)malloc(sizeof(int) * NUM_SIGNAL_TYPES);
                //brain_nodes[currentNeuronIdx].num_nerve_inputs = (int*)malloc(sizeof(int) * NUM_SIGNAL_TYPES);
                for (int j = 0; j < NUM_SIGNAL_TYPES; j++)
                {
                    brain_nodes[currentNeuronIdx].num_nerve_outputs[j] = brain_nodes[currentNeuronIdx].num_nerve_inputs[j] = 0;
                }

                if (strncmp("<neuron>", line_contents, 8) == 0)
                {
                    brain_nodes[currentNeuronIdx].node_type = NEURON;
                }
                else if (strncmp("<nerve>", line_contents, 7) == 0)
                {
                    brain_nodes[currentNeuronIdx].node_type = NERVE;
                }
                else
                {
                    assert(0);
                }
                if (currentNeuronIdx >= num_brain_nodes)
                {
                    fprintf(stderr, "Too many neurons and nerves, increase number in <num_neurons> and <num_nerves>\n");
                    exit(-1);
                }
            }

            if (strncmp("</neuron>", line_contents, 9) == 0 || strncmp("</nerve>", line_contents, 8) == 0)
            {
                currentMode = NONE;
                currentNeuronIdx++;
            }

            if (strncmp("<edge>", line_contents, 6) == 0)
            {
                currentMode = EDGE;
                if (currentEdgeIdx >= num_edges)
                {
                    fprintf(stderr, "Too many edges increase number in <num_edges>\n");
                    exit(-1);
                }
                //edges[currentEdgeIdx].messageTypeWeightings = (float*)malloc(sizeof(float) * NUM_SIGNAL_TYPES);
            }

            if (strncmp("</edge>", line_contents, 7) == 0)
            {
                currentMode = NONE;
                currentEdgeIdx++;
            }

            if (strncmp("<id>", line_contents, 4) == 0)
            {
                assert(currentMode == NEURON_NERVE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                int id = atof(&s[1]);
                brain_nodes[currentNeuronIdx].id = id;
            }

            if (strncmp("<x>", line_contents, 3) == 0 || strncmp("<y>", line_contents, 3) == 0 || strncmp("<z>", line_contents, 3) == 0)
            {
                assert(currentMode == NEURON_NERVE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                float val = atof(&s[1]);
                if (strncmp("<x>", line_contents, 3) == 0)
                {
                    brain_nodes[currentNeuronIdx].x = val;
                }
                else if (strncmp("<y>", line_contents, 3) == 0)
                {
                    brain_nodes[currentNeuronIdx].y = val;
                }
                else if (strncmp("<z>", line_contents, 3) == 0)
                {
                    brain_nodes[currentNeuronIdx].z = val;
                }
                else
                {
                    assert(0);
                }
            }

            if (strncmp("<type>", line_contents, 6) == 0)
            {
                assert(currentMode == NEURON_NERVE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                if (strcmp("sensory", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = SENSORY;
                }
                else if (strcmp("motor", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = MOTOR;
                }
                else if (strcmp("unipolar", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = UNIPOLAR;
                }
                else if (strcmp("pseudounipolar", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = PSEUDOUNIPOLAR;
                }
                else if (strcmp("bipolar", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = BIPOLAR;
                }
                else if (strcmp("multipolar", &s[1]) == 0)
                {
                    brain_nodes[currentNeuronIdx].neuron_type = MULTIPOLAR;
                }
                else
                {
                    fprintf(stderr, "Neuron type of '%s' unknown for neuron %d", &s[1], brain_nodes[currentNeuronIdx].id);
                    exit(-1);
                }
            }

            if (strncmp("<from>", line_contents, 6) == 0 || strncmp("<to>", line_contents, 4) == 0)
            {
                assert(currentMode == EDGE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                int neuron_id = atoi(&s[1]);
                if (strncmp("<from>", line_contents, 6) == 0)
                {
                    edges[currentEdgeIdx].from = neuron_id;
                }
                else if (strncmp("<to>", line_contents, 4) == 0)
                {
                    edges[currentEdgeIdx].to = neuron_id;
                }
                else
                {
                    assert(0);
                }
            }

            if (strncmp("<max_value>", line_contents, 11) == 0)
            {
                assert(currentMode == EDGE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                float max_value = atof(&s[1]);
                edges[currentEdgeIdx].max_value = max_value;
            }

            if (strncmp("<direction>", line_contents, 11) == 0)
            {
                assert(currentMode == EDGE);
                char* s = strstr(line_contents, ">");
                char* e = strstr(s, "<");
                e[0] = '\0';
                if (strcmp("unidirectional", &s[1]) == 0)
                {
                    edges[currentEdgeIdx].direction = UNIDIRECTIONAL;
                }
                else if (strcmp("bidirectional", &s[1]) == 0)
                {
                    edges[currentEdgeIdx].direction = BIDIRECTIONAL;
                }
                else
                {
                    fprintf(stderr, "Direction type of '%s' unknown", &s[1]);
                    exit(-1);
                }
            }

            if (strncmp("<weighting_", line_contents, 11) == 0)
            {
                assert(currentMode == EDGE);
                char* s = strstr(line_contents, "_");
                char* e = strstr(s, ">");
                e[0] = '\0';
                int weight_idx = atoi(&s[1]);
                assert(weight_idx < NUM_SIGNAL_TYPES);
                char* c = strstr(&e[1], "<");
                c[0] = '\0';
                float val = atof(&e[1]);
                edges[currentEdgeIdx].messageTypeWeightings[weight_idx] = val;
            }
        }

        fclose(f);
    }

    // 广播全局参数
    MPI_Bcast(&num_brain_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 其他进程分配内存
    if (world_rank != 0) {
        brain_nodes = (NeuronNerveStruct*) malloc(num_brain_nodes * sizeof(struct NeuronNerveStruct));
        edges = (EdgeStruct*) malloc(num_edges * sizeof(struct EdgeStruct));
    }

    // 广播节点数据（需自定义数据类型）
    MPI_Bcast(brain_nodes, num_brain_nodes, MPI_NeuronNerveType, 0, MPI_COMM_WORLD);

    // 广播边数据（需自定义EdgeStruct类型）
    MPI_Bcast(edges, num_edges, MPI_EdgeType, 0, MPI_COMM_WORLD);
}

/**
 * Frees up memory once simulation is completed
 **/
static void freeMemory()
{
    for (int i = 0; i < num_edges; i++)
    {
        free(edges[i].messageTypeWeightings);
    }
    free(edges);

    for (int i = 0; i < num_brain_nodes; i++)
    {
        free(brain_nodes[i].signalInbox);
        free(brain_nodes[i].edges);
        free(brain_nodes[i].num_nerve_inputs);
        free(brain_nodes[i].num_nerve_outputs);
    }
    free(brain_nodes);
}

/**
 * Writes out a report to a file that summarises the simulation for each neuron and nerve
 **/
static void generateReport(const char* report_filename, struct NeuronNerveStruct* global_brain_nodes)
{
#if DEBUG_GENERATE_REPORT
    printf("[rank %d] generate report: start\n", world_size);
#endif
    FILE* output_report;
    fopen_s(&output_report, report_filename, "w");
    if (output_report == NULL)
    {
#if DEBUG_GENERATE_REPORT
        printf("[rank %d] open file failed\n", world_size);
#endif
        return;
    }

    fprintf(output_report, "Simulation ran with %d neurons, %d nerves and %d total edges until %d ns\n", num_neurons, num_nerves, num_edges, elapsed_ns);
    fprintf(output_report, "\n");
#if DEBUG_GENERATE_REPORT
    printf("[rank %d] header is written\n", world_size);
#endif

#if DEBUG_GENERATE_REPORT
    printf("start to write information of nerve nodes\n");
#endif

    int node_ctr = 0;
    for (int i = 0; i < num_brain_nodes; i++)
    {
        if (global_brain_nodes[i].node_type == NERVE)
        {
#if DEBUG_GENERATE_REPORT
            printf("[rank %d] the node %d is nerve, and write it to file", world_rank, i);
#endif
            fprintf(output_report, "Nerve number %d with brain node id: %d\n", node_ctr, global_brain_nodes[i].id);

            print_brain_node(&global_brain_nodes[i]);

            for (int j = 0; j < NUM_SIGNAL_TYPES; j++)
            {
                fprintf(output_report, "----> Signal type %d: %d firings and %d received\n", j, global_brain_nodes[i].num_nerve_inputs[j], global_brain_nodes[i].num_nerve_outputs[j]);
            }
            node_ctr++;
        }
    }
    fprintf(output_report, "\n");
    return;

#if DEBUG_GENERATE_REPORT
    printf("start to write information of neuron nodes\n");
#endif
    node_ctr = 0;
    for (int i = 0; i < num_brain_nodes; i++)
    {
        if (brain_nodes[i].node_type == NEURON)
        {
#if DEBUG_GENERATE_REPORT
            printf("[rank %d] the node %d is neuron, and write it to file", world_rank, i);
#endif
            fprintf(output_report, "Neuron number %d, brain node id %d, total signals received %d\n", node_ctr, global_brain_nodes[i].id, global_brain_nodes[i].total_signals_recieved);
            node_ctr++;
        }
    }
    fclose(output_report);
}

// proint the brian_node
//struct NeuronNerveStruct {
//    int id, num_edges, num_outstanding_signals;
//    int total_signals_recieved;
//    int* num_nerve_outputs, * num_nerve_inputs;
//    int signals_this_ns, signals_last_ns, signals_dropped;
//    float x, y, z;
//    enum NodeType node_type;
//    enum NeuronType neuron_type;
//    int* edges;
//    struct SignalStruct* signalInbox;
//};
static void print_brain_node(struct NeuronNerveStruct* brain_node)
{
    if (brain_node == NULL)
    {
        printf("the current node is NULL\n");
        return;
    }
    printf("the brian node id: %d, num_edges: %d, num_outstanding_signals: %d\n", brain_node->id, brain_node->num_edges, brain_node->num_outstanding_signals);
    printf("the total_signal_received: %d\n", brain_node->total_signals_recieved);
    printf("Signals this ns: %d\n", brain_node->signals_this_ns);
    printf("Signals last ns: %d\n", brain_node->signals_last_ns);
    printf("Signals dropped: %d\n", brain_node->signals_dropped);
    printf("Coordinates (x, y, z): (%f, %f, %f)\n", brain_node->x, brain_node->y, brain_node->z);
    printf("Node type: %d\n", brain_node->node_type);
    printf("Neuron type: %d\n", brain_node->neuron_type);

    if (brain_node->neuron_type == NERVE)
    {
        if (brain_node->num_nerve_outputs != NULL && brain_node->num_nerve_inputs != NULL) {
            printf("Nerve outputs: ");
            for (int i = 0; i < brain_node->num_edges; i++) {
                printf("%d ", brain_node->num_nerve_outputs[i]);
            }
            printf("\n");

            printf("Nerve inputs: ");
            for (int i = 0; i < brain_node->num_edges; i++) {
                printf("%d ", brain_node->num_nerve_inputs[i]);
            }
            printf("\n");
        }
    }
#if DEBUG_PRINT_NODE_MORE


    // Assuming edges is an array of the same size as num_nerve_outputs and num_nerve_inputs
    if (brain_node->edges != NULL) {
        printf("Edges: ");
        for (int i = 0; i < brain_node->num_edges; i++) {
            printf("%d ", brain_node->edges[i]);
        }
        printf("\n");
    }

    // Assuming signalInbox is an array of SignalStruct
    if (brain_node->signalInbox != NULL) {
        printf("Signal Inbox: ");
        for (int i = 0; i < brain_node->num_outstanding_signals; i++) {
            // Assuming SignalStruct has a member function or a way to print its contents
            print_signal(brain_node->signalInbox[i]);
        }
        printf("\n");
    }
#endif
    return;
}

//struct SignalStruct {
//    int type;
//    float value;
//    int target_id;
//};
static void print_signal(struct SignalStruct* signal)
{
    printf("Signal type: %d\n", signal->type);
    printf("Singal value: %f\n", signal->value);
    printf("Signal target id: %d\n", signal->target_id);
}

static void register_mpi_neuron_nerve_type() {
    int blocklengths[] = {
        7,  // int: id, num_edges, num_outstanding_signals, total_signals_recieved, signals_this_ns, signals_last_ns, signals_dropped
        NUM_SIGNAL_TYPES,  // num_nerve_outputs
        NUM_SIGNAL_TYPES,  // num_nerve_inputs
        3,  // float: x, y, z
        2,  // enum: node_type, neuron_type
        MAX_EDGES,  // edges
        SIGNAL_INBOX_SIZE * 3  // SignalStruct数组（每个SignalStruct有3个成员）
    };
    MPI_Datatype types[] = {
        MPI_INT,  // 前7个int
        MPI_INT,   // num_nerve_outputs
        MPI_INT,   // num_nerve_inputs
        MPI_FLOAT, // x, y, z
        MPI_INT,   // enums
        MPI_INT,   // edges
        MPI_INT    // SignalStruct数组（简化处理）
    };
    MPI_Aint offsets[6];
    offsets[0] = offsetof(struct NeuronNerveStruct, id);
    offsets[1] = offsetof(struct NeuronNerveStruct, num_nerve_outputs);
    offsets[2] = offsetof(struct NeuronNerveStruct, num_nerve_inputs);
    offsets[3] = offsetof(struct NeuronNerveStruct, x);
    offsets[4] = offsetof(struct NeuronNerveStruct, node_type);
    offsets[5] = offsetof(struct NeuronNerveStruct, edges);

    MPI_Type_create_struct(6, blocklengths, offsets, types, &MPI_NeuronNerveType);
    MPI_Type_commit(&MPI_NeuronNerveType);
}

static void register_mpi_signal_type() {
    int blocklengths[3] = { 1, 1, 1 };
    MPI_Datatype types[3] = { MPI_INT, MPI_FLOAT, MPI_INT };
    MPI_Aint offsets[3];
    offsets[0] = offsetof(struct SignalStruct, type);
    offsets[1] = offsetof(struct SignalStruct, value);
    offsets[2] = offsetof(struct SignalStruct, target_id);
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_SignalType);
    MPI_Type_commit(&MPI_SignalType);
}

static void register_mpi_edge_type() {
    MPI_Datatype MPI_EdgeType;
    int blocklengths[] = {
        2,  // int: from, to
        1,  // enum EdgeDirection (假设存储为int)
        NUM_SIGNAL_TYPES,  // float messageTypeWeightings[NUM_SIGNAL_TYPES]
        1   // float max_value
    };
    MPI_Datatype types[] = {
        MPI_INT,
        MPI_INT,  // enum 按int传输
        MPI_FLOAT,
        MPI_FLOAT
    };
    MPI_Aint offsets[4];

    offsets[0] = offsetof(struct EdgeStruct, from);
    offsets[1] = offsetof(struct EdgeStruct, direction);
    offsets[2] = offsetof(struct EdgeStruct, messageTypeWeightings);
    offsets[3] = offsetof(struct EdgeStruct, max_value);

    MPI_Type_create_struct(4, blocklengths, offsets, types, &MPI_EdgeType);
    MPI_Type_commit(&MPI_EdgeType);
}


#endif // __GLOBAL_H__