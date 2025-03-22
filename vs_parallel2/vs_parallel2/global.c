#include "global.h"

// For each type of neuron determines weighting to apply to signals
const float NEURON_TYPE_SIGNAL_WEIGHTS[6] = { 0.8, 1.2, 1.1, 2.6, 0.3, 1.8 };

// Holds each node (a neuron or nerve) that comprises the brain
struct NeuronNerveStruct* brain_nodes = NULL;
// The edges between nodes
struct EdgeStruct* edges = NULL;

int num_neurons = 0, num_nerves = 0, num_edges = 0, num_brain_nodes = 0;
int world_size, world_rank;
int nodes_per_proc, start_node, end_node;
int elapsed_ns = 0;

void phello()
{
	printf("Hello World!\n");
}

/**
 * Will update a specific neuron or nerve, first firing signals if it is a nerve and then
 * handling signals that have been received
 **/
void updateNodes(int node_idx)
{
    if (brain_nodes[node_idx].num_edges > 0)
    {
        if (brain_nodes[node_idx].node_type == NERVE)
        {
            // If this is a nerve then fire a random number of signals

            // randomly emit 0-20 signals
            int num_signals_to_fire = getRandomInteger(0, MAX_RANDOM_NERVE_SIGNALS_TO_FIRE);
            for (int i = 0; i < num_signals_to_fire; i++)
            {
                // get 0.0 - 100.0
                float signalValue = generateDecimalRandomNumber(MAX_SIGNAL_VALUE);
                // get random type
                int signalType = getRandomInteger(0, NUM_SIGNAL_TYPES);

                brain_nodes[node_idx].num_nerve_inputs[signalType]++;
                fireSignal(node_idx, signalValue, signalType);
            }
        }
    }
    // Now handle all outstanding (recieved) signals
    for (int i = 0; i < brain_nodes[node_idx].num_outstanding_signals; i++)
    {
        handleSignal(node_idx, brain_nodes[node_idx].signalInbox[i].value, brain_nodes[node_idx].signalInbox[i].type);
        brain_nodes[node_idx].signals_this_ns++;
    }
    brain_nodes[node_idx].total_signals_recieved += brain_nodes[node_idx].num_outstanding_signals;
    brain_nodes[node_idx].num_outstanding_signals = 0;
}

/**
 * Handles a specific signal depending on the type of node (neuron or nerve)
 **/
void handleSignal(int node_idx, float signal, int signal_type)
{
    if (brain_nodes[node_idx].node_type == NERVE)
    {
        // Nerves consume signals and do not send them on
        brain_nodes[node_idx].num_nerve_outputs[signal_type]++;
    }
    else
    {
        // A signal is modified by a specific weight depending upon the type of neuron this is
        float changeWeight = NEURON_TYPE_SIGNAL_WEIGHTS[neuronTypeToIndex(brain_nodes[node_idx].neuron_type)];
        signal *= changeWeight;
        int recentSignals = brain_nodes[node_idx].signals_last_ns + brain_nodes[node_idx].signals_this_ns;
        if (recentSignals > 500)
        {
            // If there have been lots of recent signals then the neuron is becomming overwhelmed, might drop a signal or reduce it
            if (getRandomInteger(0, 2) == 1)
                signal /= 2.0;
            if (getRandomInteger(0, 3) == 1)
                return;
        }
        fireSignal(node_idx, signal, signal_type);
    }
}

/**
 * Fires a signal from either a neuron or nerve
 **/
void fireSignal(int node_idx, float signal, int signal_type) {
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
                is_local = 1;
                break;
            }
        }

        if (!is_local) {
            struct SignalStruct remote_sig = { signal_type, signal_to_send, tgt_id };
            MPI_Send(&remote_sig, 1, MPI_SignalType, target_rank, 0, MPI_COMM_WORLD);
        }
    }
}

/**
 * Edges are read from the input file, but are not connected up. This function will associate, for each neuron or nerve,
 * the edges that go out of it (e.g. will be used to send signals).
 */
void linkNodesToEdges()
{
    for (int i = 0; i < num_brain_nodes; i++)
    {
        int neuron_id = brain_nodes[i].id;
        int number_edges = getNumberOfEdgesForNode(neuron_id);
        brain_nodes[i].num_edges = number_edges;
        brain_nodes[i].edges = (int*)malloc(sizeof(int) * number_edges);
        int edge_idx = 0;
        for (int j = 0; j < num_edges; j++)
        {
            if ((edges[j].from == neuron_id) || (edges[j].to == neuron_id && edges[j].direction == BIDIRECTIONAL))
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
int getNumberOfEdgesForNode(int node_id)
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
int neuronTypeToIndex(enum NeuronType neuron_type)
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
 * Generates a random integer between two values, including the from value up to the to value minus
 * one, i.e. from=0, to=100 will generate a random integer between 0 and 99 inclusive
 **/
int getRandomInteger(int from, int to)
{
    return (rand() % (to - from)) + from;
}

/**
 * Generates a floating point random number up to a specific integer value, e.g. providing 100 will generate
 * from 0.0 to 100.0
 **/
float generateDecimalRandomNumber(int to)
{
    return (((float)rand()) / RAND_MAX) * to;
}

/**
 * Retrieves the current time in seconds
 **/
time_t getCurrentSeconds()
{
    return time(NULL);
}

/**
 * Parses the provided brain map file and uses this to build information
 * about each neuron, nerve and edge that connects them together
 **/
void loadBrainGraph(char* filename)
{
    const char whitespace[] = " \f\n\r\t\v";
    enum ReadMode currentMode = NONE;
    int currentNeuronIdx = 0, currentEdgeIdx = 0;
    char buffer[MAX_LINE_LEN];
    FILE* f;
    printf("filename: %s\n", filename);
    fopen_s(&f, filename, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Error opening roadmap file '%s'\n", filename);
        exit(-1);
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
            brain_nodes[currentNeuronIdx].signalInbox = (struct SignalStruct*)malloc(sizeof(struct SignalStruct) * SIGNAL_INBOX_SIZE);

            brain_nodes[currentNeuronIdx].num_nerve_outputs = (int*)malloc(sizeof(int) * NUM_SIGNAL_TYPES);
            brain_nodes[currentNeuronIdx].num_nerve_inputs = (int*)malloc(sizeof(int) * NUM_SIGNAL_TYPES);
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
            edges[currentEdgeIdx].messageTypeWeightings = (float*)malloc(sizeof(float) * NUM_SIGNAL_TYPES);
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

/**
 * Frees up memory once simulation is completed
 **/
void freeMemory()
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
void generateReport(const char* report_filename, struct NodeInfo* nodeinfos)
{
    FILE* output_report;
    fopen_s(&output_report, report_filename, "w");
    if (!output_report)
    {
        fprintf(stderr, "Failed to open file %s\n", report_filename);
        return 0;
    }
    fprintf(output_report, "Simulation ran with %d neurons, %d nerves and %d total edges until %d ns\n", num_neurons, num_nerves, num_edges, elapsed_ns);
    fprintf(output_report, "\n");
    int node_ctr = 0;
    for (int i = 0; i < num_brain_nodes; i++)
    {
        if (nodeinfos[i].node_type == NERVE)
        {
            fprintf(output_report, "Nerve number %d with brain node id: %d\n", node_ctr, nodeinfos[i].id);
            for (int j = 0; j < NUM_SIGNAL_TYPES; j++)
            {
                fprintf(output_report, "----> Signal type %d: %d firings and %d received\n", j, nodeinfos[i].num_nerve_inputs[j], nodeinfos[i].num_nerve_outputs[j]);
            }
            node_ctr++;
        }
    }
    fprintf(output_report, "\n");
    node_ctr = 0;
    for (int i = 0; i < num_brain_nodes; i++)
    {
        if (nodeinfos[i].node_type == NEURON)
        {
            fprintf(output_report, "Neuron number %d, brain node id %d, total signals received %d\n", node_ctr, nodeinfos[i].id, nodeinfos[i].total_signal_recved);
            node_ctr++;
        }
    }
    fclose(output_report);
}

// register the MPI_SignalType
void register_mpi_signal_type() {
    int blocklengths[3] = { 1, 1, 1 };
    MPI_Datatype types[3] = { MPI_INT, MPI_FLOAT, MPI_INT };
    MPI_Aint offsets[3];
    offsets[0] = offsetof(struct SignalStruct, type);
    offsets[1] = offsetof(struct SignalStruct, value);
    offsets[2] = offsetof(struct SignalStruct, target_id);
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_SignalType);
    MPI_Type_commit(&MPI_SignalType);
}

// register MPI_NodeInfoType
void register_mpi_node_info_type() {  
    int blocklengths[5] = { 1, 1, NUM_SIGNAL_TYPES, NUM_SIGNAL_TYPES, 1 };  
    MPI_Datatype types[5] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT };  
    MPI_Aint offsets[5];  

    // Calculate offsets using offsetof from stddef.h  
    offsets[0] = offsetof(struct NodeInfo, id);  
    offsets[1] = offsetof(struct NodeInfo, node_type);  
    offsets[2] = offsetof(struct NodeInfo, num_nerve_inputs);  
    offsets[3] = offsetof(struct NodeInfo, num_nerve_outputs);  
    offsets[4] = offsetof(struct NodeInfo, total_signal_recved); 

    MPI_Type_create_struct(5, blocklengths, offsets, types, &MPI_NodeInfoType);   
    MPI_Type_commit(&MPI_NodeInfoType);  
}  

// Free memory and finalize MPI
void mpi_finalize() {
    MPI_Type_free(&MPI_SignalType);
    MPI_Type_free(&MPI_NodeInfoType);
    freeMemory();
    MPI_Finalize();
}


// utils for debugging

//struct NodeInfo
//{
//    int id;
//    enum NodeType node_type;
//    int num_nerve_inputs[NUM_SIGNAL_TYPES];
//    int num_nerve_outputs[NUM_SIGNAL_TYPES];
//    int total_signal_recved;
//};
void print_node_info(int rank, struct NodeInfo* nodeinfo)
{
    printf("[rank %d] nodeinfo: id: %d, nodetype: %d\n", rank, nodeinfo->id, nodeinfo->node_type);
    if (nodeinfo->node_type == NERVE)
    {
        for (int i = 0; i < NUM_SIGNAL_TYPES; i++)
        {
            printf("[rank %d] signal type %d fired %d, recved %d\n", rank, i, nodeinfo->num_nerve_inputs[i], nodeinfo->num_nerve_outputs[i]);
        }
    }
    else
    {
        printf("[rank %d] total signal recved: %d\n", rank, nodeinfo->total_signal_recved);
    }
}

//struct SignalStruct
//{
//    int type;
//    float value;
//    int target_id;
//};
void print_signal(int rank, struct SignalStruct* signal)
{
    printf("[rank %d] Signal type: %d\n", rank, signal->type);
    printf("[rank %d] Singal value: %f\n", rank, signal->value);
    printf("[rank %d] Signal target id: %d\n", rank, signal->target_id);
}
