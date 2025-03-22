#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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

// for debugging
#define DEBUG_MAIN 0
#define DEBUG_MPI_PROB 0
#define DEBUG_ITERATION_INFO 1
#define OUTPUT_INFO 1

enum ReadMode
{
	NONE,
	NEURON_NERVE,
	EDGE
};

// The differnt types of neuron
enum NeuronType
{
	SENSORY,
	MOTOR,
	UNIPOLAR,
	PSEUDOUNIPOLAR,
	BIPOLAR,
	MULTIPOLAR
};

// Whether the node type is a neuron or a nerve
enum NodeType
{
	NEURON,
	NERVE
};

// Whether an edge is one way or both ways
enum EdgeDirection
{
	BIDIRECTIONAL,
	UNIDIRECTIONAL
};

// This is a neuron or nerve, we use the same data structure to hold both
struct NeuronNerveStruct
{
	int id, num_edges, num_outstanding_signals, total_signals_recieved;
	int* num_nerve_outputs, * num_nerve_inputs;
	int signals_this_ns, signals_last_ns;
	float x, y, z;
	enum NodeType node_type;
	enum NeuronType neuron_type;
	int* edges;
	struct SignalStruct* signalInbox;
};

// Is an edge that connects neurons or nerves
struct EdgeStruct
{
	int from, to;
	enum EdgeDirection direction;
	float* messageTypeWeightings;
	float max_value;
};

// Represents a signal, which is the value and type
struct SignalStruct
{
	int type;
	float value;
	int target_id;
};

// the all information of node that needed for generating the report
struct NodeInfo
{
	int id;
	enum NodeType node_type;
	int num_nerve_inputs[NUM_SIGNAL_TYPES];
	int num_nerve_outputs[NUM_SIGNAL_TYPES];
	int total_signal_recved;
};

extern void phello();

extern void generateReport(const char*, struct NodeInfo*);
extern void linkNodesToEdges();

extern int getNumberOfEdgesForNode(int);
extern void updateNodes(int);
extern void handleSignal(int, float, int);
extern void fireSignal(int, float, int);

// utils
extern int neuronTypeToIndex(enum NeuronType);
extern void loadBrainGraph(char*);
extern void freeMemory();
extern int getRandomInteger(int, int);
extern float generateDecimalRandomNumber(int);
extern time_t getCurrentSeconds();

// MPI_type
extern void register_mpi_signal_type(); 
extern void register_mpi_node_info_type(); 

// utils for debugging
extern void print_node_info(int, struct NodeInfo*);
extern void print_signal(int, struct SignalStruct*);

// For each type of neuron determines weighting to apply to signals
extern const float NEURON_TYPE_SIGNAL_WEIGHTS[6];

// Holds each node (a neuron or nerve) that comprises the brain
extern struct NeuronNerveStruct* brain_nodes;
// The edges between nodes
extern struct EdgeStruct* edges;

extern int num_neurons;
extern int num_nerves;
extern int num_edges;
extern int num_brain_nodes;
extern int elapsed_ns;
extern int world_size, world_rank;
extern int nodes_per_proc, start_node, end_node;
MPI_Datatype MPI_SignalType;
MPI_Datatype MPI_NodeInfoType;

extern void mpi_finalize();
#endif //__GLOBAL_H__