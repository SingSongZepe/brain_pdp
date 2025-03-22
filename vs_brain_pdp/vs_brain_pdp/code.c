#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <windows.h>
#include <assert.h>
#include <time.h>
#include <ctime>

#define MAX_LINE_LEN 100
#define NUM_SIGNAL_TYPES 10
#define MIN_LENGTH_NS 2
#define SIGNAL_INBOX_SIZE 200
#define MAX_RANDOM_NERVE_SIGNALS_TO_FIRE 20
#define MAX_SIGNAL_VALUE 1000
#define OUTPUT_REPORT_FILENAME "summary_report"

enum ReadMode {
  NONE,
  NEURON_NERVE,
  EDGE
};

// The differnt types of neuron
enum NeuronType {
  SENSORY,
  MOTOR,
  UNIPOLAR,
  PSEUDOUNIPOLAR,
  BIPOLAR,
  MULTIPOLAR
};

// Whether the node type is a neuron or a nerve
enum NodeType {
  NEURON,
  NERVE
};

// Whether an edge is one way or both ways
enum EdgeDirection {
  BIDIRECTIONAL,
  UNIDIRECTIONAL
};

// This is a neuron or nerve, we use the same data structure to hold both
struct NeuronNerveStruct {
  int id, num_edges, num_outstanding_signals, total_signals_recieved;
  int * num_nerve_outputs, * num_nerve_inputs;
  int signals_this_ns, signals_last_ns;
  float x, y, z;
  enum NodeType node_type;
  enum NeuronType neuron_type;
  int * edges;
  struct SignalStruct * signalInbox;
};

// Is an edge that connects neurons or nerves
struct EdgeStruct {
  int from, to;
  enum EdgeDirection direction;
  float * messageTypeWeightings;
  float max_value;
};

// Represents a signal, which is the value and type
struct SignalStruct {
  int type;
  float value;
};

// For each type of neuron determines weighting to apply to signals
const float NEURON_TYPE_SIGNAL_WEIGHTS[6]={0.8, 1.2, 1.1, 2.6, 0.3, 1.8};

// Holds each node (a neuron or nerve) that comprises the brain
struct NeuronNerveStruct * brain_nodes=NULL;
// The edges between nodes
struct EdgeStruct * edges=NULL;

int num_neurons=0, num_nerves=0, num_edges=0, num_brain_nodes=0;
int elapsed_ns=0;

static void generateReport(const char*);
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

/**
 * Program entry point and main loop
 **/
int main(int argc, char * argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Error: You need to provide the brain graph file and number of nanoseconds to simulate as arguments\n");
    exit(-1);
  }
  time_t t;
  // Seed the random number generator
  srand((unsigned) time(&t));
  // Load brain map configuration from the file
  loadBrainGraph(argv[1]);
  printf("Loaded brain graph file '%s'\n", argv[1]);
  // Link the neurons to the edges in the data structure
  
  linkNodesToEdges();
  // Initialise time
  time_t seconds=0;
  time_t start_seconds=getCurrentSeconds();

  int num_ns_to_simulate=atoi(argv[2]);

  // Tracks performance data (number of iterations per ns)
  int total_iterations=0, current_ns_iterations=0, max_iteration_per_ns=-1, min_iteration_per_ns=-1;

  printf("Starting simulation to %d nanoseconds. Brain contains %d neurons, %d nerves and %d total edges\n", num_ns_to_simulate, num_neurons, num_nerves, num_edges);
  while (elapsed_ns < num_ns_to_simulate) {
    printf("current elapsed nanoseconds: %d\n", elapsed_ns);
    time_t current_seconds=getCurrentSeconds();
    // First checks whether the time (in nanoseconds) needs to be updated
    if (current_seconds != seconds) {
      seconds=current_seconds;
      if (seconds-start_seconds > 0) {
        if ((seconds-start_seconds) % MIN_LENGTH_NS == 0) {
          if (elapsed_ns == 0) {
            max_iteration_per_ns=min_iteration_per_ns=current_ns_iterations;
          } else {
            if (current_ns_iterations > max_iteration_per_ns) max_iteration_per_ns=current_ns_iterations;
            if (current_ns_iterations < min_iteration_per_ns) min_iteration_per_ns=current_ns_iterations;
          }
          elapsed_ns++;
          current_ns_iterations=0;
          for (int i=0;i<num_brain_nodes;i++) {
            brain_nodes[i].signals_last_ns=brain_nodes[i].signals_this_ns;
            brain_nodes[i].signals_this_ns=0;
          }
        }
      }
    }
    // Will run an update phase for each neuron and nerve
    for (int i=0;i<num_brain_nodes;i++) {
      updateNodes(i);
    }
    current_ns_iterations++;
    total_iterations++;
  }

  // Write summary report to file on termination
  generateReport(OUTPUT_REPORT_FILENAME);

  // Highlight this has finished and report performance
  printf("Finished after %d ns, full report written to `%s` file\n", elapsed_ns, OUTPUT_REPORT_FILENAME);
  printf("Performance data: %d total iterations, maximum %d iterations per nanosecond and minimum %d iterations per nanosecond\n",
          total_iterations, max_iteration_per_ns, min_iteration_per_ns);

  freeMemory();

  return 0;
}



/**
 * Will update a specific neuron or nerve, first firing signals if it is a nerve and then
 * handling signals that have been received
 **/
static void updateNodes(int node_idx) {
  if (brain_nodes[node_idx].num_edges > 0) {
    if (brain_nodes[node_idx].node_type == NERVE) {
      // If this is a nerve then fire a random number of signals
      int num_signals_to_fire=getRandomInteger(0, MAX_RANDOM_NERVE_SIGNALS_TO_FIRE);
      for (int i=0;i<num_signals_to_fire;i++) {
          float signalValue=generateDecimalRandomNumber(MAX_SIGNAL_VALUE);
          int signalType=getRandomInteger(0, NUM_SIGNAL_TYPES);
          brain_nodes[node_idx].num_nerve_inputs[signalType]++;
          fireSignal(node_idx, signalValue, signalType);
      }
    }
  }
  // Now handle all outstanding (recieved) signals
  for (int i=0;i<brain_nodes[node_idx].num_outstanding_signals;i++) {
    handleSignal(node_idx, brain_nodes[node_idx].signalInbox[i].value, brain_nodes[node_idx].signalInbox[i].type);
    brain_nodes[node_idx].signals_this_ns++;
  }
  brain_nodes[node_idx].total_signals_recieved+=brain_nodes[node_idx].num_outstanding_signals;
  brain_nodes[node_idx].num_outstanding_signals=0;
}

/**
 * Handles a specific signal depending on the type of node (neuron or nerve)
 **/
static void handleSignal(int node_idx, float signal, int signal_type) {
  if (brain_nodes[node_idx].node_type == NERVE) {
    // Nerves consume signals and do not send them on
    brain_nodes[node_idx].num_nerve_outputs[signal_type]++;
  } else {
    // A signal is modified by a specific weight depending upon the type of neuron this is
    float changeWeight=NEURON_TYPE_SIGNAL_WEIGHTS[neuronTypeToIndex(brain_nodes[node_idx].neuron_type)];
    signal*=changeWeight;
    int recentSignals=brain_nodes[node_idx].signals_last_ns + brain_nodes[node_idx].signals_this_ns;
    if (recentSignals > 500) {
      // If there have been lots of recent signals then the neuron is becomming overwhelmed, might drop a signal or reduce it
      if (getRandomInteger(0, 2) == 1) signal/=2.0;
      if (getRandomInteger(0, 3) == 1) return;
    }
    fireSignal(node_idx, signal, signal_type);
  }
}

/**
 * Fires a signal from either a neuron or nerve
 **/
static void fireSignal(int node_idx, float signal, int signal_type) {
  while (signal >= 0.001) {
    // Needs to be slightly above 0.0 due to rounding
    int edge_to_use=getRandomInteger(0, brain_nodes[node_idx].num_edges);
    int edge_idx=brain_nodes[node_idx].edges[edge_to_use];
    int tgt_neuron=edges[edge_idx].from == brain_nodes[node_idx].id ? edges[edge_idx].to : edges[edge_idx].from;

    float signal_to_send=signal;
    // Check if the signal exceeds the capacity of the edge, if so will need to be sent in multiple chunks
    if (signal_to_send > edges[edge_idx].max_value) signal_to_send=edges[edge_idx].max_value;
    signal-=signal_to_send;

    // Weight the signal based upon it's type and this edge's weighting of that
    float type_weight=edges[edge_idx].messageTypeWeightings[signal_type];
    signal_to_send*=type_weight;

    if (brain_nodes[tgt_neuron].num_outstanding_signals < SIGNAL_INBOX_SIZE) {
      // We ensure that the target neuron's inbox can hold this signal. If not then throw it away
      brain_nodes[tgt_neuron].signalInbox[brain_nodes[tgt_neuron].num_outstanding_signals].value=signal_to_send;
      brain_nodes[tgt_neuron].signalInbox[brain_nodes[tgt_neuron].num_outstanding_signals].type=signal_type;
      brain_nodes[tgt_neuron].num_outstanding_signals++;
    }
  }
}

/**
 * Edges are read from the input file, but are not connected up. This function will associate, for each neuron or nerve,
 * the edges that go out of it (e.g. will be used to send signals).
 */
static void linkNodesToEdges() {
  for (int i=0;i<num_brain_nodes;i++) {
    int neuron_id=brain_nodes[i].id;
    int number_edges=getNumberOfEdgesForNode(neuron_id);
    brain_nodes[i].num_edges=number_edges;
    brain_nodes[i].edges=(int*) malloc(sizeof(int) * number_edges);
    int edge_idx=0;
    for (int j=0;j<num_edges;j++) {
      if ((edges[j].from == neuron_id) || (edges[j].to == neuron_id && edges[j].direction == BIDIRECTIONAL)) {
        brain_nodes[i].edges[edge_idx]=j;
        edge_idx++;
      }
    }
  }
}

/**
 * Retrieved the number of edges associated in the out direction for a neuron or nerve
 **/
static int getNumberOfEdgesForNode(int node_id) {
  int counted_edges=0;
  for (int i=0;i<num_edges;i++) {
    if (edges[i].from == node_id) {
      counted_edges++;
    } else if (edges[i].to == node_id && edges[i].direction == BIDIRECTIONAL) {
      counted_edges++;
    }
  }
  return counted_edges;
}

/**
 * Maps the enumeration type to integer value for neuron type
 **/
static int neuronTypeToIndex(enum NeuronType neuron_type) {
  if (neuron_type == SENSORY) return 0;
  if (neuron_type == MOTOR) return 1;
  if (neuron_type == UNIPOLAR) return 2;
  if (neuron_type == PSEUDOUNIPOLAR) return 3;
  if (neuron_type == BIPOLAR) return 4;
  if (neuron_type == MULTIPOLAR) return 5;
  assert(0);
}

/**
 * Generates a random integer between two values, including the from value up to the to value minus
 * one, i.e. from=0, to=100 will generate a random integer between 0 and 99 inclusive
 **/
static int getRandomInteger(int from, int to) {
  return (rand() % (to-from)) + from;
}

/**
 * Generates a floating point random number up to a specific integer value, e.g. providing 100 will generate
 * from 0.0 to 100.0
 **/
static float generateDecimalRandomNumber(int to) {
  return (((float) rand()) / RAND_MAX)*to;
}

/**
 * Retrieves the current time in seconds
 **/
//static time_t getCurrentSeconds() {
//  struct timeval curr_time;
//  gettimeofday(&curr_time, NULL);
//  time_t current_seconds=curr_time.tv_sec;
//  return current_seconds;
//}

/**  
 * Retrieves the current time in seconds  
 **/  
static time_t getCurrentSeconds() {  
    return time(NULL); // time() returns the current time in seconds  
} 

/**
 * Parses the provided brain map file and uses this to build information
 * about each neuron, nerve and edge that connects them together
 **/
 static void loadBrainGraph(char * filename) {
  const char whitespace[]=" \f\n\r\t\v";
  enum ReadMode currentMode=NONE;
  int currentNeuronIdx=0, currentEdgeIdx=0;
  char buffer[MAX_LINE_LEN];
  FILE * f;
  printf("filename: %s\n", filename);
  fopen_s(&f, filename, "r");
  if (f == NULL) {
    fprintf(stderr, "Error opening roadmap file '%s'\n", filename);
    exit(-1);
  }

  while(fgets(buffer, MAX_LINE_LEN, f)) {
    if (buffer[0]=='%') continue;
    char * line_contents=buffer+strspn(buffer, whitespace);
    if (strncmp("<num_neurons>", line_contents, 13)==0) {
      char * s=strstr(line_contents, ">");
      num_neurons=atoi(&s[1]);

    }
    if (strncmp("<num_nerves>", line_contents, 12)==0) {
      char * s=strstr(line_contents, ">");
      num_nerves=atoi(&s[1]);
    }
    if (brain_nodes == NULL && num_neurons > 0 && num_nerves > 0) {
      num_brain_nodes=num_neurons + num_nerves;
      brain_nodes=(struct NeuronNerveStruct*) malloc(sizeof(struct NeuronNerveStruct) * num_brain_nodes);
    }
    if (strncmp("<num_edges>", line_contents, 11)==0) {
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      num_edges=atoi(&s[1]);
      edges=(struct EdgeStruct*) malloc(sizeof(struct EdgeStruct) * num_edges);
    }
    if (strncmp("<neuron>", line_contents, 8)==0 || strncmp("<nerve>", line_contents, 7)==0) {
      currentMode=NEURON_NERVE;
      brain_nodes[currentNeuronIdx].num_edges=0;
      brain_nodes[currentNeuronIdx].num_outstanding_signals=0;
      brain_nodes[currentNeuronIdx].signals_this_ns=0;
      brain_nodes[currentNeuronIdx].signals_last_ns=0;
      brain_nodes[currentNeuronIdx].total_signals_recieved=0;
      brain_nodes[currentNeuronIdx].signalInbox=(struct SignalStruct*) malloc(sizeof(struct SignalStruct) * SIGNAL_INBOX_SIZE);

      brain_nodes[currentNeuronIdx].num_nerve_outputs=(int*) malloc(sizeof(int)*NUM_SIGNAL_TYPES);
      brain_nodes[currentNeuronIdx].num_nerve_inputs=(int*) malloc(sizeof(int)*NUM_SIGNAL_TYPES);
      for (int j=0;j<NUM_SIGNAL_TYPES;j++) {
        brain_nodes[currentNeuronIdx].num_nerve_outputs[j]=brain_nodes[currentNeuronIdx].num_nerve_inputs[j]=0;
      }

      if (strncmp("<neuron>", line_contents, 8)==0) {
        brain_nodes[currentNeuronIdx].node_type=NEURON;
      } else if (strncmp("<nerve>", line_contents, 7)==0) {
        brain_nodes[currentNeuronIdx].node_type=NERVE;
      } else {
        assert(0);
      }
      if (currentNeuronIdx >= num_brain_nodes) {
        fprintf(stderr, "Too many neurons and nerves, increase number in <num_neurons> and <num_nerves>\n");
        exit(-1);
      }
    }

    if (strncmp("</neuron>", line_contents, 9)==0 || strncmp("</nerve>", line_contents, 8)==0) {
      currentMode=NONE;
      currentNeuronIdx++;
    }

    if (strncmp("<edge>", line_contents, 6)==0) {
      currentMode=EDGE;
      if (currentEdgeIdx >= num_edges) {
        fprintf(stderr, "Too many edges increase number in <num_edges>\n");
        exit(-1);
      }
      edges[currentEdgeIdx].messageTypeWeightings=(float*) malloc(sizeof(float) * NUM_SIGNAL_TYPES);
    }

    if (strncmp("</edge>", line_contents, 7)==0) {
      currentMode=NONE;
      currentEdgeIdx++;
    }

    if (strncmp("<id>", line_contents, 4)==0) {
      assert(currentMode == NEURON_NERVE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      int id=atof(&s[1]);
      brain_nodes[currentNeuronIdx].id=id;
    }

    if (strncmp("<x>", line_contents, 3)==0 || strncmp("<y>", line_contents, 3)==0 || strncmp("<z>", line_contents, 3)==0) {
      assert(currentMode == NEURON_NERVE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      float val=atof(&s[1]);
      if (strncmp("<x>", line_contents, 3)==0) {
        brain_nodes[currentNeuronIdx].x=val;
      } else if (strncmp("<y>", line_contents, 3)==0) {
        brain_nodes[currentNeuronIdx].y=val;
      } else if (strncmp("<z>", line_contents, 3)==0) {
        brain_nodes[currentNeuronIdx].z=val;
      } else {
        assert(0);
      }
    }

    if (strncmp("<type>", line_contents, 6)==0) {
      assert(currentMode == NEURON_NERVE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      if (strcmp("sensory", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=SENSORY;
      } else if (strcmp("motor", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=MOTOR;
      } else if (strcmp("unipolar", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=UNIPOLAR;
      } else if (strcmp("pseudounipolar", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=PSEUDOUNIPOLAR;
      } else if (strcmp("bipolar", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=BIPOLAR;
      } else if (strcmp("multipolar", &s[1]) == 0) {
        brain_nodes[currentNeuronIdx].neuron_type=MULTIPOLAR;
      } else {
        fprintf(stderr, "Neuron type of '%s' unknown for neuron %d", &s[1], brain_nodes[currentNeuronIdx].id);
        exit(-1);
      }
    }

    if (strncmp("<from>", line_contents, 6)==0 || strncmp("<to>", line_contents, 4)==0) {
      assert(currentMode == EDGE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      int neuron_id=atoi(&s[1]);
      if (strncmp("<from>", line_contents, 6)==0) {
        edges[currentEdgeIdx].from=neuron_id;
      } else if (strncmp("<to>", line_contents, 4)==0) {
        edges[currentEdgeIdx].to=neuron_id;
      } else {
        assert(0);
      }
    }

    if (strncmp("<max_value>", line_contents, 11)==0) {
      assert(currentMode == EDGE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      float max_value=atof(&s[1]);
      edges[currentEdgeIdx].max_value=max_value;
    }

    if (strncmp("<direction>", line_contents, 11)==0) {
      assert(currentMode == EDGE);
      char * s=strstr(line_contents, ">");
      char * e=strstr(s, "<");
      e[0]='\0';
      if (strcmp("unidirectional", &s[1]) == 0) {
        edges[currentEdgeIdx].direction=UNIDIRECTIONAL;
      } else if (strcmp("bidirectional", &s[1]) == 0) {
        edges[currentEdgeIdx].direction=BIDIRECTIONAL;
      } else {
        fprintf(stderr, "Direction type of '%s' unknown", &s[1]);
        exit(-1);
      }
    }

    if (strncmp("<weighting_", line_contents, 11)==0) {
      assert(currentMode == EDGE);
      char * s=strstr(line_contents, "_");
      char * e=strstr(s, ">");
      e[0]='\0';
      int weight_idx=atoi(&s[1]);
      assert (weight_idx < NUM_SIGNAL_TYPES);
      char * c=strstr(&e[1], "<");
      c[0]='\0';
      float val=atof(&e[1]);
      edges[currentEdgeIdx].messageTypeWeightings[weight_idx]=val;
    }

  }
  fclose(f);
}

/**
 * Frees up memory once simulation is completed
 **/
static void freeMemory() {
  for (int i=0;i<num_edges;i++) {
    free(edges[i].messageTypeWeightings);
  }
  free(edges);

  for (int i=0;i<num_brain_nodes;i++) {
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
 static void generateReport(const char * report_filename) {
  FILE * output_report=fopen(report_filename, "w");
  fprintf(output_report, "Simulation ran with %d neurons, %d nerves and %d total edges until %d ns\n", num_neurons, num_nerves, num_edges, elapsed_ns);
  fprintf(output_report, "\n");
  int node_ctr=0;
  for (int i=0;i<num_brain_nodes;i++) {
    if (brain_nodes[i].node_type == NERVE) {
      fprintf(output_report, "Nerve number %d with brain node id: %d\n", node_ctr, brain_nodes[i].id);
      for (int j=0;j<NUM_SIGNAL_TYPES;j++) {
        fprintf(output_report, "----> Signal type %d: %d firings and %d received\n", j, brain_nodes[i].num_nerve_inputs[j], brain_nodes[i].num_nerve_outputs[j]);
      }
      node_ctr++;
    }
  }
  fprintf(output_report, "\n");
  node_ctr=0;
  for (int i=0;i<num_brain_nodes;i++) {
    if (brain_nodes[i].node_type == NEURON) {
      fprintf(output_report, "Neuron number %d, brain node id %d, total signals received %d\n", node_ctr, brain_nodes[i].id, brain_nodes[i].total_signals_recieved);
      node_ctr++;
    }
  }
  fclose(output_report);
}