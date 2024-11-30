#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ia_model.h"


// INITIALIZATION

// Glorot uniform initialization
float glorot_uniform(int nb_input, int nb_output) {
    float limit = sqrt(6.0 / (nb_input + nb_output)); 
    return ((float)rand() / RAND_MAX) * limit * 2 - limit;
}
// Initialize a layer
void initialize_layer(Layer *layer, int input_size, int output_size) {

    // Check the input/output parameters
    if (input_size < 0 || output_size < 0)
    {
       perror("Incorrect input/output values"); 
       exit(EXIT_FAILURE); 
    }

    layer->input_size = input_size;
    layer->output_size = output_size;

    // Calculate total number of weights = input_size * output_size
    int connections = input_size * output_size;

    // Allocate memory for weights
    layer->weights = malloc(sizeof(float) * connections);
    if (!layer->weights) {
        perror("Failed to allocate memory for weights");
        exit(EXIT_FAILURE);
    }

    layer->biases = malloc(sizeof(float) * output_size);
    if (!layer->biases) {
        perror("Failed to allocate memory for biases");
        free(layer->weights);
        exit(EXIT_FAILURE);
    }

    // Initialize weights and biases
    for (int i = 0; i < connections; i++) {
        layer->weights[i] = glorot_uniform(input_size, output_size);
    }

    for (int j = 0; j < output_size; j++) {
        layer->biases[j] = 0.0f;
    }
}

// Initialize a network
void initialize_network(Network *network, Layer **layers, int total_layers, int input_layer_idx, int output_layer_idx) {
    // Check if the parameters are incorrects
    if (total_layers <= 0 || !layers || input_layer_idx < 0 || output_layer_idx < 0) {
        perror("Incorrects inputs");
        exit(EXIT_FAILURE);
    }

    network->layers = layers;
    network->total_layers = total_layers;
    network->input_layer = layers[input_layer_idx];
    network->output_layer = layers[output_layer_idx];
    network->nb_hidden_layers = total_layers - 2;

    // if the neuron needs to have hidden layer
    if (network->nb_hidden_layers > 0) {
        // allocate memory for hidden layers
        network->hidden_layers = malloc(sizeof(Layer *) * network->nb_hidden_layers);
        if (!network->hidden_layers) {
            perror("Failed to allocate memory for hidden layers");
            exit(EXIT_FAILURE);
        }

        for (int i = 0, j = 0; i < total_layers; i++) {
            if (i != input_layer_idx && i != output_layer_idx) {
                network->hidden_layers[j++] = layers[i];
            }
        }
    } else {
        network->hidden_layers = NULL;
    }
}

// Initialization of arrays of values for each layer of the network
void initialize_output_layer_values(Network *network, float ***output_values) {
    // The total number of arrays
    int total_layers = network->total_layers;
    // Allocate memory for each output values array
    *output_values = malloc(sizeof(float *) * total_layers);
    if (!*output_values) {
        perror("Failed to allocate memory for output values array");
        exit(EXIT_FAILURE);
    }

    // Loop through each array to create dynamic 0-value initiated
    for (int i = 0; i < total_layers; i++) {
        (*output_values)[i] = calloc(network->layers[i]->output_size, sizeof(float));
        // If an allocation of a specific array fails, we free all previous arrays that succeeded
        if (!(*output_values)[i]) {
            perror("Failed to allocate memory for layer output values");
            for (int j = 0; j < i; j++) {
                free((*output_values)[j]);
            }
            // then we free the pointer to matrice
            free(*output_values);
            exit(EXIT_FAILURE);
        }
    }
}



// FORWARD 

// Leaky ReLU 
void leaky_relu(float * input_values, int hidden_layer_size, float coefficient) {
 if (coefficient <= 0 || coefficient >= 0.1) {
    perror("Bad use of LeakyRelu coefficient"); 
    exit(EXIT_FAILURE); 
 }
 for (int i = 0; i < hidden_layer_size ; i++) {
     input_values[i] = (input_values[i] < 0 ) ? input_values[i] * coefficient : input_values[i]; 

 }
}

// Softmax
void softmax(float * input_values, int output_layer_size) {
    // Define a sum
    float sum = 0.0f;  
    // define a max
    float max = input_values[0] ; 
    for(int i = 1; i<output_layer_size; i++) 
        max = input_values[i] > max ? input_values[i] : max; 

    // collect the sum apply
    for(int i = 0; i<output_layer_size; i++) {
        input_values[i]=exp(input_values[i]- max); 
        sum += input_values[i]; 
    }

    // assign values applied to softmax 
    for(int i = 0; i<output_layer_size; i++) {
        input_values[i] /= sum; 
    }

}

// Forward propagation 
void forward_propagation(Layer * layer, float * input, float * output) {
    
    if (!layer || !layer->weights || !layer->biases) {
        perror("Invalid layer structure");
        exit(EXIT_FAILURE);
    }
    if (!output|| !input) {
        perror("Error output/input array empty !"); 
        exit(EXIT_FAILURE); 
    }    
    
    int previous_output_nodes = layer->input_size; 
    int current_output_nodes = layer->output_size;
    
    for (int i = 0; i<current_output_nodes; i++) {
        // Start to initialize outputs with biases
        output[i] = layer->biases[i];  
        for (int j = 0 ; j < previous_output_nodes; j ++ ){
            // associate 
            output[i] +=  input[j] * layer->weights[ j * current_output_nodes + i ];
        }
    }
}



void forward_pass(Network *network, float *input_values, float **output_values, float leaky_relu_coefficient) {
    if (!network || !input_values || !output_values) {
        perror("Invalid arguments passed to forward_pass.");
        exit(EXIT_FAILURE);
    }

    float *current_input = input_values; // Start with the input layer

    // Print initial input values
    printf("Input values:\n");
    for (int i = 0; i < network->layers[0]->input_size; i++) {
        printf("%f ", input_values[i]);
    }
    printf("\n");

    for (int i = 0; i < network->total_layers; i++) {
        // Perform forward propagation
        forward_propagation(network->layers[i], current_input, output_values[i]);

        // Debug: Print output values before activation
        printf("Layer %d output (before activation):\n", i + 1);
        for (int j = 0; j < network->layers[i]->output_size; j++) {
            printf("%f ", output_values[i][j]);
        }
        printf("\n");

        // Apply LeakyReLU to hidden layers (not the last layer)
        if (i < network->total_layers - 1) {
            leaky_relu(output_values[i], network->layers[i]->output_size, leaky_relu_coefficient);

            // Debug: Print output values after LeakyReLU activation
            printf("Layer %d output (after LeakyReLU):\n", i + 1);
            for (int j = 0; j < network->layers[i]->output_size; j++) {
                printf("%f ", output_values[i][j]);
            }
            printf("\n");
        }

        // Update the current input for the next layer
        current_input = output_values[i];
    }

    // Apply softmax to the output layer (last layer's output)
    softmax(output_values[network->total_layers - 1], network->layers[network->total_layers - 1]->output_size);

    // Debug: Print final output values after softmax
    printf("Final output values (after softmax):\n");
    for (int i = 0; i < network->layers[network->total_layers - 1]->output_size; i++) {
        printf("%f ", output_values[network->total_layers - 1][i]);
    }
    printf("\n");
}


// RESULTS AND LOSS 

// Output gradient
void output_gradient(float * output_values, float * output_gradient, int label) {



}

// FREE FUNCTIONS 

// Free a single layer
void free_layer(Layer *layer) {
    if (layer) {
        free(layer->weights);
        free(layer->biases);
        free(layer); // Libère la mémoire allouée pour la couche elle-même
    }
}

// Free all layers in a network
void free_layers(Layer **layers, int total_layers) {
    for (int i = 0; i < total_layers; i++) {
        free_layer(layers[i]); 
    }
    free(layers); 
}

// Free a network
void free_network(Network *network) {
    free_layers(network->layers, network->total_layers);
    if (network->hidden_layers) {
        free(network->hidden_layers);
    }
    free(network); 
}
// Free output layer values
void free_output_layer_values(Network *network, float **output_values) {
    if (output_values) {
        for (int i = 0; i < network->total_layers; i++) {
            free(output_values[i]);
        }
        free(output_values);
    }
}


// Main function for testing
int main(int argc, char **argv) {
    srand(time(NULL));

    // Allocate and initialize layers
    Layer **layers = malloc(sizeof(Layer *) * 4);
    layers[0] = malloc(sizeof(Layer));
    layers[1] = malloc(sizeof(Layer));
    layers[2] = malloc(sizeof(Layer));
    layers[3] = malloc(sizeof(Layer));

    // Initialize layers with input and output sizes
    initialize_layer(layers[0], 3, 5); // Input layer: 3 inputs, 5 outputs
    initialize_layer(layers[1], 5, 4); // Hidden layer 1: 5 inputs, 4 outputs
    initialize_layer(layers[2], 4, 2); // Hidden layer 2: 4 inputs, 2 outputs
    initialize_layer(layers[3], 2, 4); // Output layer: 2 inputs, 4 outputs

    // Allocate and initialize network
    Network *network = malloc(sizeof(Network));
    initialize_network(network, layers, 4, 0, 3);

    // Display neural network initialized
    printf("Network has %d total layers and %d hidden layers\n", network->total_layers, network->nb_hidden_layers);

    // Initialize output values for each layer
    float **output_values = NULL;
    initialize_output_layer_values(network, &output_values);

    // Input values to the network
    float input_values[] = {11.0, -2.0, 4.0};

    // Perform forward pass
    float leaky_relu_coefficient = 0.01;
    forward_pass(network, input_values, output_values, leaky_relu_coefficient);

    // Display final output
    printf("Final output values after softmax:\n");
    for (int i = 0; i < network->layers[network->total_layers - 1]->output_size; i++) {
        printf("%f ", output_values[network->total_layers - 1][i]);
    }
    printf("\n");

    // Free all resources
    free_output_layer_values(network, output_values);
    free_network(network);

    return 0;
}
