#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ia_model.h"

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

// Main function for testing
int main(int argc, char **argv) {
    srand(time(NULL));

    // Allocate and initialize layers 
    Layer **layers = malloc(sizeof(Layer *) * 4);
    layers[0] = malloc(sizeof(Layer));
    layers[1] = malloc(sizeof(Layer));
    layers[2] = malloc(sizeof(Layer));
    layers[3] = malloc(sizeof(Layer));
    
    initialize_layer(layers[0], 3, 5);
    initialize_layer(layers[1], 5, 4);
    initialize_layer(layers[2], 4, 2);
    initialize_layer(layers[3],2,4); 

    // Allocate and initialize network
    Network *network = malloc(sizeof(Network));
    initialize_network(network, layers, 4, 0, 3);

    // Test the network 
    printf("Network has %d total layers and %d hidden layers\n", network->total_layers, network->nb_hidden_layers);

    // Free all resources
    free_network(network);

    return 0;
}
