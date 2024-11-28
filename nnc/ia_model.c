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
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Calculate total number of weights
    int connections = input_size * output_size;

    // Allocate memory
    layer->weights = malloc(sizeof(float) * connections);
    if (!layer->weights) {
        perror("Failed to allocate memory for weights");
        exit(EXIT_FAILURE);
    }

    layer->biases = malloc(sizeof(float) * output_size);
    if (!layer->biases) {
        perror("Failed to allocate memory for biases");
        free(layer->weights); // Libérer les poids avant de quitter
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
    if (total_layers <= 0 || !layers || input_layer_idx < 0 || output_layer_idx < 0) {
        fprintf(stderr, "Invalid parameters to initialize_network\n");
        return;
    }

    network->layers = layers;
    network->total_layers = total_layers;
    network->input_layer = layers[input_layer_idx];
    network->output_layer = layers[output_layer_idx];
    network->nb_hidden_layers = total_layers - 2;

    if (network->nb_hidden_layers > 0) {
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

// Free a layer
void free_layer(Layer *layer) {
    if (layer) {
        free(layer->weights);
        free(layer->biases);
    }
}

// Free a network
void free_network(Network *network) {
    if (network->hidden_layers) free(network->hidden_layers);
}

// Main function for testing
int main(int argc, char **argv) {
    srand(time(NULL)); // Initialize random seed

    // Example network initialization
    Layer input_layer, hidden_layer, output_layer;
    initialize_layer(&input_layer, 3, 5);
    initialize_layer(&hidden_layer, 5, 4);
    initialize_layer(&output_layer, 4, 2);

    Layer *layers[] = {&input_layer, &hidden_layer, &output_layer};
    Network network;
    initialize_network(&network, layers, 3, 0, 2);

    // Test the network structure
    printf("Network has %d total layers and %d hidden layers\n", network.total_layers, network.nb_hidden_layers);

    // Free resources
    free_layer(&input_layer);
    free_layer(&hidden_layer);
    free_layer(&output_layer);
    free_network(&network);

    return 0;
}
