#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ia_model.h"


// glorot uniform init
float glorot_uniform(int nb_input, int nb_output) {

    float limit = sqrt(0.6/(nb_input * nb_output)); 
    return ((float)rand()/RAND_MAX) * limit * 2 - limit;

}


// write the initialize_layer function
void initialize_layer(Layer * layer, int input_size, int output_size) {

    if ( input_size < 0 || output_size < 0 || layer == NULL) {

        fprintf(stderr, "Invalid values to init layer"); 
        return; 
    } 
    layer->input_size = input_size; 
    layer->output_size = output_size; 

    // get the total connections number
    int connections = input_size * output_size; 
    // intialize weights and biases
    // allocate memory 
    layer->weights = malloc(sizeof(float) * connections) ; 
    layer->biases = malloc(sizeof(float) * output_size ); 

    for (int i = 0 ; i < connections ; i++ ) {
        layer->weights[i] = glorot_uniform(input_size,output_size); 
    }

    for (int j = 0; j < output_size; j ++ ) {
        layer->biases[j] = 0.0f; 
    }

}
//
void initialize_network(Network * network, Layer ** layers, int input_layer_idx, int output_layer_idx, int total_layers) {

    if ( input_layer_idx < 0 || output_layer_idx < 0 || layers == NULL || network == NULL ) {

        fprintf(stderr, "Cannot initialize the network : incorrect indexes\n"); 
        return ; 
    }
    if ( input_layer_idx >= total_layers || output_layer_idx >= total_layers ) {

        fprintf(stderr, "Cannot initialize the network : indexes out of bounds \n");  
    }

    // initialize layer
    network->layers = layers; 
    network->input_layer = layers[input_layer_idx]; 
    network->output_layer = layers[output_layer_idx]; 

    network->total_layers = total_layers; 
    network->nb_hidden_layers = total_layers - 2; 

    if (network->nb_hidden_layers > 0 ) {

        // then allocate memory to save hidden layers 

        network->hidden_layers = malloc(sizeof(Layer *) * network->nb_hidden_layers); 


        if (network->hidden_layers == NULL) {
            fprintf(stderr, "Cannot Initialize the network - Hidden layer issue\n");
            return; 
        }
        for (int i = 1; i< network->nb_hidden_layers; i++ ) {

            network->hidden_layers[i-1]=layers[i]; 
        }
    } else {

        // if the neuron doesn't have hidden layer - make it null
        network->hidden_layers = NULL; 
    }


}





int main(int argc, char ** argv) {

return 0; 


}


