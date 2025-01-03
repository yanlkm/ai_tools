#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "ia_model.h"
#include <stdbool.h>


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


    // Initialize the layer
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

    // Allocate memory for biases
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

    // Initialize the network
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
    // Define a max
    float max = input_values[0] ; 
    for(int i = 1; i<output_layer_size; i++) 
        max = input_values[i] > max ? input_values[i] : max; 

    // Collect the sum apply
    for(int i = 0; i<output_layer_size; i++) {
        input_values[i]=exp(input_values[i]- max); 
        sum += input_values[i]; 
    }

    // Assign values applied to softmax 
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
    // if previous output size is 0 then we have an input layer so previous_output_nodes = current_output_nodes
    int previous_output_nodes = 0;
    if (layer->input_size == 0) {
        previous_output_nodes = layer->output_size;
    } else {
     previous_output_nodes = layer->input_size; 
    }
    int current_output_nodes = layer->output_size;
    
    
    if (layer->input_size == 0) {
    
        // Copy input values to output for the input layer
        for (int i = 0; i < current_output_nodes; i++) {
            output[i] = input[i];
        }
    } else {
    
    for (int i = 0; i<current_output_nodes; i++) {
        // Start to initialize outputs with biases
        output[i] = layer->biases[i];  
        for (int j = 0 ; j < previous_output_nodes; j ++ ){
            // Add the weights multiplied by the input values 
            output[i] +=  input[j] * layer->weights[ j * current_output_nodes + i ];
        }
    }
    }

}


// Forward pass
void forward_pass(Network *network, float *input_values, float **output_values, float leaky_relu_coefficient) {
    
    // Check if the network is valid
    if (!network || !input_values || !output_values) {
        perror("Invalid arguments passed to forward_pass.");
        exit(EXIT_FAILURE);
    }
    // Start with the input layer
    float *current_input = input_values; 

    for (int i = 0; i < network->total_layers; i++) {
        // Perform forward propagation
        forward_propagation(network->layers[i], current_input, output_values[i]);

        // Apply LeakyReLU to hidden layers (not the last layer)
        if (i < network->total_layers - 1) {
            leaky_relu(output_values[i], network->layers[i]->output_size, leaky_relu_coefficient);
        }

        // Update the current input for the next layer
        current_input = output_values[i];
    }

    // Apply softmax to the output layer (last layer's output)
    softmax(output_values[network->total_layers - 1], network->layers[network->total_layers - 1]->output_size);
}


// BACKWARD
// Output gradient
void output_gradient(float *output_values, float *output_gradient, float label, int output_size) {
    // Check if the output values, gradient, and size are valid
    if (!output_values || !output_gradient || output_size <= 0) {
        perror("Incorrect output values, size or gradient");
        exit(EXIT_FAILURE);
    }

    // Compute outgradient
    for (int i = 0; i < output_size; i++) {
        output_gradient[i] = output_values[i] - (i == (int)label ? 1.0f : 0.0f);
    }

}

// Leaky ReLU derivative
float leaky_relu_derivative(float coefficient, float logit) {
    return (logit > 0) ? 1.0f : coefficient;
}

// Activation value gradient
float activation_value_gradient(float next_layer_gradient, float *weights, int current_output_nodes) {
    float gradient = 0.0f;
    for (int i = 0; i < current_output_nodes; i++) {
        gradient += next_layer_gradient * weights[i]; 
    }
    return gradient;
}


// Computed value gradient
float computed_value_gradient(float activated_gradient, float leaky_relu_coefficient, float logit) {
    float leaky_relu_grad = leaky_relu_derivative(leaky_relu_coefficient, logit);
    return activated_gradient * leaky_relu_grad;
}

// Backward propagation
void backward_propagation(Layer *layer, float *input_values, float *next_layer_activated_gradients,
                          float *current_layer_activated_gradients, float derivative_lrelu_coefficient, float learning_rate, bool isLastLayer) {
    // Check if the arguments are valid
    if (!layer || !input_values || !next_layer_activated_gradients || !current_layer_activated_gradients) {
        perror("Invalid arguments for backward_propagation");
        exit(EXIT_FAILURE);
    }

    // check if is the input layer
    if (layer->input_size == 0) {
        printf("The input layer has no weights or biases to update.\n");
    } else {

    // Get the number of nodes in the previous and current layers
    int previous_output_nodes = layer->input_size;
    int current_output_nodes = layer->output_size;

    // Allocate temporary gradients
    float *temp_gradients = calloc(current_output_nodes, sizeof(float));
    if (!temp_gradients) {
        perror("Memory allocation failed for temp_gradients");
        exit(EXIT_FAILURE);
    }


    // Step 1: Compute gradients of activated values
    for (int i = 0; i < current_output_nodes; i++) {
        // Compute the gradient of the activation value : dL/dy(l-1) = dL/dzl * w
        if (isLastLayer) {
            temp_gradients[i] = next_layer_activated_gradients[i];
        } else {
        temp_gradients[i] = activation_value_gradient(next_layer_activated_gradients[i], layer->weights + i * current_output_nodes, current_output_nodes);
        }
    }

    // Step 2: Compute gradients of computed values
    for (int i = 0; i < current_output_nodes; i++) {
        // Compute the gradient of the computed value
        current_layer_activated_gradients[i] = computed_value_gradient(temp_gradients[i], derivative_lrelu_coefficient, input_values[i]);
    }

    // Step 3: Update weights and biases
    for (int i = 0; i < previous_output_nodes; i++) {
        for (int j = 0; j < current_output_nodes; j++) {
            layer->weights[i * current_output_nodes + j] -=
                learning_rate * current_layer_activated_gradients[j] * input_values[i];
        }
    }

    // Update biases
    for (int i = 0; i < current_output_nodes; i++) {
        layer->biases[i] -= learning_rate * current_layer_activated_gradients[i];
    }

    // Free temporary gradients
    free(temp_gradients);

    }
}

// Backward pass
void backward_pass(Network *network, float **output_values, float leaky_relu_coefficient, float learning_rate, float label) {
    // Check if the arguments are valid
    if (!network || !output_values) {
        perror("Invalid arguments for backward_pass");
        exit(EXIT_FAILURE);
    }

    // Allocate gradients for each layer without using malloc or calloc to avoid memory leaks 
    // Get the maximum size of the output layer
    int max = 0;
    for (int i = 0; i < network->total_layers; i++) {
        max = network->layers[i]->output_size > max ? network->layers[i]->output_size : max;
    }
    // Initialize gradients for each layer
    float gradients[network->total_layers][max];
    for (int i = 0; i < network->total_layers; i++) {
        for (int j = 0; j < network->layers[i]->output_size; j++) {
            gradients[i][j] = 0.0f;
        }
    }
    
    // Perform output gradient computation for the last layer
    output_gradient(
        output_values[network->total_layers - 1],       // Output values of the last layer
        gradients[network->total_layers - 1],          // Gradient storage for the last layer
        label,                                          // Target label
        network->layers[network->total_layers - 1]->output_size
    );

    // Propagate gradients backwards through all layers
    for (int layer_idx = network->total_layers - 1; layer_idx > 0; layer_idx--) {
        bool isLastLayer = layer_idx == network->total_layers - 1;
        backward_propagation(
            network->layers[layer_idx],                 // Current layer
            output_values[layer_idx - 1],               // Inputs to the current layer
            gradients[layer_idx],                       // Gradients from the next layer
            gradients[layer_idx - 1],                   // Gradients for the current layer
            leaky_relu_coefficient,                     // Leaky ReLU coefficient
            learning_rate,                              // Learning rate
            isLastLayer                                 // Is this the last layer?
        );
    }
}



// TRAINING

// SAVE TRAINING

/// Load training from file and initialize the network
void load_train(Network *network, char *filename) {
    // Open the file
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file for loading");
        exit(EXIT_FAILURE);
    }

    // Read general information about the network
    if (fscanf(file, "%d %d", &(network)->nb_hidden_layers, &(network)->total_layers) != 2) {
        perror("Failed to read network metadata");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the layers array
    (network)->layers = malloc(sizeof(Layer *) * (network)->total_layers);
    if (!(network)->layers) {
        perror("Failed to allocate memory for layers");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read each layer's information and allocate memory for weights and biases
    for (int i = 0; i < (network)->total_layers; i++) {
        // Allocate memory for the current layer
        (network)->layers[i] = malloc(sizeof(Layer));
        Layer *layer = (network)->layers[i];
        if (!layer) {
            perror("Failed to allocate memory for layer");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Read layer dimensions
        if (fscanf(file, "%d %d", &layer->input_size, &layer->output_size) != 2) {
            perror("Failed to read layer dimensions");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Allocate and read biases
        layer->biases = malloc(sizeof(float) * layer->output_size);
        if (!layer->biases) {
            perror("Failed to allocate memory for biases");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < layer->output_size; j++) {
            if (fscanf(file, "%f", &layer->biases[j]) != 1) {
                perror("Failed to read biases");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }

        // Allocate and read weights
        layer->weights = malloc(sizeof(float) * layer->input_size * layer->output_size);
        if (!layer->weights) {
            perror("Failed to allocate memory for weights");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < layer->input_size; j++) {
            for (int k = 0; k < layer->output_size; k++) {
                if (fscanf(file, "%f", &layer->weights[j * layer->output_size + k]) != 1) {
                    perror("Failed to read weights");
                    fclose(file);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    // Close the file
    fclose(file);
    printf("Training loaded successfully from %s\n", filename);
}


// Save training

void save_train(Network *network, char *filename) {
    // Open the file
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file for saving");
        exit(EXIT_FAILURE);
    }

    // Write general information about the network
    fprintf(file, "%d %d\n", network->nb_hidden_layers, network->total_layers);

    // Save each layer's information (input_size, output_size, weights, biases)
    for (int i = 0; i < network->total_layers; i++) {
        Layer *layer = network->layers[i];

        // First line: layer info (input_size, output_size)
        fprintf(file, "%d %d\n", layer->input_size, layer->output_size);

        // Second line: biases
        for (int j = 0; j < layer->output_size; j++) {
            fprintf(file, "%f ", layer->biases[j]);
        }
        fprintf(file, "\n");

        // Third line: weights
        for (int j = 0; j < layer->input_size; j++) {
            for (int k = 0; k < layer->output_size; k++) {
                fprintf(file, "%f ", layer->weights[j * layer->output_size + k]);
            }
        }
        fprintf(file, "\n");
    }
    // Close the file
    fclose(file);
    printf("Training saved to %s\n", filename);
}

// bool check if a file has saves training structure
bool is_saved(char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return false;
    }

    // check if the file is empty
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fclose(file);
        return false;
    }
    // Close the file
    fclose(file);
    return true;
}

// Application : read mnist data
float **read_mnist_images(const char *filename, int *num_images, int *image_size) {
    // Open the file
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // read the magic number, number of images, number of rows, and number of columns (source of code : https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c)
    int magic_number = 0, num_rows = 0, num_cols = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number);

    fread(num_images, sizeof(int), 1, file);
    *num_images = __builtin_bswap32(*num_images);

    fread(&num_rows, sizeof(int), 1, file);
    num_rows = __builtin_bswap32(num_rows);

    fread(&num_cols, sizeof(int), 1, file);
    num_cols = __builtin_bswap32(num_cols);

    *image_size = num_rows * num_cols;

    //Allocate memory for images
    float **images = malloc(*num_images * sizeof(float *));
    for (int i = 0; i < *num_images; i++) {
        images[i] = malloc(*image_size * sizeof(float));
    }

    // Read images : convert pixel values to float
    unsigned char *buffer = malloc(*image_size);
    for (int i = 0; i < *num_images; i++) {
        fread(buffer, sizeof(unsigned char), *image_size, file);
        for (int j = 0; j < *image_size; j++) {
            images[i][j] = buffer[j] > 127 ? 1.0f : 0.0f;
        }
    }
    // Free buffer and close the file
    free(buffer);
    fclose(file);

    return images;
}

// Read MNIST labels
float *read_mnist_labels(const char *filename, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Read headers
    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    // Convert to big-endian
    magic_number = __builtin_bswap32(magic_number); 
    // Check if the magic number is valid : 2049 because it is the label file
    if (magic_number != 2049) {
        fprintf(stderr, "Invalid magic number: %d\n", magic_number);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read number of labels
    fread(num_labels, sizeof(int), 1, file);
    *num_labels = __builtin_bswap32(*num_labels);
    // Check if the number of labels is valid
    if (*num_labels <= 0) {
        fprintf(stderr, "Invalid number of labels: %d\n", *num_labels);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // Display the magic number and number of labels
    printf("Magic number: %d\n", magic_number);
    printf("Number of labels: %d\n", *num_labels);

    // Read labels
    unsigned char *temp_labels = malloc(*num_labels * sizeof(unsigned char));
    if (fread(temp_labels, sizeof(unsigned char), *num_labels, file) != *num_labels) {
        fprintf(stderr, "Failed to read all labels\n");
        free(temp_labels);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Convert to float array
    float *labels = malloc(*num_labels * sizeof(float));
    for (int i = 0; i < *num_labels; i++) {
        labels[i] = (float)temp_labels[i];
    }
    // Free temporary labels and close the file
    free(temp_labels);
    fclose(file);

    return labels;
}

// Training function 
void training(Network *network, float learning_rate, int epochs, float ***output_values, char *save_file_name, const char *images_file, const char *labels_file) {
    // Initialize output values for each layer
    initialize_output_layer_values(network, output_values);

    // Read MNIST images and labels
    int num_images, image_size, num_labels;
    float **images = read_mnist_images(images_file, &num_images, &image_size);
    float *labels = read_mnist_labels(labels_file, &num_labels);

    if (num_images != num_labels) {
        perror("Number of images and labels don't match");
        exit(EXIT_FAILURE);
    }

    // Define batch size
    int batch_size = 32; // Common convention for batch size : 32, 64, 128, 256, etc.
    int num_batches = (num_images + batch_size - 1) / batch_size; // Round up : ceil(num_images / batch_size)

    // Training loop : for each epoch and batch size : forward pass, backward pass, and update
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Initialize epoch loss
        float epoch_loss = 0.0f;

        printf("Epoch %d/%d\n", epoch + 1, epochs);
        // Loop through each batch
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * batch_size;
            int current_batch_size = (batch_start + batch_size > num_images) ? (num_images - batch_start) : batch_size;
            // Initialize batch loss
            float batch_loss = 0.0f;
            // Loop through each image in the batch
            for (int i = 0; i < current_batch_size; i++) {
                int idx = batch_start + i;
                float *input_values = images[idx];
                float label = labels[idx];

                // Perform forward pass
                float leaky_relu_coefficient = 0.01;
                forward_pass(network, input_values, *output_values, leaky_relu_coefficient);

                // Compute cross-entropy loss
                float *output = (*output_values)[network->total_layers - 1];
                float loss = -log(output[(int)label] + 1e-9); // Add small value to avoid log(0)
                batch_loss += loss;

                // Perform backward pass
                backward_pass(network, *output_values, leaky_relu_coefficient, learning_rate, label);
            }
                // Save the training
                save_train(network, save_file_name);
                printf("The training has been saved into %s\n", save_file_name);

            // Normalize batch loss
            batch_loss /= current_batch_size;
            epoch_loss += batch_loss;

            // Display progress
            printf("  Batch %d/%d - Loss: %f\n", batch + 1, num_batches, batch_loss);
        }

        // Compute average loss for the epoch
        epoch_loss /= num_batches;

        // Display epoch loss
        printf("Epoch %d - Average Loss: %f\n", epoch + 1, epoch_loss);
    }



    // Free MNIST images and labels
    for (int i = 0; i < num_images; i++) {
        free(images[i]);
    }
    free(images);
    free(labels);
}


// TESTING

// TESTING AND ACCURACY (Test does not update weights and biases - its not a training function)
void test(Network *network, float **output_values, float *input_values, float *label_values, float * score) {
    // Perform forward pass
    float leaky_relu_coefficient = 0.01;
    forward_pass(network, input_values, output_values, leaky_relu_coefficient);

    // Determine the predicted label
    float max_value = -1.0f;
    int max_index = -1;
    for (int i = 0; i < network->layers[network->total_layers - 1]->output_size; i++) {
        if (output_values[network->total_layers - 1][i] > max_value) {
            max_value = output_values[network->total_layers - 1][i];
            max_index = i;
        }
    }
    // Determine the score
    *score = (max_index == (int)*label_values) ? 1.0f : 0.0f;
}


// Massive test
void massive_test(Network *network, float **output_values, float **input_values, float *label_values, int total_tests, float *score) {
    // Initialize the score
    *score = 0.0f;

    // Perform tests
    for (int i = 0; i < total_tests; i++) {
        test(network, output_values, input_values[i], label_values + i, score);
    }

    // Compute the accuracy
    *score /= total_tests;
}



// FREE FUNCTIONS 

// Free a single layer
void free_layer(Layer *layer) {
    if (layer) {
        if (layer->weights) {
            free(layer->weights);
            layer->weights = NULL;
        }
        if (layer->biases) {
            free(layer->biases);
            layer->biases = NULL;
        }
        free(layer);
        layer = NULL;
    }
}

// Free all layers in a network
void free_layers(Layer **layers, int total_layers) {
    if (layers) {
        for (int i = 0; i < total_layers; i++) {
            if (layers[i]) {
                free_layer(layers[i]);
                layers[i] = NULL;
            }
        }
        free(layers);
        layers = NULL;
    }
}

// Free a network
void free_network(Network *network) {
    if (network) {
        if (network->layers) {
            free_layers(network->layers, network->total_layers);
            network->layers = NULL;
        }
        if (network->hidden_layers) {
            free(network->hidden_layers);
            network->hidden_layers = NULL;
        }
        free(network);
        network = NULL;
    }
}

// Free output layer values
void free_output_layer_values(Network *network, float **output_values) {
    if (output_values) {
        for (int i = 0; i < network->total_layers; i++) {
            if (output_values[i]) {
                free(output_values[i]);
                output_values[i] = NULL;
            }
        }
        free(output_values);
        output_values = NULL;
    }
}


// Main function for testing
int main(int argc, char **argv) {
    srand(time(NULL));
    // Files definition : save, images and labels
    char * fileName = "save.bin";
    char * images_file = "data/train-images-idx3-ubyte/train-images-idx3-ubyte"; 
    char * labels_file = "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    // Allocate ( and initialize) network
    Network *network = malloc(sizeof(Network)); 
 
    // Check if the training is already saved
    if (is_saved(fileName)) {

        load_train(network,fileName); 

    } else {
    // Allocate and initialize layers
    Layer **layers = malloc(sizeof(Layer *) * 4);
    layers[0] = malloc(sizeof(Layer));
    layers[1] = malloc(sizeof(Layer));
    layers[2] = malloc(sizeof(Layer));
    layers[3] = malloc(sizeof(Layer));

    // Initialize layers with 784 input nodes, 128 hidden nodes, 64 hidden nodes, and 10 output nodes
    initialize_layer(layers[0], 0, 784);
    initialize_layer(layers[1], 784, 128);
    initialize_layer(layers[2], 128, 64);
    initialize_layer(layers[3], 64, 10);

    initialize_network(network, layers, 4, 0, 3);
    }
    // Display neural network initialized
    printf("Network has %d total layers and %d hidden layers\n", network->total_layers, network->nb_hidden_layers);

    // define the learning rate for 60000 images
    float learning_rate = 0.0005;
    // define the number of epochs
    int epochs = 6;

    // Allocate memory for output values
    float **output_values = NULL;
    initialize_output_layer_values(network, &output_values);
    // Training
    //training(network, learning_rate, epochs, &output_values, fileName, images_file, labels_file);

    // read test files : 
    char * images_file_test = "data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"; 
    char * labels_file_test = "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    // get the first image to test 
    int num_images, image_size, num_labels;

    float **images = read_mnist_images(images_file_test, &num_images, &image_size);
    float *labels = read_mnist_labels(labels_file_test, &num_labels);


    if (num_images != num_labels) {
        perror("Number of images and labels don't match");
        exit(EXIT_FAILURE);
    }

    // re intialized the output values
    free_output_layer_values(network, output_values);
    initialize_output_layer_values(network, &output_values);

    // test the neural network 
    test(network, output_values, images[1], &labels[1], &learning_rate);

    // interpret the result
    printf("The score is : %f\n", learning_rate);
    // detail the result 
    printf(" According to the label, the image is a : %d\n", (int)labels[1]);
    printf("the ouput of the network is : \n");
    for (int i = 0; i < 10; i++) {
        printf(" %f of propability for digit : %d\n", output_values[network->total_layers - 1][i],i);
    }

    // Free all resources
    free_output_layer_values(network, output_values);
    free_network(network);

    return 0;
}
