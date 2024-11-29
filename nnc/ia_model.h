#define HIDDEN_LAYERS 2
#define INPUT_SIZE 784
#define OUTPUT_SIZE 9

typedef struct neural_layer {

    float * weights ; 
    float * biases;
    int input_size;// number of previous nodes/ouputs layer
    int output_size;// current layer nodes/outputs 

} Layer; 

typedef struct neural_network {
    Layer ** layers; 
    Layer * input_layer; 
    Layer * output_layer; 
    Layer ** hidden_layers; 
    int total_layers; 
    int nb_hidden_layers; 
}Network;   

// initialization functions : 
// uniform random initialization : Glorot Uniform
float glorot_uniform(int nb_input, int nb_output);

// network & layer init
void initialize_network(Network * network, Layer ** layers, int input_layer_idx, int output_layer_idx, int total_layers);

void initialize_layer(Layer * layer, int input_size, int output_size); 

// forward propagation
// hidden layer activation function : Leaky ReLU
void leaky_relu(float * input_values, int hidden_layer_size, float coefficient); 
// forward function
void forward_propagation (Layer * layer, float * input, float * output); 
// output layer activation : softmax
void softmax(float * input_values, int output_layer_size); 
