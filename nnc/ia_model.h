#define HIDDEN_LAYERS 2
#define INPUT_SIZE 784
#define OUTPUT_SIZE 9
#include <stdbool.h>

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

// Initialize output arrays for each layer according to layer-> output (nodes)
void initialize_output_layer_values(Network *network, float ***output_values); 
// forward propagation
// hidden layer activation function : Leaky ReLU
void leaky_relu(float * input_values, int hidden_layer_size, float coefficient); 
// forward function
void forward_propagation (Layer * layer, float * input, float * output); 
// output layer activation : softmax
void softmax(float * input_values, int output_layer_size); 

// forward pass : includes input_values, output_values array and leaky relu coefficient
void forward_pass(Network * network, float * input_values, float ** output_values, float leaky_relu_coefficient);  

// results and loss

// if logit z > 0 then leaky_relu_derivative is 1 if not is coefficient
float leaky_relu_derivative( float coefficient, float logit);  
// output gradient : call it once 
void output_gradient(float * output_values, float * output_gradient, float label, int output_size); 

// Activation gradient is deravative of loss function by activated value : perform first the sum of the logit gradiant of the next layer with the related nodes weights to these logits dL/dy(l-1) = dL/dzl * w 
float activation_value_gradient(float next_layer_gradient, float *weights, int current_output_nodes) ;///
// Computed value gradient is the derivative of loss function by computed value : the product of the activation of "next" layer value (l+1) with the derivative activation function value dL/dzl = dL/dy(l+1) * dy/dz , dy/dz depends on the "logit" (z) value (ex : if z < 0 => derivative coeff is applied if not, is not) 
float computed_value_gradient(float activated_value_gradient, float derivative_lrelu_coefficient, float  logit ); 
// backward propagation on specific Layer, with activated_values_gradient array, 
void backward_propagation(Layer *layer, float *input_values, float *next_layer_activated_gradients, float *current_layer_activated_gradients, float derivative_lrelu_coefficient, float learning_rate, bool isLastLayer);
// backward pass : includes output_values, leaky_relu_coefficient, learning_rate and label
void backward_pass(Network * network, float ** output_values, float leaky_relu_coefficient, float learning_rate, float label);
// train and save training
void training(Network *network, float learning_rate, int epochs, float ***output_values, char * save_file_name);
// saves
// load network already trained
void load_train(Network *network, char *filename); 
// save a network trained
void save_train(Network *network, char *filename); 
// check if saving file has saves or not
bool is_saved(char *filename);
