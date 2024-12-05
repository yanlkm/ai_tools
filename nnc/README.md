# Neural Network in C 

## Introduction 

This is an implementation of a Feedforward Neural Network to perform reading of hand-written digit MNIST dataset. We will describe in this readme each meaning part of the algorithm and bring some mathematical concepts (linear algebra). 

## Overview

### Neural network : meaning & use
A neural network is a collection of units connected called neurons. They are arranged in layers and they can proceed data to make predictions and decisions. They are fundamental in the concept of deep learning subsets of artificial intelligence. Here we are using a Feedforward Neural Network, it does mean that we need to proceed the information flows in a single direction — from the input layer, through the hidden layers, to the output layer. There are no loops or cycles in the connections between nodes. This type of neural network is widely used for tasks like classification and regression.

#### 1. Input Layer
* The first layer, where the data is fed into the network. In our case, it consists of **784 neurons**, each corresponding to a pixel in a 28x28 grayscale image from the MNIST dataset.

#### 2. Hidden Layers 
* Two layers of neurons are used to extract complex features from the data. Each neuron applies a weighted sum of its inputs, adds a bias, and passes the result through an activation function which is ``LeakyReLU``.

#### 3. Output Layer 
* The final layer has **10 neurons**, corresponding to the 10 possible digit classes (0–9). The outputs represent probabilities for each class, computed using a softmax activation function.


## Key Concepts of this algorithm

### Data Structures

### 1. Layer 
```c
typedef struct neural_layer {
    float * weights; 
    float * bias; 
    int input_size; 
    int output_size; 

} Layer; 
```
Layer is the key structure of our algorithm, it does mean to be the set of a collections of neurons/nodes. ``weights`` refers to the
 connections between neurons of the previous layer and the current layer, ``bias`` is a parameter in the neuron that is used to adjust the output along with the weighted sum of the inputs. ``input_size`` and ``output_size`` are the number of neurons in the previous layer and the current layer, respectively.

 ### 2. Network 
```c
typedef struct {
    Layer ** layers; 
    Layer * input_layer; 
    Layer * output_layer; 
    Layer ** hidden_layers; 
    int total_layers; 
    int nb_hidden_layers; 
} Network; 
```
Network is a structure that contains an array of layers and the number of layers in the network. It represents the entire neural network model.
``layers`` is an array of pointers to the layers in the network (two dimensions array). ``input_layer`` and ``output_layer`` are pointers to the input and output layers of the network. ``hidden_layers`` is an array of pointers to the hidden layers in the network. ``total_layers`` is the total number of layers in the network, and ``nb_hidden_layers`` is the number of hidden layers.

## Initialization

### Initialize Layer
```c
void initialize_layer(Layer * layer, int input_size, int output_size) {
    ...
    // Set the input and output sizes
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Calculate total number of weights = input_size * output_size
    int connections = input_size * output_size;

    // Allocate memory for weights and biases
    layer->weights = (float *)malloc(connections * sizeof(float));
    layer->bias = (float *)malloc(output_size * sizeof(float));
    
    // Initialize weights and biases
    // Initialize weights with Glorot Uniform initialization
    for (int i = 0; i < connections; i++) {
        layer->weights[i] = glorot_uniform(input_size, output_size);
    }
    for (int j = 0; j < output_size; j++) {
        layer->biases[j] = 0.0f;
    }
}
```
This function initializes a layer by setting its input and output sizes, allocating memory for weights and biases, and initializing them with small random values using the **Glorot Uniform** initialization method.

#### Parameters:
- `Layer * layer`: A pointer to the layer that needs to be initialized.
- `int input_size`: The number of neurons in the previous layer.
- `int output_size`: The number of neurons in the **current** layer.

#### Glorot Uniform Initialization:
The Glorot Uniform initialization method, also known as Xavier initialization, is used to initialize the weights. This method maintains the variance of activations and gradients throughout the layers. The weights are initialized from a uniform distribution within the range:

$$
W \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right)
$$

where $n_{\text{in}}$ is the number of input units and $n_{\text{out}}$ is the number of output units. This initialization helps in preventing the vanishing and exploding gradient problems, ensuring that the network trains efficiently.


The biases are just initialized to zero 



### Initialize Network
```c
void initialize_network(Network *network, Layer **layers, int total_layers, int input_layer_idx, int output_layer_idx); 
```
This function initializes the entire network by initializing each layer with the specified sizes.
- ``network`` : Pointer to the Network structure to be initialized.
- ``layers`` :  Array of pointers to Layer structures representing the layers of the network.
- ``total_laters`` : Total number of layers in the network.
- ``input_layer_idx`` : Index of the input layer in the layers array.
- ``output_layer_idx``: Index of the output layer in the layers array.

## Forward Propagation
```c
void forward_propagation(Network * network, float * input) 
```
This function performs forward propagation through the network, computing the activations for each layer using the weights, biases, and activation function.

### Process

1. **Input Layer**: The input data is fed into the network.

2. **Hidden Layers**: For each hidden layer, the following steps are performed:
- Compute the weighted sum of inputs (logits):
$$
z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}
$$

where:
  - $z^{(l)}$ is the vector of logits for layer $l$
  - $W^{(l)}$ is the weight matrix for layer $l$
  - $a^{(l-1)}$ is the activation from the previous layer
  - $b^{(l)}$ is the bias vector for layer $l$

Apply the LeakyReLU activation function to compute the activated value:
$$ 
a^{(l)} = \text{LeakyReLU}(z^{(l)})
$$
where LeakyReLU is defined as:
$$
\text{LeakyReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$
and $\alpha$ is a small constant (``leaky_relu_coefficient``) (e.g., 0.00001).

3. **Output Layer**: The final layer computes the logits and applies the softmax activation function to produce probabilities:
$$
z^{(L)} = W^{(L)} \cdot a^{(L-1)} + b^{(L)}
$$
$$
\hat{y} = \text{softmax}(z^{(L)}) = \frac{e^{z^{(L)}}}{\sum_{j} e^{z^{(L)}_j}}
$$
    where $\hat{y}$ is the output probability vector.


## Backward Propagation
```c
// Backward propagation
void backward_propagation(Layer *layer, float *input_values, float *next_layer_activated_gradients float *current_layer_activated_gradients, float derivative_lrelu_coefficient, float learning_rate bool isLastLayer) ;
```
This function performs backward propagation, computing the error at each layer and updating the weights and biases using gradient descent.

### Parameters:
- `Layer *layer`: A pointer to the current layer.
- `float *input_values`: The input values to the current layer (activations from the previous layer or input data).
- `float *next_layer_activated_gradients`: The gradients of the activated values from the next layer.
- `float *current_layer_activated_gradients`: The gradients of the activated values for the current layer.
- `float derivative_lrelu_coefficient`: The coefficient for the derivative of the LeakyReLU activation function.
- `float learning_rate`: The learning rate for gradient descent.
- `bool isLastLayer`: A flag indicating if the current layer is the last layer.

### Process

1. **Output Gradient**: If the current layer is the output layer, compute the output gradient:
    ```c
    void output_gradient(float *output_values, float *output_gradient, float label, int output_size) {
        for (int i = 0; i < output_size; i++) {
            output_gradient[i] = output_values[i] - (i == (int)label ? 1.0f : 0.0f);
        }
    }
    ```
$$
\delta^{(L)} = \hat{y} - y
$$
    where $\hat{y}$ is the predicted output probability vector and $y$ is the true label (one-hot encoded, e.g., [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).

2. **Activation Value Gradient**: Compute the gradient of the activation values for the current layer:
    ```c
    float activation_value_gradient(float next_layer_gradient, float *weights, int current_output_nodes) {
        float gradient = 0.0f;
        for (int i = 0; i < current_output_nodes; i++) {
            gradient += next_layer_gradient * weights[i]; 
        }
        return gradient;
    }
    ```
$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)}) => \delta^{(l)} = \sum_{i=1}^{n} W^{(l+1)}_i \delta^{(l+1)}_i \odot f'(z^{(l)})$
    where $\odot$ denotes element-wise multiplication and $f'(z^{(l)})$ is the derivative of the activation function. The sum $\sum_{i=1}^{n} W^{(l+1)}_i \delta^{(l+1)}_i$ is computed for each neuron in the next layer, it means that we are computing the gradient of the activated values of the current layer using the gradients of the activated values of the next layer.

3. **Computed Value Gradient**: Compute the gradient of the computed values using the derivative of the LeakyReLU activation function:
    ```c
    float computed_value_gradient(float activated_gradient, float leaky_relu_coefficient, float logit) {
        float leaky_relu_grad = (logit > 0) ? 1.0f : leaky_relu_coefficient;
        return activated_gradient * leaky_relu_grad;
    }
    ```
$$
\frac{\partial L}{\partial z^{(l)}} = \delta^{(l)} \odot f'(z^{(l)})
$$
    where $f'(z^{(l)})$ is the derivative of the LeakyReLU activation function. This step computes the gradient of the computed values of the current layer using the gradient of the activated values of the current layer. The derivative of the LeakyReLU activation function is used to compute the gradient of the computed values.

4. **Update Weights and Biases**: Update the weights and biases using the computed gradients and the learning rate:
    ```c
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->weights[i * layer->input_size + j] -= learning_rate * current_layer_activated_gradients[i] * input_values[j];
        }
        layer->bias[i] -= learning_rate * current_layer_activated_gradients[i];
    }
    ```
   - $W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$ 
   - $b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$
       where $\eta$ is the learning rate. The weights and biases are updated using the computed gradients and the learning rate. Here we are using the gradient descent optimization algorithm to update the weights and biases.

This process ensures that the network learns by minimizing the loss function through gradient descent.


## Training

### Training Function
```c
void training(Network *network, float learning_rate, int epochs, float ***output_values, char *save_file_name, const char *images_file, const char *labels_file) {
    ...
}
```
This function trains the neural network using the MNIST dataset, adjusting the weights and biases through multiple epochs.

### Parameters:
- `Network *network`: Pointer to the neural network structure to be trained.
- `float learning_rate`: The learning rate for gradient descent.
- `int epochs`: The number of times the entire training dataset is passed through the network.
- `float ***output_values`: Pointer to store the output values of the network after training.
- `char *save_file_name`: The name of the file to save the trained network parameters.
- `const char *images_file`: The path to the MNIST images file.
- `const char *labels_file`: The path to the MNIST labels file.

### Process

1. **Load Data**: Load the MNIST dataset images and labels.
    ```c
    // Read MNIST images and labels
    int num_images, image_size, num_labels;
    float **images = read_mnist_images(images_file, &num_images, &image_size);
    float *labels = read_mnist_labels(labels_file, &num_labels);
    ```

2. **Training Loop**: Iterate through the dataset for a specified number of epochs.
    ```c
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
        // Forward propagation
        forward_pass(network, input_values, *output_values, leaky_relu_coefficient);
        ...
        // Compute loss
        float loss = compute_loss(network->output_layer->output, labels[i]);
        ...
        // Perform backward pass
        backward_pass(network, *output_values, leaky_relu_coefficient, learning_rate, label);
        }
    }
    ```

3. **Forward Pass**: Compute the activations for each layer using the weights, biases, and activation functions.
    ```c
    void forward_pass(Network *network, float *input_values, float **output_values, float leaky_relu_coefficient) 
    ```

4. **Compute Loss**: Calculate the cross-entropy loss for classification tasks.
    ```c
        float batch_loss = 0.0f;
        ... 
        float *output = (*output_values)[network->total_layers - 1];
        float loss = -log(output[(int)label] + 1e-9); // Add small value to avoid log(0)
        batch_loss += loss;
    ```
    - $L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$
          where $y$ is the true label and $\hat{y}$ is the predicted output probability vector.

5. **Backward Pass**: Adjust the weights and biases using gradient descent for each layer.
    ```c
    void backward_pass(Network *network, float **output_values, float leaky_relu_coefficient, float learning_rate, float label)
    ```
- $W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$
- $b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$
    where $\eta$ is the learning rate. The weights and biases are updated using the computed gradients and the learning rate.

6. **Save Training**: Save the trained network parameters to a file.
    ```c
    // Save the training
    void save_train(Network *network, char *save_file_name)
    ```
    The trained network parameters are saved to a file for future use : 
    - first line : network properties (number of layers, input size, output size, number of hidden layers)
    - next lines : weights and biases of each layer

### MNIST Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). Each image is flattened into a 784-dimensional vector and fed into the input layer of the network. The corresponding label is a one-hot encoded vector representing the digit class.

This process ensures that the network learns to classify the MNIST digits accurately by minimizing the loss function through gradient descent.
