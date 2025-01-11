# imports 
import torch.nn as nn 
import torch 
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# choose the device to run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## create a RNN class from scratch 


# input_size : size of the input
# hidden_size : size of the hidden state at (t-1) time
# output_size : size of the output of the neural network 
# batch_size : amount of data to be proceed
class RNN(nn.Module) : 
    def __init__(self, input_size : int, hidden_size : int, output_size : int, batch_size : int)  -> None :
        super().__init__()
        # parameters 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # initialize layers 
        # put biais to false on the first dense connection
        self.input_to_hidden = nn.Linear(in_features= input_size, out_features = hidden_size, bias = False) 
        self.hidden_to_hidden = nn.Linear(in_features = hidden_size, out_features = hidden_size )
        self.hidden_to_output = nn.Linear(in_features = hidden_size, out_features = output_size)

        # activation functions 
        # tanh to determine the new hidden state 
        self.tanh_from_hidden = nn.Tanh()

    # create a hidden_state tensor with 0-value initialed
    def hidden_to_zeros(self, batch_size=1) -> torch.Tensor : 
        return torch.zeros(batch_size, self.hidden_size, requires_grad= False)

    def forward(self, hidden_state  : torch.tensor , input_data : torch.tensor ) -> tuple[torch.tensor, torch.tensor] :
        ## pass through the input_to_hidden, hidden_to_hidden, hidden_to_output, tanh_from_hidden, softmax_to_output
        # input to hidden
        input_data = self.input_to_hidden(input_data)
        # hidden to hidden
        hidden_state = self.hidden_to_hidden(hidden_state)
        # new hidden state
        hidden_state = self.tanh_from_hidden(hidden_state + input_data)
        # hidden to output
        output = self.hidden_to_output(hidden_state)
        # return output and hidden state
        return output, hidden_state

# data formalization

# Data formalization
def load_and_process_data(csv_file_path):
    # load csv file
    data = pd.read_csv(csv_file_path)

    # handle missing or invalid data
    data = data.dropna(subset=["AvgTemperature"])

    # remove rows with temperature values less than -28 or greather than 92
    data = data[(data["AvgTemperature"] >= -28) & (data["AvgTemperature"] <= 92)]

    # filter by country and year
    grouped_data = data.groupby(["Country", "Year"])

    # create a list of sequences
    sequences = {}

    # loop over the grouped data
    for (country, year), group in grouped_data : 
        # sort the data by month and day
        group = group.sort_values(by=["Month", "Day"])
        # get the temperature values
        temperatures = group["AvgTemperature"].values
        # add the sequence to the dictionary
        sequences[(country, year)] = temperatures
    
    return sequences

# create a subsequence function
def create_subsequences(sequence, seq_length):
    # create a list to hold the subsequences
    subsequences = []
    # loop over the sequence
    for i in range(len(sequence) - seq_length):
        subsequences.append(sequence[i:i + seq_length + 1])
    # return the list of subsequences
    return np.array(subsequences)


# generate the input and output data
def generate_subsequences_for_single_sequence(sequence, seq_length):
    # check if the sequence is long enough
    if len(sequence) <= seq_length:  
        return None, None
    # create the subsequences
    subsequences = create_subsequences(sequence, seq_length)
    # normalize the sequence in intervals of 0 and 1
    subsequences = normalize_sequence(subsequences)
    # return the input and output data formatted
    X = subsequences[:, :-1]
    y = subsequences[:, -1]
    return X, y

# normalize the sequence
def normalize_sequence(sequence):
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    return (sequence - min_val) / (max_val - min_val)


# use the data formalization functions to load and process the data
csv_file_path = "data/city_temperature.csv"  

# define the sequence length wanted
seq_length = 10 

# create sequences carrying the temperature values on countries and years
sequences = load_and_process_data(csv_file_path)

# train the model
def train( model : RNN, sequences : dict, seq_length : int, epochs :int, optimizer : optim.Optimizer, loss : nn.Module) -> None : 
    # training variables
    average_loss = 0.0
    # store the gradients norms
    gradient_norms = []
    # loop over the epochs
    for epoch in range(epochs):
        # loop over the (country, year) pairs in the sequences
        for (country, year), sequence in sequences.items():  # Correct unpacking
            # generate the subsequences
            X, y = generate_subsequences_for_single_sequence(sequence, seq_length)
            # check if there is enough data
            if X is None or y is None:
                print(f"Skipping {country} - {year}: insufficient data.")
                continue
            # initialize the hidden state
            hidden_state = model.hidden_to_zeros()
            # loop over the subsequences
            for i in range(X.shape[0]) : 
                # get the input and target values
                input_data_array = X[i].reshape(1, -1)
                target_array = y[i].reshape(1, -1)
                # convert the input and target to tensors
                input_data = torch.tensor(input_data_array, dtype=torch.float32, device=device)
                target = torch.tensor(target_array, dtype=torch.float32, device=device)
                # zero the gradients
                optimizer.zero_grad()
                # forward pass
                output, hidden_state = model(hidden_state, input_data)
                # detach the hidden state from the computation graph
                hidden_state = hidden_state.detach()
                # calculate the loss
                loss_value = loss(output, target)
                # sum the loss
                average_loss += loss_value.item()
                # backward pass
                loss_value.backward()

                # compute and store gradient norms 
                # initialize the total norm
                total_norm = 0.0
                # loop over the parameters
                for param in model.parameters():
                    # if the parameter has a gradient
                    if param.grad is not None:
                        # then the norm of the gradient is calculated   
                        param_norm = param.grad.data.norm(2) 
                        # add the square of the norm to the total norm (L2 norm)
                        total_norm += param_norm.item() ** 2
                # calculate the square root of the total norm
                total_norm = total_norm ** 0.5
                # store the total norm
                gradient_norms.append(total_norm)
                # update the parameters
                optimizer.step()
        # print the average loss
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss / len(sequences)}")
    # plot the gradient norms
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label="Gradient Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Evolution During Training")
    plt.legend()
    plt.grid(True)
    plt.show()


# train the model 
model = RNN(input_size=seq_length, hidden_size=20, output_size=1, batch_size=1)
# move the model to the device
model.to(device)
# create an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# create a loss function
loss = nn.MSELoss()

# train the model
train(model, sequences, seq_length, 1, optimizer, loss)

