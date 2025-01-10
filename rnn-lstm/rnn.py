# imports 
import torch.nn as nn 
import torch 
import pandas as pd
import numpy as np

## create a RNN class from scratch 


# input_size : size of the input
# hidden_size : size of the hidden state at (t-1) time
# output_size : size of the output of the neural network 
# batch_size : amount of data to be proceed
class RNN(nn.Module) : 
    def __init__(self, input_size : int, hidden_size : int, output_size : int, batch_size : int)  -> None :
        super.__init__()
        # parameters 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # initialize layers 
        # put biais to false on the first dense connection
        self.input_to_hidden = nn.Linear(in_features= input_size, out_features = hidden_size, biais = False) 
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

    # remove rows with temperature values less than -50 or greater than 50
    data = data[(data["AvgTemperature"] >= -60) & (data["AvgTemperature"] <= 60)]    

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
def generate_subsequences(sequences, seq_length):
    all_subsequences = []
    for sequence in sequences.values():
        if len(sequence) > seq_length: # ensure the sequence is long enough
            subsequences = create_subsequences(sequence, seq_length)
            all_subsequences.append(subsequences)

    # concatenate all the subsequences
    all_subsequences = np.concatenate(all_subsequences, axis=0)
    X = all_subsequences[:, :-1]
    y = all_subsequences[:, -1]
    return X, y


