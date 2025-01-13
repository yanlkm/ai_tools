# imports 
import torch.nn as nn
import torch
import torch.optim as optim
import utils as utils


## create a LSTM class from scratch
# input_size : size of the input
# hidden_size : size of the hidden state at (t-1) time
# output_size : size of the output of the neural network

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # forget gate : forget the information that is not needed in percentage
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # cell state candidate : the cell state is where the information is stored
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        self.b_C = nn.Parameter(torch.zeros(hidden_size))

        # output gate : retain the information that will be output
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # output layer : hidden to output
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, input_data, hidden_state, cell_state):
        combined = torch.cat((hidden_state, input_data), dim=1)

        # forget gate
        f_t = torch.sigmoid(self.W_f(combined) + self.b_f)

        # input gate
        i_t = torch.sigmoid(self.W_i(combined) + self.b_i)
        C_tilde = torch.tanh(self.W_C(combined) + self.b_C)

        # update cell state
        C_t = f_t * cell_state + i_t * C_tilde

        # output gate
        o_t = torch.sigmoid(self.W_o(combined) + self.b_o)
        h_t = o_t * torch.tanh(C_t)

        # compute output
        output = self.hidden_to_output(h_t)

        return output, h_t, C_t

    def init_hidden_and_cell(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))


# train the model
def train_lstm(model: LSTM, sequences: dict, seq_length: int, epochs: int, optimizer: optim.Optimizer,
               loss: nn.Module, device: torch.device) -> list:
    # training variables
    gradient_norms = []  # Store the gradient norms for visualization

    # loop over the epochs
    for epoch in range(epochs):
        # reset average loss for this epoch
        average_loss = 0.0

        # loop over the (country, year) pairs in the sequences
        for (country, year), sequence in sequences.items():
            # generate the subsequences
            X, y = utils.generate_subsequences_for_single_sequence(sequence, seq_length)

            # check if there is enough data
            if X is None or y is None:
                print(f"Skipping {country} - {year}: insufficient data.")
                continue

            # initialize the hidden and cell states
            hidden_state, cell_state = model.init_hidden_and_cell(batch_size=1)

            # loop over the subsequences
            for i in range(X.shape[0]):
                # get the input and target values
                input_data_array = X[i].reshape(1, -1)
                target_array = y[i].reshape(1, -1)

                # convert the input and target to tensors
                input_data = torch.tensor(input_data_array, dtype=torch.float32, device=device)
                target = torch.tensor(target_array, dtype=torch.float32, device=device)

                # zero the gradients
                optimizer.zero_grad()

                # forward pass
                output, hidden_state, cell_state = model(input_data, hidden_state, cell_state)

                # detach the hidden and cell states from the computation graph
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                # calculate the loss
                loss_value = loss(output, target)

                # sum the loss
                average_loss += loss_value.item()

                # compute the gradients norm
                total_norm = 0.0
                # for each parameter in the model (weights and biases)
                for param in model.parameters():
                    # if the gradient is not None
                    if param.grad is not None:
                        # compute the L2 norm of the gradient
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                # compute the square root of the total norm (L2 norm of the gradients)
                total_norm = total_norm ** 0.5
                # append the total norm to the list
                gradient_norms.append(total_norm)
                optimizer.step()
    return gradient_norms
