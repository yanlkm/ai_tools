# imports 
import torch.nn as nn
import torch
import torch.optim as optim
import utils as utils


# create an RNN class from scratch


# input_size : size of the input
# hidden_size : size of the hidden state at (t-1) time
# output_size : size of the output of the neural network 
# batch_size : amount of data to be proceeded
class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int) -> None:
        super().__init__()
        # parameters 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # initialize layers 
        # put biais to false on the first dense connection
        self.input_to_hidden = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.hidden_to_output = nn.Linear(in_features=hidden_size, out_features=output_size)

        # activation functions 
        # tanh to determine the new hidden state 
        self.tanh_from_hidden = nn.Tanh()

    # create a hidden_state tensor with 0-value initialed
    def hidden_to_zeros(self, batch_size=1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)

    def forward(self, hidden_state: torch.tensor, input_data: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        # pass through the input_to_hidden, hidden_to_hidden, hidden_to_output, tanh_from_hidden, softmax_to_output
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


# train the model
def train(model: RNN, sequences: dict, seq_length: int, epochs: int, optimizer: optim.Optimizer,
          loss: nn.Module, device: torch.device) -> list:
    # training variables
    gradient_norms = []

    # loop over the epochs
    for epoch in range(epochs):
        # loop over the (country, year) pairs in the sequences
        for (country, year), sequence in sequences.items():
            X, y = utils.generate_subsequences_for_single_sequence(sequence, seq_length)
            if X is None or y is None:
                continue
            # initialize the hidden state
            hidden_state = model.hidden_to_zeros()
            # loop over the subsequences (x.shape[0] is the number of rows in X which is the number of subsequences)
            for i in range(X.shape[0]):
                # convert the data to tensors
                input_data = torch.tensor(X[i].reshape(1, -1), dtype=torch.float32, device=device)
                target = torch.tensor(y[i].reshape(1, -1), dtype=torch.float32, device=device)
                # zero the gradients
                optimizer.zero_grad()
                # forward pass
                output, hidden_state = model(hidden_state, input_data)
                # detach the hidden state to prevent backpropagation through time (BPTT)
                hidden_state = hidden_state.detach()
                # compute the loss and backpropagation
                loss_value = loss(output, target)
                loss_value.backward()

                # Compute gradient norms
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
