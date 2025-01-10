# imports 
import torch.nn as nn 
import torch 



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
        # softmax to predict output probabilities
        self.softmax_to_output = nn.Softmax()

    # create a hidden_state tensor with 0-value initialed
    def hidden_to_zeros(self, batch_size=1) -> torch.Tensor : 
        return torch.zeros(batch_size, self.hidden_size, requires_grad= False)

    def forward(self, hiddden_state  : torch.tensor , input_data : torch.tensor ) -> tuple[torch.tensor, torch.tensor] :
        ## pass through the input_to_hidden, hidden_to_hidden, hidden_to_output, tanh_from_hidden, softmax_to_output
        # input to hidden
        input_data = self.input_to_hidden(input_data)
        # hidden to hidden
        hidden_state = self.hidden_to_hidden(hidden_state)
        # new hidden state
        hidden_state = self.tanh_from_hidden(hidden_state + input_data)
        # hidden to output
        output = self.hidden_to_output(hidden_state)
        # output probabilities
        output = self.softmax_to_output(output)



