import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import rnn as rnn
import lstm as lstm
import utils

# choose the device to run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# variables

# load the data
# use the data formalization functions to load and process the data
csv_file_path = "data/city_temperature.csv"

# define the sequence length wanted
seq_length = 10

# create sequences carrying the temperature values on countries and years
sequences = utils.load_and_process_data(csv_file_path)


# train the model
model = lstm.LSTM(input_size=seq_length, hidden_size=20, output_size=1)
# move the model to the device
model.to(device)
# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# define the loss function
loss = nn.MSELoss()
# train the model
lstm_gradient_norms = lstm.train_lstm(model, sequences, seq_length, 1, optimizer, loss, device)

# train the model
rnn_model = rnn.RNN(input_size=seq_length, hidden_size=20, output_size=1, batch_size=1)
# move the model to the device
rnn_model.to(device)
# define the optimizer
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.001)
# define the loss function
loss_rnn = nn.MSELoss()
# train the model
rnn_gradient_norms = rnn.train(rnn_model, sequences, seq_length, 1, optimizer_rnn, loss_rnn , device)

# plot the gradient norms
plt.figure(figsize=(10, 6))
plt.plot(rnn_gradient_norms, label="RNN Gradient Norm", color='blue')
plt.plot(lstm_gradient_norms, label="LSTM Gradient Norm", color='green')
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Evolution During Training (RNN vs LSTM)")
plt.legend()
plt.grid(True)
plt.show()


