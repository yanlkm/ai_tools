# imports
import io

import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pandas as pd
import os
from PIL import Image

# Define constants

# Here we assume the files are as follows:
# project-root/
#   data/
#     cat/
#       cat.0.jpg
#       cat.1.jpg
#       ...
#     not-cat/
#       not-cat.0.jpg
#       not-cat.1.jpg
#       ...
#     labels.csv
#   saves/
#   cnn.py
#   /dataset_maker


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the variable learning rate, number of epochs, batch size, path to the date and number of classes
learning_rate = 0.001
num_epochs = 1
batch_size = 32
data_dir = 'data'
num_classes = 1
momentum = 0.9

# define the csv file path
csv_file_path = f'{data_dir}/labels.csv'


# Define the CatIdentifier convolutional neural network
class CatDataset(torch.utils.data.Dataset):
    def __init__(self, csv_data, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = csv_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and label at the specified index
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        # Open the image using PIL and convert it to RGB (3 channels)
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation to the image
        if self.transform:
            image = self.transform(image)
        return image, label


# Define the CatIdentifier neural network model
class CatIdentifier(nn.Module):
    def __init__(self, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the input layer : take 3 channels of 128x128 pixels and 32 filters as output
        self.conv1 = nn.Conv2d(
            in_channels=3,  # number of channels (depth of the input volume)
            out_channels=32,  # number of filters
            kernel_size=3  # size of a single filter [3x3]
        )
        # The size of a single channel is 128(Height) - 3(kernel_size) +2*0(padding) / 1 (stride) + 1 = 126
        # Add a ReLU activation function
        self.relu1 = nn.ReLU()
        # Add a max pooling layer 2x2 filter
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # After the pooling layer, the size of the output is 126/2 = 63
        # Define the second layer :  take 32 channels of 64x64 pixels and 64 filters & output
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        # The size of a single channel is 63(Height) - 3(kernel_size) +2*0(padding) / 1 (stride) + 1 = 61
        # Add a ReLU activation function
        self.relu2 = nn.ReLU()
        # Add a max pooling layer 2x2 filter
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # After the pooling layer, the size of the output is 61/2 = 30
        # Define the third layer : take 64 channels of 32x32 pixels and 128 filters & output
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        # The size of a single channel is 30(Height) - 3(kernel_size) +2*0(padding) / 1 (stride) + 1 = 28
        # Add a ReLU activation function
        self.relu3 = nn.ReLU()
        # Add a max pooling layer 2x2 filter
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # After the pooling layer, the size of the output is 28/2 = 14
        # Define a flatten layer
        self.flatten_size = self._get_flatten_size()
        # Define a fully connected layer with 128 nodes as input and another 64 node as output
        self.fc1 = nn.Linear(self.flatten_size, 64)
        # Define a ReLU activation function
        self.relu4 = nn.ReLU()
        # Define a fully connected layer with 64 nodes as input and another 1 node as output which is the final output
        self.fc2 = nn.Linear(
            in_features=64,  # This remains the same
            out_features=num_classes,
            bias=True
        )
        # Define a sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    # Get dynamically the size
    def _get_flatten_size(self):
        # Pass a dummy tensor to calculate the output size after convolution and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.pool3(self.relu3(
                self.conv3(self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy_input)))))))))
            # We need to get the size of the features dimension after flattening.
            return dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]

    # Perform the forward pass
    def forward(self, x):
        # Perform the forward pass through the first layer
        out = self.pool1(self.relu1(self.conv1(x)))
        # Perform the forward pass through the second layer
        out = self.pool2(self.relu2(self.conv2(out)))
        # Perform the forward pass through the third layer
        out = self.pool3(self.relu3(self.conv3(out)))
        # Perform the forward pass through the fourth layer after got flattened
        # Reshape the output before feeding to fc1 using "view" :
        out = out.view(-1, self.flatten_size)
        out = self.relu4(self.fc1(out))
        # Perform the forward pass through the fifth layer and return the output passed through the sigmoid
        # activation function
        out = self.sigmoid(self.fc2(out))
        return out


# Implement the training logic

# Define function of the transformation to resize the image to a 3x128x128 tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Normalize the image to have a mean of 0.5 and a standard deviation of 0.5
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data from the csv file and create a DataFrame considering the head of the csv file
#data = pd.read_csv(filepath_or_buffer=csv_file_path, header=2, usecols=['image_path', 'label'])

# Create an instance of the CatDataset class
#dataset = CatDataset(data, transform=transform)

# Create a DataLoader object to load the data in batches for training
#train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the model : if there is no save, create an instance
if os.path.exists('saves/model.pth'):
    model = CatIdentifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('saves/model.pth', weights_only=True))
else:
    model = CatIdentifier(num_classes=num_classes).to(device)

# Define the loss function (use binary cross-entry loss)
criterion = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = 1  #len(train_loader)


# Define the train_model function
def train_model(num_epochs, train_loader, model, criterion, optimizer):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            # Calculate the loss using the labels un-squeezed to 1 dimension
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            # Save the model checkpoint through the torch.save function
            torch.save(model.state_dict(), 'saves/model.pth')


# Call the train_model function
#train_model(num_epochs, train_loader, model, criterion, optimizer)


# Function to predict the class cat of an image
def predict_cat_image(image_bytes_input):
    # Open the image using PIL and convert it to RGB (3 channels)
    image_input = Image.open(io.BytesIO(image_bytes_input)).convert('RGB')

    # Resize the image before applying the transform
    image_input = image_input.resize((128, 128))

    # Apply the same transformations used during training
    image_input = transform(image_input)

    # Load the model and set it to evaluation mode
    cat_identifier_model = CatIdentifier(num_classes=num_classes).to(device)
    cat_identifier_model.load_state_dict(torch.load('saves/model.pth', weights_only=True))
    cat_identifier_model.eval()

    # Perform the forward pass
    with torch.no_grad():
        # Move the image to the device
        image_input = image_input.to(device)

        outputs = cat_identifier_model(image_input)
        # Get the predicted class
        cat_prediction = outputs[0][0].item()
        if cat_prediction >= 0.5:
            return 'cat'
        else:
            return 'not-cat'


# Test a prediction
image_path = 'my-cat.jpg'
image = Image.open(image_path)
image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
image_bytes = image_bytes.getvalue()
print(predict_cat_image(image_bytes))
