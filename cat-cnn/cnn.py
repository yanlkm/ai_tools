# imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the CatIdentifier convolutional neural network
class CatIdentifier(nn.Module):
        def __init__(self, csv_data, transform=None):
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

    # Create an instance of the CatDataset class

    dataset = CatDataset(data, transform=transform)

    # Create a DataLoader object to load the data in batches for training

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Define the CatIdentifier neural network model
    class CatIdentifier(nn.Module):
        def __init__(self, num_classes=1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Define the input layer : take 3 channels of 128x128 pixels and 32 filters as output
            self.conv1 = nn.Conv2d(
                in_channels=3, # number of channels (depth of the input volume)
                out_channels=32, # number of filters
                kernel_size=3 # size of a single filter [3x3]
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
            self.flatten = nn.Flatten()
            # Define a fully connected layer with 128 nodes as input and another 64 nodes as output
            self.fc1 = nn.Linear(
                in_features=128*14*14,
                out_features=64,
                bias=True
            )
            # Define a ReLU activation function
            self.relu4 = nn.ReLU()
            # Define a fully connected layer with 64 nodes as input and another 1 node as output which is the final output
            self.fc2 = nn.Linear(
                in_features=64,
                out_features=num_classes,
                bias=True
            )
            # Define a sigmoid activation function
            self.sigmoid = nn.Sigmoid()

        # Perform the forward pass
        def forward(self, x):
            # Perform the forward pass through the first layer
            out = self.pool1(self.relu1(self.conv1(x)))
            # Perform the forward pass through the second layer
            out = self.pool2(self.relu2(self.conv2(out)))
            # Perform the forward pass through the third layer
            out = self.pool3(self.relu3(self.conv3(out)))
            # Flatten the output of the third
            out = self.flatten(out)
            # Perform the forward pass through the fourth layer
            out = self.relu4(self.fc1(out))
            # Perform the forward pass through the fifth layer and return the output passed through the sigmoid activation function
            out = self.sigmoid(self.fc2(out))
            return out