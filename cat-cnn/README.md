# Cat Identifier Convolutional Neural Network

## Overview
This project implements a Convolutional Neural Network (CNN) to identify whether an image contains a cat or not. The CNN is trained using custom data and is designed to run on environments like Google Colab or Kaggle Jupyter Notebook.

## How CNN Works
A Convolutional Neural Network (CNN) is a type of deep learning model that processes structured grid data such as images. The main components of a CNN are:

1. **Convolutional Layers**: These layers apply a set of filters/kernels to the input image to extract features such as edges, textures, and shapes (the outputs of these layers are called feature maps).
2. **Activation Functions**: Non-linear functions like ReLU (Rectified Linear Unit) are applied to introduce non-linearity into the model.
3. **Pooling Layers**: These layers reduce the spatial dimensions of the feature maps, retaining the most important information (**not to confuse with the flattening process**).
4. **Fully Connected Layers**: These layers are used to make predictions based on the features extracted by the convolutional and pooling layers.
5. **Output Layer**: The final layer that provides the probability of the image being a cat or not.

## Labeling Data with `dataset_maker`
The `dataset_maker` program is used to label the images in the dataset. It processes directories containing images of cats and non-cats, and generates a CSV file with the image paths and corresponding labels.

### Steps:
1. **Specify Directories**: Define the directories containing cat and non-cat images in `dataset_maker.c`.
2. **Generate CSV**: The program reads the images, assigns labels (1 for cat, 0 for non-cat), and writes the image paths and labels to a CSV file.
3. **Compile and Run**: Use the `make` command to compile and run the `dataset_maker` program.

## Main Logic of `cnn.py`
The `cnn.py` script implements the CNN model and handles the training process.

### Steps:
1. **Define Constants**: Set parameters like learning rate, number of epochs, batch size, and data directory.
2. **Define Dataset Class**: Create a custom dataset class to load and preprocess the images.
3. **Define CNN Model**: Implement the CNN architecture with convolutional, pooling, and fully connected layers.
4. **Load Data**: Read the labeled data from the CSV file and create a DataLoader for batch processing.
5. **Define Loss and Optimizer**: Use CrossEntropyLoss and Adam optimizer for training.
6. **Train the Model**: Iterate through the dataset, perform forward and backward passes, and update the model weights.

## Training the Model
The training process involves the following steps:

1. **Load Data**: Load the images and labels from the CSV file using the custom dataset class.
2. **Initialize Model**: Create an instance of the CNN model and move it to the appropriate device (CPU or GPU).
3. **Define Loss and Optimizer**: Set up the loss function and optimizer.
4. **Train**: Loop through the dataset for a specified number of epochs, performing forward and backward passes, and updating the model weights.
5. **Save Checkpoints**: Save the model checkpoints periodically during training.

### Training set up on Google Colab:
To train the model on Google Colab, you first need to upload the dataset directory (containing cat and non-cat images) to your Google Drive. Then, you can mount your Google Drive in the Colab notebook and access the dataset files.
**Advices**: I recommend to upload .zip file containing the dataset directory to Google Drive and unzip it in the Colab notebook using code. 

1. **Mount Google Drive**: Use the following code snippet to mount your Google Drive in the Colab notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. **Access Dataset Files**: You can access the dataset files in your Google Drive using the path `/content/drive/My Drive/path_to_dataset`.
3. **Unzip Dataset**: If you uploaded a .zip file, you can unzip it using the following code:
```python
import zipfile
with zipfile.ZipFile('/content/drive/My Drive/path_to_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/catcnn')
```
4. **Move to Dataset Directory**: Change the current working directory to the dataset directory.
5. **Train the Model**: Run the `cnn.py` script to train the CNN model on the dataset.

### Example Training Loop:
Here we assume that the model, loss function, optimizer are already defined and the training setup is complete. The training loop looks like this:
```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
    # Save the model checkpoint
    torch.save(model.state_dict(), 'saves/model.ckpt')
```

This README provides a comprehensive guide to understand and use the Cat Identifier CNN.