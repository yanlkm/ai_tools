# DataSet Maker 
Sometimes, we encounter the need to format the labels of our dataset. This can pass through a variety of operations such as attribute a numerical value to a file name that indicates a class.
Here is a dataset maker that creates a CSV file on the root of data training directory, with the image path and the label of the image.

** Notes ** : The dataset maker is designed to work with only 2 classes, but you can customize it to work with more classes by just adding in main the new class directory and the label.
* eg : 
    ```c
    // Add the new class directory and the label
    char *class_2 = "../data/class_2";
    int label_2 = 2;
    ```
## Step of dataset maker logic 

1. Specify the directories of the images in the `dataset_maker.c` file.
2. The program will read the images in the specified directories and generate a CSV file with the image paths and labels.
3. The program will create a CSV file with the following format:
    ```
    image_path,label
    data/cat/cat.0.jpg,1
    data/not-cat/not-cat.0.jpg,0
    ...
    ```

## Step to run the dataset maker

1. **Compile the Program**: Use the `make` command to compile the `dataset_maker.c` file. This will generate an executable named `dataset_maker`.

    ```sh
    make
    ```

2. **Run the Program**: Execute the compiled program to create the dataset. The program will process the images in the specified directories and generate a CSV file with the image paths and labels.

    ```sh
    ./dataset_maker
    ```

3. **Clean the CSV File**: If you need to clean the CSV file, use the `make clean` command. This will run the program with the `--clean` flag, which clears the content of the CSV file.

    ```sh
    make clean
    ```

4. **Remove the Executable**: To remove the compiled executable, use the `make remove` command.

    ```sh
    make remove
    ```

5. **Clean and Remove**: To clean the CSV file and remove the compiled executable, use the `make clean_all` command.

    ```sh
    make clean_all
    ```

6. **Run and Remove**: To compile, run the program, and then remove the compiled executable, use the `make run` command.

    ```sh
    make run
    ```

### Directory Structure

The program expects the following directory structure:

```
project-root/
  data/
    cats/
      cat.0.jpg
      cat.1.jpg
      ...
    not-cats/
      not-cat.0.jpg
      not-cat.1.jpg
      ...
    labels.csv
  dataset_maker/
    dataset_maker.c
    Makefile
```

### CSV File

The generated CSV file will have the following format:

```
image_path,label
data/cat/cat.0.jpg,1
data/not-cat/not-cat.0.jpg,0
...
```
