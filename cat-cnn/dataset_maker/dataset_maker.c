#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

// Check if a file is an image
int is_image(const char *filename) {
    const char *extensions[] = {".jpg", ".jpeg", ".png", ".bmp"};
    size_t num_extensions = sizeof(extensions) / sizeof(extensions[0]);

    for (size_t i = 0; i < num_extensions; i++) {
        if (strstr(filename, extensions[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

// Process a directory :
void process_directory(const char *dir_path, int label, FILE *csv_file) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);

    if (dir == NULL) {
        perror("Error opening directory");
        exit(EXIT_FAILURE);
    }

    while ((entry = readdir(dir)) != NULL) {
        // Iggnore special entries : . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // Check if an entry is a valid image
        if (is_image(entry->d_name)) {
            // build the full path of the file
            char filepath[1024];
            snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, entry->d_name);

            // remove all "../" before "data" directory to avoid path traversal
            char *data = strstr(filepath, "data");
            if (data == NULL) {
                continue;
            } else {
                // Overwrite the file path with the new one
                int output = snprintf(filepath, sizeof(filepath), "%s", data);
                if (output < 0) {
                    perror("Error processing file path");
                    exit(EXIT_FAILURE);
                }

            // Write the file path and label to the CSV file
            fprintf(csv_file, "%s,%d\n", filepath, label);
            }
        }
    }

    closedir(dir);
}


// check is the CSV file is already created with content
int is_csv_created(const char *csv_file_path) {
    FILE *file = fopen(csv_file_path, "r");
    if (file == NULL) {
        return 0;
    }

    // Check if the file is empty
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);

    if (size > 0) {
        return 1;
    }

    return 0;
}

// Clean the CSV file
void clean_csv(const char *csv_file_path) {
    // Open the CSV file in write mode to clear its content
    FILE *file = fopen(csv_file_path, "w");
    if (file == NULL) {
        perror("Error cleaning CSV");
        exit(EXIT_FAILURE);
    }
    // Close the file
    fclose(file);
}

int main( int argc, char *argv[] )  {

    // Directotry paths
    const char *cat_dir = "../data/cats";
    const char *non_cat_dir = "../data/cats";

    // CSV file path
    const char *csv_file_path = "../data/labels.csv";

    // if the user wants to clean the CSV file
    if (argc > 1) {
        if (strcmp(argv[1], "--clean") == 0) {
            clean_csv(csv_file_path);
            printf("CSV file cleaned\n");
            return 0;
        }
    }


    // Open the CSV file
    FILE *csv_file = fopen(csv_file_path, "w");
    if (csv_file == NULL) {
        perror("Error opening CSV");
        exit(EXIT_FAILURE);
    }

    // Check if the CSV file is already created
    if (is_csv_created(csv_file_path)) {
        printf("labeled dataset already created\n");
        return 0;
    }

    // Add header to the CSV file : image_path, label
    fprintf(csv_file, "image_path,label\n");

    // Process the directories
    process_directory(cat_dir, 1, csv_file);
    process_directory(non_cat_dir, 0, csv_file);

    // Close the CSV file
    fclose(csv_file);

    printf("Dataset created successfully\n");

    return 0;
}