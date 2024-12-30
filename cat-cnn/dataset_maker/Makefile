# Variables

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -O2

# Source files
SRC = dataset_maker.c

# Output executable
OUT = dataset_maker

# Regular targets

# Default target: compile and run the program
all: $(OUT)
	./$(OUT)

# Target to clean the CSV file : use "clean" flag
clean: $(OUT)
	./$(OUT) --clean

# Compile the C program
$(OUT): $(SRC)
	$(CC) $(CFLAGS) -o $(OUT) $(SRC)

# Phony targets

# Remove the compiled executable: use "remove" flag
.PHONY: remove
remove:
	rm -f $(OUT)

# Remove the compiled executable and clean the CSV file
.PHONY: clean_all
clean_all: clean remove

# Compile and run the program and delete the compiled executable
.PHONY: run
run: all remove


# Phony targets
.PHONY: all clean