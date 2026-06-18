OUTPUT_FILE = main.out
SRC_FILES = main.cu neighbors.cu KernelScorer.cu

CC = nvcc

FLAGS = --std=c++17 -Xcompiler -Wall -Wextra

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SRC_FILES)
	$(CC) $(FLAGS) -o $(OUTPUT_FILE) $(SRC_FILES)

clean:
	rm -f $(OUTPUT_FILE)