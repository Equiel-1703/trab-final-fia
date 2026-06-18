OUTPUT_FILE = brute_force.out
SRC_FILES = brute_force.cu

CC = nvcc

FLAGS = --std=c++17 -Xcompiler -Wall

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SRC_FILES)
	$(CC) -o $(OUTPUT_FILE) $(SRC_FILES) $(FLAGS)

clean:
	rm -f $(OUTPUT_FILE)