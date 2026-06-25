OUTPUT_FILE = main.out
SRC_FILES = main.cu neighbors.cu
HEADERS = $(wildcard *.hpp)

CC = nvcc

FLAGS = --std=c++17 -Xcompiler -Wall

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SRC_FILES) $(HEADERS)
	$(CC) -o $(OUTPUT_FILE) $(SRC_FILES) $(FLAGS)

clean:
	rm -f $(OUTPUT_FILE)