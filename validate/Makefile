OUTPUT_FILE = validate.out
SRC_FILES = validate.cu

CC = nvcc

FLAGS = --std=c++17 -Xcompiler -Wall

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SRC_FILES)
	$(CC) -o $(OUTPUT_FILE) $(SRC_FILES) $(FLAGS)

clean:
	rm -f $(OUTPUT_FILE)