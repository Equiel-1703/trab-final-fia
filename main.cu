#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "neighbors.hpp"
#include "KernelScorer.hpp"

// ======================================== Simple kernel for testing ========================================
__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n)
  {
    c[tid] = a[tid] + b[tid];
  }
}
// ===========================================================================================================

int main(int argc, const char *args[])
{
  const int INPUT_SIZE = 1024;
  const dim3 input_dimensions(INPUT_SIZE, 1);

  dim3 block(16, 16);
  dim3 grid(
      (INPUT_SIZE + block.x - 1) / block.x,
      (INPUT_SIZE + block.y - 1) / block.y);

  KernelScorer scorer(vector_add, input_dimensions);
  double final_score = scorer.score(block, grid);

  std::cout << "\n========== Kernel ==========\n";
  std::cout << "Kernel: vector_add\n";

  std::cout << "\n========== Launch Configuration ==========\n";
  std::cout << "Block: " << block.x << " x " << block.y << '\n';
  std::cout << "Grid: " << grid.x << " x " << grid.y << '\n';
  std::cout << "Threads per block: " << block.x * block.y << '\n';
  std::cout << "Total blocks: " << grid.x * grid.y << '\n';

  std::cout << "\n========== Final Score ==========\n";
  std::cout << "Score: " << final_score << '\n';

  return EXIT_SUCCESS;
}
