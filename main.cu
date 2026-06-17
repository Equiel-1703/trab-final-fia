#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

cudaDeviceProp get_device_properties()
{
  cudaDeviceProp deviceProp;

  // Fetch properties for the first GPU
  cudaError_t status = cudaGetDeviceProperties(&deviceProp, 0);

  if (status == cudaSuccess)
  {
    std::cout << "Number of Streaming Multiprocessors (SMs): " << deviceProp.multiProcessorCount << std::endl;
    return deviceProp;
  }
  else
  {
    std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}

// ============================================= Metricas =============================================

double warp_efficiency(int num_threads)
{
  int warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
  int waste = (warps * WARP_SIZE) - num_threads;

  return 1.0 - static_cast<double>(waste) / static_cast<double>(warps * WARP_SIZE);
}

double sm_usage(int blocks, const cudaDeviceProp &prop)
{
  return std::min(static_cast<double>(blocks) / static_cast<double>(prop.multiProcessorCount), 1.0);
}

template <typename Kernel>
double occupancy(Kernel kernel, int threads_per_block, const cudaDeviceProp &prop)
{
  int activeBlocks = 0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, kernel, threads_per_block, 0);

  int warpsPerBlock = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;

  int activeWarps = activeBlocks * warpsPerBlock;

  int maxWarps = prop.maxThreadsPerMultiProcessor / WARP_SIZE;

  return static_cast<double>(activeWarps) / static_cast<double>(maxWarps);
}

double grid_balance(int grid_x, int grid_y)
{
  int largest = std::max(grid_x, grid_y);

  int smallest = std::min(grid_x, grid_y);

  if (largest == 0)
    return 0.0;

  return static_cast<double>(smallest) / static_cast<double>(largest);
}

double memory_coalescing_score(int block_x)
{
  return std::min(static_cast<double>(block_x) / 32.0, 1.0);
}

double score(double occupancy, double sm_usage, double warp_efficiency, double balance, double coalescing)
{
  return 0.35 * occupancy + 0.25 * sm_usage + 0.20 * warp_efficiency + 0.10 * balance + 0.10 * coalescing;
}

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
  cudaDeviceProp prop = get_device_properties();

  const int INPUT_SIZE = 1024;

  dim3 block(16, 16);
  dim3 grid(
      (INPUT_SIZE + block.x - 1) / block.x,
      (INPUT_SIZE + block.y - 1) / block.y);

  int threads_per_block = block.x * block.y;
  int total_blocks = grid.x * grid.y;

  double occ = occupancy(vector_add, threads_per_block, prop);
  double sm = sm_usage(total_blocks, prop);
  double warpEff = warp_efficiency(threads_per_block);
  double balance = grid_balance(grid.x, grid.y);
  double coalescing = memory_coalescing_score(block.x);
  
  double finalScore = score(occ, sm, warpEff, balance, coalescing);

  std::cout << "\n========== Kernel ==========\n";
  std::cout << "Kernel: vector_add\n";

  std::cout << "\n========== Launch Configuration ==========\n";
  std::cout << "Block: " << block.x << " x " << block.y << '\n';
  std::cout << "Grid: " << grid.x << " x " << grid.y << '\n';
  std::cout << "Threads per block: " << threads_per_block << '\n';
  std::cout << "Total blocks: " << total_blocks << '\n';

  std::cout << "\n========== Heuristics ==========\n";
  std::cout << "Occupancy: " << occ << '\n';
  std::cout << "SM Usage: " << sm << '\n';
  std::cout << "Warp Efficiency: " << warpEff << '\n';
  std::cout << "Grid Balance: " << balance << '\n';
  std::cout << "Memory Coalescing Score: " << coalescing << '\n';

  std::cout << "\n========== Final Score ==========\n";
  std::cout << "Score: " << finalScore << '\n';

  return EXIT_SUCCESS;
}
