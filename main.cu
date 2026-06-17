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

/*
  Verifica se o número total de threads é perfeitamente divisível em warps, sem desperdiçar nenhuma thread.
*/
double warp_efficiency(int num_threads)
{
  int warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
  int waste = (warps * WARP_SIZE) - num_threads;

  return 1.0 - static_cast<double>(waste) / static_cast<double>(warps * WARP_SIZE);
}

/*
  Muitas vezes, 100% de ocupação causa thrashing no cache L1 e esgota registradores, diminuindo o paralelismo em nível
  de instrução (ILP). A literatura de otimização de CUDA mostra que, geralmente, a partir de 50% a 60% de ocupação,
  a latência de memória já está bem escondida.

  Por isso, em vez de um relacionamento linear (onde 1.0 é o melhor), usamos uma função por partes que atinge o ápice
  em 50% e decai levemente se for em direção aos 100% para penalizar o excesso de concorrência.
*/
template <typename Kernel>
double occupancy(Kernel kernel, int threads_per_block, int *active_blocks_per_sm, const cudaDeviceProp &prop)
{
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_blocks_per_sm, kernel, threads_per_block, 0);

  int warpsPerBlock = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
  int activeWarps = (*active_blocks_per_sm) * warpsPerBlock;
  int maxWarps = prop.maxThreadsPerMultiProcessor / WARP_SIZE;

  double raw_occupancy = static_cast<double>(activeWarps) / static_cast<double>(maxWarps);

  if (raw_occupancy <= 0.5)
  {
    return raw_occupancy / 0.5; // Cresce de 0.0 até 1.0
  }
  else
  {
    return 1.0 - 0.2 * (raw_occupancy - 0.5); // Cai levemente até 0.9 em 100% de ocupação
  }
}

/*
  Calcula quantos blocos cabem na GPU inteira de uma vez e avalia o desperdício na última onda.
*/
double wave_quantization_efficiency(int total_blocks, int active_blocks_per_sm, const cudaDeviceProp &prop)
{
  int sm_count = prop.multiProcessorCount;
  int max_concurrent_blocks = sm_count * active_blocks_per_sm;

  if (max_concurrent_blocks == 0)
    return 0.0;

  int waves = (total_blocks + max_concurrent_blocks - 1) / max_concurrent_blocks; // teto da divisão
  double ideal_blocks = waves * max_concurrent_blocks;

  return static_cast<double>(total_blocks) / ideal_blocks;
}

/*
  Verifica quão bem o bloco está alinhado com os warps no eixo x. Desalinhamento de warps causa penalidade severa
  no desempenho.
*/
double memory_coalescing_score(int block_x)
{
  if (block_x % WARP_SIZE == 0)
  {
    return 1.0; // Alinhamento perfeito
  }
  else if (block_x >= WARP_SIZE / 2)
  {
    return 0.4; // Meio warp (penalidade pesada, gera 2 transações em vez de 1)
  }
  else
  {
    return 0.1; // Desalinhamento severo (péssima performance)
  }
}

double score(double occupancy, double wave_efficiency, double warp_efficiency, double coalescing)
{
  return 0.35 * occupancy + 0.35 * wave_efficiency + 0.20 * warp_efficiency + 0.10 * coalescing;
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
  int active_blocks_per_sm = 0;

  dim3 block(16, 16);
  dim3 grid(
      (INPUT_SIZE + block.x - 1) / block.x,
      (INPUT_SIZE + block.y - 1) / block.y);

  int threads_per_block = block.x * block.y;
  int total_blocks = grid.x * grid.y;

  double occ = occupancy(vector_add, threads_per_block, &active_blocks_per_sm, prop);
  double wave_eff = wave_quantization_efficiency(total_blocks, active_blocks_per_sm, prop);
  double warp_eff = warp_efficiency(threads_per_block);
  double coalescing = memory_coalescing_score(block.x);

  double final_score = score(occ, warp_eff, warp_eff, coalescing);

  std::cout << "\n========== Kernel ==========\n";
  std::cout << "Kernel: vector_add\n";

  std::cout << "\n========== Launch Configuration ==========\n";
  std::cout << "Block: " << block.x << " x " << block.y << '\n';
  std::cout << "Grid: " << grid.x << " x " << grid.y << '\n';
  std::cout << "Threads per block: " << threads_per_block << '\n';
  std::cout << "Total blocks: " << total_blocks << '\n';

  std::cout << "\n========== Heuristics ==========\n";
  std::cout << "Occupancy: " << occ << '\n';
  std::cout << "Wave Efficiency: " << wave_eff << '\n';
  std::cout << "Warp Efficiency: " << warp_eff << '\n';
  std::cout << "Memory Coalescing Score: " << coalescing << '\n';

  std::cout << "\n========== Final Score ==========\n";
  std::cout << "Score: " << final_score << '\n';

  return EXIT_SUCCESS;
}
