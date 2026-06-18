#include "KernelScorer.hpp"

#include <iostream>
#include <cmath>

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

// Constructor
template <typename KernelType>
KernelScorer<KernelType>::KernelScorer(KernelType &kernel, dim3 input_dimensions) : kernel(kernel), input_dimensions(input_dimensions)
{
  this->device_properties = get_device_properties();
}

/*
  Verifica se o número de threads por bloco é perfeitamente divisível em warps, sem desperdiçar nenhuma thread.
*/
template <typename KernelType>
double KernelScorer<KernelType>::warp_efficiency(int threads_per_block)
{
  int warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
  int waste = (warps * WARP_SIZE) - threads_per_block;

  return 1.0 - static_cast<double>(waste) / static_cast<double>(warps * WARP_SIZE);
}

/*
  Avalia o desperdício de threads nas bordas da grid quando o tamanho
  do problema não é perfeitamente divisível pelo tamanho do bloco.
  Retorna 1.0 para encaixe perfeito e valores menores para desperdício.
*/
template <typename KernelType>
double KernelScorer<KernelType>::boundary_efficiency(dim3 &block, dim3 &grid)
{
  // Total de threads reais que farão trabalho útil (dentro dos limites do IF)
  double useful_threads = static_cast<double>(input_dimensions.x) * static_cast<double>(input_dimensions.y);

  // Total de threads alocadas e lançadas pelo hardware
  double launched_threads = static_cast<double>(grid.x * block.x) * static_cast<double>(grid.y * block.y);

  if (launched_threads == 0.0)
    return 0.0;

  return useful_threads / launched_threads;
}

/*
  Muitas vezes, 100% de ocupação causa thrashing no cache L1 e esgota registradores, diminuindo o paralelismo em nível
  de instrução (ILP). A literatura de otimização de CUDA mostra que, geralmente, a partir de 50% a 60% de ocupação,
  a latência de memória já está bem escondida.

  Por isso, em vez de um relacionamento linear (onde 1.0 é o melhor), usamos uma função por partes que atinge o ápice
  em 50% e decai levemente se for em direção aos 100% para penalizar o excesso de concorrência.
*/
template <typename KernelType>
double KernelScorer<KernelType>::occupancy(int threads_per_block)
{
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&(this->active_blocks_per_sm), this->kernel, threads_per_block, 0);

  int warps_per_block = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
  int active_warps = this->active_blocks_per_sm * warps_per_block;
  int max_warps = this->device_properties.maxThreadsPerMultiProcessor / WARP_SIZE;

  double raw_occupancy = static_cast<double>(active_warps) / static_cast<double>(max_warps);

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
template <typename KernelType>
double KernelScorer<KernelType>::wave_quantization_efficiency(int total_blocks)
{
  int sm_count = this->device_properties.multiProcessorCount;
  int max_concurrent_blocks = sm_count * this->active_blocks_per_sm;

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
template <typename KernelType>
double KernelScorer<KernelType>::memory_coalescing_score(dim3 &block)
{
  if (block.x % WARP_SIZE == 0)
  {
    return 1.0; // Alinhamento perfeito
  }
  else if (block.x >= WARP_SIZE / 2)
  {
    return 0.4; // Meio warp (penalidade pesada, gera 2 transações em vez de 1)
  }
  else
  {
    return 0.1; // Desalinhamento severo (péssima performance)
  }
}

template <typename KernelType>
double KernelScorer<KernelType>::score(dim3 &block, dim3 &grid)
{
  int threads_per_block = block.x * block.y * block.z;
  int total_blocks = grid.x * grid.y * grid.z;
  
  double occupancy_score = this->occupancy(threads_per_block);
  double warp_eff = this->warp_efficiency(threads_per_block);
  double wave_eff = this->wave_quantization_efficiency(total_blocks);
  double boundary_eff = this->boundary_efficiency(block, grid);
  double coalescing = this->memory_coalescing_score(block);

  return 0.35 * occupancy_score + 0.35 * warp_eff + 0.10 * wave_eff + 0.10 * boundary_eff + 0.10 * coalescing;
}