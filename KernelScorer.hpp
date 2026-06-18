#pragma once

#include <cuda_runtime.h>

template <typename KernelType>
class KernelScorer
{
private:
  const KernelType &kernel;
  const int WARP_SIZE = 32;
  const dim3 input_dimensions;

  cudaDeviceProp device_properties;
  int active_blocks_per_sm = 0;

  double warp_efficiency(int threads_per_block);
  double boundary_efficiency(dim3 &block, dim3 &grid);
  double occupancy(int threads_per_block);
  double wave_quantization_efficiency(int total_blocks);
  double memory_coalescing_score(dim3 &block);

public:
  KernelScorer(KernelType &kernel, dim3 input_dimensions);

  double score(dim3 &block, dim3 &grid);
};
