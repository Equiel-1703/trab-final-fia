#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#define WARP_SIZE 32

double inline warp_waste_ratio(int num_threads)
{
    int warps = (int) std::ceil((float) num_threads / (float) WARP_SIZE);
    int waste = (warps * WARP_SIZE) - num_threads;

    return static_cast<double>(waste) / static_cast<double>(warps * WARP_SIZE);
}

double inline sm_usage(int blocks)
{
    int sm_count = 0;

    int deviceId = 0; // Target the first GPU
    cudaDeviceProp deviceProp;

    // Fetch properties for the specified device
    cudaError_t status = cudaGetDeviceProperties(&deviceProp, deviceId);

    if (status == cudaSuccess)
    {
        std::cout << "Number of Streaming Multiprocessors (SMs): " << deviceProp.multiProcessorCount << std::endl;
        sm_count = deviceProp.multiProcessorCount;
    }
    else
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    return std::min(static_cast<double>(blocks) / static_cast<double>(sm_count), 1.0);
}

int main(int argc, const char *args[])
{
    std::cout << "Warp waste ratio (T: 250) = " << warp_waste_ratio(250) << std::endl;
    std::cout << "SM Usage (blocks 40): " << sm_usage(40) << std::endl;

    return EXIT_SUCCESS;
}
