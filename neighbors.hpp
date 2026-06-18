#pragma once

#include <vector>
#include <cuda_runtime.h>

std::vector<dim3> generate_neighbors(dim3 current_block);
