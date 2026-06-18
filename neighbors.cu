#include "neighbors.hpp"

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

std::vector<int> generate_valid_dims()
{
    std::vector<int> valid_dims;
    int dim = 1;  // Inicia em 1
    int step = 8; // Múltiplos de 8

    for (int i = 1; dim <= 1024; i++)
    {
        valid_dims.push_back(dim);
        dim = i * step; // Incrementa para o próximo múltiplo de 8
    }

    return valid_dims;
}
/*
    Meyers' Singleton Pattern para garantir que a lista de dimensões válidas seja gerada
    apenas uma vez e reutilizada em chamadas subsequentes.
*/
const std::vector<int> &get_valid_dims()
{
    static std::vector<int> valid_dims = generate_valid_dims();
    return valid_dims;
}

int get_index(int value)
{
    const auto &VALID_DIMS = get_valid_dims();

    auto it = std::find(VALID_DIMS.begin(), VALID_DIMS.end(), value);
    if (it != VALID_DIMS.end())
    {
        return std::distance(VALID_DIMS.begin(), it);
    }
    return -1;
}

std::vector<dim3> generate_neighbors(dim3 current_block)
{
    const auto &VALID_DIMS = get_valid_dims();
    std::vector<dim3> valid_neighbors;

    int idx_x = get_index(current_block.x);
    int idx_y = get_index(current_block.y);

    // Movimentos nos índices: -1 (diminuir), 0 (manter), +1 (aumentar)
    int d_idx[] = {-1, 0, 1};

    for (int dx : d_idx)
    {
        for (int dy : d_idx)
        {
            // Ignorar a combinação (0,0) pois é o próprio estado atual
            if (dx == 0 && dy == 0)
                continue;

            int new_idx_x = idx_x + dx;
            int new_idx_y = idx_y + dy;

            // Garantir que não vamos acessar índices fora da nossa lista
            if (new_idx_x >= 0 && new_idx_x < (int)VALID_DIMS.size() &&
                new_idx_y >= 0 && new_idx_y < (int)VALID_DIMS.size())
            {

                int new_x = VALID_DIMS[new_idx_x];
                int new_y = VALID_DIMS[new_idx_y];

                // Nunca ultrapassar o limite de 1024 threads por bloco
                if (new_x * new_y <= 1024)
                {
                    valid_neighbors.push_back(dim3(new_x, new_y, 1));
                }
            }
        }
    }
    return valid_neighbors;
}
