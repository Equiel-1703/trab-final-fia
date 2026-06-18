#pragma once

#include <cmath>
#include <vector>
#include <random>
#include <iostream>

#include "neighbors.hpp"
#include "KernelScorer.hpp"

template <typename K>
class SimulatedAnnealing
{
private:
  KernelScorer<K> &scorer;

  dim3 calculate_grid(const dim3 &block)
  {
    const dim3 &input_dims = scorer.get_input_dimensions();
    return dim3(
        (input_dims.x + block.x - 1) / block.x,
        (input_dims.y + block.y - 1) / block.y);
  }

public:
  SimulatedAnnealing(KernelScorer<K> &scorer) : scorer(scorer) {};

  /*
    Executa o algoritmo de Simulated Annealing para encontrar uma configuração de lançamento que maximize a pontuação do kernel.
    Retorna a melhor configuração encontrada e sua pontuação.
  */
  std::pair<dim3, double> simulated_annealing(const dim3 &initial_block)
  {
    // Inicialização da geração de números aleatórios (Padrão moderno do C++)
    std::random_device rd;
    std::mt19937 rng(rd()); // Mersenne Twister para boa aleatoriedade
    std::uniform_real_distribution<double> chance_dist(0.0, 1.0);

    // Parâmetros do SA
    double T = 1.0;        // Temperatura inicial alta (combina com nossos scores 0.0 a 1.0)
    double T_min = 0.0001; // Critério de parada
    double alpha = 0.95;   // Taxa de resfriamento (cooling rate)

    // Estado Atual
    dim3 current_block = initial_block;
    double current_score = scorer.score(current_block, calculate_grid(current_block));

    // MEMÓRIA DO MELHOR ESTADO (Crucial!)
    // Como o SA pula para estados piores, ele pode terminar a execução fora do pico.
    // Precisamos salvar a melhor configuração que ele viu no caminho.
    dim3 best_block = current_block;
    double best_score = current_score;

    while (T > T_min)
    {
      // Gerar vizinhos válidos
      std::vector<dim3> neighbors = generate_neighbors(current_block);
      if (neighbors.empty())
        break; // Proteção de segurança

      // Sortear UM vizinho aleatório para testar
      std::uniform_int_distribution<int> neighbor_dist(0, neighbors.size() - 1);
      dim3 next_block = neighbors[neighbor_dist(rng)];

      // Avaliar o vizinho
      double next_score = scorer.score(next_block, calculate_grid(next_block));
      double delta = next_score - current_score;

      // Critério de Aceitação
      if (delta > 0)
      {
        // É melhor! Aceita imediatamente.
        current_block = next_block;
        current_score = next_score;

        // Atualiza o melhor global, se aplicável
        if (current_score > best_score)
        {
          best_score = current_score;
          best_block = current_block;
        }
      }
      else
      {
        // É pior. Aceitamos com base na probabilidade de Metropolis.
        double probability = std::exp(delta / T);
        if (chance_dist(rng) < probability)
        {
          current_block = next_block; // Aceitou o estado pior
          current_score = next_score;
        }
      }

      // Resfriamento
      T *= alpha;
    }

    std::cout << "Busca concluída! Melhor configuração: Block(" << best_block.x << ", " << best_block.y << ") com score " << best_score << std::endl;
    return std::make_pair(best_block, best_score);
  }
};
