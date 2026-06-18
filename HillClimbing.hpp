#pragma once

#include <vector>
#include <iostream>

#include "neighbors.hpp"
#include "KernelScorer.hpp"

template <typename K>
class HillClimbing
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
  HillClimbing(KernelScorer<K> &scorer) : scorer(scorer) {};

  /*
    Executa o Steepest Ascent Hill Climbing. Este é uma variação do Hill Climbing tradicional
    que em vez de testar um vizinho aleatório e já pular para ele se for melhor, o algoritmo
    olha para todos os vizinhos válidos ao redor, encontra qual deles tem a maior nota, e
    só então dá o passo na direção desse "melhor vizinho". Ele repete isso até chegar em
    um ponto onde nenhum vizinho é melhor que o estado atual.
  */
  std::pair<dim3, double> hill_climbing(const dim3 &initial_block)
  {
    // Estado Atual
    dim3 current_block = initial_block;
    double current_score = scorer.score(current_block, calculate_grid(current_block));

    // Loop infinito que só é quebrado quando chegamos no ótimo local
    while (true)
    {
      std::vector<dim3> neighbors = generate_neighbors(current_block);
      if (neighbors.empty())
        break; // Proteção de segurança caso não haja vizinhos

      double best_neighbor_score = -1.0;
      dim3 best_neighbor_block = current_block;

      // Olhar ao redor: Avalia TODOS os vizinhos possíveis
      for (const dim3 &neighbor : neighbors)
      {
        double neighbor_score = scorer.score(neighbor, calculate_grid(neighbor));

        if (neighbor_score > best_neighbor_score)
        {
          best_neighbor_score = neighbor_score;
          best_neighbor_block = neighbor;
        }
      }

      // O melhor vizinho é melhor que o meu estado atual? Esse algoritmo é "greedy"
      if (best_neighbor_score > current_score)
      {
        // Se sim, dá um passo morro acima.
        current_block = best_neighbor_block;
        current_score = best_neighbor_score;
      }
      else
      {
        // Se não, então todos os vizinhos são piores ou iguais.
        // Chegamos no topo de um ótimo local, e não há mais para onde subir.
        // Fim do algoritmo.
        break;
      }
    }

    std::cout << "Hill Climbing concluído! Pico encontrado: Block("
              << current_block.x << ", " << current_block.y
              << ") com score " << current_score << std::endl;

    return std::make_pair(current_block, current_score);
  }
};