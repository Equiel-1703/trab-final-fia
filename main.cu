#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelScorer.hpp"
#include "SimulatedAnnealing.hpp"
#include "HillClimbing.hpp"

// ======================================== Multiplicação de Matrizes Ingênuo ========================================
__global__ void matrix_multiply(const float *A, const float *B, float *C, int N)
{
  // Mapeamento 2D das threads para as coordenadas da matriz
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary Check (Proteção de Limites)
  // Garante que apenas as threads dentro do tamanho válido N façam trabalho.
  // As threads "excedentes" nas bordas falham silenciosamente aqui, ativando
  // o desperdício que a boundary_efficiency penaliza
  if (row < N && col < N)
  {
    float sum = 0.0f;

    // Algoritmo Ingênuo: Produto escalar da linha de A pela coluna de B
    for (int k = 0; k < N; ++k)
    {
      // A é percorrido por linha (acesso contíguo na memória, bom coalescimento)
      // B é percorrido por coluna (strides longos, péssimo coalescimento)
      sum += A[row * N + k] * B[k * N + col];
    }

    // Escrita do resultado
    C[row * N + col] = sum;
  }
}
// ===================================================================================================================

int main(int argc, const char *args[])
{
  if (argc != 2)
  {
    std::cerr << "Uso: " << args[0] << " <tamanho_da_matriz>" << std::endl;
    return EXIT_FAILURE;
  }

  // Pega tamanho da matriz a partir do argumento na linha de comando
  const int INPUT_SIZE = std::stoi(args[1]);
  const dim3 input_dimensions(INPUT_SIZE, INPUT_SIZE, 1);

  // Instanciando o avaliador com o kernel da multiplicação de matrizes e as dimensões de entrada
  KernelScorer<decltype(matrix_multiply)> scorer(matrix_multiply, input_dimensions);

  // Instanciando os algoritmos de IA
  SimulatedAnnealing<decltype(matrix_multiply)> sa(scorer);
  HillClimbing<decltype(matrix_multiply)> hc(scorer);

  // Começamos de propósito com um bloco pequeno e ruim
  dim3 initial_block(8, 8);

  std::cout << "\n===== Iniciando Algoritmos =====\n";
  std::cout << "Tamanho da Matriz: " << INPUT_SIZE << " x " << INPUT_SIZE << "\n";
  std::cout << "Bloco Inicial da Busca: (" << initial_block.x << ", " << initial_block.y << ")\n\n";

  std::cout << "\n===== Executando Hill Climbing =====\n";
  auto hc_result = hc.hill_climbing(initial_block);

  dim3 hc_best_block = hc_result.first;
  double hc_best_score = hc_result.second;

  std::cout << "\n\t* Melhor bloco (Hill Climbing): " << "(" << hc_best_block.x << ", " << hc_best_block.y << ")\n";
  std::cout << "\t* Pontuação (Hill Climbing): " << hc_best_score << " (0.0 a 1.0, onde 1.0 é o ideal)\n";

  std::cout << "\n===== Executando Simulated Annealing =====\n";
  auto sa_result = sa.simulated_annealing(initial_block);

  dim3 sa_best_block = sa_result.first;
  double sa_best_score = sa_result.second;

  std::cout << "\n\t* Melhor bloco (Simulated Annealing): " << "(" << sa_best_block.x << ", " << sa_best_block.y << ")\n";
  std::cout << "\t* Pontuação (Simulated Annealing): " << sa_best_score << " (0.0 a 1.0, onde 1.0 é o ideal)\n";

  return EXIT_SUCCESS;
}
