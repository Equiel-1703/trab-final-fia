#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "KernelScorer.hpp"
#include "SimulatedAnnealing.hpp"
#include "HillClimbing.hpp"

// ======================================== Multiplicação de Matrizes Ingênuo ========================================
/*
  Matriz A com dimensão M x K
  Matriz B com dimensão K x N
  Matriz C com dimensão M x N

  Regra da multiplicação de matrizes: A(MxK) * B(KxN) = C(MxN)
*/
__global__ void matrix_multiply(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
  // col e row referem-se à matriz de SAÍDA (C), que tem dimensão M x N
  int col = blockIdx.x * blockDim.x + threadIdx.x; // Eixo X vai até N (largura)
  int row = blockIdx.y * blockDim.y + threadIdx.y; // Eixo Y vai até M (altura)

  // Validação de índices
  if (row < M && col < N)
  {
    float sum = 0.0f;

    // k percorre a dimensão interna (K) comum entre A e B
    for (int k = 0; k < K; ++k)
    {
      // Matriz A: K colunas
      // Matriz B: N colunas
      sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
  }
}
// ===================================================================================================================

int main(int argc, const char *args[])
{
  if (argc != 4)
  {
    std::cout << "Multiplicação de Matrizes A x B = C, onde A é MxK, B é KxN e C é MxN" << std::endl;
    std::cout << "Uso: " << args[0] << " <M> <N> <K>" << std::endl;
    return EXIT_FAILURE;
  }

  // Pega tamanho das matrizes a partir dos argumentos na linha de comando
  const int M = std::stoi(args[1]);
  const int N = std::stoi(args[2]);
  const int K = std::stoi(args[3]);

  // As dimensões que estamos interessados são apenas da matriz de resultado C, que é M x N
  // O valor K só é utilizado pelo kernel para iterar sobre as linhas e colunas de A e B, mas é irrelevante
  // para a configuração de blocos e grids, que é o que estamos otimizando.
  const dim3 dimensions(N, M, 1);

  // Instanciando o avaliador com o kernel da multiplicação de matrizes e as dimensões de entrada
  KernelScorer<decltype(matrix_multiply)> scorer(matrix_multiply, dimensions);

  // Instanciando os algoritmos de IA
  SimulatedAnnealing<decltype(matrix_multiply)> sa(scorer);
  HillClimbing<decltype(matrix_multiply)> hc(scorer);

  // Começamos de propósito com um bloco pequeno e ruim
  dim3 initial_block(8, 8);

  std::cout << "\n===== Iniciando Algoritmos =====\n";
  std::cout << "Matriz A: " << M << " x " << K << std::endl;
  std::cout << "Matriz B: " << K << " x " << N << std::endl;
  std::cout << "Matriz C: " << M << " x " << N << std::endl;
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
