#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

// Função de checagem de erros CUDA
inline void check_cuda_error(const cudaError_t &err)
{
  if (err != cudaSuccess)
  {
    std::cerr << "Erro CUDA: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

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

int main(int argc, char const *argv[])
{
  if (argc != 6)
  {
    std::cout << "Multiplicação de Matrizes A x B = C, onde A é MxK, B é KxN e C é MxN" << std::endl;
    std::cout << "Uso: " << argv[0] << " <M> <N> <K> <block_x> <block_y>" << std::endl;
    return EXIT_FAILURE;
  }

  // Ler os argumentos da linha de comando
  const int M = std::stoi(argv[1]);
  const int N = std::stoi(argv[2]);
  const int K = std::stoi(argv[3]);
  const int block_x = std::stoi(argv[4]);
  const int block_y = std::stoi(argv[5]);

  // Configurando grid e block
  dim3 block(block_x, block_y);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // Imprimindo configurações do lançamento do kernel
  std::cout << "Matriz A: " << M << "x" << K << std::endl;
  std::cout << "Matriz B: " << K << "x" << N << std::endl;
  std::cout << "Matriz C: " << M << "x" << N << std::endl;
  std::cout << "Tamanho do bloco: " << block.x << "x" << block.y << std::endl;
  std::cout << "Tamanho do grid: " << grid.x << "x" << grid.y << std::endl;

  // Alocando memória para as matrizes A, B e C na CPU
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];

  // Inicializando as matrizes A e B com valores aleatórios
  std::mt19937 rng(42); // Semente fixa para reprodutibilidade
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Inicializando a matriz A (M x K)
  for (int i = 0; i < M * K; ++i)
  {
    h_A[i] = dist(rng);
  }

  // Inicializando a matriz B (K x N)
  for (int i = 0; i < K * N; ++i)
  {
    h_B[i] = dist(rng);
  }

  // Alocando memória para as matrizes A, B e C na GPU
  float *d_A, *d_B, *d_C;
  check_cuda_error(cudaMalloc(&d_A, M * K * sizeof(float)));
  check_cuda_error(cudaMalloc(&d_B, K * N * sizeof(float)));
  check_cuda_error(cudaMalloc(&d_C, M * N * sizeof(float)));

  // Copiando as matrizes A e B da CPU para a GPU
  check_cuda_error(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Medindo o tempo de execução do kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Executando o kernel de multiplicação de matrizes
  matrix_multiply<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  // Verifica se houve algum erro durante o lançamento do kernel
  check_cuda_error(cudaGetLastError());
  check_cuda_error(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Tempo de execução: " << milliseconds << " ms" << std::endl;

  // Copiando o resultado da GPU para a CPU
  check_cuda_error(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Liberando a memória alocada na GPU
  check_cuda_error(cudaFree(d_A));
  check_cuda_error(cudaFree(d_B));
  check_cuda_error(cudaFree(d_C));

  // Liberando a memória alocada na CPU
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
