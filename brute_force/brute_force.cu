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

int main(int argc, char const *argv[])
{
  if (argc != 4)
  {
    std::cerr << "Uso: " << argv[0] << " <tamanho_da_matriz> <tamanho_do_bloco_x> <tamanho_do_bloco_y>" << std::endl;
    return EXIT_FAILURE;
  }

  // Ler os argumentos da linha de comando
  int N = std::stoi(argv[1]);
  int block_x = std::stoi(argv[2]);
  int block_y = std::stoi(argv[3]);

  // Configurando grid e block
  dim3 block(block_x, block_y);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // Imprimindo configurações do lançamento do kernel
  std::cout << "Tamanho da matriz: " << N << "x" << N << std::endl;
  std::cout << "Tamanho do bloco: " << block.x << "x" << block.y << std::endl;
  std::cout << "Tamanho do grid: " << grid.x << "x" << grid.y << std::endl;

  // Alocando memória para as matrizes A, B e C na CPU
  float *h_A = new float[N * N];
  float *h_B = new float[N * N];
  float *h_C = new float[N * N];

  // Inicializando as matrizes A e B com valores aleatórios
  std::mt19937 rng(42); // Semente fixa para reprodutibilidade
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < N * N; ++i)
  {
    h_A[i] = dist(rng);
    h_B[i] = dist(rng);
  }

  // Alocando memória para as matrizes A, B e C na GPU
  float *d_A, *d_B, *d_C;
  check_cuda_error(cudaMalloc(&d_A, N * N * sizeof(float)));
  check_cuda_error(cudaMalloc(&d_B, N * N * sizeof(float)));
  check_cuda_error(cudaMalloc(&d_C, N * N * sizeof(float)));

  // Copiando as matrizes A e B da CPU para a GPU
  check_cuda_error(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

  // Medindo o tempo de execução do kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Executando o kernel de multiplicação de matrizes
  matrix_multiply<<<grid, block>>>(d_A, d_B, d_C, N);

  // Verifica se houve algum erro durante o lançamento do kernel
  check_cuda_error(cudaGetLastError());
  check_cuda_error(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Tempo de execução: " << milliseconds << " ms" << std::endl;

  // Copiando o resultado da GPU para a CPU
  check_cuda_error(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

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
