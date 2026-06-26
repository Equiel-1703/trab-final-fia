# Otimizando Lançamento de Kernels CUDA com Algoritmos Clássicos de IA

![CUDA](https://img.shields.io/badge/CUDA-C++-76B900?style=for-the-badge&logo=nvidia)
![C++17](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)

Este projeto investiga a utilização de algoritmos de busca clássicos de **Inteligência Artificial** para encontrar configurações eficientes de lançamento de kernels CUDA (`blockDim` e `gridDim`) em problemas de Computação de Alto Desempenho.

Neste estudo de caso, utilizamos **Hill Climbing** e **Simulated Annealing** na otimização do lançamento de um kernel bidimensional de Multiplicação de Matrizes Ingênuo (Naive Matrix Multiplication). Os algoritmos exploram as combinações possíveis de blocos avaliando a qualidade de cada configuração com uma **Função de Avaliação Heurística** que modela o comportamento da GPU para cada cenário.

Nossa abordagem é puramente analítica, nós não lançamos o kernel de fato. Isso permite explorar o espaço de busca de forma extremamente rápida e com baixo custo computacional.

## O Problema

Otimizar configurações de blocos e grids em GPGPU geralmente envolve rodar o kernel milhares de vezes empíricamente e selecionar a configuração ótima (auto-tuning). Esse processo é muito custoso. Neste projeto, modelamos o espaço de busca (dimensões $X \times Y$ de um bloco da GPU) como estados, e substituímos a execução real por uma **Função de Avaliação Heurística**. A IA explora o espaço de blocos possíveis e avalia a qualidade de cada configuração de forma puramente analítica (sem rodar o kernel de fato). Isso é muito mais barato e rápido do que o auto-tuning tradicional.

## Algoritmos de Busca Implementados

1. **Steepest Ascent Hill Climbing:** É uma variação do Hill Climbing tradicional. Este é um algoritmo de busca local gulosa que avalia **TODOS** os vizinhos válidos e sempre caminha na direção da maior melhoria, parando ao encontrar o pico de um ótimo local.
2. **Simulated Annealing (Têmpera Simulada):** É também um algoritmo de busca local. Ele utiliza uma "temperatura" virtual para aceitar probabilisticamente soluções piores. Isso permite escapar de ótimos locais e explorar o espaço de dimensões da GPU de forma mais ampla antes de estabilizar na melhor configuração geométrica.

## A Função de Avaliação Heurística

Nós implementamos uma classe chamada `KernelScorer` que avalia a qualidade de uma configuração de bloco para um kernel previamente fornecido à classe. A pontuação do bloco (score) vai de 0.0 a 1.0, onde 0.0 é a pior configuração e 1.0 é a melhor. O score é obtido através de uma composição ponderada das seguintes métricas:

* **Occupancy Score:** Utilizando a API CUDA, calculamos a ocupação dos SMs (Streaming Multiprocessors) da GPU com base no kernel e na configuração do bloco. A occupancy é uma métrica que avalia a proporção de warps ativos em cada SM em relação ao máximo possível;
* **Warp Efficiency:** Avalia o desperdício causado por blocos cujas threads totais não preenchem warps completos (múltiplos de 32);
* **Wave Quantization Efficiency:** Avalia o "Tail Effect" (efeito de cauda) na GPU. Essa métrica avalia o número total de blocos lançados e o máximo de blocos que a GPU consegue processar de uma vez. Se o número de blocos lançados não for um múltiplo do máximo de blocos processáveis, haverá desperdício;
* **Boundary Efficiency:** Penaliza grades que lançam muitas threads fora dos limites úteis da matriz computada (threads mortas em `if (row < M)`);
* **Memory Coalescing:** Avalia rigorosamente o alinhamento de memória global no eixo X do bloco, exigindo múltiplos perfeitos do Warp Size (32 bytes) para maximizar a largura de banda;
* **Spatial Locality (Cache L1):** Avalia a proporção Área-Perímetro (Intensidade Aritmética) do bloco, premiando geometrias bidimensionais (como 16x16 ou 32x16) que promovem a reutilização massiva de dados no cache L1 durante a multiplicação de matrizes;
* **Matrix Alignment:** Avalia o grau de similaridade geométrica entre a proporção da matriz de saída ($M \times N$) e a geometria do bloco;

## Como Compilar e Rodar

### Pré-requisitos

* NVIDIA CUDA Toolkit instalado (`nvcc`);

* Compilador compatível com C++17;

### Rodando os Algoritmos de IA

Para executar os algoritmos de busca, primeiro compile o programa utilizando o Makefile fornecido:

```bash
make clean # Apenas para limpar arquivos antigos
make
```

Depois, execute o programa fornecendo as dimensões `<M> <N> <K>` das matrizes de entrada $A$ e $B$, onde $A$ é de dimensão $M \times K$ e $B$ é de dimensão $K \times N$, resultando em uma matriz $C$ de dimensão $M \times N$.

O programa irá otimizar a configuração de lançamento do kernel para multiplicar essas matrizes. No exemplo a seguir, estamos rodando o programa para otimizar a multiplicação de duas matrizes, onde a primeira tem tamanho $2000 \times 500$ e a segunda tem tamanho $500 \times 3000$:

```bash
$ ./main.out 2000 3000 500
Number of Streaming Multiprocessors (SMs): 46

===== Iniciando Algoritmos =====
Matriz A: 2000 x 500
Matriz B: 500 x 3000
Matriz C: 2000 x 3000
Bloco Inicial da Busca: (8, 8)


===== Executando Hill Climbing =====

  * Melhor bloco (Hill Climbing): (32, 24)
  * Pontuação (Hill Climbing): 0.979403 (0.0 a 1.0, onde 1.0 é o ideal)

===== Executando Simulated Annealing =====

  * Melhor bloco (Simulated Annealing): (32, 24)
  * Pontuação (Simulated Annealing): 0.979403 (0.0 a 1.0, onde 1.0 é o ideal)
```

Para validar se essas configurações de blocos são de fato a melhor configuração possível, você pode compilar e executar o programa de validação na pasta `validate/`. Este programa irá lançar o kernel de multiplicação de matrizes com a configuração de blocos fornecida e medir o tempo de execução real.

Para compilar o programa de validação, siga os passos abaixo:

```bash
cd validate # Vai para a pasta do programa de validação
make clean # Apenas para limpar arquivos antigos
make        # Compila o programa de validação
```

Para executar, forneça as dimensões das matrizes de entrada (`<M> <N> <K>`) e a configuração de blocos (`<blockDimX> <blockDimY>`). No exemplo abaixo, estamos medindo o tempo de execução da multiplicação de matrizes com as mesmas dimensões do exemplo anterior e com a configuração de blocos encontrada pelos algoritmos de IA (32, 24):

```bash
$ ./validate.out 2000 3000 500 32 24
Matriz A: 2000x500
Matriz B: 500x3000
Matriz C: 2000x3000
Tamanho do bloco: 32x24
Tamanho do grid: 94x84
Tempo de execução: 3.45603 ms
```

O exemplo acima foi executado em uma máquina equipada com uma GPU NVIDIA RTX 4070, que possui 46 SMs.
