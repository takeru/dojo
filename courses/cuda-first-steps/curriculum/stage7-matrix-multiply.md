# Stage 7: 行列積で本気を出す

## 目標

このステージを完了すると、生徒は：
- 行列の乗算をGPUで実装できる
- 2次元データ構造を扱えるようになる
- GPUの圧倒的な速度優位性を実感できる（数十倍〜数百倍）

## 前提知識

- Stage 6完了（2次元グリッドが使える）
- 行列積の基本的な計算方法
- 2次元配列の1次元配列への変換

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **matrix-speedup** - タイリング手法でさらに10倍速くする
- **memory-access-tricks** - メモリアクセスパターンの最適化
- **visual-profiling** - Nsightでボトルネックを可視化
- **tuning-basics** - さらなる性能チューニング

## 教え方ガイド

### 導入（なぜこれを学ぶか）

行列の乗算は、機械学習（ニューラルネットワーク）、画像処理、科学計算などで最も重要な演算の1つです。CPUでは時間がかかりすぎる大きな行列も、GPUなら一瞬で計算できます。

このステージでは、行列積を実装して、**GPUの本領発揮**を体験します。数十倍から数百倍の高速化が期待できます！

### 説明の流れ

1. **行列積の基本**

   2つの行列A（M×K）とB（K×N）を掛けて、C（M×N）を計算します：

   ```
   C[i][j] = Σ(k=0...K-1) A[i][k] * B[k][j]
   ```

   例: 2×2 行列
   ```
   A = [1, 2]    B = [5, 6]    C = [1*5+2*7, 1*6+2*8]   = [19, 22]
       [3, 4]        [7, 8]        [3*5+4*7, 3*6+4*8]     [43, 50]
   ```

2. **CPU版の実装**

   ```c
   void matrixMulCPU(float *A, float *B, float *C, int M, int K, int N) {
       for (int i = 0; i < M; i++) {
           for (int j = 0; j < N; j++) {
               float sum = 0.0f;
               for (int k = 0; k < K; k++) {
                   sum += A[i * K + k] * B[k * N + j];
               }
               C[i * N + j] = sum;
           }
       }
   }
   ```

   **計算量**: O(M × N × K)
   例えば、1000×1000の行列なら、10億回の計算！

3. **GPU版の実装（基本版）**

   各スレッドが1つの要素C[i][j]を担当します：

   ```cuda
   __global__ void matrixMulGPU(float *A, float *B, float *C, int M, int K, int N) {
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int col = blockIdx.x * blockDim.x + threadIdx.x;

       if (row < M && col < N) {
           float sum = 0.0f;
           for (int k = 0; k < K; k++) {
               sum += A[row * K + k] * B[k * N + col];
           }
           C[row * N + col] = sum;
       }
   }
   ```

4. **完全なプログラム**

   ```cuda
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   #include <cuda_runtime.h>

   // CPU版
   void matrixMulCPU(float *A, float *B, float *C, int M, int K, int N) {
       for (int i = 0; i < M; i++) {
           for (int j = 0; j < N; j++) {
               float sum = 0.0f;
               for (int k = 0; k < K; k++) {
                   sum += A[i * K + k] * B[k * N + j];
               }
               C[i * N + j] = sum;
           }
       }
   }

   // GPU版
   __global__ void matrixMulGPU(float *A, float *B, float *C, int M, int K, int N) {
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int col = blockIdx.x * blockDim.x + threadIdx.x;

       if (row < M && col < N) {
           float sum = 0.0f;
           for (int k = 0; k < K; k++) {
               sum += A[row * K + k] * B[k * N + col];
           }
           C[row * N + col] = sum;
       }
   }

   int main() {
       int M = 1024, K = 1024, N = 1024;  // 1024×1024 行列

       size_t size_A = M * K * sizeof(float);
       size_t size_B = K * N * sizeof(float);
       size_t size_C = M * N * sizeof(float);

       // CPU側メモリ確保
       float *h_A = (float*)malloc(size_A);
       float *h_B = (float*)malloc(size_B);
       float *h_C_cpu = (float*)malloc(size_C);
       float *h_C_gpu = (float*)malloc(size_C);

       // 行列を初期化
       for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
       for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

       // CPU版の実行と測定
       printf("CPU版を実行中...\n");
       clock_t start_cpu = clock();
       matrixMulCPU(h_A, h_B, h_C_cpu, M, K, N);
       clock_t end_cpu = clock();
       double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

       // GPU側メモリ確保
       float *d_A, *d_B, *d_C;
       cudaMalloc((void**)&d_A, size_A);
       cudaMalloc((void**)&d_B, size_B);
       cudaMalloc((void**)&d_C, size_C);

       // GPU版の実行と測定
       printf("GPU版を実行中...\n");
       clock_t start_gpu = clock();

       cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
       cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

       dim3 blockSize(16, 16);  // 16×16 = 256スレッド/ブロック
       dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                     (M + blockSize.y - 1) / blockSize.y);

       matrixMulGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

       cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

       clock_t end_gpu = clock();
       double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

       // 結果の検証（一部をチェック）
       printf("\n=== 結果検証 ===\n");
       int errors = 0;
       for (int i = 0; i < M * N; i++) {
           if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 0.01f) {
               errors++;
               if (errors < 5) {  // 最初の数個だけ表示
                   printf("不一致: index=%d, CPU=%.2f, GPU=%.2f\n",
                          i, h_C_cpu[i], h_C_gpu[i]);
               }
           }
       }

       if (errors == 0) {
           printf("✓ CPU版とGPU版の結果が一致しました！\n");
       } else {
           printf("✗ %d 個の要素が不一致\n", errors);
       }

       // 速度比較
       printf("\n=== 性能比較 ===\n");
       printf("行列サイズ: %d × %d\n", M, N);
       printf("CPU時間: %.6f 秒\n", time_cpu);
       printf("GPU時間: %.6f 秒\n", time_gpu);
       printf("高速化率: %.2f倍\n", time_cpu / time_gpu);

       // メモリ解放
       free(h_A);
       free(h_B);
       free(h_C_cpu);
       free(h_C_gpu);
       cudaFree(d_A);
       cudaFree(d_B);
       cudaFree(d_C);

       return 0;
   }
   ```

5. **実行結果の例**

   ```
   CPU版を実行中...
   GPU版を実行中...

   === 結果検証 ===
   ✓ CPU版とGPU版の結果が一致しました！

   === 性能比較 ===
   行列サイズ: 1024 × 1024
   CPU時間: 12.345678 秒
   GPU時間: 0.123456 秒
   高速化率: 100.00倍
   ```

   **100倍速い！**

6. **重要ポイント**
   - ブロックサイズは 16×16（=256）が一般的
   - 行列が大きいほど、GPUの高速化率が上がる
   - 2048×2048なら200倍以上の高速化も可能

### よくある間違い

- **行と列を逆にする**: `blockIdx.x` が列、`blockIdx.y` が行
- **インデックス計算ミス**: `row * K + k` と `k * N + col` を間違える
- **境界チェック忘れ**: `if (row < M && col < N)` が必須
- **floatとintを混同**: 行列計算は通常float型

## 演習課題

### 課題1: 行列積の実装
上記のプログラムを実行し、CPU版とGPU版で同じ結果が得られることを確認してください。

### 課題2: 行列サイズを変えて測定
行列サイズを 256, 512, 1024, 2048 と変えて、高速化率がどう変わるか調べてください。

### 課題3: 結果の一部を表示
計算結果の左上3×3の要素を表示して、正しく計算されているか目視確認してください。

### 課題4（発展）: 非正方行列
M=512, K=1024, N=256 のような非正方行列の乗算を試してください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] 行列積のGPU版が正しく実行できる
- [ ] CPU版とGPU版の結果が一致する
- [ ] 1024×1024以上の行列で50倍以上の高速化を達成できる
- [ ] 2次元グリッドとブロックを正しく使えている

## ヒント集

### ヒント1（軽め）
まずは小さい行列（例: 4×4）で試して、計算が正しいか確認しましょう。

### ヒント2（中程度）
2次元のスレッドインデックスを正しく計算してください：
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列
```

### ヒント3（具体的）
行列の要素へのアクセス：
```cuda
// A[row][k] → A[row * K + k]
// B[k][col] → B[k * N + col]
// C[row][col] → C[row * N + col]
```

## 補足・発展トピック

ステージクリア後、生徒が「もっと知りたい」を選んだら、`/dojo:topic` コマンドで以下のトピックを選択できます：

- **matrix-speedup** - シェアードメモリを使ったタイリング手法で10倍高速化
- **memory-access-tricks** - メモリアクセスパターンの最適化
- **visual-profiling** - Nsightでプログラムの動作を可視化

### 参考リンク

- [CUDA C++ Programming Guide - Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/) - NVIDIA提供の超高速行列演算ライブラリ
