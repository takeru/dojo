# Stage 4: ベクトル加算で基礎固め

## 目標

このステージを完了すると、生徒は：
- 2つの配列を使った処理を実装できる
- ベクトル加算（要素ごとの足し算）をGPUで実装できる
- 大きなデータでGPUの速度優位性を実感できる

## 前提知識

- Stage 3完了（速度測定ができる）
- カーネル関数の基本的な書き方
- スレッドインデックスの計算

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **sum-it-up** - 配列の全要素を合計するReductionパターン
- **memory-access-tricks** - メモリアクセスを最適化する方法

## 教え方ガイド

### 導入（なぜこれを学ぶか）

これまでは1つの配列を操作していましたが、実際の計算では複数の配列を扱うことが多いです。ベクトル加算（`C[i] = A[i] + B[i]`）は、機械学習やシミュレーションで頻繁に使われる基本演算です。

このステージでは、2つの配列を足し合わせる処理を通じて、複数のデータを扱う方法を学びます。

### 説明の流れ

1. **ベクトル加算とは**

   ベクトル加算は、2つの配列の対応する要素を足し合わせて、新しい配列を作る処理です：

   ```
   A = [1, 2, 3, 4, 5]
   B = [10, 20, 30, 40, 50]
   C = A + B = [11, 22, 33, 44, 55]
   ```

2. **CPU版の実装**

   ```c
   void vectorAddCPU(int *a, int *b, int *c, int n) {
       for (int i = 0; i < n; i++) {
           c[i] = a[i] + b[i];
       }
   }
   ```

3. **GPU版の実装**

   ```cuda
   __global__ void vectorAddGPU(int *a, int *b, int *c, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }
   ```

4. **完全なプログラム例**

   ```cuda
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   #include <cuda_runtime.h>

   // GPU版
   __global__ void vectorAddGPU(int *a, int *b, int *c, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }

   // CPU版
   void vectorAddCPU(int *a, int *b, int *c, int n) {
       for (int i = 0; i < n; i++) {
           c[i] = a[i] + b[i];
       }
   }

   int main() {
       int n = 10000000;  // 1000万要素

       // CPU側のメモリ確保
       int *h_a = (int*)malloc(n * sizeof(int));
       int *h_b = (int*)malloc(n * sizeof(int));
       int *h_c_cpu = (int*)malloc(n * sizeof(int));
       int *h_c_gpu = (int*)malloc(n * sizeof(int));

       // 配列を初期化
       for (int i = 0; i < n; i++) {
           h_a[i] = i;
           h_b[i] = i * 2;
       }

       // CPU版の実行と測定
       clock_t start_cpu = clock();
       vectorAddCPU(h_a, h_b, h_c_cpu, n);
       clock_t end_cpu = clock();
       double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

       // GPU側のメモリ確保
       int *d_a, *d_b, *d_c;
       cudaMalloc((void**)&d_a, n * sizeof(int));
       cudaMalloc((void**)&d_b, n * sizeof(int));
       cudaMalloc((void**)&d_c, n * sizeof(int));

       // GPU版の実行と測定
       clock_t start_gpu = clock();

       // データ転送 CPU → GPU
       cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

       // カーネル実行
       int blockSize = 256;
       int gridSize = (n + blockSize - 1) / blockSize;
       vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

       // データ転送 GPU → CPU
       cudaMemcpy(h_c_gpu, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

       clock_t end_gpu = clock();
       double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

       // 結果の検証（最初の10要素）
       printf("最初の10要素:\n");
       printf("A: ");
       for (int i = 0; i < 10; i++) printf("%d ", h_a[i]);
       printf("\nB: ");
       for (int i = 0; i < 10; i++) printf("%d ", h_b[i]);
       printf("\nCPU版 C: ");
       for (int i = 0; i < 10; i++) printf("%d ", h_c_cpu[i]);
       printf("\nGPU版 C: ");
       for (int i = 0; i < 10; i++) printf("%d ", h_c_gpu[i]);
       printf("\n\n");

       // 全要素の一致チェック
       int match = 1;
       for (int i = 0; i < n; i++) {
           if (h_c_cpu[i] != h_c_gpu[i]) {
               match = 0;
               printf("不一致: index=%d, CPU=%d, GPU=%d\n", i, h_c_cpu[i], h_c_gpu[i]);
               break;
           }
       }

       if (match) {
           printf("✓ CPU版とGPU版の結果が一致しました！\n\n");
       }

       // 速度比較
       printf("配列サイズ: %d\n", n);
       printf("CPU時間: %.6f 秒\n", time_cpu);
       printf("GPU時間: %.6f 秒\n", time_gpu);
       printf("高速化率: %.2f倍\n", time_cpu / time_gpu);

       // メモリ解放
       free(h_a);
       free(h_b);
       free(h_c_cpu);
       free(h_c_gpu);
       cudaFree(d_a);
       cudaFree(d_b);
       cudaFree(d_c);

       return 0;
   }
   ```

5. **重要ポイント**
   - GPU側に3つの配列（a, b, c）を確保
   - 入力配列（a, b）だけをCPU→GPUに転送
   - 出力配列（c）だけをGPU→CPUに転送
   - ブロックサイズは256が一般的（32の倍数が推奨）

### よくある間違い

- **出力配列cを転送し忘れる**: GPU → CPU のコピーを忘れると、結果が0のまま
- **全ての配列を両方向転送**: 入力はCPU→GPU、出力はGPU→CPUだけで良い
- **ブロックサイズが大きすぎる**: 最大1024まで（256が無難）
- **gridSizeの計算ミス**: `(n + blockSize - 1) / blockSize` で切り上げ

## 演習課題

### 課題1: ベクトル加算の実装
上記のサンプルコードを実行し、CPU版とGPU版で同じ結果が得られることを確認してください。

### 課題2: ベクトル減算に変更
`c[i] = a[i] + b[i]` を `c[i] = a[i] - b[i]` に変更し、ベクトル減算を実装してください。

### 課題3: スカラー倍の実装
`c[i] = a[i] * 3` のように、配列の各要素を定数倍するプログラムを実装してください。

### 課題4（発展）: 3つの配列の加算
`d[i] = a[i] + b[i] + c[i]` のように、3つの配列を足すプログラムを実装してください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] ベクトル加算のGPU版が正しく実行できる
- [ ] CPU版とGPU版の結果が一致する
- [ ] 1000万要素でGPUの方が速いことを確認できる
- [ ] 結果の最初の数要素が正しい値（例: `[0, 3, 6, 9, 12, ...]`）

## ヒント集

### ヒント1（軽め）
ベクトル加算では、3つの配列を扱います。GPU側にも3つ分のメモリを確保してください。

### ヒント2（中程度）
データ転送は、必要なものだけにしましょう：
- CPU → GPU: 入力配列（a, b）のみ
- GPU → CPU: 出力配列（c）のみ

### ヒント3（具体的）
gridSizeの計算は切り上げが必要です：
```cuda
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;  // 切り上げ除算
```

例: n=1000, blockSize=256 の場合
- `gridSize = (1000 + 255) / 256 = 1255 / 256 = 4`
- 合計スレッド数 = 4 × 256 = 1024（> 1000なのでOK）

