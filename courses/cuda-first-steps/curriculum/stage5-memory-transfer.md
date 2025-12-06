# Stage 5: メモリ転送のコストを知る

## 目標

このステージを完了すると、生徒は：
- CPU-GPU間のデータ転送時間を測定できる
- 計算時間と転送時間を分けて分析できる
- 転送がボトルネックになる場合を理解できる

## 前提知識

- Stage 4完了（ベクトル加算が実装できる）
- `cudaMemcpy` の使い方
- 時間測定の方法

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **easy-memory-management** - Unified Memoryで転送を簡単にする
- **fast-memory-transfer** - Pinned Memoryで転送を高速化
- **memory-playground** - GPU内のメモリの種類と使い分け
- **multiple-tasks** - ストリームで転送と計算を並行実行

## 教え方ガイド

### 導入（なぜこれを学ぶか）

前のステージで「GPUは速い」ことを体感しましたが、実は**データ転送**が遅いという問題があります。どんなに計算が速くても、データを運ぶのに時間がかかったら意味がありません。

このステージでは、転送時間と計算時間を分けて測定し、どちらがボトルネックかを分析します。

### 説明の流れ

1. **転送時間を個別に測定する**

   これまでは全部まとめて測っていましたが、細かく分けてみます：

   ```cuda
   #include <stdio.h>
   #include <stdlib.h>
   #include <time.h>
   #include <cuda_runtime.h>

   __global__ void vectorAddGPU(int *a, int *b, int *c, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }

   int main() {
       int n = 10000000;  // 1000万要素
       size_t bytes = n * sizeof(int);

       // メモリ確保
       int *h_a = (int*)malloc(bytes);
       int *h_b = (int*)malloc(bytes);
       int *h_c = (int*)malloc(bytes);
       int *d_a, *d_b, *d_c;

       // 初期化
       for (int i = 0; i < n; i++) {
           h_a[i] = i;
           h_b[i] = i * 2;
       }

       cudaMalloc((void**)&d_a, bytes);
       cudaMalloc((void**)&d_b, bytes);
       cudaMalloc((void**)&d_c, bytes);

       // === 転送時間の測定 ===
       clock_t start_h2d = clock();
       cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
       cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
       clock_t end_h2d = clock();
       double time_h2d = (double)(end_h2d - start_h2d) / CLOCKS_PER_SEC;

       // === 計算時間の測定 ===
       int blockSize = 256;
       int gridSize = (n + blockSize - 1) / blockSize;

       clock_t start_kernel = clock();
       vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
       cudaDeviceSynchronize();  // カーネル完了まで待機
       clock_t end_kernel = clock();
       double time_kernel = (double)(end_kernel - start_kernel) / CLOCKS_PER_SEC;

       // === 転送時間の測定（GPU→CPU） ===
       clock_t start_d2h = clock();
       cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
       clock_t end_d2h = clock();
       double time_d2h = (double)(end_d2h - start_d2h) / CLOCKS_PER_SEC;

       // 結果表示
       printf("データサイズ: %d 要素 (%.2f MB)\n", n, bytes / (1024.0 * 1024.0));
       printf("\n=== 詳細な時間測定 ===\n");
       printf("CPU→GPU転送時間: %.6f 秒\n", time_h2d);
       printf("カーネル実行時間: %.6f 秒\n", time_kernel);
       printf("GPU→CPU転送時間: %.6f 秒\n", time_d2h);
       printf("合計GPU時間:      %.6f 秒\n", time_h2d + time_kernel + time_d2h);
       printf("\n=== 内訳 ===\n");
       printf("転送時間の割合: %.1f%%\n", (time_h2d + time_d2h) / (time_h2d + time_kernel + time_d2h) * 100);
       printf("計算時間の割合: %.1f%%\n", time_kernel / (time_h2d + time_kernel + time_d2h) * 100);

       // メモリ解放
       free(h_a);
       free(h_b);
       free(h_c);
       cudaFree(d_a);
       cudaFree(d_b);
       cudaFree(d_c);

       return 0;
   }
   ```

2. **実行結果の例**

   ```
   データサイズ: 10000000 要素 (38.15 MB)

   === 詳細な時間測定 ===
   CPU→GPU転送時間: 0.008234 秒
   カーネル実行時間: 0.000512 秒
   GPU→CPU転送時間: 0.007891 秒
   合計GPU時間:      0.016637 秒

   === 内訳 ===
   転送時間の割合: 96.9%
   計算時間の割合: 3.1%
   ```

   **重要**: 転送が全体の96%以上を占めている！

3. **計算量を増やすとどうなる？**

   ベクトル加算は計算が軽すぎるので、もっと重い処理にしてみます：

   ```cuda
   __global__ void heavyComputation(int *a, int *b, int *c, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           float result = a[idx] + b[idx];
           // 重い計算を追加
           for (int i = 0; i < 1000; i++) {
               result = result * 1.0001f + 0.0001f;
           }
           c[idx] = (int)result;
       }
   }
   ```

   すると：
   ```
   CPU→GPU転送時間: 0.008234 秒
   カーネル実行時間: 0.102345 秒  ← 増えた！
   GPU→CPU転送時間: 0.007891 秒

   転送時間の割合: 13.7%
   計算時間の割合: 86.3%
   ```

   計算が重くなると、転送時間の割合が減ります。

4. **帯域幅の計算**

   転送速度（帯域幅）を計算してみます：

   ```
   データ量 = 38.15 MB
   転送時間 = 0.008234 秒
   帯域幅 = 38.15 / 0.008234 = 4.63 GB/秒
   ```

   PCIe 3.0 x16の理論値は約12 GB/秒なので、実効速度は理論値の約40%です。

### よくある間違い

- **cudaDeviceSynchronize()を忘れる**: カーネルは非同期なので、測定には同期が必要
- **メモリ確保時間を含めてしまう**: `cudaMalloc` は初回が遅いので、測定から外す
- **初回実行を測る**: GPU初期化があるため、2回目以降を測るべき
- **転送方向を間違える**: `cudaMemcpyHostToDevice` と `DeviceToHost` を取り違える

## 演習課題

### 課題1: 転送時間の測定
上記のプログラムを実行し、転送時間と計算時間の内訳を確認してください。

### 課題2: データサイズを変えて測定
データサイズを 100万、1000万、1億 と変えて、転送時間の変化を観察してください。

### 課題3: 帯域幅の計算
転送時間から帯域幅（GB/秒）を計算してください。

### 課題4（発展）: 転送を減らす工夫
配列cを初期化せずに、GPU上でだけ使う場合、CPU→GPU転送を省略できます。試してみてください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] CPU→GPU転送時間を測定できる
- [ ] カーネル実行時間を測定できる
- [ ] GPU→CPU転送時間を測定できる
- [ ] 転送時間が計算時間より長いことを確認できる

## ヒント集

### ヒント1（軽め）
時間測定を細かく分けて、どこで時間がかかっているか調べましょう。

### ヒント2（中程度）
カーネルの実行時間を測る際は、`cudaDeviceSynchronize()` を呼んでGPUの処理が終わるまで待ってください：
```cuda
clock_t start = clock();
kernel<<<...>>>(...);
cudaDeviceSynchronize();
clock_t end = clock();
```

### ヒント3（具体的）
転送時間の割合を計算するには：
```c
double total = time_h2d + time_kernel + time_d2h;
double transfer_percent = (time_h2d + time_d2h) / total * 100.0;
```

