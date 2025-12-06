# Stage 3: 速度を測ってみる

## 目標

このステージを完了すると、生徒は：
- CPU版とGPU版の実行時間を測定できる
- 速度差を数値で確認できる
- データサイズによる速度の変化を観察できる

## 前提知識

- Stage 2完了（CPU版とGPU版の実装ができる）
- 基本的な時間計測の概念

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **memory-playground** - メモリアクセスのパターンが速度に与える影響
- **tuning-basics** - さらに高速化するための基礎知識

## 教え方ガイド

### 導入（なぜこれを学ぶか）

「GPUは速い」と言われても、実際にどれくらい速いのか数字で見ないと実感が湧きません。このステージでは、時間を測定してCPUとGPUの速度差を**数値で**確認します。

最初は「あれ、GPUの方が遅い？」となるかもしれませんが、データサイズを大きくしていくと、徐々にGPUの本領が発揮されます。

### 説明の流れ

1. **C言語での時間計測方法**

   `<time.h>` を使った基本的な計測方法：

   ```c
   #include <time.h>

   clock_t start = clock();
   // 計測したい処理
   clock_t end = clock();
   double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
   printf("実行時間: %.6f 秒\n", elapsed);
   ```

2. **CPU版とGPU版の速度比較プログラム**

   ```cuda
   #include <stdio.h>
   #include <time.h>
   #include <cuda_runtime.h>

   // CPU版
   void doubleArrayCPU(int *arr, int n) {
       for (int i = 0; i < n; i++) {
           arr[i] = arr[i] * 2;
       }
   }

   // GPU版
   __global__ void doubleArrayGPU(int *arr, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           arr[idx] = arr[idx] * 2;
       }
   }

   int main() {
       int n = 1000000;  // 100万要素
       int *h_arr_cpu = (int*)malloc(n * sizeof(int));
       int *h_arr_gpu = (int*)malloc(n * sizeof(int));
       int *d_arr;

       // 配列を初期化
       for (int i = 0; i < n; i++) {
           h_arr_cpu[i] = i;
           h_arr_gpu[i] = i;
       }

       // CPU版の速度測定
       clock_t start_cpu = clock();
       doubleArrayCPU(h_arr_cpu, n);
       clock_t end_cpu = clock();
       double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

       // GPU版の速度測定
       clock_t start_gpu = clock();

       cudaMalloc((void**)&d_arr, n * sizeof(int));
       cudaMemcpy(d_arr, h_arr_gpu, n * sizeof(int), cudaMemcpyHostToDevice);

       int blockSize = 256;
       int gridSize = (n + blockSize - 1) / blockSize;
       doubleArrayGPU<<<gridSize, blockSize>>>(d_arr, n);

       cudaMemcpy(h_arr_gpu, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
       cudaFree(d_arr);

       clock_t end_gpu = clock();
       double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

       // 結果表示
       printf("配列サイズ: %d\n", n);
       printf("CPU時間: %.6f 秒\n", time_cpu);
       printf("GPU時間: %.6f 秒\n", time_gpu);
       printf("高速化率: %.2f倍\n", time_cpu / time_gpu);

       free(h_arr_cpu);
       free(h_arr_gpu);

       return 0;
   }
   ```

3. **データサイズを変えて測定**

   データサイズを変えながら、どこからGPUが速くなるか調べてみます：

   ```
   サイズ      CPU時間    GPU時間    高速化率
   ------------------------------------------------
   1,000       0.000010   0.001000   0.01x (遅い)
   10,000      0.000100   0.001100   0.09x (遅い)
   100,000     0.001000   0.001500   0.67x (遅い)
   1,000,000   0.010000   0.002000   5.00x (速い!)
   10,000,000  0.100000   0.010000   10.0x (速い!)
   ```

   **ポイント**: 小さいデータではGPUは遅いが、大きくなると圧倒的に速くなる！

4. **なぜ小さいデータではGPUが遅いのか**

   GPUには「オーバーヘッド」があります：
   - メモリ確保（`cudaMalloc`）
   - データ転送（`cudaMemcpy`）
   - カーネル起動のコスト

   小さいデータでは、このオーバーヘッドの方が大きいため、CPU版より遅くなります。

### よくある間違い

- **GPU時間に転送時間が含まれていない**: `cudaMemcpy` も含めて測定しないと不公平
- **初回実行が遅い**: GPUの初回起動は初期化があるため遅い → 2回目以降を測る
- **最適化オプションをつけていない**: `nvcc -O3` でコンパイルすると高速化される
- **cudaDeviceSynchronize()を忘れる**: カーネルは非同期なので、測定前に同期が必要

## 演習課題

### 課題1: 速度測定プログラムの実行
上記のサンプルコードを実行し、CPU版とGPU版の速度を比較してください。

### 課題2: データサイズを変えて測定
データサイズを 1,000 / 10,000 / 100,000 / 1,000,000 / 10,000,000 と変えながら、それぞれの実行時間を記録してください。

### 課題3: 結果を表にまとめる
測定結果を以下のような表にまとめてください：

```
データサイズ | CPU時間 | GPU時間 | 高速化率
------------|---------|---------|--------
1,000       |         |         |
10,000      |         |         |
...
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] CPU版とGPU版の実行時間を測定できる
- [ ] データサイズを変えながら複数回測定できる
- [ ] どのくらいのデータサイズからGPUが速くなるか確認できる
- [ ] 大きなデータ（100万要素以上）でGPUの方が速いことを確認できる

## ヒント集

### ヒント1（軽め）
時間測定には `clock()` 関数を使います。処理の前後で `clock()` を呼び、差を `CLOCKS_PER_SEC` で割ると秒数が得られます。

### ヒント2（中程度）
GPU版の時間測定では、以下を含めて測定してください：
```cuda
clock_t start = clock();
cudaMalloc(...);
cudaMemcpy(..., ..., cudaMemcpyHostToDevice);
kernel<<<...>>>(...);
cudaMemcpy(..., ..., cudaMemcpyDeviceToHost);
cudaFree(...);
clock_t end = clock();
```

### ヒント3（具体的）
もっと正確に測定したい場合は、CUDAのイベントAPIを使います：
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// GPU処理
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU時間: %.6f 秒\n", milliseconds / 1000.0);
```

