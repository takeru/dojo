# Stage 2: CPUとGPUで同じことをやってみる

## 目標

このステージを完了すると、生徒は：
- CPU版とGPU版で同じ処理を実装できる
- カーネル関数の基本的な書き方を理解できる
- 両方の実装結果が一致することを確認できる

## 前提知識

- Stage 1完了（CUDAの環境が整っている）
- `cudaMalloc`, `cudaMemcpy`, `cudaFree` の使い方
- `__global__` キーワードの意味

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **debugging-cuda** - GPU版の結果が合わない時の確認方法
- **rookie-mistakes** - よくあるミスを事前に知っておく

## 教え方ガイド

### 導入（なぜこれを学ぶか）

GPUプログラミングを始めるとき、いきなりGPUだけで書くとデバッグが大変です。まずはCPU版で正しい実装を書き、それと同じことをGPU版でやってみる、という流れが王道です。

このステージでは、**配列の各要素を2倍にする**という簡単な処理を、CPU版とGPU版の両方で実装してみます。

### 説明の流れ

1. **CPU版の実装（ベースライン）**

   まず普通のC言語で書いてみます：

   ```c
   #include <stdio.h>

   void doubleArrayCPU(int *arr, int n) {
       for (int i = 0; i < n; i++) {
           arr[i] = arr[i] * 2;
       }
   }

   int main() {
       int n = 10;
       int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

       printf("元の配列: ");
       for (int i = 0; i < n; i++) printf("%d ", arr[i]);
       printf("\n");

       doubleArrayCPU(arr, n);

       printf("2倍にした配列: ");
       for (int i = 0; i < n; i++) printf("%d ", arr[i]);
       printf("\n");

       return 0;
   }
   ```

   出力例：
   ```
   元の配列: 1 2 3 4 5 6 7 8 9 10
   2倍にした配列: 2 4 6 8 10 12 14 16 18 20
   ```

2. **GPU版の実装**

   同じ処理をGPUでやってみます：

   ```cuda
   #include <stdio.h>
   #include <cuda_runtime.h>

   // GPU版：各スレッドが1要素を担当
   __global__ void doubleArrayGPU(int *arr, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           arr[idx] = arr[idx] * 2;
       }
   }

   int main() {
       int n = 10;
       int h_arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
       int *d_arr;

       printf("元の配列: ");
       for (int i = 0; i < n; i++) printf("%d ", h_arr[i]);
       printf("\n");

       // GPUメモリ確保
       cudaMalloc((void**)&d_arr, n * sizeof(int));

       // CPU → GPU にデータをコピー
       cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

       // カーネル実行（1ブロック、10スレッド）
       doubleArrayGPU<<<1, n>>>(d_arr, n);

       // GPU → CPU にデータをコピー
       cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

       printf("2倍にした配列: ");
       for (int i = 0; i < n; i++) printf("%d ", h_arr[i]);
       printf("\n");

       // メモリ解放
       cudaFree(d_arr);

       return 0;
   }
   ```

3. **CPUとGPU両方を持つバージョン**

   両方を1つのプログラムにまとめて、結果を比較してみます：

   ```cuda
   #include <stdio.h>
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
       int n = 10;
       int h_arr_cpu[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
       int h_arr_gpu[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
       int *d_arr;

       // CPU版実行
       doubleArrayCPU(h_arr_cpu, n);

       // GPU版実行
       cudaMalloc((void**)&d_arr, n * sizeof(int));
       cudaMemcpy(d_arr, h_arr_gpu, n * sizeof(int), cudaMemcpyHostToDevice);
       doubleArrayGPU<<<1, n>>>(d_arr, n);
       cudaMemcpy(h_arr_gpu, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
       cudaFree(d_arr);

       // 結果を比較
       printf("CPU版: ");
       for (int i = 0; i < n; i++) printf("%d ", h_arr_cpu[i]);
       printf("\n");

       printf("GPU版: ");
       for (int i = 0; i < n; i++) printf("%d ", h_arr_gpu[i]);
       printf("\n");

       // 一致チェック
       int match = 1;
       for (int i = 0; i < n; i++) {
           if (h_arr_cpu[i] != h_arr_gpu[i]) {
               match = 0;
               break;
           }
       }

       if (match) {
           printf("✓ CPU版とGPU版の結果が一致しました！\n");
       } else {
           printf("✗ 結果が一致しません\n");
       }

       return 0;
   }
   ```

4. **重要ポイント**
   - GPU版では、forループの代わりに**各スレッドが1要素を担当**する
   - `idx = threadIdx.x + blockIdx.x * blockDim.x` で自分の担当インデックスを計算
   - `if (idx < n)` で配列外アクセスを防ぐ（スレッド数 > 配列サイズの場合）

### よくある間違い

- **結果が0になる**: `cudaMemcpy` の方向が逆（`cudaMemcpyDeviceToHost` と `cudaMemcpyHostToDevice` を間違えている）
- **一部の要素だけ更新される**: スレッド数が足りない（`<<<1, n>>>` のnが配列サイズより小さい）
- **結果が変わらない**: カーネル実行後に `cudaMemcpy` を忘れている
- **ランダムな値が出る**: GPU版で配列を初期化していない（CPU → GPU のコピーを忘れている）

## 演習課題

### 課題1: CPU版とGPU版を実装
上記のサンプルコードを実行し、CPU版とGPU版で同じ結果が得られることを確認してください。

### 課題2: 3倍にする処理に変更
`* 2` を `* 3` に変更し、配列の各要素を3倍にするプログラムに変更してください。

### 課題3: 配列サイズを100に増やす
配列サイズを10から100に増やし、同じように動作することを確認してください。ただし、ブロックサイズは最大1024なので、1ブロックで対応できます。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] CPU版のプログラムが正しく実行できる
- [ ] GPU版のプログラムが正しく実行できる
- [ ] CPU版とGPU版の結果が一致する
- [ ] 配列の各要素が2倍になっている（例: `[2, 4, 6, 8, 10, ...]`）

## ヒント集

### ヒント1（軽め）
まずCPU版だけを書いてみて、正しく動くことを確認しましょう。その後GPU版に挑戦するとスムーズです。

### ヒント2（中程度）
GPU版では、forループを書く代わりに、各スレッドが自分のインデックスを計算します：
```cuda
int idx = threadIdx.x;  // 0, 1, 2, 3, ... と各スレッドで異なる値
```

### ヒント3（具体的）
GPU版の基本パターン：
```cuda
// 1. GPU メモリ確保
cudaMalloc((void**)&d_arr, n * sizeof(int));

// 2. CPU → GPU コピー
cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

// 3. カーネル実行
myKernel<<<1, n>>>(d_arr, n);

// 4. GPU → CPU コピー
cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

// 5. メモリ解放
cudaFree(d_arr);
```

