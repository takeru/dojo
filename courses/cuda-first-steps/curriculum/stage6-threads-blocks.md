# Stage 6: スレッドとブロックを理解する

## 目標

このステージを完了すると、生徒は：
- `threadIdx`, `blockIdx`, `blockDim` の意味を理解できる
- 複数ブロックを使ったカーネルを実装できる
- 2次元グリッドを使った処理を書ける

## 前提知識

- Stage 5完了（メモリ転送の仕組みを理解している）
- 1次元配列の処理ができる
- `<<<gridSize, blockSize>>>` の基本的な使い方

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **how-gpu-runs** - GPUが何千ものスレッドを動かす仕組み
- **if-statement-trap** - GPU でif文を使うと遅くなる理由
- **memory-playground** - シェアードメモリを使った高速化

## 教え方ガイド

### 導入（なぜこれを学ぶか）

これまで `<<<1, n>>>` のように「1ブロック、nスレッド」で書いてきましたが、これには限界があります。GPUは1ブロックあたり最大1024スレッドまでしか起動できません。

もっと大きなデータを処理するには、**複数のブロック**を使う必要があります。このステージでは、スレッドとブロックの階層構造を理解し、大規模な並列処理を書けるようになります。

### 説明の流れ

1. **GPUのスレッド階層**

   GPUのスレッドは階層構造になっています：

   ```
   Grid（グリッド）
     └── Block（ブロック）複数個
           └── Thread（スレッド）複数個
   ```

   例: `<<<4, 256>>>` の場合
   - グリッドサイズ = 4ブロック
   - ブロックサイズ = 256スレッド
   - 合計スレッド数 = 4 × 256 = 1024スレッド

2. **組み込み変数**

   カーネル内で使える変数：

   | 変数 | 意味 | 例（`<<<4, 256>>>` の場合） |
   |------|------|---------------------------|
   | `threadIdx.x` | ブロック内のスレッド番号 | 0〜255 |
   | `blockIdx.x` | グリッド内のブロック番号 | 0〜3 |
   | `blockDim.x` | ブロックのサイズ | 256 |
   | `gridDim.x` | グリッドのサイズ | 4 |

3. **グローバルインデックスの計算**

   複数ブロックを使う場合、各スレッドの「グローバルなインデックス」を計算します：

   ```cuda
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   ```

   **例**:
   ```
   ブロック0: idx = 0, 1, 2, ..., 255
   ブロック1: idx = 256, 257, ..., 511
   ブロック2: idx = 512, 513, ..., 767
   ブロック3: idx = 768, 769, ..., 1023
   ```

4. **複数ブロックを使った例**

   ```cuda
   #include <stdio.h>
   #include <cuda_runtime.h>

   __global__ void printThreadInfo(int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           printf("idx=%d: blockIdx=%d, threadIdx=%d\n",
                  idx, blockIdx.x, threadIdx.x);
       }
   }

   int main() {
       int n = 20;
       int blockSize = 8;
       int gridSize = (n + blockSize - 1) / blockSize;  // = 3ブロック

       printf("グリッドサイズ: %d, ブロックサイズ: %d\n", gridSize, blockSize);
       printThreadInfo<<<gridSize, blockSize>>>(n);
       cudaDeviceSynchronize();

       return 0;
   }
   ```

   出力例：
   ```
   グリッドサイズ: 3, ブロックサイズ: 8
   idx=0: blockIdx=0, threadIdx=0
   idx=1: blockIdx=0, threadIdx=1
   ...
   idx=7: blockIdx=0, threadIdx=7
   idx=8: blockIdx=1, threadIdx=0
   ...
   idx=19: blockIdx=2, threadIdx=3
   ```

5. **2次元グリッドとブロック**

   画像処理などでは、2次元のグリッド/ブロックが便利です：

   ```cuda
   #include <stdio.h>
   #include <cuda_runtime.h>

   __global__ void process2D(int *arr, int width, int height) {
       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;

       if (x < width && y < height) {
           int idx = y * width + x;  // 2次元→1次元変換
           arr[idx] = x + y;
       }
   }

   int main() {
       int width = 16;
       int height = 16;
       int n = width * height;

       int *h_arr = (int*)malloc(n * sizeof(int));
       int *d_arr;
       cudaMalloc((void**)&d_arr, n * sizeof(int));

       // 2次元のブロックとグリッドを定義
       dim3 blockSize(8, 8);  // 8×8 = 64スレッド/ブロック
       dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);

       printf("gridSize: (%d, %d), blockSize: (%d, %d)\n",
              gridSize.x, gridSize.y, blockSize.x, blockSize.y);

       process2D<<<gridSize, blockSize>>>(d_arr, width, height);

       cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

       // 結果を表示（一部）
       printf("最初の4x4:\n");
       for (int y = 0; y < 4; y++) {
           for (int x = 0; x < 4; x++) {
               printf("%3d ", h_arr[y * width + x]);
           }
           printf("\n");
       }

       free(h_arr);
       cudaFree(d_arr);
       return 0;
   }
   ```

6. **重要ポイント**
   - ブロックサイズの上限は1024スレッド
   - 2次元の場合も、8×8=64, 16×16=256など、積が1024以下
   - gridSizeは制限がほぼないので、大きなデータでも対応可能

### よくある間違い

- **ブロックサイズが1024を超える**: `<<<1, 2048>>>` はエラー
- **2次元で積が1024を超える**: `dim3(32, 32)` = 1024は超えてないが、`(32, 33)` はエラー
- **インデックス計算のミス**: `threadIdx.x + blockIdx.x` と書いてしまう（`* blockDim.x` を忘れる）
- **境界チェック忘れ**: 2次元では `if (x < width && y < height)` が必要

## 演習課題

### 課題1: 複数ブロックの動作確認
上記の `printThreadInfo` プログラムを実行し、スレッドとブロックの関係を確認してください。

### 課題2: 100万要素の配列処理
100万要素の配列を、256スレッド/ブロックで処理するプログラムを書いてください。

### 課題3: 2次元配列の処理
2次元の配列（例: 32×32）を、8×8のブロックで処理するプログラムを書いてください。

### 課題4（発展）: 3次元グリッド
`dim3` は3次元も対応しています。3次元配列の処理を試してみてください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `threadIdx.x`, `blockIdx.x`, `blockDim.x` の意味を説明できる
- [ ] グローバルインデックス `idx = threadIdx.x + blockIdx.x * blockDim.x` を理解している
- [ ] 複数ブロックを使ったプログラムが実行できる
- [ ] 2次元グリッドを使ったプログラムが実行できる

## ヒント集

### ヒント1（軽め）
まずは1次元で複数ブロックを使ってみましょう。`<<<4, 256>>>` のように、グリッドサイズを1より大きくしてみてください。

### ヒント2（中程度）
グローバルインデックスの計算式を覚えましょう：
```cuda
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

これで、すべてのスレッドが一意なインデックスを持ちます。

### ヒント3（具体的）
2次元の場合：
```cuda
dim3 blockSize(8, 8);
dim3 gridSize((width + 7) / 8, (height + 7) / 8);
process<<<gridSize, blockSize>>>(...);

// カーネル内：
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int idx = y * width + x;  // 1次元インデックスに変換
```

## 補足・発展トピック

ステージクリア後、生徒が「もっと知りたい」を選んだら、`/dojo:topic` コマンドで以下のトピックを選択できます：

- **how-gpu-runs** - スレッド・ブロック・Warpの詳細
- **if-statement-trap** - 分岐がパフォーマンスに与える影響
- **memory-playground** - シェアードメモリの使い方

### 参考リンク

- [CUDA C++ Programming Guide - Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- [dim3 Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)
