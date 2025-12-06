# Stage 1: 環境構築と Hello GPU

## 目標

このステージを完了すると、生徒は：
- CUDA Toolkitをインストールし、開発環境を構築できる
- nvccコンパイラを使ってCUDAプログラムをコンパイル・実行できる
- 最初のGPUプログラムを動かし、GPUが動作していることを確認できる

## 前提知識

- C/C++の基本的な構文（変数、関数、ポインタの基礎）
- ターミナル（コマンドライン）の基本操作
- CUDAの知識は一切不要

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **nvcc-basics** - nvccコンパイラの仕組みと裏側
- **cuda-toolkit-tour** - CUDA Toolkitに何が入っているか探検
- **debugging-cuda** - エラーが出たときの対処法

## 教え方ガイド

### 導入（なぜこれを学ぶか）

GPUを使ったプログラミングは、CPUだけでは時間がかかりすぎる計算を劇的に高速化できる技術です。ディープラーニング、画像処理、科学計算など、様々な分野で使われています。CUDAはNVIDIA製GPUでプログラムを実行するためのプラットフォームで、初めての人でも意外と簡単に始められます。

このステージでは、まず環境を整えて、GPUに「Hello」させてみましょう。

### 説明の流れ

1. **CUDA Toolkitのインストール**
   - CUDA Toolkitは、GPUプログラミングに必要なコンパイラやライブラリのセットです
   - 公式サイト: https://developer.nvidia.com/cuda-downloads
   - OSに応じたインストーラーをダウンロード（Windows/Linux/macOS）

   **注意**: macOSの場合、CUDAは古いバージョンしかサポートされていません。最新のMacではCUDAは動きません。

2. **インストール確認**
   ```bash
   nvcc --version
   ```
   CUDA Compiler Driverのバージョンが表示されればOK

3. **GPUの確認**
   まず、使えるGPUがあるか確認します：
   ```bash
   nvidia-smi
   ```
   GPU名、ドライバーバージョン、メモリ容量などが表示されます。

4. **最初のCUDAプログラム（配列の初期化）**

   シンプルな例：GPUで配列を初期化してみます。

   ```cuda
   #include <stdio.h>
   #include <cuda_runtime.h>

   // GPU上で実行される関数（カーネル）
   __global__ void initArray(int *arr, int n) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < n) {
           arr[idx] = idx;  // 配列の各要素にインデックスを代入
       }
   }

   int main() {
       int n = 10;
       int *d_arr;  // GPU上の配列（dはdeviceの意味）
       int h_arr[10];  // CPU上の配列（hはhostの意味）

       // GPU上にメモリを確保
       cudaMalloc((void**)&d_arr, n * sizeof(int));

       // カーネルを実行（1ブロック、10スレッド）
       initArray<<<1, n>>>(d_arr, n);

       // GPUからCPUへデータをコピー
       cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

       // 結果を表示
       printf("配列の内容: ");
       for (int i = 0; i < n; i++) {
           printf("%d ", h_arr[i]);
       }
       printf("\n");

       // GPUメモリを解放
       cudaFree(d_arr);

       return 0;
   }
   ```

5. **コンパイルと実行**
   ```bash
   nvcc hello_gpu.cu -o hello_gpu
   ./hello_gpu
   ```

   出力例：
   ```
   配列の内容: 0 1 2 3 4 5 6 7 8 9
   ```

6. **コードの重要ポイント**
   - `__global__`: GPU上で実行される関数（カーネル）の印
   - `<<<1, n>>>`: カーネル起動の構文（1ブロック、nスレッド）
   - `cudaMalloc`: GPU上にメモリ確保
   - `cudaMemcpy`: CPU⇔GPU間でデータ転送
   - `cudaFree`: GPUメモリを解放

### よくある間違い

- **nvccが見つからない**: PATHが通っていない → 環境変数を確認
- **nvidia-smiでGPUが見えない**: ドライバーが正しくインストールされていない
- **cudaMallocでエラー**: GPUメモリ不足、または他のプログラムがGPUを使用中
- **実行結果が0だらけ**: `cudaMemcpy`の引数の順番が逆（src と dst を間違えている）
- **カーネル起動後にエラー**: `cudaDeviceSynchronize()`を入れてエラーを確認

## 演習課題

### 課題1: CUDA Toolkitのインストール
CUDA Toolkitをインストールし、`nvcc --version` と `nvidia-smi` が動作することを確認してください。

### 課題2: Hello GPUプログラムの実行
上記のサンプルコードを実行し、配列の内容が正しく表示されることを確認してください。

### 課題3: 配列サイズを変更してみる
配列のサイズを10から100に変更し、同じように動作するか試してみてください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `nvcc --version` がCUDAコンパイラのバージョンを表示する
- [ ] `nvidia-smi` がGPU情報を表示する
- [ ] Hello GPUプログラムが正しくコンパイル・実行できる
- [ ] 配列の内容が `0 1 2 3 4 5 6 7 8 9` と表示される

## ヒント集

### ヒント1（軽め）
まずNVIDIAの公式サイトからCUDA Toolkitをダウンロードしましょう。自分のOSに合ったインストーラーを選んでください。

### ヒント2（中程度）
インストール後、以下のコマンドで確認してください：
```bash
nvcc --version
nvidia-smi
```
もし `command not found` と出たら、PATHに `/usr/local/cuda/bin` を追加してください。

Linuxの場合：
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### ヒント3（具体的）
サンプルコードを `hello_gpu.cu` というファイル名で保存し、以下のコマンドでコンパイル・実行します：

```bash
nvcc hello_gpu.cu -o hello_gpu
./hello_gpu
```

もしエラーが出た場合、エラーメッセージをよく読んでください。よくあるエラー：
- `error: identifier "threadIdx" is undefined` → ファイル名が `.cu` になっていない
- `cannot find -lcudart` → LD_LIBRARY_PATHが設定されていない

