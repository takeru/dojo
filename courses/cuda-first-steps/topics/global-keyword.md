# トピック: `__global__` って何？

## メタ情報

- **ID**: global-keyword
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: 言語機能

## 前提知識

- Stage 1完了

## このトピックで学べること

- `__global__`, `__device__`, `__host__` の違い
- 関数がどこで実行されるかの指定方法
- カーネル関数の呼び出し方

## 関連ステージ

- Stage 1: 環境構築と Hello GPU

## 要点（ドキュメント形式用）

CUDAでは、関数がどこで実行されるかを明示する必要があります：

### 修飾子の種類

| 修飾子 | 実行場所 | 呼び出し元 |
|--------|----------|-----------|
| `__global__` | GPU | CPU |
| `__device__` | GPU | GPU |
| `__host__` | CPU | CPU |

### 使用例

```cuda
// GPU上で実行、CPUから呼び出し（カーネル）
__global__ void myKernel(int *arr, int n) {
    // ...
}

// GPU上で実行、GPU内からのみ呼び出し
__device__ int helper(int x) {
    return x * 2;
}

// CPU上で実行（普通のC関数、__host__は省略可能）
void normalFunction() {
    // ...
}
```

### `__global__` 関数の特徴

- 戻り値は `void` のみ
- `<<<grid, block>>>` 構文で起動
- 非同期実行される

## 対話形式の教え方ガイド（先生用）

### 導入

「関数の前に `__global__` ってついてるの気づいた？これがCUDAの魔法のキーワードなんだ」

なぜこれを知っておくと便利か：
- GPUで実行される関数とCPUで実行される関数を区別できる
- ヘルパー関数を書くときに適切な修飾子を選べる

### 説明の流れ

1. **3つの修飾子を説明**
   - `__global__`: カーネル（CPUからGPUを起動）
   - `__device__`: GPUヘルパー関数
   - `__host__`: 普通のCPU関数

2. **コード例で確認**
   ```cuda
   __global__ void kernel() {
       // GPUで実行
   }

   // CPUから呼び出し
   kernel<<<1, 10>>>();
   ```

### 実践課題（オプション）

1. `__device__` 関数を作成してカーネルから呼び出す

## クリア条件（オプション）

- [ ] 3つの修飾子の違いを説明できる
- [ ] `__global__` 関数の特徴を理解している

## 補足情報

### `__host__ __device__` の組み合わせ

両方につけると、CPUとGPU両方のコードが生成されます：

```cuda
__host__ __device__ int add(int a, int b) {
    return a + b;
}
// CPU/GPUどちらからでも呼べる
```

### 参考リンク

- [CUDA C++ Programming Guide - Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers)
