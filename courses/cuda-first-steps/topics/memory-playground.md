# トピック: GPUの記憶の仕組み：グローバルとシェアードメモリで遊んでみる

## メタ情報

- **ID**: memory-playground
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: メモリ管理

## 前提知識

- Stage 5,6完了
- カーネル関数とメモリ転送の基礎

## このトピックで学べること

- GPUメモリの種類（グローバル、シェアード、レジスタ）
- それぞれの速度と容量の違い
- シェアードメモリを使った高速化の基礎

## 関連ステージ

- Stage 5: メモリ転送
- Stage 6: スレッドとブロック

## 要点（ドキュメント形式用）

GPUには複数の種類のメモリがあり、それぞれ速度と容量が異なります。

### メモリの種類と特徴

| メモリ | 速度 | 容量 | スコープ | 用途 |
|--------|------|------|----------|------|
| レジスタ | 超高速 | 極小 | スレッド内 | ローカル変数 |
| シェアードメモリ | 高速 | 小（48KB程度） | ブロック内 | ブロック内共有 |
| グローバルメモリ | 遅い | 大（GB単位） | 全スレッド | 主記憶 |

### グローバルメモリ（これまで使ってきたもの）

```cuda
int *d_arr;
cudaMalloc(&d_arr, size);  // グローバルメモリに確保
```

- すべてのスレッドからアクセス可能
- 容量は大きいが遅い
- `cudaMalloc`で確保

### シェアードメモリ（ブロック内で共有）

```cuda
__global__ void kernel() {
    __shared__ int shared_arr[256];  // ブロック内で共有

    // ブロック内のすべてのスレッドが同じ配列を見る
    shared_arr[threadIdx.x] = threadIdx.x;
    __syncthreads();  // 同期が必要
}
```

- ブロック内のスレッド間で共有
- グローバルメモリの100倍速い
- 容量は小さい（48KB程度）

### シェアードメモリの使用例（リダクション）

```cuda
__global__ void sumReduction(int *input, int *output, int n) {
    __shared__ int shared[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // グローバル → シェアード にコピー
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // シェアードメモリ内でリダクション
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // ブロックの結果をグローバルに書き戻し
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「GPUのメモリって、実は何種類もあるんだ。速いけど小さいメモリと、遅いけど大きいメモリ。使い分けると劇的に速くなるよ」

### 説明の流れ

1. **メモリ階層の説明**

   「CPUにもキャッシュがあるように、GPUにも複数の記憶領域がある。近いメモリほど速い！」

2. **シェアードメモリを試す**

   簡単な例で、シェアードメモリの効果を体感

3. **同期の重要性**

   `__syncthreads()`の説明

### 実践課題（オプション）

1. シェアードメモリを使ったベクトル加算
2. `__syncthreads()`を外すとどうなるか試す

## クリア条件（オプション）

- [ ] グローバルメモリとシェアードメモリの違いを説明できる
- [ ] `__shared__`キーワードを使える
- [ ] `__syncthreads()`の必要性を理解している

## 補足情報

### 参考リンク

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
