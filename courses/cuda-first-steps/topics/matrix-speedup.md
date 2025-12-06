# トピック: 行列積をさらに10倍速くする（タイリング手法）

## メタ情報

- **ID**: matrix-speedup
- **難易度**: 上級
- **所要時間**: 15-20分
- **カテゴリ**: 最適化

## 前提知識

- Stage 7完了
- シェアードメモリの理解

## このトピックで学べること

- タイリング手法による最適化
- シェアードメモリを使った高速化
- グローバルメモリアクセスの削減

## 関連ステージ

- Stage 7: 行列積

## 要点（ドキュメント形式用）

行列積をシェアードメモリを使ったタイリング手法で高速化します。

### 基本版の問題点

```cuda
__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
        // ↑ グローバルメモリアクセス（遅い）
    }
    C[row * N + col] = sum;
}
```

N回のグローバルメモリアクセス！

### タイリング版（シェアードメモリ使用）

```cuda
#define TILE_SIZE 16

__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // タイルごとに処理
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // グローバル → シェアード（1回だけ）
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // シェアードメモリで計算（高速）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

グローバルメモリアクセスが激減！

### 高速化の理由

- グローバルメモリアクセス: N回 → N/TILE_SIZE回
- シェアードメモリは100倍速い

### 実測例

```
1024×1024行列:
基本版:       0.150秒
タイリング版:  0.015秒
高速化率:      10倍
```

## 対話形式の教え方ガイド（先生用）

### 導入

「Stage 7の行列積、まだ10倍速くできるんだ」

### 実践課題

1. タイリング版を実装
2. 基本版と速度比較
3. TILE_SIZEを変えて最適値を探す

## クリア条件

- [ ] タイリング手法を実装できる
- [ ] シェアードメモリで10倍高速化を達成できる

## 補足情報

### 参考リンク

- [Matrix Multiplication with Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
