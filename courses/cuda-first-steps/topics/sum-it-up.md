# トピック: 全要素の合計を超高速で計算する（Reductionパターン）

## メタ情報

- **ID**: sum-it-up
- **難易度**: 中級
- **所要時間**: 12-15分
- **カテゴリ**: アルゴリズム

## 前提知識

- Stage 4,7完了
- シェアードメモリの基礎

## このトピックで学べること

- Reductionパターンの実装
- 並列アルゴリズムの考え方
- 木構造での集約

## 関連ステージ

- Stage 4: ベクトル加算
- Stage 7: 行列積

## 要点（ドキュメント形式用）

配列の全要素の合計を並列で計算するパターンをReductionと呼びます。

### ナイーブな実装（遅い）

```cuda
__global__ void sumNaive(int *arr, int *result, int n) {
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += arr[i];  // 1スレッドで全部やる（遅い）
        }
        *result = sum;
    }
}
```

### 並列Reduction（速い）

```cuda
__global__ void sumReduction(int *input, int *output, int n) {
    __shared__ int shared[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // グローバル → シェアード
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // 木構造で集約
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // ブロックの結果
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}
```

### 実行イメージ（8スレッドの場合）

```
初期: [1, 2, 3, 4, 5, 6, 7, 8]

stride=4:
[1+5, 2+6, 3+7, 4+8, 5, 6, 7, 8]
= [6, 8, 10, 12, _, _, _, _]

stride=2:
[6+10, 8+12, _, _, _, _, _, _]
= [16, 20, _, _, _, _, _, _]

stride=1:
[16+20, _, _, _, _, _, _, _]
= [36, _, _, _, _, _, _, _]

結果: 36
```

## 対話形式の教え方ガイド（先生用）

### 導入

「配列の合計を全スレッドで並列に計算してみよう」

### 実践課題

1. 100万要素の合計をCPU版とGPU版で速度比較
2. ブロック数を変えて性能測定

## クリア条件

- [ ] Reductionパターンを実装できる
- [ ] シェアードメモリを使った集約ができる

## 補足情報

### 参考リンク

- [Reduction Optimization](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
