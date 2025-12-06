# トピック: gridSizeの切り上げ計算

## メタ情報

- **ID**: gridsize-ceiling
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: 実装テクニック

## 前提知識

- Stage 4完了

## このトピックで学べること

- 切り上げ除算のテクニック
- gridSize計算が必要な理由
- 余分なスレッドの処理方法

## 関連ステージ

- Stage 4: ベクトル加算で基礎固め

## 要点（ドキュメント形式用）

`(n + blockSize - 1) / blockSize` は**切り上げ除算**のテクニックです。

### なぜ切り上げが必要か

普通の除算は切り捨てになります：

```c
int gridSize = n / blockSize;  // 切り捨て
```

例: n=1000, blockSize=256
- `1000 / 256 = 3`（切り捨て）
- 合計スレッド = 3 × 256 = 768（< 1000 なので足りない！）

### 切り上げ除算

```c
int gridSize = (n + blockSize - 1) / blockSize;
```

例: n=1000, blockSize=256
- `(1000 + 255) / 256 = 1255 / 256 = 4`
- 合計スレッド = 4 × 256 = 1024（> 1000 なのでOK）

### 余分なスレッドの処理

余分な24スレッド（1024 - 1000）は `if (idx < n)` で弾かれます：

```cuda
__global__ void kernel(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {  // 範囲外のスレッドは何もしない
        arr[idx] = arr[idx] * 2;
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`(n + blockSize - 1) / blockSize` って何やってるの？」

### 説明の流れ

1. **普通の除算の問題点**
   - 切り捨てになる
   - スレッドが足りなくなる

2. **切り上げ除算の仕組み**
   - `n + blockSize - 1` で調整してから割る
   - 具体例で計算してみる

3. **実際のコードでの使い方**
   - gridSizeを計算
   - カーネル内で範囲チェック

## クリア条件（オプション）

- [ ] 切り上げ除算の計算ができる
- [ ] なぜ切り上げが必要か説明できる

## 補足情報

### 別の書き方

C++11以降では `ceil` 関数も使えます：

```cpp
#include <cmath>
int gridSize = (int)ceil((double)n / blockSize);
```

ただし、整数演算の方が速いので `(n + blockSize - 1) / blockSize` が好まれます。

### 参考リンク

- [CUDA C++ Programming Guide - Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)
