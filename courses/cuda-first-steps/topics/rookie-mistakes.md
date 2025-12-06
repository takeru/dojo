# トピック: みんなここでハマる！CUDA初心者あるある集

## メタ情報

- **ID**: rookie-mistakes
- **難易度**: 初級
- **所要時間**: 5-8分
- **カテゴリ**: Tips・トラブルシューティング

## 前提知識

- Stage 1-2完了

## このトピックで学べること

- よくあるミスとその対処法
- ハマりポイントを事前に知る
- デバッグの時短テクニック

## 関連ステージ

- すべてのステージ

## 要点（ドキュメント形式用）

CUDA初心者が必ずハマる典型的なミスを紹介します。

### ミス1: cudaMemcpyの方向を間違える

**症状**: 結果が0のままor ゴミデータ

```cuda
// ❌ 間違い
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyDeviceToHost);  // 逆！

// ✅ 正しい
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);  // CPU → GPU
```

**覚え方**: 第1引数が「宛先」、第2引数が「元」

### ミス2: カーネル実行後にcudaMemcpyを忘れる

**症状**: GPU で計算したのに結果が反映されない

```cuda
// ❌ 間違い
kernel<<<...>>>(d_arr, n);
printf("%d\n", h_arr[0]);  // GPU の結果を見てない！

// ✅ 正しい
kernel<<<...>>>(d_arr, n);
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
printf("%d\n", h_arr[0]);
```

### ミス3: ブロックサイズが1024を超える

**症状**: `invalid configuration argument`

```cuda
// ❌ 間違い
kernel<<<1, 2048>>>(arr, n);  // 1024超え

// ✅ 正しい
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(arr, n);
```

### ミス4: 配列外アクセス

**症状**: ランダムな値、クラッシュ

```cuda
// ❌ 間違い
__global__ void kernel(int *arr, int n) {
    int idx = threadIdx.x;
    arr[idx] = idx * 2;  // n より大きいidxがある！
}

// ✅ 正しい
__global__ void kernel(int *arr, int n) {
    int idx = threadIdx.x;
    if (idx < n) {  // 境界チェック
        arr[idx] = idx * 2;
    }
}
```

### ミス5: cudaFreeを忘れる

**症状**: メモリリーク、徐々にGPUメモリ不足

```cuda
// ❌ 間違い
cudaMalloc(&d_arr, size);
// ... 使う ...
// cudaFreeを忘れる！

// ✅ 正しい
cudaMalloc(&d_arr, size);
// ... 使う ...
cudaFree(d_arr);  // 必ず解放
```

### ミス6: blockDim.x を忘れる

**症状**: スレッド0しか動かない、結果がおかしい

```cuda
// ❌ 間違い
int idx = threadIdx.x + blockIdx.x;  // blockDim.x 忘れ

// ✅ 正しい
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

### ミス7: ファイル拡張子を .c にする

**症状**: `identifier "threadIdx" is undefined`

```bash
# ❌ 間違い
nvcc program.c -o program  // .c だとC言語として処理

# ✅ 正しい
nvcc program.cu -o program  // .cu が必須
```

### ミス8: エラーチェックをしない

**症状**: 何も起きない、結果が0

```cuda
// ❌ 間違い
cudaMalloc(&d_arr, size);  // エラーを無視
kernel<<<...>>>(d_arr, n);

// ✅ 正しい
cudaError_t err = cudaMalloc(&d_arr, size);
if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「これからよくあるミスを紹介するから、覚えておくと時短になるよ」

### 説明の流れ

1. 各ミスの症状と原因を説明
2. 実際にミスを再現して見せる（オプション）
3. 正しい書き方を示す

### 実践課題（オプション）

1. 意図的にミスを入れて、エラーメッセージを確認
2. デバッグして修正

## クリア条件（オプション）

- [ ] よくあるミス8個を知っている
- [ ] cudaMemcpyの方向を正しく使える
- [ ] 配列外アクセスを防げる

## 補足情報

### デバッグチェックリスト

エラーが出たら以下を確認：

- [ ] ファイルは `.cu` になっている？
- [ ] cudaMemcpyの方向は正しい？
- [ ] if (idx < n) の境界チェックはある？
- [ ] cudaGetLastError() でエラーチェックした？
- [ ] cudaDeviceSynchronize() の後にエラーチェック？
- [ ] cudaFree を忘れていない？
- [ ] ブロックサイズは1024以下？

### 参考リンク

- [CUDA C++ Programming Guide - Common Mistakes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
