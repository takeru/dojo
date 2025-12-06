# トピック: GPUに複数の仕事を同時にさせる（Streams）

## メタ情報

- **ID**: multiple-tasks
- **難易度**: 中級
- **所要時間**: 12-15分
- **カテゴリ**: 並行実行

## 前提知識

- Stage 5完了

## このトピックで学べること

- CUDAストリームの仕組み
- 転送と計算の並行実行
- 複数カーネルの同時実行

## 関連ステージ

- Stage 5: メモリ転送

## 要点（ドキュメント形式用）

CUDAストリームを使うと、複数の処理を並行実行できます。

### 従来（直列実行）

```
転送1 → カーネル1 → 転送2 → カーネル2
```

### ストリーム使用（並列実行）

```
Stream 0: 転送1 → カーネル1
Stream 1:          転送2 → カーネル2
```

### ストリームの作成と使用

```cuda
// ストリーム作成
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// ストリーム指定で実行
cudaMemcpyAsync(d1, h1, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d1, n);

cudaMemcpyAsync(d2, h2, size, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream2>>>(d2, n);

// 同期
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// 破棄
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 転送と計算のオーバーラップ

```cuda
for (int i = 0; i < numChunks; i++) {
    cudaMemcpyAsync(..., stream[i]);
    kernel<<<..., stream[i]>>>(...);
    cudaMemcpyAsync(..., stream[i]);
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「今まで1つずつ順番に処理してたけど、実は同時にやれるんだ」

### 実践課題

1. ストリームあり・なしで速度比較
2. 2つのカーネルを並行実行

## クリア条件

- [ ] cudaStreamCreateを使える
- [ ] 非同期転送（cudaMemcpyAsync）を使える

## 補足情報

### 参考リンク

- [CUDA Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
