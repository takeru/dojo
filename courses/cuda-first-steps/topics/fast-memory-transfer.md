# トピック: データ転送を爆速にする裏ワザ（Pinned Memory）

## メタ情報

- **ID**: fast-memory-transfer
- **難易度**: 中級
- **所要時間**: 8-10分
- **カテゴリ**: メモリ管理

## 前提知識

- Stage 5完了

## このトピックで学べること

- Pinned Memoryの仕組み
- 転送速度を2倍にする方法

## 関連ステージ

- Stage 5: メモリ転送

## 要点（ドキュメント形式用）

通常のmallocではなく、cudaHostAllocを使うと転送が2倍速くなります。

### 通常のメモリ（Pageable Memory）

```cuda
int *h_arr = (int*)malloc(size);  // 遅い
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
```

### Pinned Memory（Page-Locked Memory）

```cuda
int *h_arr;
cudaHostAlloc(&h_arr, size, cudaHostAllocDefault);  // 速い！
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
cudaFreeHost(h_arr);
```

### 速度比較（典型的な例）

```
Pageable Memory: 6 GB/秒
Pinned Memory:   12 GB/秒
```

約2倍速い！

### なぜ速いのか

通常のメモリは仮想メモリ（ページング可能）なので、転送前にページロックが必要。Pinned Memoryは最初から物理メモリに固定されているため、すぐに転送できる。

### 注意点

- Pinned Memoryは物理メモリを専有するので、使いすぎるとシステムが遅くなる
- 少量のデータには効果が薄い

## 対話形式の教え方ガイド（先生用）

### 導入

「メモリ転送が遅いって言ってたけど、実は2倍速くする方法があるんだ」

### 実践課題

1. 同じデータをmallocとcudaHostAllocで転送して速度比較

## クリア条件

- [ ] cudaHostAllocを使える
- [ ] Pinned Memoryのメリット・デメリットを説明できる

## 補足情報

### 参考リンク

- [Pinned Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
