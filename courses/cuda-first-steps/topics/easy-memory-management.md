# トピック: 面倒なメモリ管理をサボる方法（Unified Memory）

## メタ情報

- **ID**: easy-memory-management
- **難易度**: 初級
- **所要時間**: 5-8分
- **カテゴリ**: メモリ管理

## 前提知識

- Stage 5完了（cudaMalloc, cudaMemcpyを使ったことがある）

## このトピックで学べること

- Unified Memoryの仕組み
- cudaMallocManagedの使い方
- どんな時に便利か

## 関連ステージ

- Stage 5: メモリ転送

## 要点（ドキュメント形式用）

Unified Memoryを使うと、`cudaMemcpy`を書かずにCPU⇔GPU間のデータ転送を自動化できます。

### 従来の方法（手動管理）

```cuda
int *h_arr = (int*)malloc(size);
int *d_arr;
cudaMalloc(&d_arr, size);
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
kernel<<<...>>>(d_arr, n);
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
cudaFree(d_arr);
free(h_arr);
```

### Unified Memory（自動管理）

```cuda
int *arr;
cudaMallocManaged(&arr, size);  // CPU/GPU両方からアクセス可能

// CPUで初期化
for (int i = 0; i < n; i++) arr[i] = i;

// GPUで計算（自動転送）
kernel<<<...>>>(arr, n);
cudaDeviceSynchronize();

// CPUで結果を使う（自動転送）
printf("%d\n", arr[0]);

cudaFree(arr);
```

### メリット・デメリット

**メリット**:
- コードがシンプル
- cudaMemcpyを書かなくて良い
- デバッグしやすい

**デメリット**:
- 転送タイミングが自動なので、予想外のオーバーヘッド
- 明示的管理より遅い場合がある

### 使い所

- プロトタイピング（まずは動かす）
- データサイズが小さい場合
- 転送回数が少ない場合

## 対話形式の教え方ガイド（先生用）

### 導入

「毎回 cudaMalloc, cudaMemcpy を書くの面倒だよね。実はもっと楽な方法があるんだ」

### 実践課題

1. 既存のプログラムをUnified Memoryに書き換え
2. 速度を比較

## クリア条件

- [ ] cudaMallocManagedを使える
- [ ] 従来方法との違いを説明できる

## 補足情報

### 参考リンク

- [Unified Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
