# トピック: cudaMemcpyの方向

## メタ情報

- **ID**: memcpy-direction
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: メモリ管理

## 前提知識

- Stage 2完了

## このトピックで学べること

- `cudaMemcpyHostToDevice` と `cudaMemcpyDeviceToHost` の違い
- 転送方向を間違えた時の症状
- 正しい転送パターン

## 関連ステージ

- Stage 2: CPUとGPUで同じことをやってみる

## 要点（ドキュメント形式用）

データの転送方向を正しく指定する必要があります。

### 転送方向

- **`cudaMemcpyHostToDevice`**: CPU → GPU（カーネル実行前）
  ```cuda
  cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
  ```

- **`cudaMemcpyDeviceToHost`**: GPU → CPU（カーネル実行後）
  ```cuda
  cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
  ```

### 引数の順序

```cuda
cudaMemcpy(dst, src, size, direction);
//         ↑    ↑
//       宛先  元
```

覚え方: 「dst = src」の順

### よくある間違い

```cuda
// 間違い: CPU → GPU なのに DeviceToHost
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyDeviceToHost);  // ✗

// 正しい
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);  // ✓
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`cudaMemcpyHostToDevice` と `cudaMemcpyDeviceToHost` って何が違うの？」

### 説明の流れ

1. **Host = CPU, Device = GPU を確認**
2. **矢印で覚える**
   - HostToDevice: CPU → GPU
   - DeviceToHost: GPU → CPU

3. **間違えた時の症状**
   - データが0のまま
   - ゴミデータが入る

## クリア条件（オプション）

- [ ] 転送方向の違いを説明できる
- [ ] 正しい方向を選べる

## 補足情報

### その他の転送方向

```cuda
cudaMemcpyDeviceToDevice  // GPU → GPU（GPU内コピー）
cudaMemcpyHostToHost      // CPU → CPU（通常は使わない）
```

### 参考リンク

- [cudaMemcpy Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
