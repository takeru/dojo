# トピック: 入出力の転送を最小限にする

## メタ情報

- **ID**: minimize-transfer
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: パフォーマンス

## 前提知識

- Stage 4完了

## このトピックで学べること

- 転送を減らす基本原則
- 入力と出力の区別
- 無駄な転送の見つけ方

## 関連ステージ

- Stage 4: ベクトル加算で基礎固め
- Stage 5: メモリ転送のコストを知る

## 要点（ドキュメント形式用）

転送は時間がかかるので、**必要最小限**にします。

### 基本原則

- **入力データ**: CPU → GPU のみ
- **出力データ**: GPU → CPU のみ

### 良い例

```cuda
// 入力だけ転送
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

kernel<<<...>>>(d_a, d_b, d_c, n);

// 出力だけ転送
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

### 悪い例（無駄な転送）

```cuda
// 出力配列を転送（不要！）
cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);  // ✗

kernel<<<...>>>(d_a, d_b, d_c, n);

// 入力配列を転送（不要！）
cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);  // ✗
cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);  // ✗
```

### 判断基準

| 配列 | 役割 | CPU→GPU | GPU→CPU |
|------|------|---------|---------|
| a, b | 入力 | ○ 必要 | ✗ 不要 |
| c    | 出力 | ✗ 不要 | ○ 必要 |

## 対話形式の教え方ガイド（先生用）

### 導入

「全部の配列を往復転送しなくていいの？」

### 説明の流れ

1. **入力と出力を区別**
   - 入力: 最初からデータがあるもの
   - 出力: 計算結果を書き込むもの

2. **転送の方向を決める**
   - 入力は CPU → GPU
   - 出力は GPU → CPU

3. **具体例で確認**
   ベクトル加算 `c = a + b` では
   - a, b: 入力
   - c: 出力

## クリア条件（オプション）

- [ ] 入力と出力の区別ができる
- [ ] 必要な転送だけを選べる

## 補足情報

### 中間結果の扱い

連続で複数のカーネルを実行する場合、中間結果はGPU上に残します：

```cuda
kernel1<<<...>>>(d_a, d_intermediate, n);  // 中間結果
kernel2<<<...>>>(d_intermediate, d_c, n);  // 最終結果

// 中間結果は転送不要！最終結果だけ転送
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

### 参考リンク

- [CUDA C++ Best Practices Guide - Data Transfer](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
