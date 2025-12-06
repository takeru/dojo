# トピック: 転送を減らす方法

## メタ情報

- **ID**: reduce-transfer-strategies
- **難易度**: 初級〜中級
- **所要時間**: 5-10分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: パフォーマンス

## 前提知識

- Stage 5完了

## このトピックで学べること

- 転送を減らす5つの戦略
- 各戦略の使いどころ
- 次に学ぶべきトピック

## 関連ステージ

- Stage 5: メモリ転送のコストを知る

## 要点（ドキュメント形式用）

転送がボトルネックなら、転送を減らすのが王道です。

### 戦略1: 必要なデータだけ転送

入力と出力を区別し、必要最小限に：

```cuda
// 入力だけ転送
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

// 計算
kernel<<<...>>>(d_a, d_b, n);

// 出力だけ転送
cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
```

### 戦略2: 中間結果はGPU上に残す

連続処理では転送を1回に：

```cuda
// 一度だけ転送
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// GPU上で連続処理（転送なし）
kernel1<<<...>>>(d_data, d_temp, n);
kernel2<<<...>>>(d_temp, d_result, n);
kernel3<<<...>>>(d_result, d_final, n);

// 一度だけ転送
cudaMemcpy(h_result, d_final, size, cudaMemcpyDeviceToHost);
```

### 戦略3: Unified Memory

転送を自動管理（後で学習）：

```cuda
int *data;
cudaMallocManaged(&data, size);  // 自動転送

// CPU/GPU両方からアクセス可能
```

### 戦略4: Pinned Memory

高速転送（後で学習）：

```cuda
int *h_data;
cudaMallocHost(&h_data, size);  // ページロックメモリ
// 転送速度が向上
```

### 戦略5: Streams（並行実行）

転送と計算を同時実行（後で学習）：

```
時間 →
Stream 1: [転送A][計算A][転送A']
Stream 2:        [転送B][計算B][転送B']
```

## 対話形式の教え方ガイド（先生用）

### 導入

「転送が遅いのはわかった。じゃあどうすればいい？」

### 説明の流れ

1. **まずは基本から**
   - 必要なものだけ転送
   - 中間結果はGPU上に残す

2. **発展的な方法を紹介**
   - Unified Memory
   - Pinned Memory
   - Streams

3. **次のステップを示す**
   「これらは別のトピックで詳しく学べるよ」

## クリア条件（オプション）

- [ ] 基本的な転送削減方法を2つ以上説明できる
- [ ] 発展的な方法があることを知っている

## 補足情報

### 関連トピック

- **easy-memory-management**: Unified Memoryの詳細
- **fast-memory-transfer**: Pinned Memoryの詳細
- **multiple-tasks**: Streamsの詳細

### 判断基準

| 状況 | 推奨戦略 |
|------|----------|
| 初心者・プロトタイプ | Unified Memory |
| 最大性能が必要 | Pinned Memory + Streams |
| 連続処理 | 中間結果をGPUに残す |

### 参考リンク

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
