# トピック: メモリの読み方次第で10倍速くなる話

## メタ情報

- **ID**: memory-access-tricks
- **難易度**: 中級
- **所要時間**: 10-12分
- **カテゴリ**: 最適化

## 前提知識

- Stage 7完了

## このトピックで学べること

- Coalesced Memory Accessの仕組み
- 効率的なメモリアクセスパターン

## 関連ステージ

- Stage 7: 行列積

## 要点（ドキュメント形式用）

GPUでは、隣接するスレッドが連続したメモリアドレスにアクセスすると高速化されます（Coalesced Access）。

### 悪い例（バラバラにアクセス）

```cuda
// ストライドアクセス（遅い）
int idx = threadIdx.x;
int value = arr[idx * 32];  // 0, 32, 64, 96, ...
```

### 良い例（連続アクセス）

```cuda
// 連続アクセス（速い）
int idx = threadIdx.x;
int value = arr[idx];  // 0, 1, 2, 3, ...
```

### 行列転置での例

**非効率**:
```cuda
out[j][i] = in[i][j];  // 書き込みがバラバラ
```

**効率的**:
```cuda
// シェアードメモリ経由で連続アクセスに
```

## 対話形式の教え方ガイド（先生用）

### 導入

「メモリの読み方を変えるだけで10倍速くなることもあるんだ」

### 実践課題

1. ストライドアクセスと連続アクセスの速度比較

## クリア条件

- [ ] Coalesced Accessの概念を理解している

## 補足情報

### 参考リンク

- [Coalesced Memory Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
