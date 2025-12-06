# トピック: GPUでif文を使うと遅くなる？Warp分岐の罠

## メタ情報

- **ID**: if-statement-trap
- **難易度**: 中級
- **所要時間**: 8-10分
- **カテゴリ**: 最適化

## 前提知識

- Stage 6完了
- Warpの概念（how-gpu-runs推奨）

## このトピックで学べること

- Warp Divergence（分岐発散）とは
- if文がパフォーマンスに与える影響
- 分岐を減らす工夫

## 関連ステージ

- Stage 6: スレッドとブロック

## 要点（ドキュメント形式用）

Warpの32スレッドで分岐（if文）が発生すると、両方のパスを順次実行するため遅くなります。

### Warp Divergence（分岐発散）

```cuda
if (threadIdx.x < 16) {
    // 処理A
} else {
    // 処理B
}
```

Warp内で半分が処理A、半分が処理Bに分かれると：
1. まず16スレッドが処理Aを実行（残り16スレッドは待機）
2. 次に16スレッドが処理Bを実行（最初の16スレッドは待機）

→ **直列実行になる！**

### 悪い例（Warp内で分岐）

```cuda
int idx = threadIdx.x;
if (idx % 2 == 0) {  // 偶数スレッド
    // 重い処理
}
```

### 良い例（Warp単位で分岐）

```cuda
int warpId = threadIdx.x / 32;
if (warpId == 0) {  // Warp全体で同じ分岐
    // 全スレッドが同じパス
}
```

### 分岐を避ける工夫

```cuda
// 悪い
if (condition) {
    result = a + b;
} else {
    result = 0;
}

// 良い（分岐なし）
result = condition ? (a + b) : 0;
// または
result = (a + b) * condition;  // conditionが0/1の場合
```

## 対話形式の教え方ガイド（先生用）

### 導入

「CPUでは普通に使うif文が、GPUでは遅くなることがあるんだ」

### 実践課題

1. Warp内分岐とWarp単位分岐の速度比較

## クリア条件

- [ ] Warp Divergenceの概念を理解している
- [ ] 分岐を減らす工夫を知っている

## 補足情報

### 参考リンク

- [CUDA C++ Best Practices - Control Flow](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#control-flow)
