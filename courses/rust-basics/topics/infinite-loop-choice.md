# トピック: 無限ループの使い分け

## メタ情報

- **ID**: infinite-loop-choice
- **難易度**: 初級
- **所要時間**: 4-6分（対話形式）/ 2分（読み物）
- **カテゴリ**: 制御フロー

## 前提知識

- Stage 5完了（loop, whileの基本）

## このトピックで学べること

- loopとwhile trueの違い
- どちらを使うべきか
- コンパイラ最適化の観点

## 関連ステージ

- Stage 5: 制御フロー（ここで登場）

## 要点（ドキュメント形式用）

### loopを使うべき理由

無限ループには `loop` を使いましょう。

```rust
// ✅ 推奨
loop {
    // 処理
    if done { break; }
}

// ❌ 避ける
while true {
    // 処理
    if done { break; }
}
```

### なぜloopが良いのか

1. **意図が明確**: 「無限ループ」であることが一目でわかる
2. **コンパイラに優しい**: コンパイラが無限ループと認識できる
3. **値を返せる**: `break` で値を返すことができる

### while trueの問題点

```rust
// コンパイラは condition が常に true か判断しにくい
while true {
    // ...
}

// loopなら無限ループであることが確定
loop {
    // ...
}
```

### 条件がある場合

条件付きで繰り返す場合は `while` を使います：

```rust
// 条件がある場合は while
while count < 10 {
    count += 1;
}

// 無限ループなら loop
loop {
    if count >= 10 { break; }
    count += 1;
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`loop` と `while true`、どっちを使うべきか迷ったことはないかな？」

### 説明の流れ

1. **両方の書き方を見せる**
   ```rust
   // どちらも無限ループだが...
   loop { }
   while true { }
   ```

2. **loopを推奨する理由を説明**
   「意図が明確で、コンパイラにも優しいのじゃ」

3. **値を返せる点を強調**
   「loopは式なので値を返せるが、whileは返せないぞ」

## クリア条件（オプション）

- [ ] 無限ループにはloopを使うべきことを理解している
- [ ] loopとwhile trueの違いを説明できる

## 補足情報

### Clippyの警告

Rustの静的解析ツール `clippy` は `while true` に警告を出します：

```bash
cargo clippy
# warning: denote infinite loops with `loop { ... }`
```

### 参考リンク

- Clippy lint: https://rust-lang.github.io/rust-clippy/master/index.html#while_true
