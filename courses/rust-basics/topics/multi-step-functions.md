# トピック: 関数内での複数ステップ

## メタ情報

- **ID**: multi-step-functions
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 2分（読み物）
- **カテゴリ**: 関数

## 前提知識

- Stage 4完了（関数の基本）
- 文と式の違い

## このトピックで学べること

- 関数内で複数の処理を書く方法
- 最後の式が戻り値になる仕組み
- 中間変数の活用

## 関連ステージ

- Stage 4: 関数（ここで登場）

## 要点（ドキュメント形式用）

### 基本パターン

関数内では複数の文を書けます。重要なのは**最後が式**であることです。

```rust
fn calculate(x: i32) -> i32 {
    let doubled = x * 2;        // 文
    let incremented = doubled + 1;  // 文
    incremented                 // 式（戻り値）
}
```

### よくある間違い

```rust
fn calculate_wrong(x: i32) -> i32 {
    let doubled = x * 2;
    let incremented = doubled + 1;
    incremented;                // ← ; があると文になる
    // 戻り値がない！コンパイルエラー
}
```

### 中間変数を使う理由

複雑な計算は中間変数に分けると読みやすくなります：

```rust
// 読みにくい
fn area_bad(width: i32, height: i32, margin: i32) -> i32 {
    (width + margin * 2) * (height + margin * 2)
}

// 読みやすい
fn area_good(width: i32, height: i32, margin: i32) -> i32 {
    let total_width = width + margin * 2;
    let total_height = height + margin * 2;
    total_width * total_height
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「関数内で複雑な処理をするときはどうするのか、見ていくぞ」

### 説明の流れ

1. **基本パターンを示す**
   ```rust
   fn process(x: i32) -> i32 {
       let step1 = x * 2;
       let step2 = step1 + 10;
       step2
   }
   ```

2. **よくある間違いを見せる**
   「最後にセミコロンをつけるとエラーになるのじゃ」

3. **中間変数の利点を説明**
   「複雑な式は分けると読みやすくなるぞ」

## クリア条件（オプション）

- [ ] 関数内で複数の文を書ける
- [ ] 最後の式が戻り値になることを理解している
- [ ] 中間変数を適切に使える

## 補足情報

### デバッグに便利

中間変数があると、途中の値を確認しやすい：

```rust
fn calculate(x: i32) -> i32 {
    let step1 = x * 2;
    println!("step1: {}", step1);  // デバッグ用
    let step2 = step1 + 10;
    println!("step2: {}", step2);  // デバッグ用
    step2
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch03-03-how-functions-work.html
