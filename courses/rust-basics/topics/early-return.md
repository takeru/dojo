# トピック: 早期リターン

## メタ情報

- **ID**: early-return
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 2分（読み物）
- **カテゴリ**: 関数

## 前提知識

- Stage 4完了（関数の基本）

## このトピックで学べること

- returnキーワードの使い方
- 早期リターンのパターン
- returnを省略できる場合

## 関連ステージ

- Stage 4: 関数（ここで登場）

## 要点（ドキュメント形式用）

### returnキーワード

`return` を使うと、関数の途中から値を返せます。

```rust
fn check_positive(x: i32) -> &'static str {
    if x > 0 {
        return "positive";  // ここで関数を抜ける
    }
    if x < 0 {
        return "negative";
    }
    "zero"
}
```

### 早期リターンのパターン

条件を満たさない場合に早めに抜ける「ガード節」パターン：

```rust
fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        return None;  // 早期リターン
    }
    Some(a / b)
}
```

### returnを省略できる場合

最後の式であれば `return` は不要です：

```rust
// returnあり
fn add(x: i32, y: i32) -> i32 {
    return x + y;
}

// returnなし（推奨）
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

### 条件分岐での使い分け

```rust
// 早期リターン（ガード節）が有効な場合
fn process(x: Option<i32>) -> i32 {
    if x.is_none() {
        return 0;
    }
    x.unwrap() * 2
}

// 式として書ける場合（こちらが推奨）
fn process(x: Option<i32>) -> i32 {
    match x {
        Some(n) => n * 2,
        None => 0,
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「関数の途中で結果を返したいときはどうするか、知っておくと便利じゃぞ」

### 説明の流れ

1. **returnの基本**
   ```rust
   fn greet(name: &str) -> String {
       if name.is_empty() {
           return String::from("Hello, stranger!");
       }
       format!("Hello, {}!", name)
   }
   ```

2. **ガード節パターンを紹介**
   「条件チェックを最初にやって、早めに抜けるパターンじゃ」

3. **省略できる場合を説明**
   「最後の式ならreturnは不要。Rustっぽい書き方じゃ」

## クリア条件（オプション）

- [ ] returnで途中から値を返せる
- [ ] 早期リターンの使いどころを理解している
- [ ] returnを省略できる場合を知っている

## 補足情報

### 戻り値なしの早期リターン

戻り値がない関数でも早期リターンできます：

```rust
fn print_if_positive(x: i32) {
    if x <= 0 {
        return;  // 何も返さずに終了
    }
    println!("{}", x);
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch03-03-how-functions-work.html
