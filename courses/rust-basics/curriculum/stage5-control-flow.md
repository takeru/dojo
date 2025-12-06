# Stage 5: 制御フロー

## 目標

このステージを完了すると、生徒は：
- `if`/`else` 式で条件分岐できる
- `loop` でループを実装できる
- `while` で条件ループを実装できる
- `for` で範囲ループを実装できる
- `match` でパターンマッチングできる

## 前提知識

- Stage 1-4完了（環境、Hello World、変数と型、関数）
- 真偽値（`bool`）の理解

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **rust-pattern-matching** - パターンマッチングの詳細（計画中）
- **rust-iterators** - イテレータとループ（計画中）

## 教え方ガイド

### 導入（なぜこれを学ぶか）

制御フローはプログラムの流れを制御するための基本的な仕組みです。Rustの `if` は式であり、値を返します。また、`match` はRustのパターンマッチングの基本であり、これはRustの非常に強力な機能です。スイッチ文ではなく、`match` を理解することで、Rustの本当の力が見えてきます。

### 説明の流れ

1. **条件分岐 - `if` 式**
   ```rust
   let number = 5;
   if number > 5 {
       println!("number is bigger than 5");
   }
   ```
   - 条件は真偽値である必要がある
   - 中括弧 `{}` は必須

2. **`else` と `else if`**
   ```rust
   let number = 5;
   if number > 5 {
       println!("greater than 5");
   } else if number == 5 {
       println!("equal to 5");
   } else {
       println!("less than 5");
   }
   ```

3. **`if` は式（値を返す）**
   ```rust
   let condition = true;
   let number = if condition { 5 } else { 6 };
   println!("{}", number);  // 5
   ```
   - `if` の結果を変数に代入できる
   - 型は一致している必要がある

4. **無限ループ - `loop`**
   ```rust
   let mut count = 0;
   loop {
       println!("{}", count);
       count += 1;
       if count == 5 {
           break;
       }
   }
   ```
   - `break` でループを抜ける
   - `continue` でスキップ

5. **条件ループ - `while`**
   ```rust
   let mut number = 3;
   while number != 0 {
       println!("{}!", number);
       number -= 1;
   }
   println!("LIFTOFF!!!");
   ```

6. **範囲ループ - `for`**
   ```rust
   for number in 1..=5 {
       println!("{}", number);
   }

   for number in (1..=5).rev() {
       println!("{}", number);
   }
   ```
   - `1..5` は 1-4（5は含まない）
   - `1..=5` は 1-5（5を含む）
   - `.rev()` で逆順にできる

7. **パターンマッチング - `match`**
   ```rust
   let number = 3;
   match number {
       1 => println!("one"),
       2 => println!("two"),
       3 => println!("three"),
       _ => println!("other"),
   }
   ```
   - `_` はどの値でもマッチ（ワイルドカード）
   - すべてのケースをカバーする必要がある
   - `match` も式（値を返す）

8. **`match` での複数値**
   ```rust
   match number {
       1 | 2 => println!("one or two"),
       3..=5 => println!("three to five"),
       _ => println!("other"),
   }
   ```

### よくある間違い

- `if` の条件が真偽値でない → 型チェックエラー
- `match` で全ケースをカバーしない → コンパイルエラー
- ループの中で `break` がない無限ループ → プログラムが止まらない
- `match` の戻り値の型が異なる → 型チェックエラー

## 演習課題

### 課題1: 条件分岐
数字を受け取り、それが正、負、ゼロかを判定する関数を実装してください：
```rust
fn check_number(n: i32) {
    if n > 0 {
        println!("positive");
    } else if n < 0 {
        println!("negative");
    } else {
        println!("zero");
    }
}
```

### 課題2: `if` 式で値を返す
2つの数字を受け取り、大きい方を返す関数を実装してください：
```rust
fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
```

### 課題3: ループ
1から5までをカウントアップするプログラムを以下の方法で実装してください：
- `loop` を使用
- `while` を使用
- `for` を使用

### 課題4: `match` の基本
1-5の数字を受け取り、英語で表示する関数を実装してください：
```rust
fn number_to_word(n: i32) -> &'static str {
    match n {
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        _ => "other",
    }
}
```

### 課題5: `match` での複数値
1-10の数字を受け取り、以下のように分類する関数を実装してください：
- 1-3: "small"
- 4-7: "medium"
- 8-10: "large"

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `if`/`else` で条件分岐できた
- [ ] `if` 式で値を返せた
- [ ] `loop` でループを実装できた
- [ ] `while` で条件ループを実装できた
- [ ] `for` で範囲ループを実装できた
- [ ] `match` で基本的なパターンマッチングができた

## ヒント集

### ヒント1（軽め）
`if` の条件は真偽値（`true` または `false`）である必要があります。比較演算子（`>`, `<`, `==`, `!=`）を使って真偽値を作ります。

```rust
if x > 5 {
    // x が 5 より大きい場合
}
```

### ヒント2（中程度）
`match` を使うときは、すべてのケースをカバーする必要があります。わからないケースは `_` を使います。

```rust
match color {
    "red" => println!("Red!"),
    "blue" => println!("Blue!"),
    _ => println!("Other color"),
}
```

### ヒント3（具体的）
`match` で複数値や範囲をマッチさせる場合は、以下のように記述します：

```rust
match score {
    90..=100 => println!("A"),
    80..=89 => println!("B"),
    70..=79 => println!("C"),
    _ => println!("F"),
}

// または複数値
match grade {
    1 | 2 | 3 => println!("Elementary"),
    4 | 5 | 6 => println!("Middle"),
    _ => println!("Other"),
}
```

## 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch03-05-control-flow.html
- Rust by Example: https://doc.rust-lang.org/rust-by-example/flow_control.html
