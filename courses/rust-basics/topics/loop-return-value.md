# トピック: loopから値を返す

## メタ情報

- **ID**: loop-return-value
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 2分（読み物）
- **カテゴリ**: 制御フロー

## 前提知識

- Stage 5完了（loopの基本）

## このトピックで学べること

- loopが式であること
- breakで値を返す方法
- 実践的な使用例

## 関連ステージ

- Stage 5: 制御フロー（ここで登場）

## 要点（ドキュメント形式用）

### loopは式

Rustでは `loop` も式なので、値を返すことができます。

```rust
let mut count = 0;
let result = loop {
    count += 1;
    if count == 10 {
        break count * 2;  // 20を返す
    }
};
println!("{}", result);  // 20
```

### breakで値を返す

`break` の後に値を書くと、その値がloop式の結果になります。

```rust
let found = loop {
    // 何かを探す処理
    if condition {
        break true;   // 見つかった
    }
    if timeout {
        break false;  // タイムアウト
    }
};
```

### 実践例: リトライ処理

```rust
fn try_connect() -> Option<Connection> {
    let connection = loop {
        match attempt_connection() {
            Ok(conn) => break Some(conn),
            Err(_) => {
                if max_retries_reached() {
                    break None;
                }
                // リトライ
            }
        }
    };
    connection
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「ループの結果を変数に代入できるって知っておったか？」

### 説明の流れ

1. **基本例を示す**
   ```rust
   let result = loop {
       break 42;
   };
   println!("{}", result);
   ```

2. **条件付きの例**
   ```rust
   let mut i = 0;
   let found = loop {
       i += 1;
       if i == 5 { break i; }
   };
   ```

3. **なぜ便利か説明**
   「一時変数なしで結果を取得できるのじゃ」

## クリア条件（オプション）

- [ ] loopが式であることを理解している
- [ ] breakで値を返せる
- [ ] 実践的な場面で使える

## 補足情報

### whileやforとの違い

`while` や `for` は常に `()` を返すので、値を返せません：

```rust
// これはできない
let result = while condition {
    break 42;  // エラー！
};

// loopを使う
let result = loop {
    if !condition { break 42; }
};
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch03-05-control-flow.html#returning-values-from-loops
