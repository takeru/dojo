# トピック: ガード（条件付きマッチ）

## メタ情報

- **ID**: match-guards
- **難易度**: 中級
- **所要時間**: 6-8分（対話形式）/ 3分（読み物）
- **カテゴリ**: 制御フロー

## 前提知識

- Stage 5完了（matchの基本）

## このトピックで学べること

- マッチガードの構文
- 複雑な条件の表現
- 実践的な使用例

## 関連ステージ

- Stage 5: 制御フロー（ここで登場）

## 要点（ドキュメント形式用）

### マッチガードとは

パターンマッチに追加の条件（`if` 節）をつけられます。

```rust
let number = 4;

match number {
    x if x < 0 => println!("負の数"),
    x if x == 0 => println!("ゼロ"),
    x if x > 0 => println!("正の数"),
    _ => unreachable!(),
}
```

### 基本構文

```
パターン if 条件 => 式
```

### 実践例

```rust
let pair = (2, -2);

match pair {
    (x, y) if x == y => println!("等しい"),
    (x, y) if x + y == 0 => println!("和がゼロ"),
    (x, _) if x > 0 => println!("xが正"),
    _ => println!("その他"),
}
// 出力: 和がゼロ
```

### 複数パターンとガード

ガードは全パターンに適用されます：

```rust
let x = 4;
let y = false;

match x {
    4 | 5 | 6 if y => println!("yes"),  // 4, 5, 6 全てに y がチェックされる
    _ => println!("no"),
}
// 出力: no（yがfalseなので）
```

### Option との組み合わせ

```rust
let num = Some(4);

match num {
    Some(x) if x < 5 => println!("5未満: {}", x),
    Some(x) => println!("5以上: {}", x),
    None => println!("なし"),
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「matchに追加条件をつけたいことがあるじゃろう？ガードを使うのじゃ」

### 説明の流れ

1. **基本構文を示す**
   ```rust
   match x {
       n if n > 0 => println!("positive"),
       _ => println!("not positive"),
   }
   ```

2. **なぜ必要かを説明**
   「パターンだけでは表現できない条件があるからじゃ」

3. **実践例を示す**
   「Optionと組み合わせると便利じゃぞ」

## クリア条件（オプション）

- [ ] マッチガードの構文を使える
- [ ] ガードを使う場面を理解している
- [ ] 複数パターンとガードを組み合わせられる

## 補足情報

### ガードでの変数束縛

ガード内で束縛した変数を使えます：

```rust
let num = Some(10);

match num {
    Some(x) if x > 5 => println!("大きい: {}", x),
    Some(x) => println!("小さい: {}", x),
    None => (),
}
```

### 注意点

ガードはコンパイラの網羅性チェックに含まれません：

```rust
let x = 5;

match x {
    n if n > 0 => println!("positive"),
    // コンパイラは n <= 0 のケースを知らない
    _ => println!("other"),  // これが必要
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html#extra-conditionals-with-match-guards
