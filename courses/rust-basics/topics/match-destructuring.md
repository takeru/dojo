# トピック: matchでの分解

## メタ情報

- **ID**: match-destructuring
- **難易度**: 中級
- **所要時間**: 8-10分（対話形式）/ 4分（読み物）
- **カテゴリ**: 制御フロー

## 前提知識

- Stage 5完了（matchの基本）

## このトピックで学べること

- タプルの分解
- 構造体の分解
- 部分的なマッチング

## 関連ステージ

- Stage 5: 制御フロー（ここで登場）

## 要点（ドキュメント形式用）

### タプルの分解

`match` でタプルの中身を取り出せます。

```rust
let point = (0, 5);

match point {
    (0, y) => println!("y軸上: y = {}", y),
    (x, 0) => println!("x軸上: x = {}", x),
    (x, y) => println!("座標: ({}, {})", x, y),
}
```

### 構造体の分解

```rust
struct Point {
    x: i32,
    y: i32,
}

let p = Point { x: 0, y: 7 };

match p {
    Point { x: 0, y } => println!("y軸上: y = {}", y),
    Point { x, y: 0 } => println!("x軸上: x = {}", x),
    Point { x, y } => println!("座標: ({}, {})", x, y),
}
```

### 部分的なマッチング

すべてをキャプチャする必要はありません：

```rust
let numbers = (2, 4, 8, 16, 32);

match numbers {
    (first, _, third, _, fifth) => {
        println!("{}, {}, {}", first, third, fifth);
    }
}
// 出力: 2, 8, 32
```

### 範囲でのマッチング

```rust
let x = 5;

match x {
    1..=5 => println!("1から5"),
    6..=10 => println!("6から10"),
    _ => println!("その他"),
}
```

### 複数パターン

```rust
let x = 1;

match x {
    1 | 2 => println!("1か2"),
    3 | 4 | 5 => println!("3, 4, 5のどれか"),
    _ => println!("その他"),
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「matchで値を取り出せることを知っておるか？」

### 説明の流れ

1. **タプルの分解を示す**
   ```rust
   let pair = (1, 2);
   match pair {
       (0, y) => println!("y = {}", y),
       (x, y) => println!("({}, {})", x, y),
   }
   ```

2. **_で無視することを紹介**
   「興味のない部分は _ で無視できるのじゃ」

3. **構造体でもできることを示す**
   「構造体のフィールドも分解できるぞ」

## クリア条件（オプション）

- [ ] タプルをmatchで分解できる
- [ ] _で不要な値を無視できる
- [ ] 複数パターン（|）を使える

## 補足情報

### ネストした構造の分解

```rust
let nested = ((1, 2), (3, 4));

match nested {
    ((a, _), (_, d)) => println!("{}, {}", a, d),
}
// 出力: 1, 4
```

### 参照の分解

```rust
let reference = &10;

match reference {
    &val => println!("値: {}", val),
}

// または ref を使う
match *reference {
    val => println!("値: {}", val),
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html
