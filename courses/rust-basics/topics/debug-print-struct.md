# トピック: 構造体へのデバッグプリント

## メタ情報

- **ID**: debug-print-struct
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 2分（読み物）
- **カテゴリ**: 構造体

## 前提知識

- Stage 6完了（構造体の基本）

## このトピックで学べること

- #[derive(Debug)]の使い方
- {:?}と{:#?}の違い
- 他のderiveマクロ

## 関連ステージ

- Stage 6: 構造体（ここで登場）

## 要点（ドキュメント形式用）

### 構造体をprintln!で出力

デフォルトでは構造体を `{}` で出力できません。

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

let rect = Rectangle { width: 30, height: 50 };
// println!("{}", rect);  // エラー！
```

### #[derive(Debug)]

`Debug` トレイトを自動実装すると、`{:?}` で出力できます。

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

let rect = Rectangle { width: 30, height: 50 };
println!("{:?}", rect);
// 出力: Rectangle { width: 30, height: 50 }
```

### {:?} と {:#?}

| フォーマット | 説明 | 用途 |
|-------------|------|------|
| `{:?}` | コンパクト表示 | 1行で確認 |
| `{:#?}` | 整形表示 | 複雑な構造を見やすく |

```rust
#[derive(Debug)]
struct Point { x: i32, y: i32 }

#[derive(Debug)]
struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

let rect = Rectangle {
    top_left: Point { x: 0, y: 0 },
    bottom_right: Point { x: 10, y: 10 },
};

println!("{:?}", rect);
// Rectangle { top_left: Point { x: 0, y: 0 }, bottom_right: Point { x: 10, y: 10 } }

println!("{:#?}", rect);
// Rectangle {
//     top_left: Point {
//         x: 0,
//         y: 0,
//     },
//     bottom_right: Point {
//         x: 10,
//         y: 10,
//     },
// }
```

### dbg!マクロ

より便利なデバッグ用マクロ：

```rust
let x = 5;
let y = dbg!(x * 2);  // [src/main.rs:2] x * 2 = 10
```

- ファイル名と行番号が表示される
- 値を返すので式の中で使える

## 対話形式の教え方ガイド（先生用）

### 導入

「構造体を println! で出力したいことがあるじゃろう？」

### 説明の流れ

1. **エラーを見せる**
   「そのままでは出力できないのじゃ」

2. **derive(Debug)を追加**
   ```rust
   #[derive(Debug)]
   struct Point { x: i32, y: i32 }
   ```

3. **{:?}と{:#?}を試す**
   「複雑な構造には{:#?}が見やすいぞ」

4. **dbg!を紹介**
   「デバッグ時はdbg!が便利じゃ」

## クリア条件（オプション）

- [ ] #[derive(Debug)]を使える
- [ ] {:?}と{:#?}の違いを理解している
- [ ] dbg!マクロを使える

## 補足情報

### よく使うderiveマクロ

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}
```

| マクロ | 効果 |
|--------|------|
| `Debug` | `{:?}`で出力 |
| `Clone` | `.clone()`でコピー |
| `PartialEq` | `==`で比較 |
| `Default` | デフォルト値を生成 |

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-02-example-structs.html#adding-useful-functionality-with-derived-traits
