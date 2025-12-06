# トピック: 複数のimplブロック

## メタ情報

- **ID**: multiple-impl-blocks
- **難易度**: 初級
- **所要時間**: 4-6分（対話形式）/ 2分（読み物）
- **カテゴリ**: 構造体

## 前提知識

- Stage 6完了（構造体とimplの基本）

## このトピックで学べること

- 複数のimplブロックを書ける理由
- 整理のパターン
- トレイト実装との関係

## 関連ステージ

- Stage 6: 構造体（ここで登場）

## 要点（ドキュメント形式用）

### 複数のimplブロックは有効

同じ構造体に複数の `impl` ブロックを書くことができます。

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl Rectangle {
    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}
```

### なぜ複数のimplブロックが許可されているか

1. **コードの整理**: 関連するメソッドをグループ化
2. **トレイト実装**: 各トレイトを別のimplブロックで実装
3. **条件付きコンパイル**: 特定の条件でのみメソッドを追加

### 整理のパターン

```rust
// コンストラクタ
impl Rectangle {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    fn square(size: u32) -> Self {
        Self { width: size, height: size }
    }
}

// 計算メソッド
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}

// 変更メソッド
impl Rectangle {
    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
```

### トレイト実装

各トレイトは別の `impl` ブロックで実装します：

```rust
impl Display for Rectangle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Debug for Rectangle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Rectangle {{ width: {}, height: {} }}",
               self.width, self.height)
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「同じ構造体に impl を複数書くことができるのじゃ。知っておると便利じゃぞ」

### 説明の流れ

1. **複数書けることを示す**
   ```rust
   impl Rectangle { fn area(&self) -> u32 { ... } }
   impl Rectangle { fn perimeter(&self) -> u32 { ... } }
   ```

2. **なぜ便利かを説明**
   「メソッドを整理したり、トレイトごとに分けたりできるのじゃ」

3. **実際の使われ方を紹介**
   「標準ライブラリでもよく使われておるぞ」

## クリア条件（オプション）

- [ ] 複数のimplブロックを書ける
- [ ] 整理のパターンを理解している

## 補足情報

### ジェネリクスと組み合わせ

ジェネリクスを使うと、特定の型にだけメソッドを追加できます：

```rust
impl<T> Vec<T> {
    fn len(&self) -> usize { ... }
}

impl Vec<i32> {
    fn sum(&self) -> i32 { ... }  // i32のVecにだけ追加
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-03-method-syntax.html
