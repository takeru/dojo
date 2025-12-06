# トピック: self, &self, &mut selfの違い

## メタ情報

- **ID**: self-variants
- **難易度**: 中級
- **所要時間**: 8-10分（対話形式）/ 4分（読み物）
- **カテゴリ**: 構造体

## 前提知識

- Stage 6完了（構造体とimplの基本）

## このトピックで学べること

- メソッドでのselfの3つの形式
- それぞれの使いどころ
- 所有権との関係

## 関連ステージ

- Stage 6: 構造体（ここで登場）

## 要点（ドキュメント形式用）

### 3つのself

メソッドの第一引数の形式で、所有権の扱いが変わります。

| 形式 | 説明 | 用途 |
|------|------|------|
| `&self` | 不変借用 | 読み取り専用（最も一般的） |
| `&mut self` | 可変借用 | フィールドを変更 |
| `self` | 所有権取得 | 構造体を消費 |

### &self - 参照（読み取り専用）

```rust
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

let rect = Rectangle { width: 30, height: 50 };
let a = rect.area();
println!("{}", rect.width);  // OK、まだ使える
```

### &mut self - 可変参照

```rust
impl Rectangle {
    fn grow(&mut self, amount: u32) {
        self.width += amount;
        self.height += amount;
    }
}

let mut rect = Rectangle { width: 30, height: 50 };
rect.grow(10);  // 変更される
println!("{}x{}", rect.width, rect.height);  // 40x60
```

### self - 所有権を取得

```rust
impl Rectangle {
    fn into_square(self) -> Rectangle {
        let size = self.width.min(self.height);
        Rectangle { width: size, height: size }
    }
}

let rect = Rectangle { width: 30, height: 50 };
let square = rect.into_square();
// rect はもう使えない！所有権が移動した
```

### 使い分けの指針

1. **デフォルトは `&self`**: 読み取りだけなら参照で十分
2. **変更が必要なら `&mut self`**: フィールドを更新するとき
3. **`self` は特殊ケース**: 変換や消費が目的のとき

## 対話形式の教え方ガイド（先生用）

### 導入

「メソッドで `&self` と `self` を見かけるじゃろう。これの違いは重要じゃぞ」

### 説明の流れ

1. **&selfを説明**
   「参照だから、呼び出し後も元の値を使えるのじゃ」

2. **&mut selfを説明**
   「可変参照だから、フィールドを変更できるのじゃ」

3. **selfを説明**
   「所有権を取るから、呼び出し後は元の値が使えないのじゃ」

4. **使い分けを説明**
   「迷ったら&selfを使うのじゃ。変更が必要なら&mut selfじゃ」

## クリア条件（オプション）

- [ ] 3つのselfの違いを説明できる
- [ ] それぞれの使いどころを理解している
- [ ] 所有権との関係を理解している

## 補足情報

### 命名規則

- `self` を取るメソッドは `into_*` という名前が多い
- `&mut self` を取るメソッドはオブジェクトを変更する

```rust
impl String {
    fn into_bytes(self) -> Vec<u8> { ... }  // 所有権を取る
    fn push_str(&mut self, s: &str) { ... }  // 変更する
    fn len(&self) -> usize { ... }           // 読み取りだけ
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-03-method-syntax.html
