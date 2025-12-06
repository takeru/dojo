# トピック: 構造体の所有権と借用

## メタ情報

- **ID**: struct-ownership
- **難易度**: 中級
- **所要時間**: 10-12分（対話形式）/ 5分（読み物）
- **カテゴリ**: 構造体・所有権

## 前提知識

- Stage 6完了（構造体の基本）

## このトピックで学べること

- 構造体を関数に渡すときの所有権
- 参照を使った借用
- 所有権システムの予習

## 関連ステージ

- Stage 6: 構造体（ここで登場）

## 要点（ドキュメント形式用）

### 構造体を関数に渡すと...

構造体を関数に渡すと、所有権が移動します。

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn process(rect: Rectangle) {
    println!("{:?}", rect);
}  // rect の所有権がここで終わる

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    process(rect);
    // println!("{:?}", rect);  // エラー！rect の所有権は失われた
}
```

### 参照を使えば借用できる

参照を使うと、所有権を失わずに渡せます。

```rust
fn process_ref(rect: &Rectangle) {
    println!("{:?}", rect);
}

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    process_ref(&rect);  // 参照を渡す
    println!("{:?}", rect);  // OK！まだ使える
}
```

### 可変参照で変更

```rust
fn resize(rect: &mut Rectangle, width: u32, height: u32) {
    rect.width = width;
    rect.height = height;
}

fn main() {
    let mut rect = Rectangle { width: 30, height: 50 };
    resize(&mut rect, 100, 200);
    println!("{:?}", rect);  // 100x200
}
```

### メソッドでの自動参照

メソッド呼び出しでは、Rustが自動的に参照を作ります：

```rust
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

let rect = Rectangle { width: 30, height: 50 };
let a = rect.area();      // &rect.area() と同じ
let a = (&rect).area();   // 明示的に書いた場合
```

### まとめ

| 渡し方 | 所有権 | 変更 | 渡した後に使える |
|--------|--------|------|-----------------|
| `rect` | 移動 | - | ✗ |
| `&rect` | 借用 | ✗ | ✓ |
| `&mut rect` | 可変借用 | ✓ | ✓ |

## 対話形式の教え方ガイド（先生用）

### 導入

「構造体を関数に渡すとどうなるか、これは所有権を学ぶ前哨戦じゃ」

### 説明の流れ

1. **所有権の移動を体験**
   ```rust
   fn process(rect: Rectangle) { ... }
   process(rect);
   // rect は使えない！
   ```

2. **参照で解決**
   ```rust
   fn process_ref(rect: &Rectangle) { ... }
   process_ref(&rect);
   // rect はまだ使える！
   ```

3. **メソッドの便利さを説明**
   「メソッドなら & を書かなくても自動でやってくれるのじゃ」

4. **所有権への橋渡し**
   「これは所有権システムの基礎じゃ。後で詳しく学ぶぞ」

## クリア条件（オプション）

- [ ] 構造体を関数に渡すと所有権が移動することを理解している
- [ ] 参照を使って借用できる
- [ ] 可変参照で変更できる

## 補足情報

### Copy トレイト

整数のような単純な型は `Copy` トレイトを持ち、自動的にコピーされます：

```rust
let x = 5;
let y = x;  // コピー
println!("{}", x);  // OK

let s = String::from("hello");
let t = s;  // 移動
// println!("{}", s);  // エラー！
```

構造体にも `Copy` を実装できます（すべてのフィールドが `Copy` の場合）：

```rust
#[derive(Copy, Clone)]
struct Point {
    x: i32,
    y: i32,
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
