# トピック: 関連関数とメソッドの違い

## メタ情報

- **ID**: associated-vs-method
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 2分（読み物）
- **カテゴリ**: 構造体

## 前提知識

- Stage 6完了（構造体とimplの基本）

## このトピックで学べること

- メソッドと関連関数の違い
- それぞれの呼び出し方
- new()パターン

## 関連ステージ

- Stage 6: 構造体（ここで登場）

## 要点（ドキュメント形式用）

### メソッドと関連関数の違い

| 種類 | selfを受け取る | 呼び出し方 |
|------|---------------|-----------|
| メソッド | ✓ | `instance.method()` |
| 関連関数 | ✗ | `Type::function()` |

### メソッド

```rust
impl Rectangle {
    // メソッド: &selfを受け取る
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

let rect = Rectangle { width: 30, height: 50 };
let area = rect.area();  // インスタンスから呼び出す
```

### 関連関数

```rust
impl Rectangle {
    // 関連関数: selfを受け取らない
    fn new(width: u32, height: u32) -> Rectangle {
        Rectangle { width, height }
    }

    fn square(size: u32) -> Rectangle {
        Rectangle { width: size, height: size }
    }
}

let rect = Rectangle::new(30, 50);  // 型名から呼び出す
let square = Rectangle::square(10);
```

### 使い分け

**メソッドを使う場合**:
- インスタンスのデータを使う・変更するとき
- `obj.method()` の形で呼びたいとき

**関連関数を使う場合**:
- コンストラクタ（`new`）
- インスタンスを生成するファクトリ関数
- インスタンスなしで呼べるユーティリティ

### 標準ライブラリの例

```rust
// 関連関数
let s = String::new();           // 空の文字列
let s = String::from("hello");   // 文字列から生成

// メソッド
let len = s.len();               // 長さを取得
let upper = s.to_uppercase();    // 大文字に変換
```

## 対話形式の教え方ガイド（先生用）

### 導入

「メソッドと関連関数、両方 impl の中に書くが、違いがあるのじゃ」

### 説明の流れ

1. **違いを説明**
   「selfを受け取るかどうかで決まるのじゃ」

2. **呼び出し方の違いを示す**
   ```rust
   rect.area()          // メソッド
   Rectangle::new(...)  // 関連関数
   ```

3. **new()パターンを紹介**
   「コンストラクタはnew()という関連関数にするのが慣例じゃ」

## クリア条件（オプション）

- [ ] メソッドと関連関数の違いを説明できる
- [ ] 呼び出し方の違いを理解している
- [ ] new()パターンを使える

## 補足情報

### Default トレイト

引数なしのコンストラクタには `Default` トレイトを使うこともあります：

```rust
#[derive(Default)]
struct Config {
    debug: bool,
    timeout: u32,
}

let config = Config::default();
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-03-method-syntax.html#associated-functions
