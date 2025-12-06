# トピック: Stringと&strの違い（所有権視点）

## メタ情報

- **ID**: string-vs-str
- **難易度**: 初級〜中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 型・所有権

## 前提知識

- Stage 1の所有権の基礎
- Stage 2の参照と借用

## このトピックで学べること

- `String`と`&str`の違い
- 所有権の観点からの使い分け
- 関数引数での適切な選択
- 文字列リテラルの仕組み

## 関連ステージ

- Stage 1: 所有権（String型の所有権移動）
- Stage 2: 借用（&strは借用の一種）

## 要点（ドキュメント形式用）

### 基本的な違い

| 特徴 | String | &str |
|------|--------|------|
| 所有権 | 持つ（所有型） | 持たない（借用型） |
| メモリ | ヒープ | どこでも（ヒープ、スタック、静的領域） |
| サイズ | 可変 | 固定（スライス） |
| 変更 | 可能（mut） | 不可能 |

### メモリレイアウト

```
String (24バイト on 64-bit)          &str (16バイト on 64-bit)
┌──────────────────┐                 ┌──────────────────┐
│ ptr ─────────────┼──┐              │ ptr ─────────────┼──┐
│ len = 5          │  │              │ len = 5          │  │
│ capacity = 8     │  │              └──────────────────┘  │
└──────────────────┘  │                                     │
                      ▼                                     ▼
                 ┌─────────────┐                       ┌─────────────┐
                 │ h│e│l│l│o│ │ │ │                    │ h│e│l│l│o   │
                 └─────────────┘                       └─────────────┘
                    ヒープ                                どこか
```

### 変換方法

```rust
// &str → String
let s: &str = "hello";
let owned1: String = s.to_string();
let owned2: String = String::from(s);
let owned3: String = s.to_owned();

// String → &str
let s: String = String::from("hello");
let borrowed: &str = &s;        // 暗黙の変換（Deref）
let borrowed: &str = s.as_str(); // 明示的
```

### 関数引数の設計

```rust
// ❌ 制限的: Stringしか受け取れない
fn greet_bad(name: String) {
    println!("Hello, {}!", name);
}

// ✅ 柔軟: &strを受け取る（Stringも渡せる）
fn greet_good(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let owned = String::from("Alice");
    let literal = "Bob";

    // greet_bad(literal);  // エラー
    greet_bad(owned.clone());  // clone必要

    greet_good(&owned);   // OK: String → &str
    greet_good(literal);  // OK: &str そのまま
}
```

### 所有権の観点での選択

```rust
// 所有権が必要な場合 → String
struct Person {
    name: String,  // 構造体は自分のデータを所有すべき
}

// 一時的に参照するだけ → &str
fn process_name(name: &str) {
    println!("Processing: {}", name);
}

// 文字列を生成して返す → String
fn create_greeting(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

### 文字列リテラルの正体

```rust
// 文字列リテラルは &'static str
let s: &'static str = "hello";

// バイナリに埋め込まれ、プログラム全体で有効
// だから所有権を持たなくても安全に使える
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`String`と`&str`…Rustで最初に混乱する部分じゃろう。しかし所有権の視点で見れば、すっきり理解できる」

なぜこれを知っておくと便利か：
- 関数のシグネチャを適切に設計できる
- 不必要なアロケーションを避けられる
- 文字列処理のエラーが減る

### 説明の流れ

1. **基本的な違いを確認**
   ```rust
   let literal: &str = "hello";        // 借用（静的領域を参照）
   let owned: String = String::from("hello");  // 所有
   ```

   「`literal`は静的領域のデータを借りているだけ。`owned`はヒープにデータを持っている」

2. **ムーブの違いを見る**
   ```rust
   // Stringはムーブする
   let s1 = String::from("hello");
   let s2 = s1;
   // println!("{}", s1);  // エラー

   // &strはコピーする（参照だから）
   let r1: &str = "hello";
   let r2 = r1;
   println!("{}", r1);  // OK
   ```

3. **関数引数の設計指針**
   「基本原則: **読むだけなら`&str`を受け取れ**」

   ```rust
   // これで String も &str も受け取れる
   fn print_length(s: &str) {
       println!("Length: {}", s.len());
   }
   ```

4. **所有権が必要な場面**
   ```rust
   // 構造体のフィールド → 所有したい
   struct Config {
       name: String,  // &strだとライフタイムが複雑に
   }

   // 値を返す → 所有権を渡す
   fn create_name() -> String {
       String::from("generated")
   }
   ```

### 実践課題（オプション）

1. `&str`を受け取る関数を書いて、`String`と文字列リテラル両方を渡す
2. 構造体に`String`と`&str`を使い分けてみる
3. `to_string()`, `to_owned()`, `String::from()`の違いを調べる

## クリア条件（オプション）

理解度チェック：
- [ ] `String`と`&str`の違いを所有権の観点で説明できる
- [ ] 関数引数で`&str`を使うメリットを説明できる
- [ ] 文字列リテラルの型（`&'static str`）を理解している

## 補足情報

### Cow<str>: 所有と借用の両立

```rust
use std::borrow::Cow;

fn maybe_modify(s: &str, should_modify: bool) -> Cow<str> {
    if should_modify {
        Cow::Owned(s.to_uppercase())  // 新しいStringを作成
    } else {
        Cow::Borrowed(s)  // 借用のまま返す
    }
}
```

`Cow`（Clone on Write）は、必要なときだけ所有権を取る便利な型。

### into_string()パターン

```rust
// 柔軟な引数受け取り
fn greet(name: impl Into<String>) {
    let name: String = name.into();
    println!("Hello, {}!", name);
}

fn main() {
    greet("Alice");         // &str → String
    greet(String::from("Bob"));  // String そのまま
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch08-02-strings.html
- std::string::String: https://doc.rust-lang.org/std/string/struct.String.html
- std::str: https://doc.rust-lang.org/std/primitive.str.html
