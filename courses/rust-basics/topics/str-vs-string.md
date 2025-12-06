# トピック: 文字列スライス &str の型

## メタ情報

- **ID**: str-vs-string
- **難易度**: 初級〜中級
- **所要時間**: 8-10分（対話形式）/ 4分（読み物）
- **カテゴリ**: 関数・型

## 前提知識

- Stage 4完了（関数の基本）

## このトピックで学べること

- `&str` と `String` の違い
- 関数の引数でどちらを使うか
- 文字列の基本的な変換

## 関連ステージ

- Stage 4: 関数（ここで登場）

## 要点（ドキュメント形式用）

### &str と String の違い

| 型 | 説明 | 特徴 |
|----|------|------|
| `&str` | 文字列スライス | 借用、固定長、軽量 |
| `String` | 所有された文字列 | 所有、可変長、ヒープ割り当て |

### 使い分け

**関数の引数**: 通常 `&str` を使う

```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    greet("Rust");                    // &str リテラル
    greet(&String::from("Rust"));     // String の参照
}
```

**戻り値**: 新しい文字列を作る場合は `String`

```rust
fn make_greeting(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

### 変換方法

```rust
// &str → String
let s: String = "hello".to_string();
let s: String = String::from("hello");

// String → &str
let string = String::from("hello");
let slice: &str = &string;
let slice: &str = string.as_str();
```

### なぜ引数に &str を使うのか

`&str` を受け取る関数は、`&str` も `&String` も受け付けられます：

```rust
fn print_len(s: &str) {
    println!("{}", s.len());
}

let literal = "hello";           // &str
let owned = String::from("hello");  // String

print_len(literal);   // OK
print_len(&owned);    // OK（&Stringは&strに自動変換）
```

## 対話形式の教え方ガイド（先生用）

### 導入

「関数の引数で `&str` って見たことあるじゃろう。`String` との違いを理解すると、文字列の扱いが楽になるぞ」

### 説明の流れ

1. **違いを簡単に説明**
   「`&str` は借りてきた文字列、`String` は自分が持っている文字列じゃ」

2. **引数には&strを使う理由**
   「`&str` を受け取ると、両方受け付けられて便利なのじゃ」

3. **変換方法を紹介**
   ```rust
   let s = "hello";
   let owned = s.to_string();
   let borrowed = owned.as_str();
   ```

4. **所有権との関係を予告**
   「詳しくは所有権を学ぶときに分かるぞ」

## クリア条件（オプション）

- [ ] `&str` と `String` の違いを説明できる
- [ ] 関数の引数で `&str` を使う理由を理解している
- [ ] 相互変換ができる

## 補足情報

### 文字列リテラルの型

```rust
let s = "hello";  // 型は &'static str
```

`'static` は「プログラム全体で有効」という意味のライフタイムです。

### メモリ上の違い

- `&str`: スタックにポインタと長さだけ（16バイト）
- `String`: ヒープにデータ、スタックにポインタ・長さ・容量

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch04-03-slices.html
