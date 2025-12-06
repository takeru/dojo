# トピック: よくあるライフタイムエラー集

## メタ情報

- **ID**: common-lifetime-errors
- **難易度**: 中級
- **所要時間**: 15-20分（対話形式）/ 7分（読み物）
- **カテゴリ**: デバッグ・ライフタイム

## 前提知識

- Stage 3のライフタイム基礎
- 借用チェッカーの基本

## このトピックで学べること

- よくあるライフタイムエラーのパターン
- エラーメッセージの読み方
- 各エラーの解決パターン
- デバッグのコツ

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### エラー1: ダングリング参照

**エラーメッセージ**:
```
error[E0515]: cannot return reference to local variable `s`
```

**問題のコード**:
```rust
fn dangling() -> &String {
    let s = String::from("hello");
    &s  // sはこの関数で破棄される
}
```

**解決策**:
```rust
// 方法1: 所有権を返す
fn fixed() -> String {
    String::from("hello")
}

// 方法2: 'staticなデータを返す
fn static_str() -> &'static str {
    "hello"
}
```

### エラー2: 借用が長すぎる

**エラーメッセージ**:
```
error[E0597]: `x` does not live long enough
```

**問題のコード**:
```rust
fn main() {
    let r;
    {
        let x = 5;
        r = &x;  // xはここでドロップ
    }
    println!("{}", r);  // ダングリング！
}
```

**解決策**:
```rust
fn main() {
    let x = 5;  // xのスコープを広げる
    let r = &x;
    println!("{}", r);
}
```

### エラー3: ライフタイム注釈が必要

**エラーメッセージ**:
```
error[E0106]: missing lifetime specifier
```

**問題のコード**:
```rust
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

**解決策**:
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### エラー4: 戻り値のライフタイム不一致

**エラーメッセージ**:
```
error[E0623]: lifetime mismatch
```

**問題のコード**:
```rust
fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &'a str {
    y  // 'b を 'a として返そうとしている
}
```

**解決策**:
```rust
// 方法1: 同じライフタイムを使う
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 方法2: ライフタイム境界を追加
fn longest<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### エラー5: 構造体のライフタイム

**エラーメッセージ**:
```
error[E0106]: missing lifetime specifier
```

**問題のコード**:
```rust
struct Excerpt {
    part: &str,  // ライフタイムがない
}
```

**解決策**:
```rust
struct Excerpt<'a> {
    part: &'a str,
}
```

### エラー6: 不変/可変参照の競合

**エラーメッセージ**:
```
error[E0502]: cannot borrow `x` as mutable because it is also borrowed as immutable
```

**問題のコード**:
```rust
fn main() {
    let mut x = 5;
    let r = &x;
    let m = &mut x;  // 不変参照がまだ有効
    println!("{}", r);
}
```

**解決策**:
```rust
fn main() {
    let mut x = 5;
    let r = &x;
    println!("{}", r);  // rの使用を終わらせる
    let m = &mut x;     // これでOK
    *m += 1;
}
```

### エラー7: クロージャのライフタイム

**エラーメッセージ**:
```
error[E0373]: closure may outlive the current function
```

**問題のコード**:
```rust
fn returns_closure() -> impl Fn() {
    let x = 5;
    || println!("{}", x)  // xは関数終了時にドロップ
}
```

**解決策**:
```rust
fn returns_closure() -> impl Fn() {
    let x = 5;
    move || println!("{}", x)  // xの所有権をクロージャに移動
}
```

## エラーメッセージの読み方

### 重要なキーワード

| キーワード | 意味 |
|-----------|------|
| `does not live long enough` | 参照先がスコープ外 |
| `cannot return reference to local` | ローカル変数への参照を返そうとした |
| `missing lifetime specifier` | ライフタイム注釈が必要 |
| `lifetime mismatch` | ライフタイムが合わない |
| `borrowed value` | 借用された値 |
| `outlives` | より長く生きる |

### デバッグのコツ

1. **エラー箇所を特定**: `-->` の行を確認
2. **注釈を読む**: `help:` の提案を確認
3. **ライフタイムを追う**: どの参照がどこまで有効か
4. **スコープを確認**: 変数がいつ破棄されるか

## 対話形式の教え方ガイド（先生用）

### 導入

「ライフタイムエラーは最初は怖い。しかしパターンを覚えれば、実は対処しやすいのじゃ。よくあるエラーを見ていこう」

なぜこれを知っておくと便利か：
- エラーメッセージを読めるようになる
- デバッグ時間が短縮される
- ライフタイムの理解が深まる

### 説明の流れ

1. **エラーメッセージの構造**
   ```
   error[E0515]: cannot return reference to local variable `s`
    --> src/main.rs:3:5
     |
   3 |     &s
     |     ^^ returns a reference to data owned by the current function
   ```

   「エラーコード（E0515）、場所、原因が示される。`help:`があれば解決策も教えてくれる」

2. **パターンごとに対処法を示す**
   - ダングリング → 所有権を返す
   - スコープ不足 → スコープを広げる
   - 注釈不足 → ライフタイムパラメータを追加

3. **実際にエラーを起こして修正する**
   「わざとエラーを起こして、修正する練習をするのが一番じゃ」

### 実践課題（オプション）

1. 各エラーパターンを意図的に起こす
2. エラーメッセージを読んで修正する
3. 修正前後のコードを比較する

## クリア条件（オプション）

理解度チェック：
- [ ] 主要なライフタイムエラーを識別できる
- [ ] エラーメッセージから原因を特定できる
- [ ] 各エラーの解決パターンを知っている

## 補足情報

### rustc --explainで詳細を見る

```bash
rustc --explain E0515
```

エラーコードの詳細な説明と例を表示。

### cargo clippyでの早期発見

```bash
cargo clippy
```

ライフタイム関連の問題を事前に警告してくれることがある。

### 参考リンク

- Rust Error Index: https://doc.rust-lang.org/error_codes/error-index.html
- Common Rust Lifetime Misconceptions: https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
