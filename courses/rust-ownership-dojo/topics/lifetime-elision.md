# トピック: ライフタイム省略規則

## メタ情報

- **ID**: lifetime-elision
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: ライフタイム

## 前提知識

- Stage 3のライフタイム基礎
- ライフタイム注釈の書き方

## このトピックで学べること

- ライフタイム省略規則とは
- 3つの省略規則
- いつ省略できるか
- いつ明示が必要か

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### ライフタイム省略とは

多くの場合、ライフタイム注釈を省略できます。コンパイラが自動で推論してくれます。

```rust
// 書いたコード
fn first_word(s: &str) -> &str {
    &s[..s.find(' ').unwrap_or(s.len())]
}

// コンパイラが推論するコード
fn first_word<'a>(s: &'a str) -> &'a str {
    &s[..s.find(' ').unwrap_or(s.len())]
}
```

### 3つの省略規則

**規則1: 各入力参照に独自のライフタイム**

```rust
// 入力
fn foo(x: &i32)
fn bar(x: &i32, y: &str)

// 推論後
fn foo<'a>(x: &'a i32)
fn bar<'a, 'b>(x: &'a i32, y: &'b str)
```

**規則2: 入力が1つなら、出力もそのライフタイム**

```rust
// 入力
fn first(s: &str) -> &str

// 推論後
fn first<'a>(s: &'a str) -> &'a str
```

**規則3: メソッドなら、selfのライフタイムを出力に使用**

```rust
impl<'a> MyStruct<'a> {
    // 入力
    fn method(&self) -> &str

    // 推論後
    fn method(&self) -> &'a str
}
```

### 省略できる例

```rust
// 規則1 + 規則2
fn trim(s: &str) -> &str { s.trim() }

// 規則3
impl MyStruct {
    fn name(&self) -> &str { &self.name }
}

// 引数が1つの関数
fn identity<T>(x: &T) -> &T { x }
```

### 省略できない例

```rust
// 入力が2つで、出力がどちらか不明
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
// エラー: missing lifetime specifier

// 修正: 明示的に書く
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 規則の適用フロー

```
1. 入力参照に規則1を適用
   fn foo(x: &T, y: &U) → fn foo<'a, 'b>(x: &'a T, y: &'b U)

2. 規則2を試す（入力が1つ？）
   fn foo(x: &T) → 出力も 'a

3. 規則3を試す（メソッドで&self？）
   fn method(&self) → 出力は self のライフタイム

4. それでも決まらない → エラー、手動で書く必要あり
```

### 構造体でのライフタイム

構造体に参照がある場合、**省略できません**。

```rust
// エラー: missing lifetime specifier
struct Excerpt {
    part: &str,
}

// 修正: 明示的に書く
struct Excerpt<'a> {
    part: &'a str,
}
```

### implブロックでの省略

```rust
struct Excerpt<'a> {
    part: &'a str,
}

impl<'a> Excerpt<'a> {
    // 規則3により、戻り値は'aと推論される
    fn get(&self) -> &str {
        self.part
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「毎回ライフタイムを書くのは面倒じゃろう？実は、Rustには省略規則がある。これを知れば、なぜ書かなくてよいか、なぜ書く必要があるかが分かる」

なぜこれを知っておくと便利か：
- いつライフタイムを書くべきか判断できる
- コンパイラのエラーメッセージを理解できる
- 冗長なコードを避けられる

### 説明の流れ

1. **省略できる例を見せる**
   ```rust
   fn first_word(s: &str) -> &str { ... }
   ```

   「これは動く。でもライフタイムを書いていない。なぜか？」

2. **3つの規則を順番に**
   - 規則1: 入力にそれぞれライフタイム
   - 規則2: 入力1つなら出力も同じ
   - 規則3: メソッドならselfのライフタイム

3. **省略できない例**
   ```rust
   fn longest(x: &str, y: &str) -> &str { ... }
   ```

   「入力が2つで、どちらが出力に影響するか分からん。だから明示が必要」

4. **判断基準**
   「迷ったら書いてみる。省略できるならコンパイラが教えてくれる」

### 実践課題（オプション）

1. ライフタイムを省略できる関数を書く
2. 省略できない関数を書いてエラーを確認
3. 省略規則を手動で適用してみる

## クリア条件（オプション）

理解度チェック：
- [ ] 3つの省略規則を説明できる
- [ ] ライフタイム省略が適用されるか判断できる
- [ ] 省略できない場合に手動で書ける

## 補足情報

### 歴史的背景

初期のRustでは、すべてのライフタイムを明示する必要がありました。省略規則はRust 1.0で導入され、コードを簡潔にしました。

### '_（匿名ライフタイム）

```rust
// 明示的に「推論してほしい」ことを示す
impl Iterator for MyIter<'_> {
    // ...
}
```

`'_`は「コンパイラにお任せ」の意味。

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-elision
- Rust Reference: https://doc.rust-lang.org/reference/lifetime-elision.html
- RFC 141 (Lifetime Elision): https://rust-lang.github.io/rfcs/0141-lifetime-elision.html
