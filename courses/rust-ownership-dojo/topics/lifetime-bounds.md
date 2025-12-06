# トピック: ライフタイム境界

## メタ情報

- **ID**: lifetime-bounds
- **難易度**: 中級〜上級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: ライフタイム・ジェネリクス

## 前提知識

- Stage 3のライフタイム基礎
- ジェネリクス（`<T>`）の基本

## このトピックで学べること

- `T: 'a`の意味
- ライフタイム境界が必要な場面
- 複数のライフタイム境界の組み合わせ
- 構造体でのライフタイム境界

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### T: 'a の意味

「型Tに含まれるすべての参照が、少なくともライフタイム`'a`と同じかそれ以上長く生きる」

```rust
// T: 'a は「Tは'aより短いライフタイムの参照を含まない」
fn example<'a, T: 'a>(x: &'a T) -> &'a T {
    x
}
```

### なぜ必要か

```rust
struct Ref<'a, T: 'a> {  // T: 'a が必要
    value: &'a T,
}

// なぜ？
// &'a T が有効である間、T自体も有効でなければならない
// T が 'a より短いライフタイムの参照を含んでいたら、
// その参照がダングリングになる可能性がある
```

### 具体例

```rust
// Tが参照を含む場合
struct Container<'a> {
    data: &'a str,
}

struct Wrapper<'a, T: 'a> {
    inner: &'a T,
}

fn main() {
    let s = String::from("hello");
    let container = Container { data: &s };

    let wrapper = Wrapper { inner: &container };
    // Tは Container<'_> で、参照を含む
    // T: 'a により、container内の参照が
    // wrapperのライフタイム中有効であることが保証される
}
```

### ライフタイム境界の種類

```rust
// 1. 単純な境界
fn foo<'a, T: 'a>(x: &'a T) {}

// 2. 'static境界
fn bar<T: 'static>(x: T) {}  // Tは参照を含まないか、'static参照のみ

// 3. 複数の境界
fn baz<'a, 'b: 'a, T: 'b>(x: &'a T) {}  // 'b は 'a より長い

// 4. トレイト + ライフタイム境界
fn qux<'a, T: Clone + 'a>(x: &'a T) -> T {
    x.clone()
}
```

### 'b: 'a（ライフタイムの包含関係）

```rust
// 'b: 'a は「'b は 'a と同じかより長い」
fn longest<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 戻り値は'aなので、xもyも少なくとも'a以上生きる必要がある
// 'b: 'a により、yは'a以上生きることが保証される
```

### 構造体でのライフタイム境界

```rust
// Rust 2018以降、暗黙的に推論される場合が多い
struct MyStruct<'a, T> {  // T: 'a は暗黙
    data: &'a T,
}

// 明示的に書く場合
struct ExplicitStruct<'a, T: 'a> {
    data: &'a T,
}

// 実質的に同じ
```

### 境界が不要なケース

```rust
// 所有型のみを扱う場合
struct Owner<T> {
    data: T,  // ライフタイム境界不要
}

// 参照がない型パラメータ
fn takes_owned<T>(x: T) {}  // 境界不要
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`T: 'a`という記法を見たことがあるか？これはライフタイム境界じゃ。ジェネリクスとライフタイムを組み合わせるときに必要になる」

なぜこれを知っておくと便利か：
- ジェネリックな構造体を設計できる
- 複雑なライフタイム関係を表現できる
- コンパイルエラーを理解できる

### 説明の流れ

1. **問題を見せる**
   ```rust
   struct Ref<'a, T> {
       value: &'a T,
   }
   // コンパイルエラー（古いRust）または暗黙的に T: 'a が適用
   ```

   「`&'a T`という参照がある。Tが`'a`より短い参照を含んでいたら、問題になる」

2. **境界の意味**
   ```rust
   struct Ref<'a, T: 'a> {
       value: &'a T,
   }
   ```

   「`T: 'a`は『Tに含まれる参照はすべて`'a`以上生きる』という約束じゃ」

3. **具体例で理解**
   ```rust
   // Tが String（所有型）の場合
   let s = String::from("hello");
   let r: Ref<String> = Ref { value: &s };
   // String は参照を含まない → OK

   // Tが &str の場合
   let text = "hello";  // 'static
   let r: Ref<&str> = Ref { value: &text };
   // &str は 'static → 任意の 'a を満たす → OK
   ```

4. **'b: 'a の説明**
   ```rust
   fn example<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str {
       y  // yは'bだが、'b >= 'a なので'aに縮められる
   }
   ```

### 実践課題（オプション）

1. ライフタイム境界を持つ構造体を定義
2. `'b: 'a`を使った関数を書く
3. 境界が必要な場面と不要な場面を見分ける

## クリア条件（オプション）

理解度チェック：
- [ ] `T: 'a`の意味を説明できる
- [ ] ライフタイム境界が必要な理由を理解している
- [ ] `'b: 'a`の意味を説明できる

## 補足情報

### 暗黙のライフタイム境界

Rust 2018以降、多くの場合で自動推論される:

```rust
// 暗黙的に T: 'a が適用される
struct Implicit<'a, T> {
    data: &'a T,
}

// 明示的に書いても同じ
struct Explicit<'a, T: 'a> {
    data: &'a T,
}
```

### where句での記述

```rust
fn complex<'a, 'b, T, U>(x: &'a T, y: &'b U)
where
    'b: 'a,
    T: Clone + 'a,
    U: Debug + 'b,
{
    // ...
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch19-02-advanced-lifetimes.html
- Rustonomicon: https://doc.rust-lang.org/nomicon/lifetimes.html
- RFC 2093 (Implied Bounds): https://rust-lang.github.io/rfcs/2093-infer-outlives.html
