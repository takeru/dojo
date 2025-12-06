# トピック: 高階ライフタイム入門

## メタ情報

- **ID**: hrtb-intro
- **難易度**: 上級
- **所要時間**: 15-20分（対話形式）/ 7分（読み物）
- **カテゴリ**: ライフタイム・高度な型

## 前提知識

- Stage 3のライフタイム基礎
- クロージャの基本
- トレイト境界の理解

## このトピックで学べること

- 高階トレイト境界（HRTB）とは何か
- `for<'a>`構文の読み方
- クロージャでHRTBが現れる場面
- なぜHRTBが必要か

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### 高階トレイト境界とは

「任意のライフタイムに対して」という制約を表現する構文。

```rust
// for<'a> は「任意の'aについて」を意味
fn apply_to_ref<F>(f: F)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    let s = String::from("hello");
    let result = f(&s);
    println!("{}", result);
}
```

### 問題の背景

```rust
// この関数シグネチャを考える
fn call_with_ref<F>(f: F)
where
    F: Fn(&str) -> &str,  // ライフタイムが省略されている
{
    let s = String::from("hello");
    let result = f(&s);
    println!("{}", result);
}

// 実際には、Rustは以下のように解釈する
// F: for<'a> Fn(&'a str) -> &'a str
```

### for<'a>の読み方

```rust
for<'a> Fn(&'a str) -> &'a str
```

→ 「**任意の**ライフタイム`'a`について、`&'a str`を受け取り`&'a str`を返す」

### 具体例: クロージャ

```rust
fn main() {
    // このクロージャは for<'a> Fn(&'a str) -> &'a str を満たす
    let identity = |s: &str| s;

    // 異なるライフタイムで呼び出せる
    let s1 = String::from("hello");
    let r1 = identity(&s1);

    {
        let s2 = String::from("world");
        let r2 = identity(&s2);  // 別のライフタイム
    }
}
```

### HRTBが必要な場面

**1. コールバック関数を受け取る**

```rust
fn process_items<F>(items: &[String], processor: F)
where
    F: for<'a> Fn(&'a str),  // 任意のライフタイムの参照を処理
{
    for item in items {
        processor(item);
    }
}
```

**2. イテレータアダプタ**

```rust
// 標準ライブラリのfilter
fn filter<P>(self, predicate: P) -> Filter<Self, P>
where
    P: FnMut(&Self::Item) -> bool,  // 暗黙的にHRTB
```

**3. 参照を返すトレイトオブジェクト**

```rust
// Box<dyn Fn(&str) -> &str> は実際には
// Box<dyn for<'a> Fn(&'a str) -> &'a str>
```

### HRTBなしでは何が問題か

```rust
// もしHRTBがなければ...
fn bad_example<'a, F>(f: F, s: &'a str) -> &'a str
where
    F: Fn(&'a str) -> &'a str,
{
    f(s)
}

// 問題: fは特定の'aでしか呼べない
// 異なるライフタイムで複数回呼ぶことができない
```

### いつ意識するか

**ほとんどの場合、Rustが自動で推論**:

```rust
// これを書くと
fn call<F>(f: F) where F: Fn(&str) -> &str

// Rustは自動的に
fn call<F>(f: F) where F: for<'a> Fn(&'a str) -> &'a str
// に展開する
```

**明示的に書く必要があるケース**:
- 複雑なトレイト境界
- 明示的な型注釈
- エラーメッセージの理解

## 対話形式の教え方ガイド（先生用）

### 導入

「`for<'a>`という見慣れない構文を見たことがあるか？これは『高階トレイト境界』じゃ。上級者向けの概念だが、クロージャを使うなら知っておくと役立つ」

なぜこれを知っておくと便利か：
- クロージャ関連のエラーメッセージが読める
- 高度なAPIを設計できる
- ライフタイムの深い理解につながる

### 説明の流れ

1. **問題を提示**
   ```rust
   fn call_twice<F>(f: F, s1: &str, s2: &str)
   where
       F: Fn(&str) -> usize,
   {
       println!("{}", f(s1));
       println!("{}", f(s2));
   }
   ```

   「このFは、異なるライフタイムの`&str`を受け取れる。どうやって表現する？」

2. **for<'a>の導入**
   「`for<'a>`は『任意のライフタイム`'a`について』という意味じゃ」

   ```rust
   F: for<'a> Fn(&'a str) -> usize
   ```

3. **暗黙の展開**
   「実は、Rustが自動で`for<'a>`を付けてくれる。だから普段は意識しなくてよい」

   ```rust
   // 書いたコード
   F: Fn(&str) -> usize

   // Rustが解釈するコード
   F: for<'a> Fn(&'a str) -> usize
   ```

4. **エラーメッセージでの登場**
   「コンパイルエラーで`for<'a>`を見たら、『任意のライフタイムについて』と読み替えるのじゃ」

### 実践課題（オプション）

1. HRTBを明示的に書いた関数を作る
2. クロージャを受け取る関数を設計する
3. エラーメッセージで`for<'a>`を見つける

## クリア条件（オプション）

理解度チェック：
- [ ] `for<'a>`の読み方を理解している
- [ ] HRTBが必要な理由を説明できる
- [ ] 暗黙のHRTB展開を理解している

## 補足情報

### 完全な構文

```rust
// 基本形
for<'a> Fn(&'a T) -> &'a U

// 複数のライフタイム
for<'a, 'b> Fn(&'a T, &'b U) -> &'a V

// where句での記述
where F: for<'a> FnMut(&'a str)
```

### Fn/FnMut/FnOnceとHRTB

```rust
// すべてのFnトレイトでHRTBは使える
for<'a> Fn(&'a str)
for<'a> FnMut(&'a str)
for<'a> FnOnce(&'a str)
```

### 参考リンク

- Rustonomicon - HRTBs: https://doc.rust-lang.org/nomicon/hrtb.html
- The Rust Reference: https://doc.rust-lang.org/reference/trait-bounds.html#higher-ranked-trait-bounds
- Rust Blog: https://blog.rust-lang.org/2018/01/31/The-2018-Rust-roadmap.html
