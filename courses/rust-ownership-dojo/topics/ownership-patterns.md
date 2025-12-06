# トピック: 所有権を渡す3つのパターン

## メタ情報

- **ID**: ownership-patterns
- **難易度**: 初級〜中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 設計パターン

## 前提知識

- Stage 1の所有権の基礎
- Stage 2の借用の基本（参照の作り方）
- `Clone`トレイトの理解

## このトピックで学べること

- ムーブ・クローン・参照の3パターンの使い分け
- 各パターンのメリット・デメリット
- 実践的な場面での選択基準
- APIデザインの指針

## 関連ステージ

- Stage 1: 所有権
- Stage 2: 借用

## 要点（ドキュメント形式用）

関数に値を渡す方法は3つあります。それぞれの特徴を理解して使い分けましょう。

### パターン1: ムーブ（所有権を渡す）

```rust
fn take_ownership(s: String) {
    println!("{}", s);
}

fn main() {
    let s = String::from("hello");
    take_ownership(s);
    // println!("{}", s);  // エラー: sは既にムーブされた
}
```

**使いどころ**:
- 関数内で値を消費する（使い切る）とき
- 所有権を明確に移転したいとき
- 例: ファイルを閉じる、リソースを解放する

### パターン2: クローン（コピーを渡す）

```rust
fn process(s: String) {
    println!("{}", s);
}

fn main() {
    let s = String::from("hello");
    process(s.clone());  // コピーを渡す
    println!("{}", s);   // 元は使える
}
```

**使いどころ**:
- 元の値も使いたい & 関数が所有権を必要とするとき
- 値が比較的小さいとき
- 注意: ヒープコピーのコストがかかる

### パターン3: 参照（借用する）

```rust
fn borrow(s: &String) {
    println!("{}", s);
}

fn borrow_mut(s: &mut String) {
    s.push_str(" world");
}

fn main() {
    let mut s = String::from("hello");
    borrow(&s);          // 不変参照
    borrow_mut(&mut s);  // 可変参照
    println!("{}", s);   // 元も使える
}
```

**使いどころ**:
- 値を読むだけのとき（不変参照）
- 値を変更したいが所有権は不要なとき（可変参照）
- 最も一般的で効率的なパターン

### 選択フローチャート

```
関数は値を消費する必要がある？
├─ Yes → ムーブ (fn foo(s: String))
└─ No
    ├─ 値を変更する？
    │   ├─ Yes → 可変参照 (fn foo(s: &mut String))
    │   └─ No → 不変参照 (fn foo(s: &String))
    │
    └─ 呼び出し側で元の値も使いたい & 関数がムーブを要求？
        └─ clone() を使う
```

### 実践的な比較

```rust
// ❌ 非効率: 不必要なムーブ
fn print_length(s: String) -> String {
    println!("Length: {}", s.len());
    s  // 返さないと呼び出し側で使えない
}

// ✅ 効率的: 参照を使う
fn print_length(s: &String) {
    println!("Length: {}", s.len());
}

// さらに良い: &str を受け取る
fn print_length(s: &str) {
    println!("Length: {}", s.len());
}
```

### イディオム: 所有権を取って返す

```rust
// ビルダーパターン風
fn append_greeting(mut s: String) -> String {
    s.push_str(", world!");
    s
}

fn main() {
    let s = String::from("Hello");
    let s = append_greeting(s);  // 所有権を渡して受け取る
    println!("{}", s);
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「所有権を渡す方法は3つある。ムーブ、クローン、参照…この3つを自在に使い分けられれば、Rustの達人に近づくぞ」

なぜこれを知っておくと便利か：
- コンパイルエラーを避けられる
- 効率的なコードが書ける
- Rustらしい設計ができる

### 説明の流れ

1. **3パターンを並べて見せる**
   ```rust
   fn main() {
       let s = String::from("hello");

       // パターン1: ムーブ
       fn take(s: String) { println!("{}", s); }
       // take(s);  // sは使えなくなる

       // パターン2: クローン
       fn take2(s: String) { println!("{}", s); }
       // take2(s.clone());  // コピーを渡す、sは使える

       // パターン3: 参照
       fn borrow(s: &String) { println!("{}", s); }
       borrow(&s);  // 参照を渡す、sは使える
   }
   ```

2. **それぞれのコストを説明**
   「ムーブはコスト0（ポインタのコピーだけ）。クローンはヒープコピーで高コスト。参照はコスト0（ポインタのコピーだけ）」

3. **選択基準を教える**
   「基本は**参照**を使え。所有権を渡す必要があるときだけムーブ。どうしても両方必要ならクローンじゃ」

4. **実践的な例**
   ```rust
   struct Document {
       content: String,
   }

   impl Document {
       // 読むだけ → 参照
       fn word_count(&self) -> usize {
           self.content.split_whitespace().count()
       }

       // 変更する → 可変参照
       fn append(&mut self, text: &str) {
           self.content.push_str(text);
       }

       // 消費する → ムーブ
       fn into_content(self) -> String {
           self.content
       }
   }
   ```

### 実践課題（オプション）

1. 3パターンそれぞれで関数を書いて動作確認
2. 不必要にcloneしているコードを参照に書き換える
3. `into_xxx`パターンの関数を設計する

## クリア条件（オプション）

理解度チェック：
- [ ] ムーブ・クローン・参照の使い分け基準を説明できる
- [ ] 効率を意識して適切なパターンを選べる
- [ ] 関数シグネチャを見てどのパターンか判断できる

## 補足情報

### 標準ライブラリのパターン

```rust
// 所有権を取る: into_xxx
let s = String::from("hello");
let bytes = s.into_bytes();  // Stringを消費してVec<u8>に

// 参照を取る: as_xxx
let s = String::from("hello");
let slice: &str = s.as_str();  // 参照を返す、Stringは使える

// クローンを返す: to_xxx
let slice: &str = "hello";
let owned: String = slice.to_string();  // 新しいStringを作成
```

### 命名規則の慣習

| 接頭辞 | 意味 | 例 |
|--------|------|-----|
| `into_` | 所有権を取り変換 | `into_bytes()`, `into_iter()` |
| `as_` | 参照として返す | `as_str()`, `as_bytes()` |
| `to_` | 新しい所有権を作成 | `to_string()`, `to_vec()` |

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
- API Guidelines: https://rust-lang.github.io/api-guidelines/naming.html#ad-hoc-conversions-follow-as_-to_-into_-conventions-c-conv
