# トピック: 'staticライフタイム徹底解説

## メタ情報

- **ID**: static-lifetime
- **難易度**: 中級〜上級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: ライフタイム

## 前提知識

- Stage 3のライフタイム基礎
- 参照とライフタイムの関係

## このトピックで学べること

- `'static`ライフタイムの意味
- 文字列リテラルが`'static`な理由
- `Box::leak`で`'static`を作る方法
- トレイト境界での`'static`の意味
- `'static`の誤解を解く

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### 'staticの2つの意味

1. **ライフタイム注釈として**: プログラム全体の期間
2. **トレイト境界として**: 所有型、または'static参照を含む型

### ライフタイムとしての'static

```rust
// 文字列リテラルは 'static
let s: &'static str = "hello, world";

// プログラム終了まで有効
fn get_greeting() -> &'static str {
    "Hello!"  // バイナリに埋め込まれている
}
```

### なぜ文字列リテラルは'staticか

```
プログラムのバイナリ
┌─────────────────────┐
│ コードセクション    │
├─────────────────────┤
│ データセクション    │  ← "hello" はここにある
│   "hello, world"    │
│   "Hello!"          │
└─────────────────────┘

プログラム実行中ずっと存在する → 'static
```

### Box::leakで'staticを作る

```rust
fn create_static() -> &'static str {
    let s = String::from("dynamic");
    Box::leak(s.into_boxed_str())  // メモリリーク！でも'static
}

fn main() {
    let greeting: &'static str = create_static();
    println!("{}", greeting);
}
```

**警告**: `Box::leak`はメモリを解放しません。グローバル設定など、プログラム終了まで必要なデータにのみ使用。

### トレイト境界としての'static

```rust
// T: 'static は「Tがプログラム全体で有効」ではない！
// 「Tが'staticより短いライフタイムの参照を含まない」という意味

fn spawn_thread<T: Send + 'static>(data: T) {
    std::thread::spawn(move || {
        // dataはスレッドに所有権を移動
        println!("{:?}", data);
    });
}

fn main() {
    let owned = String::from("hello");  // 'staticを満たす（所有型）
    spawn_thread(owned);  // OK

    let local = String::from("local");
    let reference = &local;  // 'staticを満たさない
    // spawn_thread(reference);  // エラー！
}
```

### 'staticの誤解

**誤解**: `'static`は「永久に生きる」必要がある

**正解**: `'static`は「プログラム全体で**有効になりうる**」という意味

```rust
fn main() {
    let s: &'static str = "hello";
    drop(s);  // 変数sを捨てる
    // "hello"自体はまだバイナリにある
}

// Stringは'staticを満たす（所有型だから）
fn takes_static<T: 'static>(t: T) {
    drop(t);  // Tをドロップできる！
}

fn main() {
    takes_static(String::from("owned"));  // OK!
}
```

### いつ'staticを使うか

| ケース | 例 |
|--------|-----|
| 文字列リテラル | `let s: &'static str = "hello"` |
| グローバル設定 | `lazy_static!`で初期化 |
| スレッドに渡すデータ | `thread::spawn`の引数 |
| エラーメッセージ | `&'static str`で定数エラー |

## 対話形式の教え方ガイド（先生用）

### 導入

「`'static`…最も長いライフタイムじゃ。しかし誤解されやすい。今日はその真の意味を理解しよう」

なぜこれを知っておくと便利か：
- スレッドにデータを渡せる
- グローバル設定を扱える
- ライフタイムエラーを理解できる

### 説明の流れ

1. **文字列リテラルを見せる**
   ```rust
   let s: &'static str = "hello";
   ```

   「この"hello"はどこにある？ヒープでもスタックでもない。バイナリに埋め込まれているのじゃ。だからプログラム全体で有効」

2. **トレイト境界の意味**
   ```rust
   fn takes_static<T: 'static>(t: T) {}

   takes_static(String::from("owned"));  // OK
   takes_static(42i32);                  // OK
   // takes_static(&local_string);       // エラー
   ```

   「`T: 'static`は『Tが短いライフタイムの参照を含まない』という意味じゃ。所有型はすべて満たす」

3. **スレッドとの関係**
   ```rust
   use std::thread;

   let s = String::from("hello");
   thread::spawn(move || {
       println!("{}", s);  // 所有権がスレッドに移動
   });
   ```

   「スレッドはいつ終わるか分からん。だから`'static`な値（または所有権を持つ値）だけを渡せる」

4. **Box::leakの使い方**
   ```rust
   let s: &'static str = Box::leak(String::from("dynamic").into_boxed_str());
   ```

   「動的に作った文字列を`'static`にできる。ただしメモリリークじゃ。慎重に使え」

### 実践課題（オプション）

1. 文字列リテラルを関数から返す
2. `'static`境界を持つ関数を書く
3. `Box::leak`を使ってみる

## クリア条件（オプション）

理解度チェック：
- [ ] 文字列リテラルが`'static`な理由を説明できる
- [ ] `T: 'static`の意味を正確に説明できる
- [ ] `'static`と「永久に生きる」の違いを理解している

## 補足情報

### lazy_static / once_cell

グローバルな`'static`値を安全に初期化:

```rust
use once_cell::sync::Lazy;

static CONFIG: Lazy<String> = Lazy::new(|| {
    String::from("initialized once")
});

fn main() {
    println!("{}", *CONFIG);
}
```

### const vs static

```rust
// const: コンパイル時定数（インライン展開される）
const MESSAGE: &str = "hello";

// static: 静的変数（アドレスを持つ）
static COUNTER: AtomicI32 = AtomicI32::new(0);
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#the-static-lifetime
- Rust Reference: https://doc.rust-lang.org/reference/lifetime-elision.html
- Common Rust Lifetime Misconceptions: https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
