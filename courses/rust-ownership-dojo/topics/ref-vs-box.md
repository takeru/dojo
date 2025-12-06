# トピック: 参照とBoxの違い

## メタ情報

- **ID**: ref-vs-box
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: スマートポインタ・所有権

## 前提知識

- Stage 1の所有権の基礎
- Stage 2の参照と借用
- スタックとヒープの違い

## このトピックで学べること

- 参照（`&T`）と`Box<T>`の違い
- `Box`の使いどころ
- 所有権の観点での使い分け
- 再帰型での`Box`の必要性

## 関連ステージ

- Stage 2: 借用（参照との比較）

## 要点（ドキュメント形式用）

### 基本的な違い

| 特徴 | &T（参照） | Box<T> |
|------|-----------|--------|
| 所有権 | 借用（持たない） | 所有する |
| メモリ | 既存データを指す | ヒープに新規確保 |
| ライフタイム | 元データに依存 | 自分で管理 |
| サイズ | ポインタ1つ分 | ポインタ1つ分 |

### 参照は借用

```rust
fn main() {
    let s = String::from("hello");
    let r: &String = &s;  // sを借用
    println!("{}", r);
    println!("{}", s);  // sはまだ有効
}
```

### Boxは所有

```rust
fn main() {
    let b: Box<String> = Box::new(String::from("hello"));
    // bはヒープ上のStringを所有
    println!("{}", b);
}  // bがドロップされ、ヒープのStringも解放
```

### Boxが必要な場面

**1. 再帰型の定義**

```rust
// ❌ これはコンパイルエラー（サイズが無限大）
// enum List {
//     Cons(i32, List),
//     Nil,
// }

// ✅ Boxでサイズを固定
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
}
```

**2. 大きなデータをムーブする**

```rust
// スタック上で大きなデータを移動（コピーコストあり）
struct BigData {
    data: [u8; 1_000_000],
}

// Boxならポインタのコピーだけ（8バイト）
let big: Box<BigData> = Box::new(BigData { data: [0; 1_000_000] });
let moved = big;  // ポインタのコピーだけ
```

**3. トレイトオブジェクト**

```rust
trait Animal {
    fn speak(&self);
}

struct Dog;
struct Cat;

impl Animal for Dog {
    fn speak(&self) { println!("Woof!"); }
}

impl Animal for Cat {
    fn speak(&self) { println!("Meow!"); }
}

fn main() {
    // 異なる型を同じ型として扱う
    let animals: Vec<Box<dyn Animal>> = vec![
        Box::new(Dog),
        Box::new(Cat),
    ];

    for animal in &animals {
        animal.speak();
    }
}
```

### 参照とBoxの選択

```rust
// 読むだけ → 参照
fn print_value(v: &i32) {
    println!("{}", v);
}

// 所有権が必要 → Box（または値そのもの）
fn store_value(v: Box<i32>) -> Box<i32> {
    // 長期間保持する
    v
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`&T`と`Box<T>`…どちらもポインタに見えるが、所有権が全然違う。この違いが分かれば、Rustのスマートポインタも怖くない」

なぜこれを知っておくと便利か：
- 再帰型を定義できる
- トレイトオブジェクトが使える
- 大きなデータを効率的に扱える

### 説明の流れ

1. **所有権の違いを確認**
   ```rust
   fn main() {
       // 参照：借用
       let s = String::from("hello");
       let r = &s;
       drop(s);  // エラー！まだrが借用中

       // Box：所有
       let b = Box::new(String::from("hello"));
       let b2 = b;  // 所有権がムーブ
       // println!("{}", b);  // エラー！bはムーブ済み
   }
   ```

2. **再帰型の問題を見せる**
   ```rust
   // これはコンパイルできない
   // enum List {
   //     Cons(i32, List),  // Listのサイズが無限大
   //     Nil,
   // }
   ```

   「`List`の中に`List`がある。するとサイズが計算できん」

3. **Boxで解決**
   ```rust
   enum List {
       Cons(i32, Box<List>),  // Boxは固定サイズ（ポインタ）
       Nil,
   }
   ```

   「`Box`はポインタだから、サイズは常に8バイト（64-bit）。これで再帰が可能じゃ」

4. **使い分けのガイドライン**
   - 一時的に借りる → `&T`
   - 所有権を持ちたい & ヒープに置きたい → `Box<T>`
   - 再帰型 → `Box<T>`
   - トレイトオブジェクト → `Box<dyn Trait>`

### 実践課題（オプション）

1. 連結リストを`Box`で実装する
2. トレイトオブジェクトを`Box`で格納する
3. `Box::new`と`&`の違いを実験

## クリア条件（オプション）

理解度チェック：
- [ ] 参照と`Box`の所有権の違いを説明できる
- [ ] `Box`が必要な場面を3つ挙げられる
- [ ] 再帰型で`Box`が必要な理由を説明できる

## 補足情報

### Box::leak でstaticライフタイムを得る

```rust
fn get_static_str() -> &'static str {
    let s = String::from("hello");
    Box::leak(s.into_boxed_str())  // メモリリークだが 'static になる
}
```

### Rc, Arcとの違い

```rust
use std::rc::Rc;

// Box: 単一所有
let b: Box<i32> = Box::new(5);

// Rc: 共有所有（参照カウント）
let r: Rc<i32> = Rc::new(5);
let r2 = Rc::clone(&r);  // 参照カウント+1
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch15-01-box.html
- std::boxed::Box: https://doc.rust-lang.org/std/boxed/struct.Box.html
