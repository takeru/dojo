# トピック: CloneとCopyトレイトの違い

## メタ情報

- **ID**: clone-vs-copy
- **難易度**: 初級〜中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: トレイト・型システム

## 前提知識

- Stage 1の所有権の基礎
- ムーブとコピーの違い
- トレイトの基本概念（実装していれば尚可）

## このトピックで学べること

- `Copy`と`Clone`トレイトの違い
- どの型が`Copy`を実装できるか
- `clone()`を使うべき場面
- 自作型に`Copy`を実装する方法

## 関連ステージ

- Stage 1: 所有権（ムーブとコピーの理解を深める）

## 要点（ドキュメント形式用）

### Copy と Clone の違い

| 特徴 | Copy | Clone |
|------|------|-------|
| コピー方法 | 暗黙的（ビットコピー） | 明示的（`.clone()`) |
| コスト | 常に安価 | 高コストの可能性あり |
| 実装可能な型 | スタックのみ・固定サイズ | 任意の型 |
| 意味 | 単純なメモリコピー | 深いコピー（カスタム可能） |

### Copy トレイト

```rust
// Copy可能な型の例
let x: i32 = 5;
let y = x;  // 暗黙的にコピー
println!("{}", x);  // OK: xはまだ有効

// Copy可能な型一覧
// - 整数型: i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize
// - 浮動小数点: f32, f64
// - bool, char
// - Copy型だけを含むタプル: (i32, f64)
// - Copy型の固定長配列: [i32; 5]
```

### Clone トレイト

```rust
// Cloneは明示的に呼ぶ
let s1 = String::from("hello");
let s2 = s1.clone();  // 明示的にクローン
println!("{} {}", s1, s2);  // 両方有効

// Vecも同様
let v1 = vec![1, 2, 3];
let v2 = v1.clone();  // ヒープデータも含めてコピー
```

### Copy を実装できる条件

1. すべてのフィールドが`Copy`を実装している
2. ヒープアロケーションを持たない
3. `Drop`トレイトを実装していない

```rust
// Copy可能な構造体
#[derive(Copy, Clone)]
struct Point {
    x: i32,
    y: i32,
}

// Copy不可能な構造体（Stringがあるため）
#[derive(Clone)]  // Copyは付けられない
struct Person {
    name: String,  // Stringは Copy ではない
    age: u32,
}
```

### 使い分けのガイドライン

```rust
// 小さな値型 → Copy
#[derive(Copy, Clone)]
struct Color(u8, u8, u8);

// ヒープを使う型 → Clone のみ
#[derive(Clone)]
struct Document {
    content: String,
    metadata: Vec<String>,
}

// 使用例
fn process_color(c: Color) {  // Copyなので値渡しOK
    // ...
}

fn process_document(d: &Document) {  // 参照を渡す（効率的）
    // ...
}

fn duplicate_document(d: &Document) -> Document {
    d.clone()  // 必要な時だけclone
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`Copy`と`Clone`…似ているようで全く違うのじゃ。この2つを理解すれば、いつムーブが起こり、いつコピーが起こるかが完璧に分かるようになる」

なぜこれを知っておくと便利か：
- コンパイルエラーの原因がすぐ分かる
- パフォーマンスを意識した設計ができる
- 自作型の設計指針が明確になる

### 説明の流れ

1. **Copyの暗黙性を確認**
   ```rust
   let x = 5;
   let y = x;
   println!("{}", x);  // OK!
   ```

   「`i32`は`Copy`トレイトを実装しているから、代入しても元の変数が使える。これは**暗黙的**に起こるのじゃ」

2. **Cloneの明示性を確認**
   ```rust
   let s1 = String::from("hello");
   // let s2 = s1;  // これはムーブ
   let s2 = s1.clone();  // これはクローン
   println!("{}", s1);  // OK!
   ```

   「`String`は`Copy`ではないが`Clone`は実装している。`.clone()`と明示的に書かないとクローンされないのじゃ」

3. **なぜ分けているのか？**
   「理由は**コスト**じゃ」

   ```rust
   // Copy: 常に安価（数バイトのメモリコピー）
   let point = (100, 200);
   let another = point;  // 16バイトのコピー、一瞬

   // Clone: 高コストの可能性
   let big_vec = vec![0; 1_000_000];
   let another = big_vec.clone();  // 4MBのヒープ確保+コピー
   ```

   「大きなデータを暗黙的にコピーされたら困る。だから明示的に`.clone()`と書かせるのじゃ」

4. **自作型にCopyを実装**
   ```rust
   #[derive(Copy, Clone)]
   struct Point {
       x: i32,
       y: i32,
   }

   fn main() {
       let p1 = Point { x: 1, y: 2 };
       let p2 = p1;
       println!("{:?}", p1);  // OK!
   }
   ```

   「`derive`で簡単に付けられる。ただし、すべてのフィールドが`Copy`でないといけない」

5. **Copyを付けられない場合**
   ```rust
   #[derive(Clone)]  // Copyは付けられない
   struct Person {
       name: String,  // StringはCopyではない
       age: u32,
   }
   ```

   「`String`フィールドがあるため`Copy`は付けられん。`Clone`だけを付けるのじゃ」

### 実践課題（オプション）

1. `Point`構造体を作り、`Copy`を付けて動作確認
2. `String`フィールドを持つ構造体を作り、`Copy`を付けようとしてエラーを確認
3. 大きな`Vec`の`clone()`にかかる時間を測定

## クリア条件（オプション）

理解度チェック：
- [ ] `Copy`と`Clone`の違いを説明できる
- [ ] `Copy`を実装できる条件を説明できる
- [ ] 自作型に`Copy`または`Clone`を適切に付けられる

## 補足情報

### Copy と Clone の関係

```rust
// Copy は Clone のサブトレイト
// pub trait Copy: Clone { }

// つまり Copy を実装する型は必ず Clone も実装する必要がある
#[derive(Copy, Clone)]  // 両方必要
struct Point { x: i32, y: i32 }
```

### パフォーマンス比較

```rust
use std::time::Instant;

fn main() {
    // Copy型（高速）
    let start = Instant::now();
    let mut sum = 0i64;
    for i in 0..10_000_000 {
        let x = i;
        let y = x;  // Copy
        sum += y;
    }
    println!("Copy: {:?}", start.elapsed());

    // Clone型（やや遅い）
    let start = Instant::now();
    let mut results = Vec::new();
    for _ in 0..100_000 {
        let s = String::from("hello");
        let t = s.clone();  // Clone
        results.push(t);
    }
    println!("Clone: {:?}", start.elapsed());
}
```

### 参考リンク

- std::marker::Copy: https://doc.rust-lang.org/std/marker/trait.Copy.html
- std::clone::Clone: https://doc.rust-lang.org/std/clone/trait.Clone.html
- The Rust Book: https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html#ways-variables-and-data-interact-clone
