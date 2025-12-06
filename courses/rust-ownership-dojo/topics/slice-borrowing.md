# トピック: スライスと借用

## メタ情報

- **ID**: slice-borrowing
- **難易度**: 初級〜中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 借用・型

## 前提知識

- Stage 2の参照と借用
- 配列と`Vec`の基本

## このトピックで学べること

- スライス（`&[T]`）とは何か
- スライスと借用の関係
- 配列、Vec、Stringからスライスを作る方法
- スライスを使った効率的なAPI設計

## 関連ステージ

- Stage 2: 借用

## 要点（ドキュメント形式用）

### スライスとは

スライスは、連続したメモリ領域への**借用**です。データを所有せず、既存データの一部を参照します。

```rust
let arr = [1, 2, 3, 4, 5];
let slice: &[i32] = &arr[1..4];  // [2, 3, 4] への参照
// arr はまだ有効
```

### メモリレイアウト

```
配列 [1, 2, 3, 4, 5]
     ↑     ↑
     │     └── len = 3
     └── ptr

スライス &[i32]（16バイト）
┌─────────────────┐
│ ptr ────────────┼───→ 2 の位置
│ len = 3         │
└─────────────────┘
```

### スライスの作り方

```rust
// 配列から
let arr = [1, 2, 3, 4, 5];
let slice1: &[i32] = &arr;        // 全体
let slice2: &[i32] = &arr[1..4];  // インデックス1〜3

// Vecから
let v = vec![1, 2, 3, 4, 5];
let slice3: &[i32] = &v;          // 全体
let slice4: &[i32] = &v[..3];     // 最初から3つ

// Stringから（文字列スライス）
let s = String::from("hello");
let slice5: &str = &s;            // 全体
let slice6: &str = &s[0..2];      // "he"
```

### スライスは借用

```rust
let mut v = vec![1, 2, 3];
let slice = &v[..];

// v.push(4);  // エラー: slice がまだ借用中
println!("{:?}", slice);

// slice のスコープ終了後
v.push(4);  // OK
```

### 可変スライス

```rust
let mut arr = [1, 2, 3, 4, 5];
let slice: &mut [i32] = &mut arr[1..4];

slice[0] = 20;  // arr[1] が 20 になる
slice[1] = 30;  // arr[2] が 30 になる

println!("{:?}", arr);  // [1, 20, 30, 4, 5]
```

### スライスを使ったAPI設計

```rust
// ❌ Vec専用
fn sum_vec(v: &Vec<i32>) -> i32 {
    v.iter().sum()
}

// ✅ スライスを受け取る（配列もVecも受け付ける）
fn sum_slice(s: &[i32]) -> i32 {
    s.iter().sum()
}

fn main() {
    let arr = [1, 2, 3];
    let vec = vec![4, 5, 6];

    // sum_vec(&arr);  // エラー
    sum_slice(&arr);   // OK
    sum_slice(&vec);   // OK（Vecから&[T]へ自動変換）
}
```

### 文字列スライス &str

```rust
// &str は &[u8] の特殊版（UTF-8保証付き）
let s = String::from("hello, 世界");

let hello: &str = &s[0..5];   // "hello"
// let bad: &str = &s[0..8];  // パニック: UTF-8の境界でない

// バイト単位より文字単位で扱う方が安全
for c in s.chars() {
    println!("{}", c);
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「スライスは、データの**一部を借りる**仕組みじゃ。コピーせずにデータを参照できるから、とても効率的」

なぜこれを知っておくと便利か：
- 柔軟なAPI設計ができる
- 不必要なコピーを避けられる
- 文字列処理が効率的になる

### 説明の流れ

1. **スライスの基本**
   ```rust
   let arr = [10, 20, 30, 40, 50];

   let all: &[i32] = &arr;       // [10, 20, 30, 40, 50]
   let part: &[i32] = &arr[1..4]; // [20, 30, 40]
   let from: &[i32] = &arr[2..];  // [30, 40, 50]
   let to: &[i32] = &arr[..3];    // [10, 20, 30]
   ```

2. **借用であることを確認**
   ```rust
   let mut v = vec![1, 2, 3];
   let slice = &v[..];

   // v.push(4);  // エラー！
   println!("{:?}", slice);
   ```

   「スライスは借用じゃ。借りている間は元を変更できん」

3. **APIでの活用**
   ```rust
   // スライスを受け取る関数は柔軟
   fn first_element(s: &[i32]) -> Option<&i32> {
       s.first()
   }

   let arr = [1, 2, 3];
   let vec = vec![4, 5, 6];

   first_element(&arr);  // 配列OK
   first_element(&vec);  // VecもOK
   ```

4. **文字列スライスの注意点**
   ```rust
   let s = "こんにちは";
   // let bad = &s[0..2];  // パニック！「こ」は3バイト

   // 文字境界で切る
   let good = &s[0..3];  // "こ"
   ```

### 実践課題（オプション）

1. 配列とVec両方を受け取れる関数を書く
2. 文字列の一部を取り出す
3. 可変スライスで配列の一部を変更する

## クリア条件（オプション）

理解度チェック：
- [ ] スライスが借用であることを説明できる
- [ ] `&[T]`を受け取る関数のメリットを説明できる
- [ ] 文字列スライスの注意点を理解している

## 補足情報

### split_at: スライスを分割

```rust
let arr = [1, 2, 3, 4, 5];
let (left, right) = arr.split_at(2);
// left = [1, 2]
// right = [3, 4, 5]
```

### chunks: 固定サイズで分割

```rust
let arr = [1, 2, 3, 4, 5];
for chunk in arr.chunks(2) {
    println!("{:?}", chunk);
}
// [1, 2]
// [3, 4]
// [5]
```

### windows: スライディングウィンドウ

```rust
let arr = [1, 2, 3, 4, 5];
for window in arr.windows(3) {
    println!("{:?}", window);
}
// [1, 2, 3]
// [2, 3, 4]
// [3, 4, 5]
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch04-03-slices.html
- std::slice: https://doc.rust-lang.org/std/slice/index.html
