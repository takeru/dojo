# トピック: forループの詳細

## メタ情報

- **ID**: for-enumerate
- **難易度**: 初級
- **所要時間**: 6-8分（対話形式）/ 3分（読み物）
- **カテゴリ**: 制御フロー

## 前提知識

- Stage 5完了（forの基本）

## このトピックで学べること

- enumerateでインデックスを取得
- イテレータのメソッド
- 参照と所有権

## 関連ステージ

- Stage 5: 制御フロー（ここで登場）

## 要点（ドキュメント形式用）

### enumerate() でインデックス取得

`.enumerate()` を使うとインデックスと値の両方を取得できます。

```rust
let fruits = vec!["apple", "banana", "cherry"];

for (index, fruit) in fruits.iter().enumerate() {
    println!("{}: {}", index, fruit);
}
// 0: apple
// 1: banana
// 2: cherry
```

### iter(), iter_mut(), into_iter()

```rust
let mut numbers = vec![1, 2, 3];

// 参照を取得（元のデータは変更しない）
for n in numbers.iter() {
    println!("{}", n);
}

// 可変参照を取得（元のデータを変更できる）
for n in numbers.iter_mut() {
    *n *= 2;
}

// 所有権を取得（元のデータは使えなくなる）
for n in numbers.into_iter() {
    println!("{}", n);
}
// numbers はもう使えない
```

### 便利なメソッド

```rust
let numbers = vec![1, 2, 3, 4, 5];

// 逆順
for n in numbers.iter().rev() {
    println!("{}", n);  // 5, 4, 3, 2, 1
}

// スキップ
for n in numbers.iter().skip(2) {
    println!("{}", n);  // 3, 4, 5
}

// 最初のn個
for n in numbers.iter().take(3) {
    println!("{}", n);  // 1, 2, 3
}
```

### 範囲ループの詳細

```rust
// 1から4まで（5は含まない）
for i in 1..5 {
    println!("{}", i);  // 1, 2, 3, 4
}

// 1から5まで（5を含む）
for i in 1..=5 {
    println!("{}", i);  // 1, 2, 3, 4, 5
}

// 逆順
for i in (1..5).rev() {
    println!("{}", i);  // 4, 3, 2, 1
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「forループでインデックスも一緒に欲しいときがあるじゃろう？」

### 説明の流れ

1. **enumerateを紹介**
   ```rust
   for (i, v) in vec.iter().enumerate() {
       println!("{}: {}", i, v);
   }
   ```

2. **iter系メソッドの違いを説明**
   「iter()は借用、into_iter()は所有権を取るのじゃ」

3. **便利メソッドを紹介**
   「rev(), skip(), take()など知っておくと便利じゃぞ」

## クリア条件（オプション）

- [ ] enumerate()でインデックスを取得できる
- [ ] iter()とinto_iter()の違いを理解している
- [ ] 範囲ループ（..と..=）を使い分けられる

## 補足情報

### zip()で2つのイテレータを組み合わせ

```rust
let names = vec!["Alice", "Bob"];
let ages = vec![30, 25];

for (name, age) in names.iter().zip(ages.iter()) {
    println!("{}: {}", name, age);
}
```

### 参考リンク

- Rust by Example: https://doc.rust-lang.org/rust-by-example/flow_control/for.html
- Iterator docs: https://doc.rust-lang.org/std/iter/trait.Iterator.html
