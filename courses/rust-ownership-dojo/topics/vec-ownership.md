# トピック: Vecと所有権

## メタ情報

- **ID**: vec-ownership
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: コレクション・所有権

## 前提知識

- Stage 1の所有権の基礎
- Stage 2の借用
- `Vec<T>`の基本的な使い方

## このトピックで学べること

- `Vec`と要素の所有権の関係
- イテレータと所有権（`iter()`, `iter_mut()`, `into_iter()`）
- 要素の取り出しパターン
- コレクション操作での所有権の扱い

## 関連ステージ

- Stage 1: 所有権（コレクションへの応用）
- Stage 2: 借用（イテレータでの借用）

## 要点（ドキュメント形式用）

### Vecは要素を所有する

```rust
let mut v = Vec::new();
let s = String::from("hello");
v.push(s);  // sの所有権がvにムーブ
// println!("{}", s);  // エラー: sはムーブ済み
```

### 3種類のイテレータ

```rust
let v = vec![String::from("a"), String::from("b")];

// 1. iter(): 不変参照（&T）
for s in v.iter() {
    println!("{}", s);  // sは&String
}
// vはまだ使える

// 2. iter_mut(): 可変参照（&mut T）
let mut v = vec![String::from("a"), String::from("b")];
for s in v.iter_mut() {
    s.push_str("!");  // 変更可能
}

// 3. into_iter(): 所有権を消費（T）
let v = vec![String::from("a"), String::from("b")];
for s in v.into_iter() {
    println!("{}", s);  // sはString（所有）
}
// vはもう使えない
```

### forループの糖衣構文

```rust
let v = vec![1, 2, 3];

// これは
for x in &v { ... }
// これと同じ
for x in v.iter() { ... }

// これは
for x in &mut v { ... }
// これと同じ
for x in v.iter_mut() { ... }

// これは
for x in v { ... }
// これと同じ
for x in v.into_iter() { ... }
```

### 要素の取り出し

```rust
let mut v = vec![String::from("a"), String::from("b"), String::from("c")];

// 借用で取得
let first: &String = &v[0];
let first: Option<&String> = v.get(0);

// 所有権を取得（消費）
let last: Option<String> = v.pop();  // 最後の要素を取り出し
let first: String = v.remove(0);     // インデックス指定で取り出し
let swapped: String = v.swap_remove(0);  // 最後の要素と入れ替えて取り出し（高速）
```

### インデックスアクセスの注意

```rust
let v = vec![String::from("hello")];

// ❌ これはできない
let s = v[0];  // エラー: cannot move out of index

// ✅ 借用する
let s: &String = &v[0];

// ✅ クローンする
let s: String = v[0].clone();

// ✅ 取り出す（Vecから削除）
let mut v = vec![String::from("hello")];
let s: String = v.remove(0);
```

### イテレータとmapでの所有権

```rust
let v = vec![String::from("hello"), String::from("world")];

// 参照のまま変換
let lengths: Vec<usize> = v.iter().map(|s| s.len()).collect();
// vはまだ使える

// 所有権を消費して変換
let uppercase: Vec<String> = v.into_iter()
    .map(|s| s.to_uppercase())
    .collect();
// vはもう使えない
```

## 対話形式の教え方ガイド（先生用）

### 導入

「`Vec`は要素を**所有**する。これが分かれば、イテレータの挙動も、要素の取り出しも、すべて理解できるようになる」

なぜこれを知っておくと便利か：
- イテレータの選択を間違えない
- コレクション操作でコンパイルエラーを避けられる
- 効率的なデータ処理ができる

### 説明の流れ

1. **Vecが所有することを確認**
   ```rust
   let s = String::from("hello");
   let mut v = Vec::new();
   v.push(s);
   // println!("{}", s);  // エラー！
   ```

   「`push`した時点で`s`の所有権は`v`に移った。`s`はもう使えん」

2. **3種類のイテレータを比較**
   ```rust
   let v = vec![String::from("a"), String::from("b")];

   // 借用するだけ
   for s in &v {
       println!("{}", s);
   }
   println!("v is still valid: {:?}", v);

   // 所有権を消費
   for s in v {
       println!("{}", s);
   }
   // println!("{:?}", v);  // エラー！
   ```

3. **使い分けの指針**
   「読むだけなら`&v`（iter）。変更するなら`&mut v`（iter_mut）。消費するなら`v`（into_iter）じゃ」

4. **要素の取り出しパターン**
   ```rust
   let mut v = vec![String::from("a"), String::from("b")];

   // 最後から取り出し（O(1)）
   if let Some(last) = v.pop() {
       println!("Got: {}", last);
   }

   // 指定位置から取り出し（O(n)）
   let first = v.remove(0);
   ```

### 実践課題（オプション）

1. `iter()`, `iter_mut()`, `into_iter()`をそれぞれ使って動作確認
2. `map`と`collect`で新しい`Vec`を作る
3. `pop`と`remove`の違いを確認

## クリア条件（オプション）

理解度チェック：
- [ ] `Vec`が要素を所有することを説明できる
- [ ] 3種類のイテレータの違いを説明できる
- [ ] forループでの`&v`, `&mut v`, `v`の違いが分かる

## 補足情報

### drain: 範囲を取り出す

```rust
let mut v = vec![1, 2, 3, 4, 5];
let drained: Vec<i32> = v.drain(1..4).collect();
// drained = [2, 3, 4]
// v = [1, 5]
```

### splitパターン

```rust
let mut v = vec![1, 2, 3, 4, 5];
let second_half = v.split_off(3);
// v = [1, 2, 3]
// second_half = [4, 5]
```

### retain: 条件で残す

```rust
let mut v = vec![1, 2, 3, 4, 5];
v.retain(|&x| x % 2 == 0);
// v = [2, 4]
```

### 参考リンク

- std::vec::Vec: https://doc.rust-lang.org/std/vec/struct.Vec.html
- Iterator trait: https://doc.rust-lang.org/std/iter/trait.Iterator.html
- The Rust Book: https://doc.rust-lang.org/book/ch08-01-vectors.html
