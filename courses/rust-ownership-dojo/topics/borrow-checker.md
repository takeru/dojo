# トピック: 借用チェッカーの仕組み

## メタ情報

- **ID**: borrow-checker
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: コンパイラ・借用

## 前提知識

- Stage 2の借用の基本
- 不変参照と可変参照のルール

## このトピックで学べること

- 借用チェッカーが何をしているか
- 非レキシカルライフタイム（NLL）の仕組み
- なぜRustは安全なのか
- 借用チェッカーの限界

## 関連ステージ

- Stage 2: 借用

## 要点（ドキュメント形式用）

### 借用チェッカーとは

Rustコンパイラの一部で、以下をコンパイル時に検証します：

1. 参照がダングリングしないこと
2. 可変参照が同時に複数存在しないこと
3. 可変参照と不変参照が同時に存在しないこと

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;      // OK: 不変参照
    let r2 = &s;      // OK: 複数の不変参照
    // let r3 = &mut s;  // エラー: 不変参照がある間は可変参照不可

    println!("{} {}", r1, r2);
}
```

### 借用ルールの確認

```rust
// ルール1: 参照は常に有効
fn dangling() -> &String {
    let s = String::from("hello");
    &s  // エラー: sは関数終了時にドロップ
}

// ルール2: 複数の不変参照 OR 1つの可変参照
fn multiple() {
    let mut s = String::from("hello");

    // OK: 複数の不変参照
    let r1 = &s;
    let r2 = &s;

    // OK: 1つの可変参照（不変参照が終わった後）
    let r3 = &mut s;
}
```

### 非レキシカルライフタイム（NLL）

Rust 2018以降、参照のスコープは「最後に使われた場所」まで。

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;
    let r2 = &s;
    println!("{} {}", r1, r2);
    // r1とr2はここで終わり（最後の使用）

    let r3 = &mut s;  // OK! r1/r2はもう使われていない
    r3.push_str(" world");
}
```

**NLL以前（Rust 2015）**:
```rust
// 昔はこれがエラーだった
let r1 = &s;
println!("{}", r1);
let r2 = &mut s;  // エラー: r1のレキシカルスコープ内
```

### 借用チェッカーが見ているもの

```
1. 各変数のライフタイム
2. 参照の作成と使用
3. ムーブの発生
4. 参照間の競合

時間軸 →
s: [所有開始]────────────────[ドロップ]
r1:     [借用]────[最後の使用]
r2:             [借用]───[最後の使用]
r3:                           [借用]───[使用]
```

### 借用チェッカーの限界

借用チェッカーは保守的です。安全なコードでもエラーになることがあります。

```rust
fn get_first_or_second(v: &mut Vec<i32>, use_first: bool) -> &mut i32 {
    if use_first {
        &mut v[0]
    } else {
        &mut v[1]
    }
    // これはOK
}

fn get_both(v: &mut Vec<i32>) -> (&mut i32, &mut i32) {
    (&mut v[0], &mut v[1])
    // エラー: 借用チェッカーは同じvecへの2つの可変参照を許可しない
    // 実際には異なる要素なので安全だが...
}

// 解決策: split_at_mutを使う
fn get_both_safe(v: &mut Vec<i32>) -> (&mut i32, &mut i32) {
    let (left, right) = v.split_at_mut(1);
    (&mut left[0], &mut right[0])
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「借用チェッカーは、Rustの安全性を支える番人じゃ。コンパイル時にメモリ安全性を保証してくれる。その仕組みを理解しよう」

なぜこれを知っておくと便利か：
- エラーメッセージが理解できる
- なぜRustが安全かが分かる
- 限界を知ることで回避策を選べる

### 説明の流れ

1. **基本ルールを確認**
   ```rust
   let mut s = String::from("hello");
   let r1 = &s;
   let r2 = &mut s;  // エラー！
   ```

   「不変参照があるときに可変参照を作ろうとすると、借用チェッカーが止めてくれる」

2. **NLLの動作を見せる**
   ```rust
   let mut s = String::from("hello");
   let r1 = &s;
   println!("{}", r1);  // r1の最後の使用
   let r2 = &mut s;     // OK!
   ```

   「r1が最後に使われた後なら、可変参照を作れる。これがNLLの恩恵じゃ」

3. **安全性の保証を説明**
   - データ競合がない
   - ダングリングポインタがない
   - 二重解放がない

4. **限界を認識させる**
   「借用チェッカーは完璧ではない。安全なコードでも拒否されることがある。そんなときは回避策を使う」

### 実践課題（オプション）

1. 借用チェッカーエラーを起こして観察
2. NLLの動作を確認
3. `split_at_mut`で回避策を試す

## クリア条件（オプション）

理解度チェック：
- [ ] 借用チェッカーの役割を説明できる
- [ ] NLLの仕組みを理解している
- [ ] 借用チェッカーの限界を知っている

## 補足情報

### Polonius（次世代借用チェッカー）

より賢い借用チェッカーが開発中:

```bash
# 実験的に有効化
RUSTFLAGS="-Z polonius" cargo build
```

現在の借用チェッカーでエラーになるコードの一部が通るようになる。

### unsafeによるエスケープ

どうしても借用チェッカーを回避したい場合:

```rust
unsafe {
    // 安全性は自分で保証する必要がある
}
```

通常は避けるべき。

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
- RFC 2094 (NLL): https://rust-lang.github.io/rfcs/2094-nll.html
- Polonius: https://github.com/rust-lang/polonius
