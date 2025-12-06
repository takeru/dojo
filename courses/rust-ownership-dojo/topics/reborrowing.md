# トピック: 再借用のパターン

## メタ情報

- **ID**: reborrowing
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 借用・コンパイラの挙動

## 前提知識

- Stage 2の借用ルール
- 不変参照と可変参照の違い

## このトピックで学べること

- 再借用（reborrowing）とは何か
- コンパイラによる暗黙の再借用
- 可変参照から不変参照を作る方法
- 再借用が必要な場面

## 関連ステージ

- Stage 2: 借用

## 要点（ドキュメント形式用）

### 再借用とは

可変参照から新たに参照を作ること。元の可変参照は一時的に「凍結」されます。

```rust
fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // 再借用：&mut から &mut を作る
    let r2 = &mut *r;  // 明示的な再借用
    r2.push_str(" world");

    // r は r2 が終わるまで使えない
    println!("{}", r);  // OK: r2 のスコープ終了後
}
```

### 暗黙の再借用

コンパイラは多くの場面で自動的に再借用を行います:

```rust
fn takes_ref(s: &mut String) {
    s.push_str("!");
}

fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // 暗黙の再借用が起こる
    takes_ref(r);  // 実際は takes_ref(&mut *r)
    takes_ref(r);  // 何度でも呼べる！

    println!("{}", r);  // r はまだ有効
}
```

**再借用がなかったら**:
```rust
// もし再借用がなければ...
takes_ref(r);  // r がムーブされる
// takes_ref(r);  // エラー: r は既にムーブ済み
```

### 可変参照から不変参照へ

```rust
fn main() {
    let mut s = String::from("hello");
    let r: &mut String = &mut s;

    // &mut から & を作る（再借用）
    let r2: &String = &*r;  // または単に &r でもOK
    println!("{}", r2);

    // r2 のスコープが終わるまで r は使えない
    // r.push_str("!");  // エラー

    drop(r2);
    r.push_str("!");  // OK
}
```

### 再借用が必要な場面

**1. 関数を複数回呼ぶ**

```rust
fn process(data: &mut Vec<i32>) {
    data.push(1);
}

fn main() {
    let mut v = vec![];
    let r = &mut v;

    process(r);  // 暗黙の再借用
    process(r);  // 何度でもOK
    process(r);
}
```

**2. 一時的に不変参照が必要**

```rust
fn main() {
    let mut data = vec![1, 2, 3];
    let r = &mut data;

    // 長さを確認（不変参照でOK）
    let len = r.len();  // 暗黙的に &*r

    // その後変更
    r.push(4);
}
```

**3. メソッドチェーン**

```rust
fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // 各メソッド呼び出しで暗黙の再借用
    r.push_str(" ");
    r.push_str("world");
    r.push('!');

    println!("{}", r);
}
```

### 明示的な再借用が必要な場面

```rust
fn takes_ownership(r: &mut String) {
    // ...
}

fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // クロージャに渡す場合など、明示的な再借用が必要なことがある
    let closure = || {
        let r2 = &mut *r;  // 明示的な再借用
        r2.push_str("!");
    };
    closure();
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「可変参照を関数に渡したら、普通はムーブされて二度と使えないはず…でもRustでは何度でも渡せる。それは**再借用**という魔法のおかげじゃ」

なぜこれを知っておくと便利か：
- なぜ`&mut`を何度も渡せるのか理解できる
- コンパイラの暗黙の動作が分かる
- エラーメッセージが読みやすくなる

### 説明の流れ

1. **問題提起**
   ```rust
   fn process(r: &mut String) {
       r.push('!');
   }

   fn main() {
       let mut s = String::from("hello");
       let r = &mut s;

       process(r);
       process(r);  // なぜこれが動く？
   }
   ```

   「`&mut String`は`Copy`じゃない。なのに`r`を2回渡せる。なぜじゃ？」

2. **再借用の仕組み**
   「実は、コンパイラが`process(&mut *r)`に変換しているのじゃ。`r`をデリファレンスして新しい`&mut`を作る。これが再借用じゃ」

3. **凍結の概念**
   ```rust
   let mut s = String::from("hello");
   let r = &mut s;

   let r2 = &*r;  // 不変の再借用
   // r.push('!');  // エラー: r は凍結中
   println!("{}", r2);
   // r は再び使える
   r.push('!');
   ```

4. **なぜ必要か**
   「再借用がなければ、可変参照を一度でも関数に渡したら終わり。使い物にならん。だからコンパイラが自動で再借用してくれるのじゃ」

### 実践課題（オプション）

1. 可変参照を複数の関数に順番に渡す
2. `&mut`から`&`への変換を試す
3. 再借用が起こっている箇所を意識的に見つける

## クリア条件（オプション）

理解度チェック：
- [ ] 再借用の概念を説明できる
- [ ] なぜ`&mut`を複数回渡せるのか説明できる
- [ ] 可変参照の「凍結」を理解している

## 補足情報

### Deref coercion との組み合わせ

```rust
fn takes_str(s: &str) {}

fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // &mut String → &String → &str（Deref coercion）
    takes_str(r);
}
```

### 再借用のライフタイム

```rust
fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    {
        let r2 = &mut *r;  // 再借用
        r2.push('!');
    }  // r2 のライフタイム終了

    r.push('?');  // r は再び使用可能
}
```

### 参考リンク

- Rustonomicon - Reborrowing: https://doc.rust-lang.org/nomicon/borrow-splitting.html
- Rust Reference - Coercions: https://doc.rust-lang.org/reference/type-coercions.html
