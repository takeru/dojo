# トピック: Dropトレイトの仕組み

## メタ情報

- **ID**: drop-mechanism
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: メモリ管理・トレイト

## 前提知識

- Stage 1の所有権の基礎（スコープとドロップの概念）
- トレイトの基本概念

## このトピックで学べること

- `Drop`トレイトの仕組み
- いつ・どの順序でドロップが呼ばれるか
- カスタム`Drop`の実装方法
- RAII（Resource Acquisition Is Initialization）パターン
- `std::mem::drop`の使い方

## 関連ステージ

- Stage 1: 所有権（スコープとドロップの理解を深める）

## 要点（ドキュメント形式用）

### Dropトレイトとは

スコープを抜けるとき自動的に呼ばれるデストラクタです。

```rust
struct MyResource {
    name: String,
}

impl Drop for MyResource {
    fn drop(&mut self) {
        println!("Dropping: {}", self.name);
    }
}

fn main() {
    let r = MyResource { name: String::from("resource1") };
    println!("Created resource");
} // ここで "Dropping: resource1" が出力される
```

### ドロップの順序

**変数は宣言と逆順にドロップされる**:

```rust
fn main() {
    let a = MyResource { name: String::from("A") };
    let b = MyResource { name: String::from("B") };
    let c = MyResource { name: String::from("C") };
}
// 出力:
// Dropping: C
// Dropping: B
// Dropping: A
```

**構造体のフィールドは宣言順にドロップされる**:

```rust
struct Container {
    first: MyResource,   // 1番目にドロップ
    second: MyResource,  // 2番目にドロップ
}
```

### 早期ドロップ: std::mem::drop

スコープ終了前にドロップしたい場合:

```rust
use std::mem::drop;

fn main() {
    let r = MyResource { name: String::from("early") };
    println!("Before drop");
    drop(r);  // 明示的にドロップ
    println!("After drop");
    // println!("{}", r.name);  // エラー: rは既にドロップされた
}
```

**注意**: `drop()`は特別な関数ではなく、単に所有権を取って何もしない関数:

```rust
// std::mem::drop の実装（実質これだけ）
pub fn drop<T>(_x: T) {}
```

### RAIIパターン

リソースの取得と解放を所有権に紐付ける:

```rust
use std::fs::File;
use std::io::Write;

fn write_file() -> std::io::Result<()> {
    let mut file = File::create("test.txt")?;
    file.write_all(b"Hello")?;
    Ok(())
}  // fileはここで自動的に閉じられる（Dropで）
```

### Dropの制約

**CopyとDropは両立できない**:

```rust
// これはエラー
#[derive(Copy, Clone)]
struct Bad {
    data: i32,
}

impl Drop for Bad {
    fn drop(&mut self) {}
}
// error: the trait `Copy` may not be implemented for this type
// because the type has a destructor
```

理由: Copyはビットコピーで複製されるため、どのコピーがDropを呼ぶべきか曖昧になる。

## 対話形式の教え方ガイド（先生用）

### 導入

「Rustではスコープを抜けると値が自動的に破棄される。これを制御するのが`Drop`トレイトじゃ。C++のデストラクタ、Javaのfinalizerに相当するが、Rustの方が遥かに予測可能じゃ」

なぜこれを知っておくと便利か：
- リソース管理（ファイル、ネットワーク接続など）が理解できる
- メモリリークを防げる
- 「いつ解放されるか」が明確になる

### 説明の流れ

1. **自動ドロップを確認**
   ```rust
   struct Noisy {
       name: String,
   }

   impl Drop for Noisy {
       fn drop(&mut self) {
           println!("Bye, {}!", self.name);
       }
   }

   fn main() {
       let a = Noisy { name: String::from("Alice") };
       println!("Hello!");
   }
   ```

   「実行すると、"Hello!" の後に "Bye, Alice!" と出る。スコープ終了時に自動で呼ばれるのじゃ」

2. **ドロップ順序を確認**
   ```rust
   fn main() {
       let x = Noisy { name: String::from("first") };
       let y = Noisy { name: String::from("second") };
       println!("In scope");
   }
   // second が先にドロップされる（スタックのLIFO順）
   ```

3. **早期ドロップの使い方**
   ```rust
   use std::mem::drop;

   fn main() {
       let lock = acquire_lock();
       // クリティカルセクション
       drop(lock);  // 早めにロックを解放
       // ロック不要な処理
   }
   ```

   「ロックを早めに解放したいとき、`drop()`を使うのじゃ」

4. **RAIIの威力を体感**
   ```rust
   use std::fs::File;

   fn risky_operation() -> Result<(), std::io::Error> {
       let file = File::create("important.txt")?;
       might_fail()?;  // エラーが起きても...
       Ok(())
   }  // fileは確実に閉じられる！
   ```

   「エラーが起きても、パニックしても、Dropは必ず呼ばれる。リソースリークを防げるのじゃ」

### 実践課題（オプション）

1. カスタム`Drop`を実装してドロップ順序を確認
2. `std::mem::drop`で早期解放を試す
3. ファイルハンドルが自動で閉じられることを確認

## クリア条件（オプション）

理解度チェック：
- [ ] `Drop`トレイトの役割を説明できる
- [ ] ドロップの順序を予測できる
- [ ] `std::mem::drop`の使いどころが分かる
- [ ] RAIIパターンのメリットを説明できる

## 補足情報

### ManuallyDropでドロップを抑制

```rust
use std::mem::ManuallyDrop;

fn main() {
    let x = ManuallyDrop::new(String::from("hello"));
    // スコープを抜けてもドロップされない
    // 手動で ManuallyDrop::drop(&mut x) を呼ぶ必要がある
}
```

**注意**: メモリリークの原因になるので、特殊な場面でのみ使用。

### std::mem::forgetとの違い

```rust
use std::mem::forget;

fn main() {
    let s = String::from("hello");
    forget(s);  // Dropを呼ばずに所有権を放棄（メモリリーク）
}
```

`forget`はDropを完全にスキップする（unsafe相当の操作）。

### 参考リンク

- std::ops::Drop: https://doc.rust-lang.org/std/ops/trait.Drop.html
- The Rust Book: https://doc.rust-lang.org/book/ch15-03-drop.html
- Rustonomicon: https://doc.rust-lang.org/nomicon/destructors.html
