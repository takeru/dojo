# トピック: メソッドのself引数パターン

## メタ情報

- **ID**: method-self-variants
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: メソッド・所有権

## 前提知識

- Stage 1の所有権の基礎
- Stage 2の借用
- 構造体とメソッドの基本

## このトピックで学べること

- `self`, `&self`, `&mut self`の違い
- `Box<Self>`や`Rc<Self>`を受け取るメソッド
- 各パターンの使いどころ
- メソッドシグネチャの設計指針

## 関連ステージ

- Stage 2: 借用（メソッドでの借用）

## 要点（ドキュメント形式用）

### 4つの基本パターン

```rust
struct Counter {
    value: i32,
}

impl Counter {
    // 1. self: 所有権を取る（消費する）
    fn into_value(self) -> i32 {
        self.value
    }

    // 2. &self: 不変借用（読み取り）
    fn get(&self) -> i32 {
        self.value
    }

    // 3. &mut self: 可変借用（変更）
    fn increment(&mut self) {
        self.value += 1;
    }

    // 4. 関連関数（selfなし）
    fn new() -> Self {
        Counter { value: 0 }
    }
}
```

### 各パターンの意味

| パターン | 所有権 | 変更 | 呼び出し後 |
|----------|--------|------|------------|
| `self` | 取る | 可 | 使用不可 |
| `&self` | 借用 | 不可 | 使用可 |
| `&mut self` | 可変借用 | 可 | 使用可 |

### self（所有権を取る）

```rust
impl Counter {
    fn destroy(self) {
        println!("Destroying counter with value: {}", self.value);
    }

    fn into_inner(self) -> i32 {
        self.value
    }
}

fn main() {
    let c = Counter::new();
    let value = c.into_inner();  // c は消費される
    // println!("{}", c.value);  // エラー: c はムーブ済み
}
```

**使いどころ**:
- リソースを解放する（ファイルを閉じる等）
- 型変換（`into_xxx`メソッド）
- ビルダーパターンの最終ステップ

### &self（不変借用）

```rust
impl Counter {
    fn get(&self) -> i32 {
        self.value
    }

    fn is_positive(&self) -> bool {
        self.value > 0
    }
}

fn main() {
    let c = Counter { value: 5 };
    println!("{}", c.get());
    println!("{}", c.is_positive());
    println!("{}", c.get());  // 何度でも呼べる
}
```

**使いどころ**:
- 値を読み取るだけ
- 状態を変えない計算

### &mut self（可変借用）

```rust
impl Counter {
    fn increment(&mut self) {
        self.value += 1;
    }

    fn reset(&mut self) {
        self.value = 0;
    }
}

fn main() {
    let mut c = Counter { value: 0 };
    c.increment();
    c.increment();
    println!("{}", c.get());  // 2
}
```

**使いどころ**:
- 内部状態を変更する
- 可変操作が必要だが所有権は不要

### Box<Self>やRc<Self>を受け取る

```rust
use std::rc::Rc;

impl Counter {
    // Box経由で所有権を取る
    fn from_box(boxed: Box<Self>) -> Self {
        *boxed
    }

    // Rc経由で共有参照
    fn shared_get(this: &Rc<Self>) -> i32 {
        this.value
    }
}
```

### メソッドチェーン

```rust
struct Builder {
    name: String,
    value: i32,
}

impl Builder {
    fn new() -> Self {
        Builder { name: String::new(), value: 0 }
    }

    // &mut selfを返すパターン
    fn name(&mut self, name: &str) -> &mut Self {
        self.name = name.to_string();
        self
    }

    fn value(&mut self, value: i32) -> &mut Self {
        self.value = value;
        self
    }

    // 最後にselfを消費
    fn build(self) -> String {
        format!("{}: {}", self.name, self.value)
    }
}

fn main() {
    let result = Builder::new()
        .name("counter")
        .value(42)
        .build();
    println!("{}", result);  // "counter: 42"
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「メソッドの最初の引数`self`には、実は3つの形がある。`self`, `&self`, `&mut self`…この違いが分かれば、Rustのメソッドは完璧に理解できる」

なぜこれを知っておくと便利か：
- メソッドの設計ができる
- APIの使い方が直感的になる
- エラーメッセージが理解しやすくなる

### 説明の流れ

1. **3つのパターンを並べる**
   ```rust
   impl Counter {
       fn consume(self) {}      // 所有権を取る
       fn read(&self) {}        // 借りる
       fn modify(&mut self) {}  // 可変で借りる
   }
   ```

2. **呼び出し後の違いを見せる**
   ```rust
   let mut c = Counter::new();

   c.read();   // OK
   c.read();   // OK（何度でも）

   c.modify(); // OK
   c.read();   // OK（まだ使える）

   c.consume(); // OK
   // c.read();  // エラー！cは消費された
   ```

3. **命名規則のヒント**
   - `into_xxx`: `self`を取る（`into_string()`, `into_inner()`）
   - `as_xxx`: `&self`を取る（`as_str()`, `as_bytes()`）
   - 動詞（`push`, `pop`等）: 状態を変える可能性

4. **設計の判断基準**
   「読むだけなら`&self`。変更するなら`&mut self`。消費するなら`self`。シンプルじゃ」

### 実践課題（オプション）

1. 3つのパターンを使った構造体を設計する
2. ビルダーパターンを実装する
3. `into_xxx`メソッドを作る

## クリア条件（オプション）

理解度チェック：
- [ ] `self`, `&self`, `&mut self`の違いを説明できる
- [ ] 各パターンの使いどころを説明できる
- [ ] メソッドシグネチャを見て所有権の動きを予測できる

## 補足情報

### selfの省略形

```rust
impl Counter {
    // これらは同じ意味
    fn get(&self) -> i32 { self.value }
    fn get(self: &Self) -> i32 { self.value }
    fn get(self: &Counter) -> i32 { self.value }
}
```

### Pin<&mut Self>

非同期プログラミングで使う特殊なパターン:

```rust
use std::pin::Pin;

impl MyFuture {
    fn poll(self: Pin<&mut Self>) {
        // 自己参照型の安全な操作
    }
}
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-03-method-syntax.html
- Rust By Example: https://doc.rust-lang.org/rust-by-example/fn/methods.html
