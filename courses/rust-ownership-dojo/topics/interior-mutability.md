# トピック: 内部可変性（RefCell/Cell）

## メタ情報

- **ID**: interior-mutability
- **難易度**: 中級〜上級
- **所要時間**: 15-20分（対話形式）/ 7分（読み物）
- **カテゴリ**: スマートポインタ・借用

## 前提知識

- Stage 2の借用ルール（不変/可変参照の制約）
- 借用チェッカーの仕組み

## このトピックで学べること

- 内部可変性（Interior Mutability）の概念
- `Cell<T>`と`RefCell<T>`の使い分け
- 実行時借用チェックの仕組み
- `Rc<RefCell<T>>`パターン

## 関連ステージ

- Stage 2: 借用（借用ルールを「緩める」方法）

## 要点（ドキュメント形式用）

### 内部可変性とは

通常、`&T`（不変参照）経由では値を変更できません。しかし「内部可変性」を持つ型は、`&self`経由でも内部状態を変更できます。

```rust
// 通常の借用ルール
let x = 5;
let r = &x;
// *r = 6;  // エラー: 不変参照経由で変更不可

// 内部可変性
use std::cell::Cell;
let c = Cell::new(5);
let r = &c;
c.set(6);  // OK! &経由でも変更できる
```

### Cell<T>: Copyな型向け

```rust
use std::cell::Cell;

struct Counter {
    count: Cell<i32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }

    fn increment(&self) {  // &selfでも変更可能
        self.count.set(self.count.get() + 1);
    }

    fn get(&self) -> i32 {
        self.count.get()
    }
}

fn main() {
    let counter = Counter::new();
    counter.increment();
    counter.increment();
    println!("Count: {}", counter.get());  // 2
}
```

**特徴**:
- `Copy`を実装した型のみ
- 値をget/setで操作
- 参照は取得できない
- オーバーヘッドなし

### RefCell<T>: 任意の型向け

```rust
use std::cell::RefCell;

struct Document {
    content: RefCell<String>,
}

impl Document {
    fn new() -> Self {
        Document { content: RefCell::new(String::new()) }
    }

    fn append(&self, text: &str) {  // &selfでも変更可能
        self.content.borrow_mut().push_str(text);
    }

    fn read(&self) -> String {
        self.content.borrow().clone()
    }
}

fn main() {
    let doc = Document::new();
    doc.append("Hello");
    doc.append(" World");
    println!("{}", doc.read());  // Hello World
}
```

**特徴**:
- 任意の型に使える
- `borrow()`で不変参照、`borrow_mut()`で可変参照
- **実行時に借用ルールをチェック**
- ルール違反でパニック

### 実行時借用チェック

```rust
use std::cell::RefCell;

fn main() {
    let c = RefCell::new(5);

    let r1 = c.borrow();      // 不変参照
    let r2 = c.borrow();      // OK: 複数の不変参照
    // let r3 = c.borrow_mut();  // パニック! 不変参照がある間は可変参照不可

    drop(r1);
    drop(r2);
    let r3 = c.borrow_mut();  // OK: 不変参照が無くなった
}
```

### CellとRefCellの選択

| 特徴 | Cell<T> | RefCell<T> |
|------|---------|------------|
| 型の制約 | T: Copy | なし |
| 参照取得 | 不可（値のget/set） | 可能（borrow/borrow_mut） |
| チェック | なし | 実行時 |
| パニックの可能性 | なし | あり |
| 用途 | 単純なフラグ、カウンタ | 複雑なデータ構造 |

### Rc<RefCell<T>>パターン

複数箇所から共有かつ変更可能にする:

```rust
use std::cell::RefCell;
use std::rc::Rc;

struct Node {
    value: i32,
    children: RefCell<Vec<Rc<Node>>>,
}

fn main() {
    let root = Rc::new(Node {
        value: 1,
        children: RefCell::new(vec![]),
    });

    let child = Rc::new(Node {
        value: 2,
        children: RefCell::new(vec![]),
    });

    // 不変参照（Rc）経由で子を追加
    root.children.borrow_mut().push(Rc::clone(&child));
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「借用ルールは厳格じゃ。しかし時には、不変参照しかない状況でも値を変えたいことがある。そんなときに使うのが『内部可変性』じゃ」

なぜこれを知っておくと便利か：
- キャッシュの実装
- グラフ構造の構築
- イベントリスナーパターン

### 説明の流れ

1. **問題を見せる**
   ```rust
   struct Cache {
       data: Option<String>,
   }

   impl Cache {
       fn get(&self) -> &str {
           if self.data.is_none() {
               // self.data = Some(...);  // エラー! &selfでは変更不可
           }
           self.data.as_ref().unwrap()
       }
   }
   ```

   「キャッシュは読み取り時に初期化したい。でも`&self`では変更できん」

2. **RefCellで解決**
   ```rust
   use std::cell::RefCell;

   struct Cache {
       data: RefCell<Option<String>>,
   }

   impl Cache {
       fn get(&self) -> String {
           let mut data = self.data.borrow_mut();
           if data.is_none() {
               *data = Some(String::from("cached value"));
           }
           data.clone().unwrap()
       }
   }
   ```

3. **実行時チェックの危険性**
   ```rust
   let c = RefCell::new(5);
   let r1 = c.borrow_mut();
   let r2 = c.borrow_mut();  // パニック!
   ```

   「コンパイル時ではなく**実行時**にチェックされる。ルール違反はパニックじゃ」

4. **try_borrowで安全に**
   ```rust
   let c = RefCell::new(5);
   let r1 = c.borrow_mut();
   if let Ok(r2) = c.try_borrow_mut() {
       // 成功
   } else {
       println!("Already borrowed!");
   }
   ```

### 実践課題（オプション）

1. `Cell`でカウンタを実装
2. `RefCell`でキャッシュを実装
3. 意図的に借用ルール違反を起こしてパニックを確認

## クリア条件（オプション）

理解度チェック：
- [ ] 内部可変性の概念を説明できる
- [ ] `Cell`と`RefCell`の違いを説明できる
- [ ] `RefCell`がパニックする条件を理解している

## 補足情報

### Mutex: スレッドセーフな内部可変性

```rust
use std::sync::Mutex;

let m = Mutex::new(5);
{
    let mut guard = m.lock().unwrap();
    *guard = 6;
}  // ロック解除
```

`RefCell`はシングルスレッド向け。マルチスレッドでは`Mutex`を使う。

### OnceCell / LazyCell

```rust
use std::cell::OnceCell;

let cell = OnceCell::new();
cell.set(42).unwrap();
assert_eq!(cell.get(), Some(&42));
// cell.set(100);  // エラー: 一度だけ設定可能
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch15-05-interior-mutability.html
- std::cell: https://doc.rust-lang.org/std/cell/index.html
