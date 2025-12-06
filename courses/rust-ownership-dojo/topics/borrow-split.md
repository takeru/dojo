# トピック: 借用の分割

## メタ情報

- **ID**: borrow-split
- **難易度**: 中級〜上級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 借用・コンパイラ

## 前提知識

- Stage 2の借用ルール
- 構造体の基本

## このトピックで学べること

- 構造体のフィールド別借用
- スライスの分割借用
- コンパイラが借用を分割できる条件
- 分割できない場合の回避策

## 関連ステージ

- Stage 2: 借用

## 要点（ドキュメント形式用）

### フィールド別借用

構造体の異なるフィールドは、同時に別々に借用できます。

```rust
struct Person {
    name: String,
    age: u32,
}

fn main() {
    let mut person = Person {
        name: String::from("Alice"),
        age: 30,
    };

    // 異なるフィールドを同時に可変借用
    let name = &mut person.name;
    let age = &mut person.age;

    name.push_str(" Smith");
    *age += 1;

    println!("{}, {}", name, age);
}
```

### なぜこれが許されるのか

コンパイラは、`person.name`と`person.age`がメモリ上で別の場所にあることを認識しています。

```
person: Person
┌─────────────────┐
│ name ───────────┼──→ "Alice" (ヒープ)
│ age: 30         │
└─────────────────┘
    ↑         ↑
    │         └── &mut age（別のアドレス）
    └──────────── &mut name（別のアドレス）
```

### スライスの分割

```rust
fn main() {
    let mut arr = [1, 2, 3, 4, 5];

    // split_at_mutで分割
    let (left, right) = arr.split_at_mut(2);

    left[0] = 10;   // [10, 2]
    right[0] = 30;  // [30, 4, 5]

    println!("{:?}", arr);  // [10, 2, 30, 4, 5]
}
```

### 分割できないケース

**1. 同じフィールドへの複数の可変参照**

```rust
let mut person = Person { name: String::from("Alice"), age: 30 };

let name1 = &mut person.name;
// let name2 = &mut person.name;  // エラー！同じフィールド
```

**2. メソッド経由のアクセス**

```rust
impl Person {
    fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
    fn age_mut(&mut self) -> &mut u32 {
        &mut self.age
    }
}

fn main() {
    let mut person = Person { name: String::from("Alice"), age: 30 };

    // メソッド経由だと分割できない
    let name = person.name_mut();
    // let age = person.age_mut();  // エラー！既にpersonを可変借用中
}
```

**3. インデックスアクセス**

```rust
let mut arr = [1, 2, 3, 4, 5];

// インデックスでは分割できない
// let a = &mut arr[0];
// let b = &mut arr[1];  // エラー！arr全体を借用中

// split_at_mutを使う
let (left, right) = arr.split_at_mut(1);
let a = &mut left[0];
let b = &mut right[0];
```

### 回避策

**1. 分割して取り出す**

```rust
impl Person {
    fn split_mut(&mut self) -> (&mut String, &mut u32) {
        (&mut self.name, &mut self.age)
    }
}

fn main() {
    let mut person = Person { name: String::from("Alice"), age: 30 };
    let (name, age) = person.split_mut();

    name.push_str(" Smith");
    *age += 1;
}
```

**2. スコープを分ける**

```rust
fn main() {
    let mut person = Person { name: String::from("Alice"), age: 30 };

    {
        let name = person.name_mut();
        name.push_str(" Smith");
    }  // nameのスコープ終了

    {
        let age = person.age_mut();
        *age += 1;
    }
}
```

**3. 一時変数に分解**

```rust
fn main() {
    let mut person = Person { name: String::from("Alice"), age: 30 };

    // 一度に分解
    let Person { ref mut name, ref mut age } = person;

    name.push_str(" Smith");
    *age += 1;
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「借用ルールは厳しいが、コンパイラは賢い。同じ構造体でも、**別のフィールド**なら同時に可変借用できるのじゃ」

なぜこれを知っておくと便利か：
- 複数フィールドを同時に操作できる
- 借用エラーを回避できる
- 効率的なコードが書ける

### 説明の流れ

1. **フィールド別借用を見せる**
   ```rust
   struct Point { x: i32, y: i32 }

   let mut p = Point { x: 1, y: 2 };
   let x = &mut p.x;
   let y = &mut p.y;  // OK！別のフィールド

   *x += 10;
   *y += 20;
   ```

2. **なぜ可能かを説明**
   「`x`と`y`はメモリ上で別の場所にある。だから同時に変更しても安全じゃ」

3. **メソッド経由の問題**
   ```rust
   impl Point {
       fn x_mut(&mut self) -> &mut i32 { &mut self.x }
       fn y_mut(&mut self) -> &mut i32 { &mut self.y }
   }

   let x = p.x_mut();
   // let y = p.y_mut();  // エラー！
   ```

   「メソッドは`&mut self`を取る。コンパイラからは『Point全体を借用』に見える」

4. **回避策を示す**
   「解決策は、分割して返すメソッドを作ることじゃ」

   ```rust
   fn both_mut(&mut self) -> (&mut i32, &mut i32) {
       (&mut self.x, &mut self.y)
   }
   ```

### 実践課題（オプション）

1. 構造体のフィールド別借用を試す
2. `split_at_mut`でスライスを分割する
3. 借用エラーを回避する方法を実践

## クリア条件（オプション）

理解度チェック：
- [ ] フィールド別借用ができる条件を説明できる
- [ ] メソッド経由で分割できない理由を理解している
- [ ] `split_at_mut`の使い方を理解している

## 補足情報

### get_many_mut (nightly)

将来的には、複数インデックスを同時に可変借用できるAPIが追加される予定:

```rust
#![feature(get_many_mut)]

let mut arr = [1, 2, 3, 4, 5];
let [a, b] = arr.get_many_mut([0, 2]).unwrap();
*a = 10;
*b = 30;
```

### unsafeでの回避（非推奨）

```rust
// 危険！自己責任
let mut arr = [1, 2, 3];
let ptr = arr.as_mut_ptr();
unsafe {
    let a = &mut *ptr.add(0);
    let b = &mut *ptr.add(1);
    *a = 10;
    *b = 20;
}
```

安全なAPIがある場合は必ずそちらを使うこと。

### 参考リンク

- Rustonomicon - Splitting Borrows: https://doc.rust-lang.org/nomicon/borrow-splitting.html
- std::slice::split_at_mut: https://doc.rust-lang.org/std/primitive.slice.html#method.split_at_mut
