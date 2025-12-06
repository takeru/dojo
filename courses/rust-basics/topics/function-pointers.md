# トピック: 関数ポインタ

## メタ情報

- **ID**: function-pointers
- **難易度**: 中級
- **所要時間**: 8-10分（対話形式）/ 4分（読み物）
- **カテゴリ**: 関数

## 前提知識

- Stage 4完了（関数の基本）

## このトピックで学べること

- 関数を値として渡す方法
- 関数ポインタの型
- 高階関数の基本

## 関連ステージ

- Stage 4: 関数（ここで登場）

## 要点（ドキュメント形式用）

### 関数ポインタとは

Rustでは関数を値として渡すことができます。

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}

fn subtract(x: i32, y: i32) -> i32 {
    x - y
}

fn apply(f: fn(i32, i32) -> i32, a: i32, b: i32) -> i32 {
    f(a, b)
}

fn main() {
    let result1 = apply(add, 10, 5);      // 15
    let result2 = apply(subtract, 10, 5); // 5
    println!("{}, {}", result1, result2);
}
```

### 関数ポインタの型

`fn(引数の型) -> 戻り値の型` が関数ポインタの型です。

```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

// 関数ポインタ型の変数
let f: fn(&str) = greet;
f("Rust");
```

### 配列に関数を入れる

```rust
fn double(x: i32) -> i32 { x * 2 }
fn triple(x: i32) -> i32 { x * 3 }
fn square(x: i32) -> i32 { x * x }

fn main() {
    let operations: [fn(i32) -> i32; 3] = [double, triple, square];

    for op in operations {
        println!("{}", op(5));  // 10, 15, 25
    }
}
```

### 高階関数の例

```rust
fn map_values(values: &[i32], f: fn(i32) -> i32) -> Vec<i32> {
    let mut result = Vec::new();
    for &v in values {
        result.push(f(v));
    }
    result
}

fn main() {
    let nums = vec![1, 2, 3];
    let doubled = map_values(&nums, |x| x * 2);
    println!("{:?}", doubled);  // [2, 4, 6]
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「関数を値として渡すことができるのを知っておると、コードの柔軟性が上がるぞ」

### 説明の流れ

1. **基本的な例を示す**
   ```rust
   fn add(x: i32, y: i32) -> i32 { x + y }
   fn apply(f: fn(i32, i32) -> i32, a: i32, b: i32) -> i32 {
       f(a, b)
   }
   ```

2. **型を説明**
   「`fn(i32, i32) -> i32` が関数ポインタの型じゃ」

3. **実用例を紹介**
   「処理を切り替えたいときに便利じゃぞ」

## クリア条件（オプション）

- [ ] 関数を引数として渡せる
- [ ] 関数ポインタの型を書ける
- [ ] 高階関数の概念を理解している

## 補足情報

### クロージャとの違い

関数ポインタ（`fn`）とクロージャ（`Fn`, `FnMut`, `FnOnce`）は異なります：

```rust
// 関数ポインタ: 環境をキャプチャしない
fn add_one(x: i32) -> i32 { x + 1 }

// クロージャ: 環境をキャプチャできる
let y = 10;
let add_y = |x| x + y;  // yをキャプチャ
```

クロージャについては別のトピックで詳しく学びます。

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch19-05-advanced-functions-and-closures.html
- Rust by Example: https://doc.rust-lang.org/rust-by-example/fn/hof.html
