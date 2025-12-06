# Stage 6: 構造体

## 目標

このステージを完了すると、生徒は：
- `struct` で構造体を定義できる
- 構造体のインスタンスを作成できる
- 構造体のフィールドにアクセスできる
- 構造体に関連メソッドを実装できる
- `impl` ブロックでメソッドを定義できる

## 前提知識

- Stage 1-5完了（環境、Hello World、変数と型、関数、制御フロー）
- 関数の理解

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **rust-enums** - 列挙型とパターンマッチング（計画中）
- **rust-traits** - トレイトと多態性（計画中）

## 教え方ガイド

### 導入（なぜこれを学ぶか）

構造体はRustで最も重要なデータ型です。関連するデータとロジックを組織化するための基本的な道具です。Rustの構造体は、他の言語のクラスに近いですが、データとメソッドを分けて定義する独特な設計を持っています。これは関数型プログラミングとオブジェクト指向プログラミングの融合を表しています。

### 説明の流れ

1. **構造体の定義**
   ```rust
   struct User {
       name: String,
       email: String,
       age: u32,
   }
   ```
   - `struct` キーワードで定義
   - フィールドには型を指定

2. **インスタンスの作成**
   ```rust
   let user = User {
       name: String::from("Alice"),
       email: String::from("alice@example.com"),
       age: 30,
   };
   ```
   - 全フィールドを初期化する必要がある

3. **フィールドへのアクセス**
   ```rust
   println!("{}", user.name);
   println!("{}", user.age);
   ```
   - ドット記法でアクセス

4. **可変構造体**
   ```rust
   let mut user = User {
       name: String::from("Bob"),
       email: String::from("bob@example.com"),
       age: 25,
   };
   user.age = 26;  // 変更可能
   ```

5. **構造体の更新**
   ```rust
   let user2 = User {
       name: String::from("Carol"),
       ..user
   };
   // user のメールと年齢を使用し、名前だけ変更
   ```

6. **関連メソッド - `impl` ブロック**
   ```rust
   impl User {
       fn display_info(&self) {
           println!("Name: {}", self.name);
           println!("Age: {}", self.age);
       }
   }

   let user = User { /* ... */ };
   user.display_info();
   ```
   - `&self` で参照を受け取る（メソッド）
   - `self` で所有権を受け取る（ほとんど使わない）
   - `&mut self` で可変参照を受け取る

7. **関連関数 - `impl` 内の `new`**
   ```rust
   impl User {
       fn new(name: String, email: String, age: u32) -> User {
           User { name, email, age }
       }
   }

   let user = User::new(
       String::from("Dave"),
       String::from("dave@example.com"),
       35,
   );
   ```
   - `::` 構文で呼び出す（関連関数）
   - `self` を受け取らない

8. **短縮記法 - フィールド初期化**
   ```rust
   let name = String::from("Eve");
   let age = 28;
   let user = User {
       name,      // name: name, と同じ
       age,       // age: age, と同じ
       email: String::from("eve@example.com"),
   };
   ```

### よくある間違い

- 全フィールドを初期化しない → コンパイルエラー
- メソッドで `self` を忘れる → コンパイルエラー
- フィールドのタイプミス → コンパイルエラー
- 不変構造体を変更しようとする → コンパイルエラー

## 演習課題

### 課題1: 基本的な構造体
以下の構造体を定義して、インスタンスを作成してください：
```rust
struct Person {
    name: String,
    age: u32,
}

fn main() {
    let person = Person {
        name: String::from("Alice"),
        age: 30,
    };
    println!("{} is {} years old", person.name, person.age);
}
```

### 課題2: メソッドの実装
`Person` 構造体に、情報を表示するメソッドを追加してください：
```rust
impl Person {
    fn display_info(&self) {
        println!("{} is {} years old", self.name, self.age);
    }
}
```

### 課題3: 関連関数の実装
`Person` 構造体に `new` 関連関数を追加してください：
```rust
impl Person {
    fn new(name: String, age: u32) -> Person {
        Person { name, age }
    }
}
```

### 課題4: 複数のメソッド
`Rectangle` 構造体を定義し、以下のメソッドを実装してください：
- `width: u32, height: u32` フィールド
- `area()` - 面積を返すメソッド
- `perimeter()` - 周辺を返すメソッド

### 課題5: 構造体の変更
可変構造体を使って、フィールドを変更してください：
```rust
let mut person = Person::new(String::from("Bob"), 25);
person.age = 26;
person.display_info();
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] 構造体を定義できた
- [ ] インスタンスを作成できた
- [ ] フィールドにアクセスできた
- [ ] `impl` ブロックでメソッドを実装できた
- [ ] `&self` を使ったメソッドを作成できた
- [ ] 関連関数（`new`）を実装できた

## ヒント集

### ヒント1（軽め）
構造体を定義するには `struct` キーワードを使い、各フィールドの型を指定します。

```rust
struct Point {
    x: i32,
    y: i32,
}
```

インスタンスを作成するときは、全フィールドを指定してください。

### ヒント2（中程度）
メソッドを追加するには `impl` ブロックを使います。メソッドは第一引数に `self` または `&self` を受け取ります。

```rust
impl Point {
    fn display(&self) {
        println!("({}, {})", self.x, self.y);
    }
}
```

### ヒント3（具体的）
`Rectangle` 構造体で面積と周辺を計算する場合は、以下のように実装します：

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}

fn main() {
    let rect = Rectangle { width: 30, height: 50 };
    println!("Area: {}", rect.area());           // 1500
    println!("Perimeter: {}", rect.perimeter()); // 160
}
```

## 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch05-00-structs.html
- Rust by Example: https://doc.rust-lang.org/rust-by-example/custom_types/structs.html
