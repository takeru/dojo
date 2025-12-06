# トピック: 構造体のライフタイム設計

## メタ情報

- **ID**: lifetime-in-structs
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 設計・ライフタイム

## 前提知識

- Stage 3のライフタイム基礎
- 構造体の基本

## このトピックで学べること

- 構造体に参照を持たせるべきか判断する方法
- ライフタイム付き構造体の設計
- 所有 vs 借用の設計トレードオフ
- 複数参照を持つ構造体

## 関連ステージ

- Stage 3: ライフタイム基礎

## 要点（ドキュメント形式用）

### 基本的な選択

```rust
// 選択肢1: 所有する（シンプル）
struct Person {
    name: String,
}

// 選択肢2: 借用する（効率的だが複雑）
struct PersonRef<'a> {
    name: &'a str,
}
```

### いつ参照を使うか

**参照を使う（借用）**:
- 一時的なビュー（パーサーの結果など）
- 大きなデータのコピーを避けたい
- 構造体が元データより長生きしない

**所有型を使う（所有）**:
- 構造体が独立して存在する
- APIをシンプルにしたい
- ライフタイム管理を避けたい

### 参照を持つ構造体

```rust
struct Excerpt<'a> {
    text: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().unwrap();

    let excerpt = Excerpt { text: first_sentence };
    // excerptはnovelより長生きできない

    println!("{}", excerpt.text);
}
```

### 複数の参照

```rust
// 同じライフタイム
struct SameLifetime<'a> {
    x: &'a str,
    y: &'a str,  // 両方同じ'a
}

// 異なるライフタイム（必要な場合）
struct DifferentLifetimes<'a, 'b> {
    x: &'a str,
    y: &'b str,  // 独立したライフタイム
}
```

### 同じvs異なるライフタイムの選択

```rust
// 同じライフタイム: シンプルだが制約が厳しい
struct Pair<'a> {
    first: &'a str,
    second: &'a str,
}

fn main() {
    let s1 = String::from("hello");
    let s2 = String::from("world");

    // 両方s1とs2の短い方のライフタイムに制約される
    let pair = Pair { first: &s1, second: &s2 };
}

// 異なるライフタイム: 柔軟だが複雑
struct FlexiblePair<'a, 'b> {
    first: &'a str,
    second: &'b str,
}

// 参照の生存期間が異なっても使える
```

### メソッドのライフタイム

```rust
struct Excerpt<'a> {
    text: &'a str,
}

impl<'a> Excerpt<'a> {
    // selfのライフタイムを返す
    fn get_text(&self) -> &'a str {
        self.text
    }

    // 新しい参照を返す（selfのライフタイムとは独立）
    fn get_first_word(&self) -> &str {  // ライフタイム省略
        self.text.split_whitespace().next().unwrap_or("")
    }
}
```

### 設計パターン

**パターン1: 所有型 + 借用ビュー**

```rust
struct Document {
    content: String,
}

struct DocumentView<'a> {
    excerpt: &'a str,
    title: &'a str,
}

impl Document {
    fn view(&self, start: usize, end: usize) -> DocumentView {
        DocumentView {
            excerpt: &self.content[start..end],
            title: "Title",
        }
    }
}
```

**パターン2: Cow（必要なときだけ所有）**

```rust
use std::borrow::Cow;

struct Flexible<'a> {
    data: Cow<'a, str>,  // 借用か所有を動的に選択
}

fn maybe_modify<'a>(s: &'a str, should_modify: bool) -> Flexible<'a> {
    if should_modify {
        Flexible { data: Cow::Owned(s.to_uppercase()) }
    } else {
        Flexible { data: Cow::Borrowed(s) }
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「構造体に参照を持たせるべきか、所有型にすべきか…これは設計判断じゃ。正解は場面によって変わる」

なぜこれを知っておくと便利か：
- 効率的なデータ構造を設計できる
- ライフタイムエラーを回避できる
- 適切なトレードオフを選択できる

### 説明の流れ

1. **2つの選択肢を見せる**
   ```rust
   // 所有型
   struct Config {
       name: String,
       value: String,
   }

   // 借用型
   struct ConfigRef<'a> {
       name: &'a str,
       value: &'a str,
   }
   ```

2. **トレードオフを説明**
   「所有型はシンプルで扱いやすい。借用型は効率的だが、ライフタイム管理が必要じゃ」

3. **判断基準を示す**
   - 構造体がデータを「持つ」べきか「見る」だけか？
   - 構造体は元データより長生きするか？
   - APIをシンプルにしたいか、効率を優先するか？

4. **実践的なアドバイス**
   「迷ったら所有型から始めよ。パフォーマンスが問題になってから借用を検討するのが賢い」

### 実践課題（オプション）

1. 参照を持つ構造体を設計する
2. 同じ構造体を所有型と借用型の両方で書く
3. `Cow`を使った柔軟な構造体を試す

## クリア条件（オプション）

理解度チェック：
- [ ] 所有と借用の設計トレードオフを説明できる
- [ ] 構造体に複数のライフタイムが必要な場面を説明できる
- [ ] 自分で判断基準を適用して設計できる

## 補足情報

### 自己参照型の問題

```rust
// これはできない！
struct SelfRef {
    data: String,
    reference: &str,  // data を参照したい...
}
```

自己参照が必要な場合は `ouroboros` クレートなどを検討。

### Arc/Rc との組み合わせ

```rust
use std::sync::Arc;

// 複数の構造体で共有
struct Shared {
    data: Arc<String>,
}

// ライフタイム注釈不要で共有できる
```

### 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#lifetime-annotations-in-struct-definitions
- Rust Design Patterns: https://rust-unofficial.github.io/patterns/
