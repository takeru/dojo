# Stage 2: Hello World

## 目標

このステージを完了すると、生徒は：
- `cargo new` で新しいRustプロジェクトを作成できる
- Rustプロジェクトの基本構造を理解できる
- `cargo run` でプログラムをビルド・実行できる
- `println!` マクロで文字列を出力できる

## 前提知識

- Stage 1完了（Rustがインストール済み）
- ターミナルの基本操作

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **cargo-internals** - cargo runが内部で何をしているか
- **release-build** - デバッグビルドとリリースビルドの違い
- **macros-intro** - println!の`!`の意味（マクロ入門）

## 教え方ガイド

### 導入（なぜこれを学ぶか）

プログラミング学習の第一歩は「Hello, World!」を表示すること。Rustでは `cargo` というツールを使ってプロジェクトを作成・管理します。Cargoは単なるビルドツールではなく、依存関係の管理やテストの実行など、Rust開発の中心となるツールです。

### 説明の流れ

1. **プロジェクトの作成**
   ```bash
   cargo new hello_world
   cd hello_world
   ```
   - `cargo new` は新しいプロジェクトを作成
   - ディレクトリ名がプロジェクト名になる

2. **プロジェクト構造の説明**
   ```
   hello_world/
   ├── Cargo.toml    # プロジェクト設定ファイル
   └── src/
       └── main.rs   # メインのソースコード
   ```

3. **main.rsの中身**
   ```rust
   fn main() {
       println!("Hello, world!");
   }
   ```
   - `fn main()` はエントリーポイント
   - `println!` はマクロ（`!` がついている）
   - 文は `;` で終わる

4. **ビルドと実行**
   ```bash
   cargo run
   ```
   - コンパイルと実行を一度に行う
   - `cargo build` はビルドのみ
   - 実行ファイルは `target/debug/` に生成される

5. **出力のカスタマイズ**
   ```rust
   println!("こんにちは、{}さん！", "Rust");
   ```
   - `{}` はプレースホルダー

### よくある間違い

- `println` の後に `!` を忘れる → マクロなので `!` が必要
- 文末の `;` を忘れる → Rustでは文は `;` で終わる
- ダブルクォートの代わりにシングルクォートを使う → 文字列は `""`、文字は `''`

## 演習課題

### 課題1: プロジェクト作成
workspaceディレクトリ内に `hello_world` という名前のRustプロジェクトを作成してください。

### 課題2: 実行
`cargo run` でプログラムを実行し、「Hello, world!」が表示されることを確認してください。

### 課題3: カスタマイズ
「Hello, world!」を自分の名前に変えて表示してみてください。
例: 「Hello, Taro!」

### 課題4（発展）: 複数行出力
`println!` を複数回使って、複数行のメッセージを表示してみてください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `cargo new` でプロジェクトを作成できた
- [ ] `cargo run` でプログラムを実行できた
- [ ] 「Hello, world!」が表示された
- [ ] 出力メッセージを変更できた

## ヒント集

### ヒント1（軽め）
まず `cargo new プロジェクト名` を実行してみましょう。

### ヒント2（中程度）
```bash
cd workspace
cargo new hello_world
cd hello_world
cargo run
```
この順番で実行してみてください。

### ヒント3（具体的）
メッセージを変更するには、`src/main.rs` を編集します：
```rust
fn main() {
    println!("Hello, あなたの名前!");
}
```
編集後、再度 `cargo run` を実行すると変更が反映されます。

## 参考リンク

- The Rust Book: https://doc.rust-lang.org/book/ch01-02-hello-world.html
- Cargo Book: https://doc.rust-lang.org/cargo/
