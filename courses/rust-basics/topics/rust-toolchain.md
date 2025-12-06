# トピック: rustup、rustc、cargoの関係

## メタ情報

- **ID**: rust-toolchain
- **難易度**: 初級
- **所要時間**: 5-10分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: ツール・環境

## 前提知識

- Stage 1完了（Rustインストール済み）

## このトピックで学べること

- rustup、rustc、cargoの役割の違い
- それぞれがどう連携しているか
- 普段使うツールとその理由

## 関連ステージ

- Stage 1: 環境構築（ここで登場）
- すべてのステージで間接的に使用

## 要点（ドキュメント形式用）

Rustの開発には3つの主要ツールがあります：

### rustup（ツールチェーン管理）
- Rustのインストーラー兼バージョン管理ツール
- Rustのバージョンを切り替えたり、更新したりする
- 他のツール（rustc、cargo）をインストール・管理する

### rustc（コンパイラ）
- Rustのコンパイラ本体
- `.rs` ファイルを実行可能なバイナリに変換
- 普段は直接使わない（cargoが裏で呼び出す）

### cargo（ビルドツール）
- プロジェクト管理・ビルドツール
- 実際の開発で一番よく使う
- 依存関係の管理、ビルド、テスト実行など

### 関係性の図

```
rustup（管理ツール）
  ├─ rustc をインストール
  ├─ cargo をインストール
  └─ 標準ライブラリをインストール

cargo（日常的に使う）
  └─ 内部で rustc を呼び出してビルド
```

### 実際の使い分け

```bash
# バージョン管理・更新
rustup update
rustup install 1.70.0

# 直接コンパイル（単一ファイルのみ）
rustc main.rs

# プロジェクト開発（通常はこれ）
cargo new my_project
cargo build
cargo run
```

## 対話形式の教え方ガイド（先生用）

### 導入

「Rustを使うとき、3つのツールが出てきて混乱するかもしれんのう。rustup、rustc、cargo…。それぞれどう違うのか、わしが説明しようぞ」

なぜこれを知っておくと便利か：
- どのツールをいつ使うべきか分かる
- エラーが出たときにどのツールの問題か判断できる
- バージョン管理の仕組みが理解できる

### 説明の流れ

1. **3つのツールを確認させる**
   ```bash
   rustup --version
   rustc --version
   cargo --version
   ```
   「すべて動くじゃろう？でも役割は全然違うのじゃ」

2. **rustupの役割を説明**
   - ツールの管理者
   - 「rustupは『道場の管理人』のようなものじゃ。剣（rustc）や防具（cargo）を用意してくれる」

3. **rustcの役割を実演**
   ```bash
   # 簡単なファイルを作成
   echo 'fn main() { println!("Test"); }' > test.rs
   rustc test.rs
   ./test
   ```
   「rustcは職人じゃ。コードを実行ファイルに変える」

4. **cargoの役割を説明**
   「cargoは、rustcを使いやすくしてくれる番頭のようなものじゃ。依存関係の管理もしてくれる」

### 実践課題（オプション）

1. `rustup show` を実行して、現在のツールチェーン情報を確認
2. 単純な `.rs` ファイルを `rustc` で直接コンパイルしてみる
3. 同じものを `cargo` プロジェクトで作ってみて、違いを体感

## クリア条件（オプション）

理解度チェック：
- [ ] rustup、rustc、cargoの役割をそれぞれ説明できる
- [ ] 普段の開発でどれを主に使うか分かる
- [ ] rustupでRustを更新するコマンドが分かる

## 補足情報

### rustupでできること

```bash
rustup update              # Rust更新
rustup install 1.70.0      # 特定バージョンをインストール
rustup default 1.70.0      # デフォルトバージョンを切り替え
rustup show                # 現在の設定を表示
```

### 参考リンク

- rustup公式: https://rust-lang.github.io/rustup/
- The Rust Book: https://doc.rust-lang.org/book/ch01-01-installation.html
