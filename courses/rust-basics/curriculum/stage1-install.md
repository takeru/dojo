# Stage 1: Rustの環境構築

## 目標

このステージを完了すると、生徒は：
- rustupを使ってRustをインストールできる
- cargoとrustcが正しく動作することを確認できる
- Rustの開発環境が整った状態になる

## 前提知識

- ターミナル（コマンドライン）の基本操作
- 特にRustの知識は不要

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **rust-toolchain** - rustup、rustc、cargoの関係を詳しく学ぶ

## 教え方ガイド

### 導入（なぜこれを学ぶか）

Rustを始めるには、まず開発環境を整える必要があります。Rustには `rustup` という公式のツールチェーン管理ツールがあり、これを使うことでRustのインストール・更新・バージョン管理が簡単にできます。

### 説明の流れ

1. **rustupの説明**
   - Rustの公式インストーラー
   - バージョン管理もできる
   - https://rustup.rs/ が公式サイト

2. **インストールコマンド**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
   - macOS/Linuxの場合
   - Windowsの場合は別の手順（rustup-init.exe）

3. **PATHの設定**
   - インストール後、シェルを再起動するか `source ~/.cargo/env` を実行
   - `~/.cargo/bin` にツールがインストールされる

4. **動作確認**
   ```bash
   rustc --version
   cargo --version
   ```

### よくある間違い

- PATHが通っていない → シェル再起動または `source ~/.cargo/env`
- 古いRustがすでに入っている → `rustup update` で更新
- 権限エラー → sudoは不要、ユーザーディレクトリにインストールされる

## 演習課題

### 課題1: Rustのインストール
rustupを使ってRustをインストールしてください。

### 課題2: バージョン確認
以下のコマンドを実行して、Rustが正しくインストールされたことを確認してください：
```bash
rustc --version
cargo --version
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `rustc --version` がバージョン番号を表示する（例: rustc 1.xx.x）
- [ ] `cargo --version` がバージョン番号を表示する（例: cargo 1.xx.x）
- [ ] エラーなく実行できる

## ヒント集

### ヒント1（軽め）
まず公式サイト https://rustup.rs/ を見てみましょう。インストールコマンドが書いてあります。

### ヒント2（中程度）
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
このコマンドを実行すると、インストーラーが起動します。基本的にはデフォルト設定（1を選択）で大丈夫です。

### ヒント3（具体的）
インストール後に `command not found` と出る場合は、PATHが通っていません。
```bash
source ~/.cargo/env
```
を実行するか、ターミナルを再起動してください。

## 参考リンク

- 公式: https://www.rust-lang.org/tools/install
- The Rust Book: https://doc.rust-lang.org/book/ch01-01-installation.html
