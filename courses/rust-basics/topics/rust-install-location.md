# トピック: インストール場所

## メタ情報

- **ID**: rust-install-location
- **難易度**: 初級
- **所要時間**: 4-6分（対話形式）/ 2分（読み物）
- **カテゴリ**: 環境構築

## 前提知識

- Stage 1完了（Rustがインストール済み）

## このトピックで学べること

- Rustがどこにインストールされるか
- 主要なディレクトリの役割
- アンインストール方法

## 関連ステージ

- Stage 1: 環境構築（ここで登場）

## 要点（ドキュメント形式用）

### インストール場所

Rustは**ユーザーディレクトリ**にインストールされます（システム全体ではなく）。

| OS | 場所 |
|----|------|
| macOS/Linux | `~/.cargo/` と `~/.rustup/` |
| Windows | `%USERPROFILE%\.cargo\` と `%USERPROFILE%\.rustup\` |

### ディレクトリ構造

```
~/.cargo/
├── bin/           # 実行ファイル（cargo, rustc など）
├── registry/      # ダウンロードしたクレート（ライブラリ）
└── git/           # gitから取得した依存関係

~/.rustup/
├── toolchains/    # 各バージョンのツールチェーン
├── update-hashes/ # 更新情報
└── settings.toml  # rustupの設定
```

### パス設定

インストール時に `~/.cargo/bin` がPATHに追加されます。
これで `cargo` や `rustc` がどこからでも実行できます。

### アンインストール

Rustを完全に削除するには：

```bash
rustup self uninstall
```

これで `~/.cargo/` と `~/.rustup/` が削除されます。

## 対話形式の教え方ガイド（先生用）

### 導入

「Rustはどこにインストールされているか、知っておくと便利じゃぞ」

### 説明の流れ

1. **場所を確認**
   ```bash
   which cargo
   which rustc
   ```

2. **ディレクトリを見てみる**
   ```bash
   ls ~/.cargo/
   ls ~/.rustup/
   ```

3. **各ディレクトリの役割を説明**
   「bin/には実行ファイル、registry/にはダウンロードしたライブラリが入っておるのじゃ」

4. **アンインストール方法を紹介**
   「困ったときは`rustup self uninstall`で綺麗に消せるぞ」

## クリア条件（オプション）

- [ ] Rustのインストール場所を知っている
- [ ] `~/.cargo/bin` の役割を理解している
- [ ] アンインストール方法を知っている

## 補足情報

### キャッシュのクリア

ディスク容量を節約したい場合、キャッシュを削除できます：

```bash
# ダウンロードしたクレートのキャッシュを削除
cargo cache --autoclean
# または手動で
rm -rf ~/.cargo/registry/cache/
```

### CARGO_HOMEとRUSTUP_HOME

環境変数でインストール場所を変更できます：

```bash
export CARGO_HOME=/custom/path/.cargo
export RUSTUP_HOME=/custom/path/.rustup
```

### 参考リンク

- rustup: https://rust-lang.github.io/rustup/installation/index.html
