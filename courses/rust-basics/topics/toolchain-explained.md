# トピック: ツールチェーンって何？

## メタ情報

- **ID**: toolchain-explained
- **難易度**: 初級
- **所要時間**: 5-7分（対話形式）/ 3分（読み物）
- **カテゴリ**: 環境構築

## 前提知識

- Stage 1完了（Rustがインストール済み）

## このトピックで学べること

- ツールチェーンの概念
- stable/beta/nightlyの違い
- nightlyを使う場面

## 関連ステージ

- Stage 1: 環境構築（ここで登場）

## 要点（ドキュメント形式用）

### ツールチェーンとは

ツールチェーンは、Rustの開発に必要なツール一式のセットです。

**含まれるもの**:
- rustc（コンパイラ）
- cargo（ビルドツール）
- rustdoc（ドキュメント生成）
- 標準ライブラリ
- その他の開発ツール

### チャネルの種類

| チャネル | 説明 | 用途 |
|---------|------|------|
| **stable** | 安定版 | 普段の開発（デフォルト） |
| **beta** | 次のstable候補 | stableになる前のテスト |
| **nightly** | 最新の実験的機能 | 新機能を試したい時 |

### nightlyを使う場面

- 最新の実験的機能（unstable features）を試したい
- 特定のツール（例: rustfmt, clippy）の最新版が必要
- コンパイラの開発に貢献したい

```bash
# nightlyをインストール
rustup install nightly

# nightlyに切り替え
rustup default nightly

# stableに戻す
rustup default stable
```

## 対話形式の教え方ガイド（先生用）

### 導入

「ツールチェーンという言葉を聞いたことがあるかな？」

### 説明の流れ

1. **ツールチェーンの中身を説明**
   「コンパイラだけでなく、cargo、ドキュメント生成など全部セットになっておるのじゃ」

2. **3つのチャネルを説明**
   「stable、beta、nightlyがあるのじゃ。普段はstableを使えばよいぞ」

3. **nightlyの用途を説明**
   「実験的な機能を試したいときはnightlyを使うのじゃ」

4. **切り替え方法を実演**
   ```bash
   rustup show
   rustup install nightly
   rustup default nightly
   rustc --version  # nightly表示
   rustup default stable
   ```

## クリア条件（オプション）

- [ ] ツールチェーンに含まれるものを説明できる
- [ ] stable/beta/nightlyの違いを理解している
- [ ] nightlyをインストール・切り替えできる

## 補足情報

### 一時的にnightlyを使う

デフォルトを変えずに、一度だけnightlyで実行：

```bash
rustup run nightly cargo build
```

または、ディレクトリ内で：

```bash
rustup override set nightly  # このディレクトリだけnightly
rustup override unset        # 解除
```

### 参考リンク

- Rust公式: https://www.rust-lang.org/tools/install
- rustup: https://rust-lang.github.io/rustup/concepts/channels.html
