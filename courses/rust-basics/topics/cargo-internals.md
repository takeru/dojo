# トピック: cargo runの内部動作

## メタ情報

- **ID**: cargo-internals
- **難易度**: 初級
- **所要時間**: 5-10分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: ツール・ビルド

## 前提知識

- Stage 2完了（Hello World）
- cargo の基本的な使い方

## このトピックで学べること

- `cargo run` が内部で何をしているか
- `cargo build` との違い
- ビルドの最適化（キャッシュ）の仕組み

## 関連ステージ

- Stage 2: Hello World（ここで登場）
- Stage 5以降: パフォーマンス測定で重要

## 要点（ドキュメント形式用）

### cargo runの3ステップ

`cargo run` は以下を自動で実行します：

1. **依存関係の確認** - Cargo.tomlを読んで必要なライブラリを確認
2. **ビルド** - `cargo build` と同じ処理（rustcを呼び出してコンパイル）
3. **実行** - 生成された実行ファイルを起動

つまり、**`cargo run` = `cargo build` + 実行**です。

### 手動で実行してみると

```bash
# cargo runと同等の処理を分解
cargo build                    # ステップ1-2: ビルド
./target/debug/hello_world     # ステップ3: 実行
```

### ビルドのキャッシュ

ファイルを変更していなければ、2回目の `cargo run` は再ビルドをスキップして高速に実行されます。

```bash
# 初回: ビルド + 実行（時間がかかる）
cargo run

# 2回目: 実行のみ（すぐ終わる）
cargo run
```

### 実行ファイルの場所

- デバッグビルド: `target/debug/プロジェクト名`
- リリースビルド: `target/release/プロジェクト名`

### cargo buildとの違い

| コマンド | ビルド | 実行 |
|---------|--------|------|
| `cargo build` | ⭕ | ❌ |
| `cargo run` | ⭕ | ⭕ |

ビルドだけしたい場合（実行はしない）は `cargo build` を使います。

## 対話形式の教え方ガイド（先生用）

### 導入

「cargo runは便利じゃが、中で何が起きているか気になるじゃろう？実は3つのステップに分かれておるのじゃ」

なぜこれを知っておくと便利か：
- エラーがどの段階で起きているか分かる
- ビルド時間を短縮するコツが分かる
- 実行ファイルを直接使う場面で役立つ

### 説明の流れ

1. **まず cargo build だけ実行させる**
   ```bash
   cd workspace/hello_world
   cargo build
   ```
   「お、ビルドが終わったのう。でもまだ実行されておらん」

2. **生成物を確認させる**
   ```bash
   ls -lh target/debug/hello_world
   ```
   「ここに実行ファイルが作られておる。これを直接実行できるぞ」

3. **実行ファイルを直接起動**
   ```bash
   ./target/debug/hello_world
   ```
   「ほれ、動いたじゃろう？cargo runは、これを自動でやってくれておるのじゃ」

4. **cargo runとの違いを実感**
   ```bash
   cargo run
   ```
   「今度は一発で動いたのう。cargo runはビルドと実行を一度にやってくれる」

5. **キャッシュの仕組みを確認**
   「もう一度 cargo run を実行してみるのじゃ。何か違いに気づくか？」
   ```bash
   cargo run
   ```
   「今度は一瞬で終わったじゃろう？変更がないから、ビルドをスキップしたのじゃ」

### 実践課題（オプション）

1. main.rsを編集して、cargo runを実行（再ビルドされることを確認）
2. target/debug/ の中身を見て、どんなファイルがあるか確認
3. cargo build --verbose を実行して、詳細なログを見る

## クリア条件（オプション）

理解度チェック：
- [ ] cargo runの3ステップを説明できる
- [ ] cargo buildとcargo runの違いを説明できる
- [ ] 実行ファイルがどこに生成されるか分かる

## 補足情報

### 詳細なログを見る

```bash
cargo run --verbose
# または
cargo run -v
```

これでrustcに渡されているオプションなどが見られます。

### cargo check

ビルドせずに、コンパイルエラーだけチェックするコマンド：
```bash
cargo check
```

ビルドより高速なので、コード書いている途中の確認に便利です。

### 参考リンク

- Cargo Book: https://doc.rust-lang.org/cargo/commands/cargo-run.html
