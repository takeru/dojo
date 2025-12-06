# トピック: 実行バイナリはどこにあるの？

## メタ情報

- **ID**: binary-location
- **難易度**: 初級
- **所要時間**: 4-6分（対話形式）/ 2分（読み物）
- **カテゴリ**: 開発ツール

## 前提知識

- Stage 2完了（cargo runを使ったことがある）

## このトピックで学べること

- 実行ファイルの場所
- targetディレクトリの構造
- debug/releaseの違い

## 関連ステージ

- Stage 2: Hello World（ここで登場）

## 要点（ドキュメント形式用）

### 実行ファイルの場所

実行ファイルは `target/debug/プロジェクト名` に生成されます。

```
hello_world/
├── Cargo.toml
├── src/
│   └── main.rs
└── target/
    └── debug/
        ├── hello_world      # ← これが実行ファイル！（Unix/Mac）
        ├── hello_world.exe  # ← Windows版
        └── ... (中間ファイルなど)
```

### 直接実行

cargo runを使わずに直接実行することもできます：

```bash
# Unix/Mac
./target/debug/hello_world

# Windows
.\target\debug\hello_world.exe
```

### targetディレクトリについて

- `target/` ディレクトリは自動生成される
- gitには入れない（.gitignoreに含まれている）
- `cargo clean` で削除できる

### debug vs release

| 場所 | ビルドコマンド |
|------|--------------|
| `target/debug/` | `cargo build` |
| `target/release/` | `cargo build --release` |

## 対話形式の教え方ガイド（先生用）

### 導入

「cargo runで動くプログラムのファイルってどこにあると思う？」

### 説明の流れ

1. **targetディレクトリを見てみる**
   ```bash
   ls target/debug/
   ```

2. **直接実行してみる**
   ```bash
   ./target/debug/hello_world
   ```

3. **gitignoreを確認**
   「targetは自動生成なのでgitに入れないのじゃ」

4. **cleanコマンドを紹介**
   ```bash
   cargo clean
   ls target/  # なくなっている
   ```

## クリア条件（オプション）

- [ ] 実行ファイルの場所を知っている
- [ ] 直接実行できる
- [ ] targetディレクトリの役割を理解している

## 補足情報

### targetディレクトリのサイズ

大きなプロジェクトでは数GB以上になることも。ディスク容量が気になるときは：

```bash
cargo clean  # target全体を削除
```

### カスタム出力先

環境変数で出力先を変更できます：

```bash
CARGO_TARGET_DIR=/tmp/rust-build cargo build
```

### 参考リンク

- Cargo Book: https://doc.rust-lang.org/cargo/guide/build-cache.html
