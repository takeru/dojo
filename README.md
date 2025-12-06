# Programming Dojo - LLMプログラミング学習システム

Claude Codeでプログラミングを学ぶインタラクティブな学習環境です。
LLM先生が能動的に指導してくれます。

複数の言語・トピックをサポートし、「つかみ」のある短期集中コースで楽しく学べます。

## 使い方

### 1. このディレクトリでClaude Codeを起動

```bash
cd dojo-v1
claude
```

### 2. 覚えるコマンドは1つだけ！

```
/dojo
```

これだけです。状況に応じた選択肢メニューが表示されます。

### 初めての学習 / 続きから始める

```
/dojo
```

これだけで現在のステージから学習が始まります！

### 学習中にメニューが見たいとき

もう一度 `/dojo` を打つと選択肢が出ます：
- 💡 ヒント
- ✅ 理解度確認
- 📚 発展トピック
- その他...

## コマンド一覧

### メインコマンド

- **`/dojo`** - 状況に応じたメニュー表示（これだけ覚えればOK）

### サブコマンド（知っていれば直接打てる）

| コマンド | 説明 |
|---------|------|
| `/dojo:start` | 学習セッション開始 |
| `/dojo:stage N` | ステージNに移動 |
| `/dojo:check` | 理解度確認 |
| `/dojo:hint` | ヒントを求める |
| `/dojo:topic [ID]` | 発展トピック学習 |
| `/dojo:teacher NAME` | 先生キャラクター変更 |
| `/dojo:course [ID]` | コース切り替え |

## 学習の流れ

```
/dojo を打つだけ
  ↓
自動的にセッション開始
  ↓
Stage 1: 環境構築
  ↓
Stage 2: Hello World
  ↓
...学習中に迷ったら...
  ↓
/dojo でメニュー表示
  - 💡 ヒント
  - ✅ 理解度確認
  - 📚 発展トピック
  - その他...
  ↓
ステージクリア後
  ↓
選択肢:
- 次のステージへ
- 発展トピックを学ぶ（サブクエスト）
- 質問する
- 今日はここまで
```

## 先生キャラクター

3人の先生から選べます：

### 錆師範（さびしはん）
- タイプ: 厳しめの師匠
- 口調: 「〜じゃ」
- 特徴: 所有権を剣術に例える

### ラスタ先輩
- タイプ: 優しい先輩
- 口調: 「〜だよ」
- 特徴: 実務経験を交えて教える

### Ferris（フェリス）
- タイプ: 論理的なAI助手
- 口調: 丁寧語
- 特徴: 公式ドキュメント重視

先生の切り替え: `/dojo` → 「先生を変更」

## カリキュラム

### Rust基礎コース（rust-basics）

Stage 1-6でRustの基本構文と制御構造を学びます：

1. 環境構築（rustup, cargo）
2. Hello World（cargo new, println!）
3. 変数と型（let, mut、基本型）
4. 関数（fn, 引数, 戻り値）
5. 制御フロー（if, loop, match）
6. 構造体（struct, impl）

### Rust所有権道場（rust-ownership-dojo）

Stage 1-3でRustの所有権システムを集中的に学びます：

1. 所有権（ownership, move）
2. 借用（&, &mut）
3. ライフタイム基礎（'a）

**前提条件**: Rust基礎コースの完了

## 発展トピック（サブクエスト）

各ステージに関連する発展トピックがあります。

- **rust-toolchain** - rustup/rustc/cargoの関係
- **cargo-internals** - cargo runの内部動作
- **release-build** - デバッグ vs リリース
- **macros-intro** - マクロ入門

トピックは2つの方法で学べます：
1. **対話形式** - 先生が教える（5-15分）
2. **ドキュメント形式** - 要点を読む（2-5分）

## 複数コース対応

このシステムは複数のコースをサポートしています。

### コース切り替え

```
/dojo → 「コースを選ぶ」
```

または直接：

```
/dojo:course
```

### 利用可能なコース

- **rust-basics**: Rust基礎コース（Stage 1-6）
- **rust-ownership-dojo**: Rust所有権道場（Stage 1-3、前提: rust-basics）
- 将来追加予定：python-kindergarten、typescript-hardcore など

### コースの特徴

- **独立した進捗**: 各コースの学習進捗は個別に保存
- **いつでも切り替え**: どのタイミングでもコース切り替え可能
- **共通キャラクター**: 先生キャラクターは全コースで共通
- **コース専用キャラも可**: 各コースに専用キャラクターも設定可能

## ディレクトリ構造

```
dojo-v1/
├── .claude/
│   ├── CLAUDE.md              # 先生システム設定
│   └── commands/
│       ├── dojo.md            # メインコマンド
│       ├── dojo:start.md
│       ├── dojo:stage.md
│       ├── dojo:check.md
│       ├── dojo:hint.md
│       ├── dojo:topic.md
│       ├── dojo:teacher.md
│       └── dojo:course.md     # コース切り替え
├── courses/                   # コース一覧
│   ├── rust-basics/           # Rust基礎コース
│   │   ├── course.json        # コースメタデータ
│   │   ├── curriculum/        # ステージ別カリキュラム
│   │   │   ├── stage1-install.md
│   │   │   ├── stage2-hello.md
│   │   │   └── ...
│   │   ├── topics/            # 発展トピック
│   │   │   ├── rust-toolchain.md
│   │   │   ├── cargo-internals.md
│   │   │   └── ...
│   │   └── characters/        # コース専用キャラ（オプション）
│   │
│   └── python-kindergarten/   # 他のコース（将来追加）
│       ├── course.json
│       ├── curriculum/
│       └── topics/
│
├── characters/                # 共通キャラクター
│   ├── sabi-shihan.md
│   ├── rasta-senpai.md
│   └── ferris.md
├── state/
│   └── progress.json          # 全コースの学習進捗
└── workspace/                 # 共通作業ディレクトリ
```

## Tips

- **迷ったら `/dojo`** を打つだけ！最初は学習開始、セッション中はメニュー表示
- 先生はシェルコマンドの実行結果を見てフィードバックします
- エラーは学習機会！恐れずに試しましょう
- ステージは順番通りでなくてもOK（`/dojo:stage N`）
- トピックはいつでも学べます（サブクエスト）
- コースはいつでも切り替え可能（`/dojo:course`）

## 開発者向け

### 新しいコースの追加

1. **コースディレクトリ作成**
   ```bash
   mkdir -p courses/your-course-id/{curriculum,topics}
   ```

2. **course.json作成**
   ```json
   {
     "id": "your-course-id",
     "name": "コース名",
     "description": "コースの説明",
     "language": "python",
     "difficulty": "beginner",
     "version": "1.0.0",
     "total_stages": 5,
     "stages": [
       {
         "stage": 1,
         "file": "stage1-intro.md",
         "title": "導入"
       }
     ],
     "topics": []
   }
   ```

3. **カリキュラムファイル追加**
   `courses/your-course-id/curriculum/stage1-intro.md` などを作成

4. **初回アクセス時に自動初期化**
   progress.jsonへの追加は初回コース選択時に自動実行

### カリキュラムの追加

`courses/{course-id}/curriculum/stageN-xxx.md` を追加し、`course.json` の `stages` リストに登録

### トピックの追加

`courses/{course-id}/topics/topic-name.md` を追加し、`course.json` の `topics` リストに登録

### キャラクターの追加

- **全コース共通**: `characters/your-character.md` を追加
- **コース専用**: `courses/{course-id}/characters/your-character.md` を追加

---

## コントリビューション募集中のコース

このプロジェクトでは、「つかみ」のある短期集中コンテンツのコントリビューションを募集しています！

**「つかみ」コンテンツの特徴:**
- 3-5ステージで完結（長すぎない）
- ビジュアルまたは数値で成果が見える
- 特定技術の"すごい部分"に集中
- 基礎は最小限、体験にフォーカス
- 前提知識は1コース程度

### ビジュアル系（即座に成果が見える）

#### 🎨 **webgl-shader-dojo** - WebGLシェーダ体験コース
- **Stage 1**: フラグメントシェーダで色を塗る
- **Stage 2**: 時間でアニメーション（波、回転）
- **Stage 3**: 数式で模様生成（フラクタル、グラデーション）
- **つかみポイント**: コード3行で美しいビジュアルが動く驚き

#### 🔥 **cuda-experience** - CUDA体験コース
- **Stage 1**: CPUとGPUで画像処理速度比較
- **Stage 2**: 1万スレッド同時実行体験
- **Stage 3**: リアルタイム物理シミュレーション
- **つかみポイント**: 100倍速くなる体感

#### 🌊 **canvas-generative-art** - Canvas API 生成アート道場
- **Stage 1**: 動く粒子システム（マウス追従）
- **Stage 2**: フラクタルツリー自動生成
- **Stage 3**: 音楽可視化（周波数→波形）
- **つかみポイント**: アートが自動生成される感動

### パフォーマンス/驚き系

#### ⚡ **wasm-speed-dojo** - WebAssembly 速度体験道場
- **Stage 1**: JavaScriptとWasmで計算速度比較
- **Stage 2**: Rustから画像フィルタをWasmで実行
- **Stage 3**: ブラウザでネイティブ級ゲーム動作
- **つかみポイント**: ブラウザなのにこの速度！？

#### 🧠 **tiny-interpreter** - 簡易インタープリタ自作道場
- **Stage 1**: 電卓（四則演算）を作る
- **Stage 2**: 変数と関数を追加
- **Stage 3**: 自作言語でFizzBuzz動かす
- **つかみポイント**: 自分でプログラミング言語が作れる驚き

#### 🔐 **crypto-algorithms** - 暗号アルゴリズム実装道場
- **Stage 1**: シーザー暗号→RSA暗号の進化体験
- **Stage 2**: ハッシュ関数で改ざん検知
- **Stage 3**: デジタル署名で本人証明
- **つかみポイント**: セキュリティの仕組みが理解できる

### AI/ML系（魔法のような体験）

#### 🤖 **transformer-from-scratch** - Transformer を1から実装
- **Stage 1**: Attentionメカニズム実装
- **Stage 2**: 小規模モデルで文章生成
- **Stage 3**: 事前学習モデルのファインチューニング
- **つかみポイント**: ChatGPTの仕組みがわかる

#### 👁️ **image-recognition-dojo** - 画像認識モデル体験
- **Stage 1**: ゼロから畳み込みニューラルネット
- **Stage 2**: MNISTで手書き数字認識
- **Stage 3**: 転移学習で独自の画像分類
- **つかみポイント**: 自分で学習させたAIが判定する

#### 🎮 **reinforcement-learning-games** - 強化学習でゲームAI道場
- **Stage 1**: Q学習で迷路を解く
- **Stage 2**: Flappy Bird を自動プレイ
- **Stage 3**: DQNでブロック崩し
- **つかみポイント**: AIが自分で学習していく様子

### システムプログラミング系

#### 💾 **memory-allocator** - メモリアロケータ自作道場
- **Stage 1**: mallocの仕組み理解
- **Stage 2**: バディシステム実装
- **Stage 3**: ガベージコレクタ実装
- **つかみポイント**: メモリ管理の裏側を完全理解

#### 🌐 **http-server-from-scratch** - HTTP サーバー自作道場
- **Stage 1**: TCPソケットでHello World配信
- **Stage 2**: HTTPリクエストをパース
- **Stage 3**: ルーティング＋静的ファイル配信
- **つかみポイント**: curl でアクセスできる自作サーバー

#### 🔗 **simple-blockchain** - 簡易ブロックチェーン実装
- **Stage 1**: ハッシュチェーンでデータ改ざん防止
- **Stage 2**: Proof of Work でマイニング体験
- **Stage 3**: P2Pネットワークで分散合意
- **つかみポイント**: 仮想通貨の仕組みが理解できる

### 特殊言語機能系

#### 🔄 **go-goroutines** - Go goroutine 並行処理道場
- **Stage 1**: 1万goroutine同時起動
- **Stage 2**: channelで通信パターン
- **Stage 3**: worker poolでタスク並列処理
- **つかみポイント**: 軽量スレッドの威力

#### 🧩 **lisp-macros** - Lisp マクロ体験道場
- **Stage 1**: S式の評価器
- **Stage 2**: マクロで新しい構文を作る
- **Stage 3**: DSL（ドメイン固有言語）作成
- **つかみポイント**: コードがコードを生成する魔法

#### ⚙️ **async-programming** - 非同期プログラミング道場 (async/await)
- **Stage 1**: コールバック地獄を体験→async/awaitで解決
- **Stage 2**: 並行API呼び出しで高速化
- **Stage 3**: ストリーム処理（リアルタイムデータ）
- **つかみポイント**: 同期的に書けて非同期で動く不思議

### 超マニアック系（本で読むと大変だけど体験したい）

#### 📚 **regex-engine** - 正規表現エンジン自作道場
- **Stage 1**: NFAで単純なパターンマッチ（a|b、a*）
- **Stage 2**: バックトラック実装（グループ、後方参照）
- **Stage 3**: 最適化（NFAからDFAへの変換）
- **つかみポイント**: `/\d+/` の裏側がわかる、爆発的遅延を体験できる
- **マニアック度**: ★★★★☆

#### 🎯 **jit-compiler-intro** - JITコンパイラ入門
- **Stage 1**: バイトコードインタープリタ
- **Stage 2**: ホットスポット検出と動的コンパイル
- **Stage 3**: 機械語生成と実行（x64命令）
- **つかみポイント**: JavaScript/Pythonがなぜ速いのか体感
- **マニアック度**: ★★★★★

#### 🗄️ **database-engine** - データベースエンジン自作
- **Stage 1**: B-treeでインデックス実装
- **Stage 2**: WAL（Write-Ahead Logging）でクラッシュ耐性
- **Stage 3**: MVCC（Multi-Version Concurrency Control）
- **つかみポイント**: SQLiteの中身、トランザクションの魔法
- **マニアック度**: ★★★★★

#### 🔐 **tls-handshake** - TLS/SSL プロトコル体験
- **Stage 1**: 鍵交換（Diffie-Hellman）を実装
- **Stage 2**: 証明書チェーン検証
- **Stage 3**: 完全なTLSハンドシェイク実装
- **つかみポイント**: HTTPSの裏側、中間者攻撃を防ぐ仕組み
- **マニアック度**: ★★★★☆

#### 💾 **filesystem-basics** - ファイルシステム実装
- **Stage 1**: inode と データブロック管理
- **Stage 2**: ディレクトリツリー実装
- **Stage 3**: ジャーナリング（障害復旧）
- **つかみポイント**: `ls -la` の裏側、ファイルが消えない仕組み
- **マニアック度**: ★★★★☆

#### 🌊 **compression-algorithms** - 圧縮アルゴリズム道場
- **Stage 1**: ハフマン符号化で文字列圧縮
- **Stage 2**: LZ77/LZ78（辞書ベース圧縮）
- **Stage 3**: Deflate（ZIP）を実装
- **つかみポイント**: なぜファイルが小さくなるのか、圧縮率の限界
- **マニアック度**: ★★★☆☆

#### 🔬 **garbage-collector** - ガベージコレクタ実装道場
- **Stage 1**: Mark-and-Sweep GC
- **Stage 2**: 世代別GC（若い世代・古い世代）
- **Stage 3**: 並行GC（Stop-the-Worldを減らす）
- **つかみポイント**: Java/Goのメモリ管理、GCポーズの原因
- **マニアック度**: ★★★★★

#### 🖥️ **mini-os** - ミニOS道場（ベアメタル）
- **Stage 1**: ブートローダと画面出力
- **Stage 2**: 割り込みハンドラとキーボード入力
- **Stage 3**: メモリ管理とタスク切り替え
- **つかみポイント**: 実機で動くOS、Linuxの起動前
- **マニアック度**: ★★★★★

#### 🌐 **tcp-ip-stack** - TCP/IPスタック実装
- **Stage 1**: Ethernet/IPパケット送受信
- **Stage 2**: TCP 3-way handshake
- **Stage 3**: 再送制御とウィンドウサイズ調整
- **つかみポイント**: `ping` と `curl` の裏側、パケットロス対策
- **マニアック度**: ★★★★★

#### 🎬 **video-codec-basics** - 動画コーデック入門
- **Stage 1**: 静止画圧縮（DCT + 量子化）
- **Stage 2**: 動き補償（前フレームとの差分）
- **Stage 3**: H.264の基本要素実装
- **つかみポイント**: YouTubeが軽い理由、フレーム間予測
- **マニアック度**: ★★★★☆

#### 🧮 **float-point-deep-dive** - 浮動小数点演算の闇
- **Stage 1**: IEEE 754の構造（符号、指数、仮数）
- **Stage 2**: 丸め誤差と桁落ちを体験
- **Stage 3**: Kahan summation などの高精度計算
- **つかみポイント**: `0.1 + 0.2 != 0.3` の真実、金融計算の罠
- **マニアック度**: ★★★☆☆

#### 🔍 **debugger-internals** - デバッガ自作道場
- **Stage 1**: ptrace でプロセスをアタッチ
- **Stage 2**: ブレークポイント設定（int3命令）
- **Stage 3**: スタックトレース表示
- **つかみポイント**: gdb/lldbの仕組み、ステップ実行の裏側
- **マニアック度**: ★★★★☆

#### 📦 **container-runtime** - コンテナランタイム実装
- **Stage 1**: chroot で隔離環境
- **Stage 2**: namespace で PID/ネットワーク分離
- **Stage 3**: cgroup でリソース制限
- **つかみポイント**: Dockerの中身、仮想マシンとの違い
- **マニアック度**: ★★★★☆

#### 🎨 **raytracer-basics** - レイトレーシング入門
- **Stage 1**: 光線と球の交差判定
- **Stage 2**: 反射と影の計算
- **Stage 3**: アンチエイリアシングと被写界深度
- **つかみポイント**: 写真のようなCG、Pixarの技術
- **マニアック度**: ★★★☆☆

#### 🔊 **audio-synthesis** - 音声合成・DSP入門
- **Stage 1**: 正弦波でドレミファソラシド
- **Stage 2**: ADSR エンベロープでシンセサイザー
- **Stage 3**: フィルタとエフェクト（リバーブ、ディレイ）
- **つかみポイント**: DTMの仕組み、音の正体
- **マニアック度**: ★★★☆☆

#### 🧬 **unicode-deep-dive** - Unicode と UTF-8 の闇
- **Stage 1**: UTF-8 エンコード/デコード実装
- **Stage 2**: 正規化（NFC/NFD）と合成文字
- **Stage 3**: 絵文字の複雑さ（ZWJ、バリエーション）
- **つかみポイント**: `"Å" == "Å"` が false になる理由、絵文字の裏側
- **マニアック度**: ★★★☆☆

#### 🔗 **git-internals** - Git の内部構造
- **Stage 1**: blob, tree, commit オブジェクト
- **Stage 2**: diff アルゴリズムとマージ
- **Stage 3**: rebase の仕組みとコンフリクト解決
- **つかみポイント**: `.git` フォルダの中身、タイムマシンの仕組み
- **マニアック度**: ★★★☆☆

#### ⚡ **simd-acceleration** - SIMD命令で高速化
- **Stage 1**: SSE/AVX で並列計算
- **Stage 2**: 画像処理を4倍/8倍高速化
- **Stage 3**: 自動ベクトル化のコツ
- **つかみポイント**: CPUの隠れた能力、ゲームが速い理由
- **マニアック度**: ★★★★☆

#### 🔓 **lock-free-data-structures** - ロックフリーデータ構造
- **Stage 1**: Compare-And-Swap（CAS）の仕組み
- **Stage 2**: ロックフリーキュー実装
- **Stage 3**: ABA問題と対策
- **つかみポイント**: マルチスレッドの最速技術、ロックの限界
- **マニアック度**: ★★★★★

#### 📊 **probabilistic-data-structures** - 確率的データ構造
- **Stage 1**: Bloom Filter（存在確認が超高速）
- **Stage 2**: HyperLogLog（カウントが省メモリ）
- **Stage 3**: Count-Min Sketch（頻度推定）
- **つかみポイント**: Redis/Cassandraの裏技、メモリを100分の1に
- **マニアック度**: ★★★★☆

### コントリビューション方法

1. `/dojo:content-generator` コマンドでコース生成
2. Pull Request を作成
3. レビュー後マージ

詳しくは `COURSE_AUTHORING_GUIDE.md` をご覧ください。

---

Happy Learning! 🎓✨
