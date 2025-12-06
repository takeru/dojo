# トピック: コンパイルって何してるの？nvccの裏側を覗いてみよう

## メタ情報

- **ID**: nvcc-basics
- **難易度**: 初級
- **所要時間**: 5-8分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: ツール・環境

## 前提知識

- Stage 1完了（CUDA Toolkitインストール済み）
- nvccコマンドの基本的な使い方

## このトピックで学べること

- nvccコンパイラが何をしているのか
- CUDAコードとC++コードがどう処理されるか
- コンパイルオプションの使い方

## 関連ステージ

- Stage 1: 環境構築（nvccを初めて使う）
- すべてのステージで使用

## 要点（ドキュメント形式用）

nvccは、CUDA C/C++コードをコンパイルする専用コンパイラです。

### nvccの役割

nvccは実は**2つのコンパイラを橋渡し**しています：

```
.cuファイル
  ↓
nvcc
  ├→ GPU部分（__global__, __device__）→ PTXコード → GPUバイナリ
  └→ CPU部分（ホストコード）→ g++/cl.exe → CPUバイナリ
  ↓
実行ファイル
```

### 基本的なコンパイル

```bash
nvcc hello.cu -o hello
./hello
```

### よく使うオプション

```bash
# 最適化レベル指定
nvcc -O3 program.cu -o program

# デバッグ情報を含める
nvcc -g -G program.cu -o program

# GPUアーキテクチャを指定
nvcc -arch=sm_75 program.cu -o program

# 詳細なコンパイル情報を表示
nvcc --verbose program.cu -o program
```

### -arch オプションの重要性

GPUの世代（Compute Capability）に合わせてコンパイルすると性能が向上します：

```bash
# 自分のGPUを確認
nvidia-smi

# 例: RTX 3080 なら sm_86
nvcc -arch=sm_86 program.cu -o program
```

## 対話形式の教え方ガイド（先生用）

### 導入

「nvccを使ってコンパイルしてるけど、実は裏で何が起きてるか知ってる？実はnvccは2つのコンパイラを同時に動かしてるんだ」

なぜこれを知っておくと便利か：
- エラーメッセージの意味が分かりやすくなる
- 最適化オプションを使いこなせる
- コンパイル時間を短縮できる

### 説明の流れ

1. **nvccの2段階処理を説明**

   「CUDAプログラムには、GPU部分とCPU部分が混在してるよね。nvccはこれを自動的に分離して、それぞれ適切なコンパイラで処理するんだ」

   ```
   __global__ 関数 → GPU用コンパイラ → GPUコード
   main() など     → C++コンパイラ   → CPUコード
   ```

2. **実際にコンパイル過程を見せる**

   ```bash
   nvcc --verbose hello.cu -o hello
   ```

   「大量の出力が出るけど、よく見ると g++ が呼ばれてるのが分かるはず」

3. **最適化オプションを試す**

   ```bash
   # 最適化なし
   nvcc hello.cu -o hello_noopt

   # 最適化あり
   nvcc -O3 hello.cu -o hello_opt

   # 実行時間を比較してみよう
   time ./hello_noopt
   time ./hello_opt
   ```

4. **-arch オプションの重要性**

   「GPUには世代があって、新しいGPUほど高機能なんだ。`-arch` でGPU世代を指定すると、そのGPUの機能を最大限に使えるよ」

   ```bash
   # 汎用版（すべてのGPUで動くが遅い）
   nvcc program.cu -o program

   # 特定GPU向け（速い）
   nvcc -arch=sm_86 program.cu -o program
   ```

### 実践課題（オプション）

1. `nvcc --version` でコンパイラのバージョンを確認
2. `nvcc --verbose` で実際のコンパイル過程を観察
3. `-O3` オプションをつけて速度を比較

## クリア条件（オプション）

理解度チェック：
- [ ] nvccが何をしているか説明できる
- [ ] `-O3` オプションの役割が分かる
- [ ] `-arch` オプションの使い方が分かる

## 補足情報

### Compute Capability（sm_XX）一覧

| GPU例 | Compute Capability | -archオプション |
|-------|-------------------|----------------|
| GTX 1080 | 6.1 | sm_61 |
| RTX 2080 | 7.5 | sm_75 |
| RTX 3080 | 8.6 | sm_86 |
| RTX 4090 | 8.9 | sm_89 |
| A100 | 8.0 | sm_80 |

自分のGPUのCompute Capabilityは以下で確認：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### その他の便利なオプション

```bash
# 警告を全て表示
nvcc -Wall program.cu -o program

# PTXコードを確認（中間表現）
nvcc -ptx program.cu

# デバッグビルド
nvcc -g -G program.cu -o program_debug
```

### 参考リンク

- [nvcc Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
