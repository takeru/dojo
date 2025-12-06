# トピック: GPUって何千個のスレッドをどうやって動かしてるの？

## メタ情報

- **ID**: how-gpu-runs
- **難易度**: 中級
- **所要時間**: 10-15分
- **カテゴリ**: アーキテクチャ

## 前提知識

- Stage 6完了

## このトピックで学べること

- Warpの概念
- SM (Streaming Multiprocessor)の仕組み
- スレッド・ブロック・グリッドの実行モデル

## 関連ステージ

- Stage 6: スレッドとブロック

## 要点（ドキュメント形式用）

GPUは物理的に数千コアを持ち、スレッドは**Warp**という32スレッド単位で実行されます。

### Warp（ワープ）

- 32スレッドが1つの実行単位
- 同じ命令を同時実行（SIMT: Single Instruction, Multiple Threads）

### 階層構造

```
Grid（グリッド）
  └── Block（ブロック）複数個
        └── Warp（ワープ）複数個
              └── Thread（スレッド）32個
```

### 実行の流れ

1. カーネル起動 `<<<grid, block>>>`
2. ブロックがSM（Streaming Multiprocessor）に割り当て
3. ブロック内のスレッドが32個ずつWarpにまとめられる
4. Warpが順次実行

### Warpの重要性

- ブロックサイズは32の倍数が推奨（64, 128, 256, 512, 1024）
- 32未満だとWarpが無駄になる

## 対話形式の教え方ガイド（先生用）

### 導入

「何千ものスレッドが動いてるって言ったけど、実は32個ずつまとめて動いてるんだ」

### 実践課題

1. ブロックサイズを31, 32, 64で試して速度比較

## クリア条件

- [ ] Warpが32スレッド単位だと知っている
- [ ] ブロックサイズが32の倍数が良い理由を説明できる

## 補足情報

### 参考リンク

- [CUDA C++ Programming Guide - SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)
