# トピック: cudaDeviceSynchronize() とは

## メタ情報

- **ID**: device-synchronize
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: 同期・制御

## 前提知識

- Stage 3完了

## このトピックで学べること

- CUDAの非同期実行の仕組み
- cudaDeviceSynchronize() の役割
- 同期が必要なタイミング

## 関連ステージ

- Stage 3: 速度を測ってみる
- Stage 5: メモリ転送のコストを知る

## 要点（ドキュメント形式用）

CUDAのカーネルは**非同期実行**されます。

### 非同期実行とは

```cuda
kernel<<<...>>>(d_arr, n);  // GPUに命令を投げて即座にreturn
// ここではまだGPUが計算中かもしれない
```

CPUはカーネルを起動したら、GPUの完了を待たずに次の処理へ進みます。

### 同期が必要な場合

GPUの計算が終わるまで待つには：

```cuda
kernel<<<...>>>(d_arr, n);
cudaDeviceSynchronize();  // GPUの処理が終わるまでブロック
// ここはGPUの計算が終わっている
```

### 自動同期されるAPI

以下のAPIは自動的に同期します：
- `cudaMemcpy`（同期転送）
- `cudaDeviceSynchronize`（明示的同期）

### なぜ非同期なのか

- CPUとGPUが並行して動ける
- 効率的なパイプライン処理が可能

## 対話形式の教え方ガイド（先生用）

### 導入

「カーネルを実行したらすぐ終わるけど、本当に計算してるの？」

### 説明の流れ

1. **非同期実行を説明**
   - kernel<<<>>>() は命令を投げるだけ
   - 完了を待たない

2. **同期の方法**
   - cudaDeviceSynchronize()
   - cudaMemcpy（自動同期）

3. **タイミング測定での注意**
   「だから正確な時間を測るには同期が必要」

## クリア条件（オプション）

- [ ] 非同期実行の意味を説明できる
- [ ] cudaDeviceSynchronize() の使い所が分かる

## 補足情報

### 参考リンク

- [CUDA C++ Programming Guide - Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization)
