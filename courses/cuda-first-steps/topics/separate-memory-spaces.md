# トピック: CPUとGPUのメモリは別物

## メタ情報

- **ID**: separate-memory-spaces
- **難易度**: 初級
- **所要時間**: 3-5分（対話形式）/ 1-2分（読み物）
- **カテゴリ**: メモリ管理

## 前提知識

- Stage 1完了

## このトピックで学べること

- CPUとGPUが独立したメモリ空間を持つこと
- `cudaMemcpy` が必要な理由
- ホスト（CPU）とデバイス（GPU）の概念

## 関連ステージ

- Stage 1: 環境構築と Hello GPU

## 要点（ドキュメント形式用）

CPUとGPUは**それぞれ独立したメモリ空間**を持っています。

### メモリ空間の図

```
CPU (Host)          GPU (Device)
+----------+        +----------+
| h_arr    |        | d_arr    |
| メモリ   |        | メモリ   |
+----------+        +----------+
     ↑                   ↑
 システムメモリ      GPUメモリ（VRAM）
```

### 基本的な流れ

```cuda
// 1. GPU側にメモリ確保
cudaMalloc((void**)&d_arr, size);

// 2. CPU → GPU にデータコピー
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

// 3. GPUで計算
kernel<<<...>>>(d_arr, n);

// 4. GPU → CPU に結果コピー
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

// 5. GPUメモリ解放
cudaFree(d_arr);
```

### 命名規則

- `h_`: Host（CPU）のメモリ
- `d_`: Device（GPU）のメモリ

## 対話形式の教え方ガイド（先生用）

### 導入

「なんで `cudaMemcpy` が必要なの？って思うよね。実は、CPUとGPUは別々のメモリを持ってるんだ」

### 説明の流れ

1. **メモリ空間の分離を図示**
2. **データの流れを説明**
   - CPU → GPU（計算前）
   - GPU → CPU（計算後）

3. **Unified Memoryへの布石**
   「これが面倒な場合は、自動で転送してくれるUnified Memoryもあるよ」

## クリア条件（オプション）

- [ ] CPUとGPUのメモリが分離していることを理解している
- [ ] データ転送の流れを説明できる

## 補足情報

### 参考リンク

- [CUDA C++ Programming Guide - Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
