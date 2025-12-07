# トピック: MetalとCUDAの違い

## メタ情報

- **ID**: metal-vs-cuda
- **難易度**: 初級
- **所要時間**: 5-10分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: 概念理解

## 前提知識

- Stage 1完了（Metal環境の確認）
- GPUプログラミングの基本概念

## このトピックで学べること

- MetalとCUDAの設計思想の違い
- それぞれの強みと用途
- コード構造の対応関係

## 関連ステージ

- Stage 1: Metal環境セットアップ

## 要点（ドキュメント形式用）

### 概要比較

| 項目 | Metal | CUDA |
|------|-------|------|
| 開発元 | Apple | NVIDIA |
| 対応GPU | Apple製デバイス | NVIDIA GPU |
| プラットフォーム | macOS, iOS, iPadOS, visionOS | Windows, Linux |
| シェーダ言語 | Metal Shading Language (MSL) | CUDA C/C++ |
| メモリモデル | Unified Memory（Apple Silicon） | 分離メモリ（手動転送必要） |
| グラフィックス | 統合（描画+コンピュート） | 別途 OpenGL/Vulkan |

### 設計思想の違い

**Metal**
- グラフィックスとコンピュートの統合フレームワーク
- Apple デバイス向けに最適化
- Unified Memory で転送オーバーヘッド削減
- Swift / Objective-C との親和性

**CUDA**
- 汎用計算（GPGPU）に特化
- 科学計算・ML/DL のエコシステムが充実
- 明示的なメモリ管理
- C/C++ 中心

### コード対応表

| 概念 | CUDA | Metal |
|------|------|-------|
| カーネル関数 | `__global__` | `kernel` |
| デバイスポインタ | `float*` (device) | `device float*` |
| スレッドID | `threadIdx.x + blockIdx.x * blockDim.x` | `[[thread_position_in_grid]]` |
| メモリ確保 | `cudaMalloc()` | `device.makeBuffer()` |
| データ転送 | `cudaMemcpy()` | Unified Memory なら不要 |
| カーネル起動 | `<<<blocks, threads>>>` | `dispatchThreads()` |

### 選択の指針

**Metalを選ぶ場合**
- Mac/iOS アプリを開発している
- Apple Silicon の性能を最大限活用したい
- グラフィックスと計算を両方使う

**CUDAを選ぶ場合**
- NVIDIA GPU を持っている
- 機械学習ライブラリ（PyTorch, TensorFlow）を使う
- 科学計算（CUDA対応のライブラリが豊富）

## 対話形式の教え方ガイド（先生用）

### 導入

「CUDAとMetal、どっちを学ぶべき？」という質問をよく受けます。答えは「持っているハードウェアによる」です。

MacユーザーならMetal一択。NVIDIAのGPUを持っているならCUDA。両方持っているなら…両方学ぶのもアリです！

### 説明の流れ

1. **まず質問**
   「あなたが使っているマシンは何ですか？Mac？Windows + NVIDIA GPU？」

2. **違いを実感する例**

   **CUDAのコード**:
   ```cuda
   __global__ void add(float* a, float* b, float* c, int n) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) c[i] = a[i] + b[i];
   }

   // ホスト側
   float *d_a, *d_b, *d_c;
   cudaMalloc(&d_a, size);
   cudaMalloc(&d_b, size);
   cudaMalloc(&d_c, size);
   cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // 転送！
   cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);  // 転送！
   add<<<blocks, threads>>>(d_a, d_b, d_c, n);
   cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);  // 転送！
   ```

   **Metalのコード（Apple Silicon）**:
   ```swift
   // シェーダ
   kernel void add(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
       c[i] = a[i] + b[i];
   }

   // ホスト側（Unified Memory = 転送不要！）
   let bufferA = device.makeBuffer(bytes: a, length: size, options: .storageModeShared)
   let bufferB = device.makeBuffer(bytes: b, length: size, options: .storageModeShared)
   let bufferC = device.makeBuffer(length: size, options: .storageModeShared)
   // → 転送なしでそのまま使える
   ```

3. **重要な違い: Unified Memory**
   - Apple Silicon ではCPUとGPUが同じメモリを共有
   - `cudaMemcpy` に相当する処理が不要
   - これがMetal on Apple Siliconの大きな利点

### 実践課題（オプション）

CUDAを使ったことがある人向け:
- 自分が書いたCUDAコードをMetalに移植してみる
- 構造の違いを実感する

## クリア条件（オプション）

- [ ] MetalとCUDAの主な違いを3つ挙げられる
- [ ] Unified Memoryの利点を説明できる
- [ ] 自分の環境でどちらを使うべきか判断できる

## 補足情報

### 参考リンク

- [Metal - Apple Developer](https://developer.apple.com/metal/)
- [CUDA Toolkit - NVIDIA](https://developer.nvidia.com/cuda-toolkit)
- [Transitioning Metal Apps to Apple Silicon](https://developer.apple.com/documentation/apple-silicon/transitioning-metal-apps-to-apple-silicon)
