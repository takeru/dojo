# トピック: インストールしたCUDA、実は何が入ってる？探検してみよう

## メタ情報

- **ID**: cuda-toolkit-tour
- **難易度**: 初級
- **所要時間**: 5-8分（対話形式）/ 2-3分（読み物）
- **カテゴリ**: ツール・環境

## 前提知識

- Stage 1完了（CUDA Toolkitインストール済み）

## このトピックで学べること

- CUDA Toolkitに含まれるツールとライブラリ
- それぞれの役割と使い所
- どのツールを覚えておけば良いか

## 関連ステージ

- Stage 1: 環境構築

## 要点（ドキュメント形式用）

CUDA Toolkitには、コンパイラだけでなく多数のツールとライブラリが含まれています。

### 主要なツール

| ツール | 役割 |
|--------|------|
| nvcc | CUDAコンパイラ |
| nvidia-smi | GPU情報とモニタリング |
| nvprof | プロファイラ（旧版） |
| nsight-systems | システムプロファイラ（新版） |
| nsight-compute | カーネルプロファイラ（新版） |
| cuda-gdb | GPU デバッガ |

### 主要なライブラリ

| ライブラリ | 用途 |
|-----------|------|
| cuBLAS | 行列演算（超高速） |
| cuFFT | 高速フーリエ変換 |
| cuDNN | ディープラーニング演算 |
| Thrust | C++ STL風の並列アルゴリズム |
| cuRAND | 乱数生成 |

### インストールディレクトリ

**Linux/macOS**: `/usr/local/cuda/`

```
/usr/local/cuda/
├── bin/           # 実行ファイル（nvcc, nvidia-smi等）
├── include/       # ヘッダーファイル
├── lib64/         # ライブラリファイル
├── samples/       # サンプルプログラム
└── doc/           # ドキュメント
```

**Windows**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\`

### すぐ使えるサンプルプログラム

```bash
# サンプルディレクトリに移動
cd /usr/local/cuda/samples

# デバイス情報を表示
cd 1_Utilities/deviceQuery
make
./deviceQuery

# 帯域幅測定
cd ../bandwidthTest
make
./bandwidthTest
```

## 対話形式の教え方ガイド（先生用）

### 導入

「CUDA Toolkitをインストールしたけど、実は数GBもあったよね。nvccだけでそんなに大きいわけないんだ。実は色んなツールとライブラリが詰まってる宝箱なんだよ」

なぜこれを知っておくと便利か：
- 便利なツールの存在を知れる
- 自分で実装しなくても高速なライブラリが使える
- 問題が起きたときにデバッグツールが使える

### 説明の流れ

1. **CUDAディレクトリを探検**

   ```bash
   # CUDAのディレクトリに移動
   cd /usr/local/cuda
   ls -la
   ```

   「bin/, lib/, samples/ などがあるはずだよ」

2. **便利なツールを紹介**

   ```bash
   # GPU情報を見る
   nvidia-smi

   # より詳細な情報
   /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
   ```

3. **ライブラリの紹介**

   「例えば、行列の乗算を自分で書いたけど、実はcuBLASっていう超高速ライブラリがあるんだ。何年もかけて最適化されてるから、自分で書くより10倍速いこともあるよ」

4. **サンプルプログラムを試す**

   ```bash
   cd /usr/local/cuda/samples
   cd 1_Utilities/deviceQuery
   make
   ./deviceQuery
   ```

   「このプログラムを見ると、自分のGPUの詳細スペックが全部分かるよ」

### 実践課題（オプション）

1. CUDAのインストールディレクトリを探す
2. `deviceQuery` サンプルをコンパイル・実行
3. `bandwidthTest` でメモリ帯域幅を測定

## クリア条件（オプション）

理解度チェック：
- [ ] CUDA Toolkitに含まれる主要なツールを3つ言える
- [ ] cuBLAS, cuFFTなどのライブラリの存在を知っている
- [ ] samplesディレクトリの場所が分かる

## 補足情報

### よく使うライブラリの詳細

**cuBLAS（行列演算）**:
```cuda
#include <cublas_v2.h>

// 行列積を超高速で計算
cublasSgemm(...);
```

**Thrust（並列アルゴリズム）**:
```cuda
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// GPUで配列をソート
thrust::device_vector<int> vec(1000000);
thrust::sort(vec.begin(), vec.end());
```

### 参考リンク

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Thrust Documentation](https://nvidia.github.io/thrust/)
