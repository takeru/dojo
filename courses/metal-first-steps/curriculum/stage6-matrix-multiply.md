# Stage 6: 行列演算でニューラルネット風計算

## 目標

このステージを完了すると、生徒は：
- GPUで行列積（Matrix Multiplication）を実装できる
- ニューラルネットワークの基本演算がGPUで高速化される理由を理解できる
- 大規模な行列演算でGPUの威力を実感できる

## 前提知識

- Stage 5完了（2D処理の基本）
- 行列の積の定義（C[i][j] = Σ A[i][k] * B[k][j]）

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **threadgroup-memory** - タイリングによる高速化
- **performance-tips** - Metalパフォーマンスチューニング

## 教え方ガイド

### 導入（なぜこれを学ぶか）

ディープラーニング、機械学習、科学計算…これらの分野で最も頻繁に行われる計算が**行列積**です。

ニューラルネットワークの順伝播を簡略化すると：
```
出力 = 活性化関数( 重み行列 × 入力ベクトル + バイアス )
```

この「重み行列 × 入力」の部分が行列積であり、これがGPUで爆速になるからこそ、AIブームが可能になりました。

このステージでは、行列積をGPUで実装し、CPUとの速度差を体感しましょう。

### 説明の流れ

1. **行列積の復習**

   行列 A (M×K) と B (K×N) の積 C (M×N):
   ```
   C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
   ```

   ```
   A (4x3)       B (3x2)       C (4x2)
   [a b c]       [p q]         [? ?]
   [d e f]   ×   [r s]    =    [? ?]
   [g h i]       [t u]         [? ?]
   [j k l]                     [? ?]

   C[0][0] = a*p + b*r + c*t
   C[0][1] = a*q + b*s + c*u
   ...
   ```

2. **シンプルな実装（Shader）**

   `matmul.metal`:
   ```metal
   #include <metal_stdlib>
   using namespace metal;

   // 行列積: C = A × B
   // A: M×K, B: K×N, C: M×N
   kernel void matmul(
       device const float* A [[buffer(0)]],
       device const float* B [[buffer(1)]],
       device float* C [[buffer(2)]],
       constant uint& M [[buffer(3)]],
       constant uint& K [[buffer(4)]],
       constant uint& N [[buffer(5)]],
       uint2 pos [[thread_position_in_grid]]
   ) {
       uint row = pos.y;  // 行インデックス
       uint col = pos.x;  // 列インデックス

       // 範囲チェック
       if (row >= M || col >= N) return;

       float sum = 0.0f;
       for (uint k = 0; k < K; k++) {
           sum += A[row * K + k] * B[k * N + col];
       }
       C[row * N + col] = sum;
   }

   // ReLU活性化関数（ニューラルネット風）
   kernel void relu(
       device float* data [[buffer(0)]],
       uint id [[thread_position_in_grid]]
   ) {
       data[id] = max(0.0f, data[id]);
   }

   // ベクトル加算（バイアス加算用）
   kernel void add_bias(
       device float* matrix [[buffer(0)]],       // M×N の行列
       device const float* bias [[buffer(1)]],   // N 次元のベクトル
       constant uint& N [[buffer(2)]],
       uint id [[thread_position_in_grid]]
   ) {
       uint col = id % N;
       matrix[id] += bias[col];
   }
   ```

3. **Swift コード（matmul.swift）**

   ```swift
   import Metal
   import Foundation

   // ========== 行列ユーティリティ ==========
   func createRandomMatrix(rows: Int, cols: Int) -> [Float] {
       return (0..<rows*cols).map { _ in Float.random(in: -1...1) }
   }

   func printMatrix(_ data: [Float], rows: Int, cols: Int, name: String) {
       print("\(name) (\(rows)×\(cols)):")
       for i in 0..<min(4, rows) {  // 最初の4行のみ表示
           let row = (0..<min(4, cols)).map { j in
               String(format: "%7.3f", data[i * cols + j])
           }.joined(separator: " ")
           print("  [\(row)\(cols > 4 ? " ..." : "")]")
       }
       if rows > 4 { print("  ...") }
       print("")
   }

   // ========== CPU行列積 ==========
   func cpuMatmul(A: [Float], B: [Float], M: Int, K: Int, N: Int) -> [Float] {
       var C = [Float](repeating: 0, count: M * N)
       for i in 0..<M {
           for j in 0..<N {
               var sum: Float = 0
               for k in 0..<K {
                   sum += A[i * K + k] * B[k * N + j]
               }
               C[i * N + j] = sum
           }
       }
       return C
   }

   // ========== 設定 ==========
   let M = 512   // 行列Aの行数、行列Cの行数
   let K = 512   // 行列Aの列数、行列Bの行数
   let N = 512   // 行列Bの列数、行列Cの列数

   print("行列サイズ: A(\(M)×\(K)) × B(\(K)×\(N)) = C(\(M)×\(N))")
   print("総計算量: \(M * K * N) 回の積和演算")
   print("")

   // ========== Metal セットアップ ==========
   guard let device = MTLCreateSystemDefaultDevice() else {
       fatalError("Metal非対応")
   }
   print("GPU: \(device.name)")

   let libraryURL = URL(fileURLWithPath: "matmul.metallib")
   guard let library = try? device.makeLibrary(URL: libraryURL),
         let function = library.makeFunction(name: "matmul"),
         let pipeline = try? device.makeComputePipelineState(function: function),
         let commandQueue = device.makeCommandQueue() else {
       fatalError("Metal セットアップ失敗")
   }

   // ========== テストデータ生成 ==========
   var A = createRandomMatrix(rows: M, cols: K)
   var B = createRandomMatrix(rows: K, cols: N)

   // ========== CPU ベンチマーク ==========
   print("--- CPU 行列積 ---")
   let cpuStart = CFAbsoluteTimeGetCurrent()
   let cpuResult = cpuMatmul(A: A, B: B, M: M, K: K, N: N)
   let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
   print("CPU 時間: \(String(format: "%.4f", cpuTime)) 秒")

   // ========== GPU 行列積 ==========
   print("\n--- GPU 行列積 ---")

   // バッファ作成
   guard let bufferA = device.makeBuffer(bytes: &A, length: M * K * MemoryLayout<Float>.size, options: .storageModeShared),
         let bufferB = device.makeBuffer(bytes: &B, length: K * N * MemoryLayout<Float>.size, options: .storageModeShared),
         let bufferC = device.makeBuffer(length: M * N * MemoryLayout<Float>.size, options: .storageModeShared) else {
       fatalError("バッファ作成失敗")
   }

   var mVal = UInt32(M), kVal = UInt32(K), nVal = UInt32(N)
   guard let bufferM = device.makeBuffer(bytes: &mVal, length: MemoryLayout<UInt32>.size, options: .storageModeShared),
         let bufferK = device.makeBuffer(bytes: &kVal, length: MemoryLayout<UInt32>.size, options: .storageModeShared),
         let bufferN = device.makeBuffer(bytes: &nVal, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
       fatalError("定数バッファ作成失敗")
   }

   let gpuStart = CFAbsoluteTimeGetCurrent()

   guard let commandBuffer = commandQueue.makeCommandBuffer(),
         let encoder = commandBuffer.makeComputeCommandEncoder() else {
       fatalError("コマンド作成失敗")
   }

   encoder.setComputePipelineState(pipeline)
   encoder.setBuffer(bufferA, offset: 0, index: 0)
   encoder.setBuffer(bufferB, offset: 0, index: 1)
   encoder.setBuffer(bufferC, offset: 0, index: 2)
   encoder.setBuffer(bufferM, offset: 0, index: 3)
   encoder.setBuffer(bufferK, offset: 0, index: 4)
   encoder.setBuffer(bufferN, offset: 0, index: 5)

   // 2Dグリッドでディスパッチ（各スレッドが C の1要素を計算）
   let gridSize = MTLSize(width: N, height: M, depth: 1)
   let tgSize = MTLSize(width: 16, height: 16, depth: 1)  // 16×16 = 256スレッド/グループ
   encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)

   encoder.endEncoding()
   commandBuffer.commit()
   commandBuffer.waitUntilCompleted()

   let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
   print("GPU 時間: \(String(format: "%.4f", gpuTime)) 秒")

   // ========== 結果比較 ==========
   print("\n--- 結果 ---")
   let speedup = cpuTime / gpuTime
   print("高速化率: \(String(format: "%.2f", speedup))x")

   // 結果の検証
   let gpuResultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: M * N)
   var maxError: Float = 0
   for i in 0..<(M * N) {
       let error = abs(cpuResult[i] - gpuResultPointer[i])
       maxError = max(maxError, error)
   }
   print("最大誤差: \(maxError)")

   if maxError < 0.001 {
       print("✓ 結果が一致しています")
   } else {
       print("⚠ 誤差が大きいです")
   }

   // 結果の一部を表示
   print("")
   let gpuResult = Array(UnsafeBufferPointer(start: gpuResultPointer, count: M * N))
   printMatrix(gpuResult, rows: M, cols: N, name: "C (GPU結果)")
   ```

4. **実行方法**

   ```bash
   xcrun -sdk macosx metal -c matmul.metal -o matmul.air
   xcrun -sdk macosx metallib matmul.air -o matmul.metallib
   swift matmul.swift
   ```

5. **期待される出力**

   ```
   行列サイズ: A(512×512) × B(512×512) = C(512×512)
   総計算量: 134217728 回の積和演算

   GPU: Apple M1 Pro

   --- CPU 行列積 ---
   CPU 時間: 0.4521 秒

   --- GPU 行列積 ---
   GPU 時間: 0.0089 秒

   --- 結果 ---
   高速化率: 50.80x
   最大誤差: 0.00012207031
   ✓ 結果が一致しています

   C (GPU結果) (512×512):
     [ -2.345   1.234  -0.567   3.456 ...]
     [  0.123  -1.890   2.345  -0.789 ...]
     ...
   ```

6. **ニューラルネット風の計算**

   実際のニューラルネットワークでは：
   ```
   output = ReLU(W × input + bias)
   ```

   これをMetalで実装すると：
   ```swift
   // 1. 行列積: tmp = W × input
   // 2. バイアス加算: tmp += bias
   // 3. ReLU活性化: output = max(0, tmp)
   ```

   各ステップを別のカーネルとして実行できます。

### よくある間違い

- **インデックス計算ミス**: `A[i][k]` → `A[i * K + k]`（行優先）
- **行列サイズの不一致**: A の列数と B の行数が一致していない
- **範囲チェック漏れ**: `dispatchThreads` で余分なスレッドが起動された時のガード
- **浮動小数点誤差**: 大規模行列では累積誤差が発生するが、通常は問題ない

## 演習課題

### 課題1: 基本的な行列積
512×512 の行列積を実行し、CPUとGPUの速度を比較してください。

### 課題2: サイズを変えて実験
M, K, N を 128, 256, 512, 1024, 2048 と変えて、高速化率の変化を観察してください。

### 課題3: ニューラルネット風の計算
以下の計算をGPUで実装してください：
1. 行列積: `hidden = weights × input`
2. バイアス加算: `hidden += bias`
3. ReLU活性化: `hidden = max(0, hidden)`

### 課題4（発展）: 簡易ニューラルネット
2層のニューラルネットワークを実装してみてください：
```
hidden = ReLU(W1 × input + b1)
output = W2 × hidden + b2
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] 行列積の計算が正しく動作する（CPUとGPUの結果が一致）
- [ ] 512×512 以上の行列でGPUがCPUより速いことを確認した
- [ ] 2Dグリッドのディスパッチを理解している（幅=N、高さ=M）
- [ ] なぜニューラルネットがGPUで速くなるか説明できる

## ヒント集

### ヒント1（軽め）
行列積では、各スレッドが結果行列 C の**1要素**を計算します。つまり M×N 個のスレッドが必要です。

### ヒント2（中程度）
Shader内のインデックス計算：
- `A[row][k]` → `A[row * K + k]`
- `B[k][col]` → `B[k * N + col]`
- `C[row][col]` → `C[row * N + col]`

行優先（Row-major）でメモリに格納されていることに注意。

### ヒント3（具体的）
ニューラルネット風計算の流れ：
```swift
// 1. matmul カーネル実行
// 2. add_bias カーネル実行
// 3. relu カーネル実行

// すべて同じコマンドバッファに追加可能
let commandBuffer = commandQueue.makeCommandBuffer()!

// matmul
let encoder1 = commandBuffer.makeComputeCommandEncoder()!
encoder1.setComputePipelineState(matmulPipeline)
// ... バッファ設定 ...
encoder1.dispatchThreads(...)
encoder1.endEncoding()

// add_bias
let encoder2 = commandBuffer.makeComputeCommandEncoder()!
encoder2.setComputePipelineState(addBiasPipeline)
// ... バッファ設定 ...
encoder2.dispatchThreads(...)
encoder2.endEncoding()

// relu
let encoder3 = commandBuffer.makeComputeCommandEncoder()!
encoder3.setComputePipelineState(reluPipeline)
// ... バッファ設定 ...
encoder3.dispatchThreads(...)
encoder3.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

## 補足・発展トピック

### なぜ行列積はGPUに向いているか

1. **高い並列性**: 結果行列の各要素は独立に計算可能
2. **規則的なメモリアクセス**: キャッシュ効率が良い
3. **計算密度が高い**: メモリ転送に対して計算量が多い

### さらなる高速化：タイリング

今回の実装は「ナイーブ」な実装です。プロダクション品質の行列積では「タイリング」という手法を使います：

```
// 概念図
1. 行列を小さなタイル（例: 16×16）に分割
2. 各タイルをスレッドグループメモリにロード
3. タイル内で計算（高速メモリからアクセス）
4. 結果を書き戻し
```

これにより10倍以上の高速化が可能ですが、実装は複雑になります。

### 実用的なライブラリ

実際のアプリケーションでは、以下のライブラリを使うのが一般的です：
- **MPSMatrix**: Metal Performance Shaders（Apple提供の最適化済み実装）
- **Accelerate / vDSP**: CPU向け最適化ライブラリ

```swift
import MetalPerformanceShaders

let matrixMultiplication = MPSMatrixMultiplication(
    device: device,
    transposeLeft: false,
    transposeRight: false,
    resultRows: M,
    resultColumns: N,
    interiorColumns: K,
    alpha: 1.0,
    beta: 0.0
)
```

### 参考リンク

- [Metal Performance Shaders - Matrix Multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
- [Optimizing Matrix Operations](https://developer.apple.com/documentation/accelerate/optimizing_matrix_operations)
