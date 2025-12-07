# Stage 3: CPU vs GPU速度比較

## 目標

このステージを完了すると、生徒は：
- CPUとGPUで同じ処理を実行し、実行時間を比較できる
- GPUが得意な処理と苦手な処理の違いを理解できる
- 大規模データでのGPUの威力を実感できる

## 前提知識

- Stage 2完了（Metal Compute Shaderの基本）
- 配列操作の基本

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **buffer-management** - バッファの種類とパフォーマンス特性

## 教え方ガイド

### 導入（なぜこれを学ぶか）

「GPUは速い」とよく聞きますが、実際にどのくらい速いのでしょうか？そして、**どんな時に速いのか**？

このステージでは、同じ計算をCPUとGPUの両方で実行し、処理時間を計測します。データサイズを変えて実験することで、「GPUが本当に速くなるのはいつか」を体感しましょう。

**ネタバレ**: 小さなデータでは実はGPUの方が遅いこともあります。GPUは「大量のデータを並列処理する」ときに真価を発揮します。

### 説明の流れ

1. **ベンチマーク対象の処理**

   配列の各要素に対して、少し重い計算をします：
   ```
   result[i] = sin(data[i]) * cos(data[i]) + sqrt(abs(data[i]))
   ```

   これは各要素が独立しているので、並列化に最適です。

2. **Shaderコード（benchmark.metal）**

   ```metal
   #include <metal_stdlib>
   using namespace metal;

   kernel void heavy_compute(
       device const float* input [[buffer(0)]],
       device float* output [[buffer(1)]],
       uint id [[thread_position_in_grid]]
   ) {
       float x = input[id];
       output[id] = sin(x) * cos(x) + sqrt(abs(x));
   }
   ```

3. **Swiftコード（benchmark.swift）**

   ```swift
   import Metal
   import Foundation

   // ========== 設定 ==========
   let dataSize = 10_000_000  // 1000万要素
   print("データサイズ: \(dataSize) 要素")

   // ========== CPU計算 ==========
   func cpuCompute(_ input: [Float]) -> [Float] {
       var output = [Float](repeating: 0, count: input.count)
       for i in 0..<input.count {
           let x = input[i]
           output[i] = sin(x) * cos(x) + sqrt(abs(x))
       }
       return output
   }

   // ========== GPU計算 ==========
   func gpuCompute(_ input: [Float], device: MTLDevice, pipeline: MTLComputePipelineState, commandQueue: MTLCommandQueue) -> [Float] {
       let count = input.count
       let bufferSize = count * MemoryLayout<Float>.size

       // バッファ作成
       var inputData = input
       guard let inputBuffer = device.makeBuffer(bytes: &inputData, length: bufferSize, options: .storageModeShared),
             let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
           fatalError("バッファ作成失敗")
       }

       // コマンドバッファ作成・実行
       guard let commandBuffer = commandQueue.makeCommandBuffer(),
             let encoder = commandBuffer.makeComputeCommandEncoder() else {
           fatalError("コマンド作成失敗")
       }

       encoder.setComputePipelineState(pipeline)
       encoder.setBuffer(inputBuffer, offset: 0, index: 0)
       encoder.setBuffer(outputBuffer, offset: 0, index: 1)

       let gridSize = MTLSize(width: count, height: 1, depth: 1)
       let threadgroupSize = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
       encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
       encoder.endEncoding()

       commandBuffer.commit()
       commandBuffer.waitUntilCompleted()

       // 結果取得
       let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
       return Array(UnsafeBufferPointer(start: resultPointer, count: count))
   }

   // ========== セットアップ ==========
   guard let device = MTLCreateSystemDefaultDevice() else {
       fatalError("Metal非対応")
   }
   print("GPU: \(device.name)")

   let libraryURL = URL(fileURLWithPath: "benchmark.metallib")
   guard let library = try? device.makeLibrary(URL: libraryURL),
         let function = library.makeFunction(name: "heavy_compute"),
         let pipeline = try? device.makeComputePipelineState(function: function),
         let commandQueue = device.makeCommandQueue() else {
       fatalError("Metal セットアップ失敗")
   }

   // ========== テストデータ生成 ==========
   var inputData = [Float](repeating: 0, count: dataSize)
   for i in 0..<dataSize {
       inputData[i] = Float(i) * 0.001
   }

   // ========== CPU ベンチマーク ==========
   print("\n--- CPU 計算 ---")
   let cpuStart = CFAbsoluteTimeGetCurrent()
   let cpuResult = cpuCompute(inputData)
   let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
   print("CPU 時間: \(String(format: "%.4f", cpuTime)) 秒")

   // ========== GPU ベンチマーク ==========
   print("\n--- GPU 計算 ---")
   let gpuStart = CFAbsoluteTimeGetCurrent()
   let gpuResult = gpuCompute(inputData, device: device, pipeline: pipeline, commandQueue: commandQueue)
   let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
   print("GPU 時間: \(String(format: "%.4f", gpuTime)) 秒")

   // ========== 結果比較 ==========
   print("\n--- 結果 ---")
   let speedup = cpuTime / gpuTime
   print("高速化率: \(String(format: "%.2f", speedup))x")

   // 計算結果の検証（最初の5要素）
   print("\n計算結果の検証（最初の5要素）:")
   for i in 0..<5 {
       let diff = abs(cpuResult[i] - gpuResult[i])
       print("  [\(i)] CPU: \(cpuResult[i]), GPU: \(gpuResult[i]), 差: \(diff)")
   }
   ```

4. **実行方法**

   ```bash
   # Shaderコンパイル
   xcrun -sdk macosx metal -c benchmark.metal -o benchmark.air
   xcrun -sdk macosx metallib benchmark.air -o benchmark.metallib

   # 実行
   swift benchmark.swift
   ```

5. **期待される出力**

   ```
   データサイズ: 10000000 要素
   GPU: Apple M1 Pro

   --- CPU 計算 ---
   CPU 時間: 0.2845 秒

   --- GPU 計算 ---
   GPU 時間: 0.0312 秒

   --- 結果 ---
   高速化率: 9.12x

   計算結果の検証（最初の5要素）:
     [0] CPU: 0.0, GPU: 0.0, 差: 0.0
     [1] CPU: 0.0320016, GPU: 0.0320016, 差: 0.0
     ...
   ```

6. **データサイズと速度の関係**

   | データサイズ | CPU時間 | GPU時間 | 高速化率 |
   |------------|--------|--------|---------|
   | 1,000 | 0.0001秒 | 0.0015秒 | **0.07x** (GPUの方が遅い) |
   | 10,000 | 0.0003秒 | 0.0016秒 | **0.19x** |
   | 100,000 | 0.0029秒 | 0.0018秒 | **1.6x** |
   | 1,000,000 | 0.0285秒 | 0.0045秒 | **6.3x** |
   | 10,000,000 | 0.2845秒 | 0.0312秒 | **9.1x** |

   **重要な発見**:
   - 小さなデータではGPUの方が**遅い**（オーバーヘッドが支配的）
   - 数万要素以上からGPUが有利に
   - データが大きくなるほど高速化率が上がる

### よくある間違い

- **時間計測が不正確**: `commandBuffer.waitUntilCompleted()` を呼ばずに計測終了している
- **GPUが遅く見える**: 小さすぎるデータで測定している
- **結果が一致しない**: 浮動小数点の精度の違い（通常は問題ない程度）

## 演習課題

### 課題1: ベンチマークの実行
上記のコードを実行し、自分のMacでの高速化率を確認してください。

### 課題2: データサイズを変えて実験
`dataSize` を 1,000 / 10,000 / 100,000 / 1,000,000 / 10,000,000 と変えて、速度の変化を観察してください。GPUが有利になる閾値はどこですか？

### 課題3: 計算の重さを変える
Shaderの計算を軽くしてみてください（例: `output[id] = input[id] * 2`）。高速化率はどう変わりますか？

### 課題4（発展）: 複数回の平均を取る
同じ計算を10回繰り返し、平均時間を計測するように改良してください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] CPUとGPUの両方で同じ結果が得られることを確認した
- [ ] 1000万要素程度でGPUがCPUより速いことを確認した
- [ ] 小さなデータ（1000要素以下）ではGPUが遅いことを確認した
- [ ] GPUにはオーバーヘッドがあることを説明できる

## ヒント集

### ヒント1（軽め）
まずは `dataSize = 10_000_000` でベンチマークを実行してみましょう。Apple Silicon Macなら5〜10倍程度の高速化が見られるはずです。

### ヒント2（中程度）
GPUが遅くなる原因は**オーバーヘッド**です：
- バッファの作成
- コマンドバッファの作成・エンコード
- GPU処理の起動・完了待ち

これらの固定コストが、データサイズが小さいと相対的に大きくなります。

### ヒント3（具体的）
複数回の平均を取るには：
```swift
let iterations = 10
var totalTime: Double = 0
for _ in 0..<iterations {
    let start = CFAbsoluteTimeGetCurrent()
    _ = gpuCompute(inputData, device: device, pipeline: pipeline, commandQueue: commandQueue)
    totalTime += CFAbsoluteTimeGetCurrent() - start
}
let averageTime = totalTime / Double(iterations)
print("GPU 平均時間: \(averageTime) 秒")
```

## 補足・発展トピック

### GPUが得意な処理 vs 苦手な処理

**得意**:
- 大量の独立したデータに対する同じ処理（SIMD）
- 行列演算、画像処理、物理シミュレーション
- データ並列性が高い処理

**苦手**:
- データ量が少ない処理
- 分岐が多い処理（if文が多い）
- 逐次的な処理（前の結果が次に必要）
- メモリアクセスパターンが不規則な処理

### 参考リンク

- [Optimizing Performance with the GPU Counters Instrument](https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/optimizing_performance_with_the_gpu_counters_instrument)
