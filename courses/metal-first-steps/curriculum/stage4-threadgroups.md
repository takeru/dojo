# Stage 4: スレッドグループとディスパッチ

## 目標

このステージを完了すると、生徒は：
- スレッド、スレッドグループ、グリッドの階層構造を理解できる
- 適切なスレッドグループサイズを選択できる
- `dispatchThreads` と `dispatchThreadgroups` の違いを理解できる

## 前提知識

- Stage 3完了（GPUの基本的な使い方）
- 配列のインデックス計算の基本

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **threadgroup-memory** - スレッドグループ内でのメモリ共有

## 教え方ガイド

### 導入（なぜこれを学ぶか）

これまで「GPUは大量のスレッドを並列実行する」と説明してきましたが、実際には**スレッドはグループ単位で管理**されています。

GPUアーキテクチャを理解すると：
- 最適なスレッドグループサイズを選べる
- パフォーマンスを最大化できる
- より高度な最適化テクニックが使える

### 説明の流れ

1. **GPUの階層構造**

   ```
   Grid（グリッド）
   ├── Threadgroup 0 ─┬── Thread 0
   │                  ├── Thread 1
   │                  ├── ...
   │                  └── Thread 255
   ├── Threadgroup 1 ─┬── Thread 0
   │                  ├── ...
   │                  └── Thread 255
   └── ...
   ```

   - **Thread**: 最小の実行単位（1つの要素を担当）
   - **Threadgroup**: スレッドの集まり（同じSIMDユニットで実行）
   - **Grid**: 全スレッドの集まり（処理対象全体）

2. **Metal での ID 取得**

   Shader 内で使える属性：

   | 属性 | 意味 | 例 (Threadgroup 2, Thread 3, サイズ256) |
   |-----|-----|----------------------------------------|
   | `[[thread_position_in_grid]]` | グリッド全体での位置 | 2 * 256 + 3 = 515 |
   | `[[thread_position_in_threadgroup]]` | スレッドグループ内での位置 | 3 |
   | `[[threadgroup_position_in_grid]]` | スレッドグループの番号 | 2 |
   | `[[threads_per_threadgroup]]` | スレッドグループのサイズ | 256 |

3. **Shader コード（threadgroup_demo.metal）**

   ```metal
   #include <metal_stdlib>
   using namespace metal;

   struct ThreadInfo {
       uint global_id;
       uint local_id;
       uint group_id;
       uint group_size;
   };

   kernel void show_thread_ids(
       device ThreadInfo* output [[buffer(0)]],
       uint global_id [[thread_position_in_grid]],
       uint local_id [[thread_position_in_threadgroup]],
       uint group_id [[threadgroup_position_in_grid]],
       uint group_size [[threads_per_threadgroup]]
   ) {
       output[global_id].global_id = global_id;
       output[global_id].local_id = local_id;
       output[global_id].group_id = group_id;
       output[global_id].group_size = group_size;
   }
   ```

4. **Swift コード（threadgroup_demo.swift）**

   ```swift
   import Metal
   import Foundation

   struct ThreadInfo {
       var global_id: UInt32
       var local_id: UInt32
       var group_id: UInt32
       var group_size: UInt32
   }

   guard let device = MTLCreateSystemDefaultDevice() else {
       fatalError("Metal非対応")
   }

   let libraryURL = URL(fileURLWithPath: "threadgroup_demo.metallib")
   guard let library = try? device.makeLibrary(URL: libraryURL),
         let function = library.makeFunction(name: "show_thread_ids"),
         let pipeline = try? device.makeComputePipelineState(function: function),
         let commandQueue = device.makeCommandQueue() else {
       fatalError("セットアップ失敗")
   }

   // ========== 設定 ==========
   let totalThreads = 16        // 全スレッド数
   let threadgroupSize = 4      // 1グループあたりのスレッド数

   let bufferSize = totalThreads * MemoryLayout<ThreadInfo>.size
   guard let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
       fatalError("バッファ作成失敗")
   }

   // ========== コマンド実行 ==========
   guard let commandBuffer = commandQueue.makeCommandBuffer(),
         let encoder = commandBuffer.makeComputeCommandEncoder() else {
       fatalError("コマンド作成失敗")
   }

   encoder.setComputePipelineState(pipeline)
   encoder.setBuffer(outputBuffer, offset: 0, index: 0)

   // dispatchThreads: 総スレッド数を指定（Metalが自動でグループ分け）
   let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)
   let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
   encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)

   encoder.endEncoding()
   commandBuffer.commit()
   commandBuffer.waitUntilCompleted()

   // ========== 結果表示 ==========
   let resultPointer = outputBuffer.contents().bindMemory(to: ThreadInfo.self, capacity: totalThreads)

   print("Total threads: \(totalThreads), Threadgroup size: \(threadgroupSize)")
   print("Number of threadgroups: \(totalThreads / threadgroupSize)")
   print("")
   print("GlobalID | LocalID | GroupID | GroupSize")
   print("---------|---------|---------|----------")

   for i in 0..<totalThreads {
       let info = resultPointer[i]
       print("   \(String(format: "%2d", info.global_id))     |    \(info.local_id)    |    \(info.group_id)    |    \(info.group_size)")
   }
   ```

5. **実行結果**

   ```
   Total threads: 16, Threadgroup size: 4
   Number of threadgroups: 4

   GlobalID | LocalID | GroupID | GroupSize
   ---------|---------|---------|----------
      0     |    0    |    0    |    4
      1     |    1    |    0    |    4
      2     |    2    |    0    |    4
      3     |    3    |    0    |    4
      4     |    0    |    1    |    4
      5     |    1    |    1    |    4
      6     |    2    |    1    |    4
      7     |    3    |    1    |    4
      8     |    0    |    2    |    4
      9     |    1    |    2    |    4
     10     |    2    |    2    |    4
     11     |    3    |    2    |    4
     12     |    0    |    3    |    4
     13     |    1    |    3    |    4
     14     |    2    |    3    |    4
     15     |    3    |    3    |    4
   ```

6. **dispatchThreads vs dispatchThreadgroups**

   ```swift
   // 方法1: 総スレッド数を指定（推奨）
   // Metal が自動で必要なグループ数を計算
   encoder.dispatchThreads(
       MTLSize(width: 1000, height: 1, depth: 1),
       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
   )

   // 方法2: スレッドグループ数を指定
   // 総スレッド数 = グループ数 × グループサイズ
   // 1000要素に対して (1000 + 255) / 256 = 4 グループ必要
   encoder.dispatchThreadgroups(
       MTLSize(width: 4, height: 1, depth: 1),
       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
   )
   // この場合、4 * 256 = 1024 スレッドが起動される
   // → 1000〜1023 のスレッドは範囲外なので if (id < n) でガード
   ```

7. **最適なスレッドグループサイズ**

   ```swift
   // パイプラインが推奨する最大スレッド数
   let maxThreads = pipeline.maxTotalThreadsPerThreadgroup  // 例: 1024

   // 推奨される実行幅（SIMD幅）
   let simdWidth = pipeline.threadExecutionWidth  // Apple Silicon: 32

   // 推奨: 32の倍数（64, 128, 256, 512など）
   let recommendedSize = min(256, maxThreads)
   ```

### よくある間違い

- **スレッドグループサイズが大きすぎる**: `maxTotalThreadsPerThreadgroup` を超えると実行時エラー
- **範囲外アクセス**: `dispatchThreadgroups` 使用時に `if (id < n)` のガードを忘れる
- **1次元だけで考える**: 2D/3D問題には2D/3Dスレッドグループを使うと効率的

## 演習課題

### 課題1: スレッド情報の確認
上記のコードを実行し、各スレッドのIDがどのように割り当てられるか確認してください。

### 課題2: スレッドグループサイズを変更
`threadgroupSize` を 2, 4, 8 と変えて、結果を観察してください。

### 課題3: 2Dグリッドで実験
4x4 = 16スレッドを、2x2のスレッドグループで実行してみてください。

```metal
kernel void show_2d_ids(
    device uint2* output [[buffer(0)]],
    uint2 global_id [[thread_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    uint idx = global_id.y * 4 + global_id.x;  // 4x4グリッドの場合
    output[idx] = global_id;
}
```

### 課題4（発展）: 最適なサイズの探索
同じ計算をスレッドグループサイズ 32, 64, 128, 256, 512 で実行し、どのサイズが最も速いか計測してください。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `global_id` = `group_id * group_size + local_id` の関係を理解している
- [ ] スレッドグループサイズを変更して実行できる
- [ ] `dispatchThreads` と `dispatchThreadgroups` の違いを説明できる
- [ ] 推奨されるスレッドグループサイズ（32の倍数）を知っている

## ヒント集

### ヒント1（軽め）
まずは `totalThreads = 16`, `threadgroupSize = 4` で実行して、出力を観察しましょう。4つのグループに分かれていることが分かります。

### ヒント2（中程度）
2Dグリッドの場合、Swiftのディスパッチは以下のようになります：
```swift
let gridSize = MTLSize(width: 4, height: 4, depth: 1)
let tgSize = MTLSize(width: 2, height: 2, depth: 1)
encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
```

### ヒント3（具体的）
スレッドグループサイズの上限を確認するには：
```swift
print("最大スレッド数/グループ: \(pipeline.maxTotalThreadsPerThreadgroup)")
print("SIMD実行幅: \(pipeline.threadExecutionWidth)")
```

Apple Silicon では `threadExecutionWidth` は通常 32 です。スレッドグループサイズはこの倍数にすると効率的です。

## 補足・発展トピック

### CUDAとの対応表

| CUDA | Metal |
|------|-------|
| `threadIdx.x` | `[[thread_position_in_threadgroup]].x` |
| `blockIdx.x` | `[[threadgroup_position_in_grid]].x` |
| `blockDim.x` | `[[threads_per_threadgroup]].x` |
| `gridDim.x` | `[[threadgroups_per_grid]].x` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `[[thread_position_in_grid]].x` |

### なぜスレッドグループが必要か

1. **SIMD効率**: GPUは32スレッド単位で同じ命令を実行（SIMT）
2. **共有メモリ**: スレッドグループ内でメモリを共有して高速化
3. **同期**: スレッドグループ内でのみ `threadgroup_barrier()` で同期可能

### 参考リンク

- [Calculating Threadgroup and Grid Sizes](https://developer.apple.com/documentation/metal/calculating_threadgroup_and_grid_sizes)
