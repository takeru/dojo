# Stage 2: 最初のCompute Shader（配列の2倍）

## 目標

このステージを完了すると、生徒は：
- Metal Shading Language (MSL) でコンピュートシェーダを書ける
- SwiftからGPUにデータを送り、計算結果を受け取れる
- カーネル関数、バッファ、コマンドキューの基本概念を理解できる

## 前提知識

- Stage 1完了（Metal環境が動作すること）
- C言語に似た構文の基本理解

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **msl-basics** - Metal Shading Language の文法と特徴
- **buffer-management** - バッファの種類と使い分け

## 教え方ガイド

### 導入（なぜこれを学ぶか）

GPU計算の基本は「大量のデータに対して同じ処理を並列に実行する」ことです。今回は最もシンプルな例として、配列の各要素を2倍にする処理をGPUで実行します。

CPUで書くと `for` ループで1つずつ処理しますが、GPUでは数千のスレッドが同時に動いて、各要素を並列に処理します。この「考え方の転換」がGPUプログラミングの核心です。

### 説明の流れ

1. **Metalプログラムの構成要素**
   - **Shaderファイル (.metal)**: GPU上で実行されるプログラム
   - **Swiftファイル (.swift)**: GPU処理を起動・管理するホストプログラム
   - **バッファ**: CPU⇔GPU間でデータをやり取りするメモリ領域

2. **Shaderコード（double_array.metal）**

   ```metal
   #include <metal_stdlib>
   using namespace metal;

   // カーネル関数: 配列の各要素を2倍にする
   kernel void double_values(
       device float* data [[buffer(0)]],      // 入出力バッファ
       uint id [[thread_position_in_grid]]    // 現在のスレッドID
   ) {
       data[id] = data[id] * 2.0;
   }
   ```

   **ポイント**:
   - `kernel`: GPUで並列実行される関数の印
   - `device float*`: GPU側のメモリを指すポインタ
   - `[[buffer(0)]]`: Swift側で渡すバッファの番号
   - `[[thread_position_in_grid]]`: このスレッドが担当する要素のインデックス

3. **Shaderのコンパイル**

   Metal Shaderは事前コンパイルが必要です：
   ```bash
   xcrun -sdk macosx metal -c double_array.metal -o double_array.air
   xcrun -sdk macosx metallib double_array.air -o double_array.metallib
   ```

   - `.air`: 中間表現（Air = Apple Intermediate Representation）
   - `.metallib`: 最終的なライブラリ（Swiftから読み込む）

4. **Swiftコード（double_array.swift）**

   ```swift
   import Metal
   import Foundation

   // 1. Metalデバイスの取得
   guard let device = MTLCreateSystemDefaultDevice() else {
       fatalError("Metal非対応")
   }

   // 2. コンパイル済みシェーダの読み込み
   let libraryURL = URL(fileURLWithPath: "double_array.metallib")
   guard let library = try? device.makeLibrary(URL: libraryURL) else {
       fatalError("metallib読み込み失敗")
   }

   // 3. カーネル関数の取得
   guard let function = library.makeFunction(name: "double_values") else {
       fatalError("関数 double_values が見つかりません")
   }

   // 4. パイプライン状態の作成（GPUの実行準備）
   guard let pipeline = try? device.makeComputePipelineState(function: function) else {
       fatalError("パイプライン作成失敗")
   }

   // 5. 入力データの準備
   let count = 10
   var inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   print("入力: \(inputData)")

   // 6. バッファの作成（CPUとGPU両方からアクセス可能）
   let bufferSize = count * MemoryLayout<Float>.size
   guard let buffer = device.makeBuffer(bytes: &inputData, length: bufferSize, options: .storageModeShared) else {
       fatalError("バッファ作成失敗")
   }

   // 7. コマンドキューの作成
   guard let commandQueue = device.makeCommandQueue() else {
       fatalError("コマンドキュー作成失敗")
   }

   // 8. コマンドバッファの作成
   guard let commandBuffer = commandQueue.makeCommandBuffer() else {
       fatalError("コマンドバッファ作成失敗")
   }

   // 9. コンピュートコマンドエンコーダの設定
   guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
       fatalError("エンコーダ作成失敗")
   }

   encoder.setComputePipelineState(pipeline)
   encoder.setBuffer(buffer, offset: 0, index: 0)

   // 10. スレッド数の設定とディスパッチ
   let gridSize = MTLSize(width: count, height: 1, depth: 1)
   let threadgroupSize = MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
   encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
   encoder.endEncoding()

   // 11. GPU処理の実行と完了待ち
   commandBuffer.commit()
   commandBuffer.waitUntilCompleted()

   // 12. 結果の取得
   let resultPointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
   let result = Array(UnsafeBufferPointer(start: resultPointer, count: count))
   print("出力: \(result)")
   ```

5. **実行方法**
   ```bash
   # Shaderをコンパイル
   xcrun -sdk macosx metal -c double_array.metal -o double_array.air
   xcrun -sdk macosx metallib double_array.air -o double_array.metallib

   # Swiftプログラムを実行
   swift double_array.swift
   ```

   出力例：
   ```
   入力: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
   出力: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
   ```

6. **処理の流れを図解**
   ```
   Swift側                              GPU側
   --------                             ------
   inputData [1,2,3,...,10]
        |
        v
   makeBuffer() ──────────────────> Buffer（共有メモリ）
        |                                   |
   dispatchThreads(10) ─────────────> 10スレッドが並列起動
        |                                   |
        |                           Thread 0: data[0] *= 2
        |                           Thread 1: data[1] *= 2
        |                           ...
        |                           Thread 9: data[9] *= 2
        |                                   |
   waitUntilCompleted() <───────────── 完了
        |
        v
   result [2,4,6,...,20]
   ```

### よくある間違い

- **metallib が見つからない**: コンパイル時のパスと実行時のパスが異なる
- **関数が見つからない**: Shaderファイルの関数名とSwift側で指定した名前が一致していない
- **結果が変わらない**: `storageModeShared` を指定していない（Apple Silicon の場合）
- **Shader コンパイルエラー**: `using namespace metal;` を忘れている

## 演習課題

### 課題1: 配列を2倍にする
上記のサンプルコードを実行し、配列の各要素が2倍になることを確認してください。

### 課題2: 配列を3倍にする
Shaderコードを修正して、配列の各要素を3倍にしてみてください。

### 課題3: 配列に定数を加算する
Shaderに2つ目のバッファを追加し、加算する値をSwift側から渡すようにしてみてください。

```metal
kernel void add_value(
    device float* data [[buffer(0)]],
    constant float* addValue [[buffer(1)]],  // 定数バッファ
    uint id [[thread_position_in_grid]]
) {
    data[id] = data[id] + *addValue;
}
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] Shaderファイルを `xcrun metal` でコンパイルできる
- [ ] Swiftプログラムが正しく実行でき、配列が2倍になる
- [ ] `kernel`、`device`、`[[buffer(N)]]`、`[[thread_position_in_grid]]` の意味を説明できる
- [ ] 処理の流れ（バッファ作成→コマンドエンコード→ディスパッチ→待機→結果取得）を理解している

## ヒント集

### ヒント1（軽め）
2つのファイルを同じディレクトリに作成してください：
- `double_array.metal` (Shaderコード)
- `double_array.swift` (Swiftコード)

### ヒント2（中程度）
Shaderのコンパイルは2段階です：
```bash
# 1. .metal → .air（中間表現）
xcrun -sdk macosx metal -c double_array.metal -o double_array.air

# 2. .air → .metallib（ライブラリ）
xcrun -sdk macosx metallib double_array.air -o double_array.metallib
```

`.metallib` ファイルができたら、Swiftプログラムを実行します。

### ヒント3（具体的）
課題3の定数バッファを使う場合、Swift側では以下のように設定します：

```swift
var addValue: Float = 100.0
let constantBuffer = device.makeBuffer(bytes: &addValue, length: MemoryLayout<Float>.size, options: .storageModeShared)
encoder.setBuffer(constantBuffer, offset: 0, index: 1)  // index: 1 = [[buffer(1)]]
```

## 補足・発展トピック

### CUDAとの対応表

| CUDA | Metal |
|------|-------|
| `__global__` | `kernel` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `[[thread_position_in_grid]]` |
| `cudaMalloc` | `device.makeBuffer()` |
| `cudaMemcpy` | Apple Siliconでは不要（Unified Memory） |
| `<<<blocks, threads>>>` | `dispatchThreads()` |

### 参考リンク

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Performing Calculations on a GPU](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
