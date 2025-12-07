# Stage 5: 画像処理（グレースケール変換）

## 目標

このステージを完了すると、生徒は：
- 2Dグリッドを使った画像処理をGPUで実行できる
- テクスチャの読み書きができる
- 実際の画像ファイルを処理して結果を確認できる

## 前提知識

- Stage 4完了（スレッドグループの基本）
- 画像がピクセルの2次元配列であることの理解

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **texture-processing** - テクスチャの詳細な使い方

## 教え方ガイド

### 導入（なぜこれを学ぶか）

画像処理はGPUの最も得意な分野の一つです。画像は「2次元の大量のピクセル」であり、各ピクセルに同じ処理を適用する——これは完璧にGPU向きの問題です。

このステージでは、カラー画像をグレースケールに変換する処理をGPUで実装します。シンプルですが、画像フィルタ、エッジ検出、ぼかしなど様々な画像処理の基礎となるテクニックです。

### 説明の流れ

1. **グレースケール変換の原理**

   カラー（RGB）からグレースケールへの変換式：
   ```
   gray = 0.299 * R + 0.587 * G + 0.114 * B
   ```

   これは人間の目の感度に基づいた重み付け（緑に敏感、青に鈍感）。

2. **2つのアプローチ**

   **方法A: バッファを使う（シンプル）**
   - 画像をRGBAの1次元配列として扱う
   - `[[thread_position_in_grid]]` で直接インデックス計算

   **方法B: テクスチャを使う（本格的）**
   - Metal のテクスチャオブジェクトを使用
   - 2D座標で直感的にアクセス
   - 画像処理に最適化されたメモリレイアウト

   今回は方法Aで基本を学びます。

3. **Shader コード（grayscale.metal）**

   ```metal
   #include <metal_stdlib>
   using namespace metal;

   // RGBA画像をグレースケールに変換
   kernel void grayscale(
       device const uchar4* input [[buffer(0)]],   // 入力: RGBA (各チャンネル0-255)
       device uchar4* output [[buffer(1)]],        // 出力: RGBA (グレースケール)
       uint id [[thread_position_in_grid]]
   ) {
       uchar4 pixel = input[id];

       // グレースケール値を計算（人間の目の感度に基づく重み付け）
       float gray = 0.299f * float(pixel.r) + 0.587f * float(pixel.g) + 0.114f * float(pixel.b);

       // グレースケール値を全チャンネルに設定（アルファは保持）
       uchar grayByte = uchar(gray);
       output[id] = uchar4(grayByte, grayByte, grayByte, pixel.a);
   }

   // ネガ反転
   kernel void invert(
       device const uchar4* input [[buffer(0)]],
       device uchar4* output [[buffer(1)]],
       uint id [[thread_position_in_grid]]
   ) {
       uchar4 pixel = input[id];
       output[id] = uchar4(255 - pixel.r, 255 - pixel.g, 255 - pixel.b, pixel.a);
   }

   // セピア調
   kernel void sepia(
       device const uchar4* input [[buffer(0)]],
       device uchar4* output [[buffer(1)]],
       uint id [[thread_position_in_grid]]
   ) {
       uchar4 pixel = input[id];
       float r = float(pixel.r);
       float g = float(pixel.g);
       float b = float(pixel.b);

       float newR = min(255.0f, 0.393f * r + 0.769f * g + 0.189f * b);
       float newG = min(255.0f, 0.349f * r + 0.686f * g + 0.168f * b);
       float newB = min(255.0f, 0.272f * r + 0.534f * g + 0.131f * b);

       output[id] = uchar4(uchar(newR), uchar(newG), uchar(newB), pixel.a);
   }
   ```

4. **Swift コード（grayscale.swift）**

   ```swift
   import Metal
   import Foundation
   import CoreGraphics
   import ImageIO
   import UniformTypeIdentifiers

   // ========== 画像読み込み ==========
   func loadImage(path: String) -> (data: [UInt8], width: Int, height: Int)? {
       let url = URL(fileURLWithPath: path)
       guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
             let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
           return nil
       }

       let width = cgImage.width
       let height = cgImage.height
       let bytesPerPixel = 4  // RGBA

       var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

       let colorSpace = CGColorSpaceCreateDeviceRGB()
       let context = CGContext(
           data: &pixelData,
           width: width,
           height: height,
           bitsPerComponent: 8,
           bytesPerRow: width * bytesPerPixel,
           space: colorSpace,
           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
       )

       context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

       return (pixelData, width, height)
   }

   // ========== 画像保存 ==========
   func saveImage(data: [UInt8], width: Int, height: Int, path: String) {
       let bytesPerPixel = 4
       let colorSpace = CGColorSpaceCreateDeviceRGB()

       var mutableData = data
       guard let context = CGContext(
           data: &mutableData,
           width: width,
           height: height,
           bitsPerComponent: 8,
           bytesPerRow: width * bytesPerPixel,
           space: colorSpace,
           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
       ),
       let cgImage = context.makeImage() else {
           print("画像作成失敗")
           return
       }

       let url = URL(fileURLWithPath: path) as CFURL
       guard let destination = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil) else {
           print("保存先作成失敗")
           return
       }

       CGImageDestinationAddImage(destination, cgImage, nil)
       CGImageDestinationFinalize(destination)
   }

   // ========== メイン処理 ==========
   guard CommandLine.arguments.count >= 2 else {
       print("使い方: swift grayscale.swift <入力画像.png>")
       exit(1)
   }

   let inputPath = CommandLine.arguments[1]
   guard let (inputData, width, height) = loadImage(path: inputPath) else {
       print("画像読み込み失敗: \(inputPath)")
       exit(1)
   }

   print("画像サイズ: \(width) x \(height) = \(width * height) ピクセル")

   // ========== Metal セットアップ ==========
   guard let device = MTLCreateSystemDefaultDevice() else {
       fatalError("Metal非対応")
   }

   let libraryURL = URL(fileURLWithPath: "grayscale.metallib")
   guard let library = try? device.makeLibrary(URL: libraryURL),
         let function = library.makeFunction(name: "grayscale"),
         let pipeline = try? device.makeComputePipelineState(function: function),
         let commandQueue = device.makeCommandQueue() else {
       fatalError("Metal セットアップ失敗")
   }

   // ========== バッファ作成 ==========
   let pixelCount = width * height
   let bufferSize = pixelCount * 4  // RGBA = 4 bytes per pixel

   var mutableInputData = inputData
   guard let inputBuffer = device.makeBuffer(bytes: &mutableInputData, length: bufferSize, options: .storageModeShared),
         let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
       fatalError("バッファ作成失敗")
   }

   // ========== GPU処理実行 ==========
   let startTime = CFAbsoluteTimeGetCurrent()

   guard let commandBuffer = commandQueue.makeCommandBuffer(),
         let encoder = commandBuffer.makeComputeCommandEncoder() else {
       fatalError("コマンド作成失敗")
   }

   encoder.setComputePipelineState(pipeline)
   encoder.setBuffer(inputBuffer, offset: 0, index: 0)
   encoder.setBuffer(outputBuffer, offset: 0, index: 1)

   let gridSize = MTLSize(width: pixelCount, height: 1, depth: 1)
   let threadgroupSize = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
   encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

   encoder.endEncoding()
   commandBuffer.commit()
   commandBuffer.waitUntilCompleted()

   let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
   print("GPU処理時間: \(String(format: "%.4f", elapsedTime)) 秒")

   // ========== 結果保存 ==========
   let resultPointer = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: bufferSize)
   let outputData = Array(UnsafeBufferPointer(start: resultPointer, count: bufferSize))

   let outputPath = inputPath.replacingOccurrences(of: ".png", with: "_gray.png")
   saveImage(data: outputData, width: width, height: height, path: outputPath)
   print("出力: \(outputPath)")
   ```

5. **実行方法**

   テスト用の画像を用意（または任意のPNG画像を使用）：
   ```bash
   # Shaderコンパイル
   xcrun -sdk macosx metal -c grayscale.metal -o grayscale.air
   xcrun -sdk macosx metallib grayscale.air -o grayscale.metallib

   # 実行（ImageIO使用のため -framework オプションが必要）
   swiftc grayscale.swift -o grayscale -framework CoreGraphics -framework ImageIO -framework Metal -framework Foundation
   ./grayscale test_image.png
   ```

   出力例：
   ```
   画像サイズ: 1920 x 1080 = 2073600 ピクセル
   GPU処理時間: 0.0023 秒
   出力: test_image_gray.png
   ```

6. **2Dグリッドバージョン（より直感的）**

   ```metal
   kernel void grayscale_2d(
       device const uchar4* input [[buffer(0)]],
       device uchar4* output [[buffer(1)]],
       constant uint& width [[buffer(2)]],
       uint2 pos [[thread_position_in_grid]]  // 2D座標
   ) {
       uint id = pos.y * width + pos.x;  // 1次元インデックスに変換
       uchar4 pixel = input[id];

       float gray = 0.299f * float(pixel.r) + 0.587f * float(pixel.g) + 0.114f * float(pixel.b);
       uchar grayByte = uchar(gray);

       output[id] = uchar4(grayByte, grayByte, grayByte, pixel.a);
   }
   ```

   Swift側のディスパッチ：
   ```swift
   let gridSize = MTLSize(width: width, height: height, depth: 1)
   let tgSize = MTLSize(width: 16, height: 16, depth: 1)  // 16x16 = 256スレッド
   encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
   ```

### よくある間違い

- **画像フォーマットの不一致**: RGBA と BGRA を間違える
- **アルファチャンネルの破壊**: 処理結果で透明度を0にしてしまう
- **バッファサイズの計算ミス**: width * height * 4（RGBAの4バイト）
- **画像の上下反転**: 座標系の違い（上から始まるか下から始まるか）

## 演習課題

### 課題1: グレースケール変換
上記のコードを実行し、カラー画像がグレースケールに変換されることを確認してください。

### 課題2: ネガ反転
Shader の `invert` 関数を使って、画像のネガ（色反転）を作成してください。

### 課題3: セピア調
Shader の `sepia` 関数を使って、セピア調（古い写真風）の画像を作成してください。

### 課題4（発展）: 明るさ調整
明るさを調整するShaderを追加してください：
```metal
kernel void brightness(
    device const uchar4* input [[buffer(0)]],
    device uchar4* output [[buffer(1)]],
    constant float& factor [[buffer(2)]],  // 1.0 = 変化なし, 1.5 = 50%明るく
    uint id [[thread_position_in_grid]]
) {
    // 実装してみてください
}
```

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] カラー画像をグレースケールに変換できる
- [ ] 変換結果が視覚的に正しいことを確認した（灰色になっている）
- [ ] 2Dグリッドでのディスパッチを理解している
- [ ] ピクセルデータの構造（RGBA = 4バイト/ピクセル）を説明できる

## ヒント集

### ヒント1（軽め）
テスト用の画像がない場合、macOSの標準画像を使えます：
```bash
cp /System/Library/Desktop\ Pictures/Solid\ Colors/Cyan.png test.png
# または任意のPNG画像をダウンロード
```

### ヒント2（中程度）
コンパイル時に`-framework`オプションを忘れがちです：
```bash
swiftc grayscale.swift -o grayscale \
    -framework CoreGraphics \
    -framework ImageIO \
    -framework Metal \
    -framework Foundation
```

### ヒント3（具体的）
明るさ調整の実装例：
```metal
kernel void brightness(
    device const uchar4* input [[buffer(0)]],
    device uchar4* output [[buffer(1)]],
    constant float& factor [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uchar4 pixel = input[id];
    float r = min(255.0f, float(pixel.r) * factor);
    float g = min(255.0f, float(pixel.g) * factor);
    float b = min(255.0f, float(pixel.b) * factor);
    output[id] = uchar4(uchar(r), uchar(g), uchar(b), pixel.a);
}
```

`min(255.0f, ...)` で255を超えないようクリップしています。

## 補足・発展トピック

### テクスチャを使う場合

より高度な画像処理（フィルタリング、サンプリング）にはテクスチャが適しています：

```metal
kernel void grayscale_texture(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 pos [[thread_position_in_grid]]
) {
    float4 color = input.read(pos);
    float gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
    output.write(float4(gray, gray, gray, color.a), pos);
}
```

### 参考リンク

- [Processing a Texture in a Compute Function](https://developer.apple.com/documentation/metal/textures/processing_a_texture_in_a_compute_function)
- [Creating a Custom Image Filter](https://developer.apple.com/documentation/coreimage/creating_a_custom_image_filter)
