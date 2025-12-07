# トピック: バッファとメモリ管理

## メタ情報

- **ID**: buffer-management
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: パフォーマンス

## 前提知識

- Stage 2, 3完了（バッファの基本的な使い方）

## このトピックで学べること

- MTLBuffer のストレージモード
- Apple Silicon vs Intel Mac でのメモリモデル
- 効率的なバッファ使用法

## 関連ステージ

- Stage 2: 最初のCompute Shader
- Stage 3: CPU vs GPU速度比較

## 要点（ドキュメント形式用）

### ストレージモード

バッファ作成時に指定するメモリ配置オプション:

| モード | CPU読み | CPU書き | GPU読み | GPU書き | 用途 |
|--------|---------|---------|---------|---------|------|
| `.storageModeShared` | ✓ | ✓ | ✓ | ✓ | Apple Silicon推奨 |
| `.storageModeManaged` | ✓ | ✓ | ✓ | ✓ | Intel Mac（同期必要）|
| `.storageModePrivate` | ✗ | ✗ | ✓ | ✓ | GPU専用、最速 |

### Apple Silicon (M1/M2/M3/M4)

**Unified Memory Architecture (UMA)**:
- CPUとGPUが同じ物理メモリを共有
- データ転送のオーバーヘッドが最小
- `.storageModeShared` を使えば自動で同期

```swift
// Apple Silicon での推奨
let buffer = device.makeBuffer(
    bytes: data,
    length: size,
    options: .storageModeShared  // CPU/GPU両方からアクセス可能
)
// → cudaMemcpy 相当の処理は不要！
```

### Intel Mac (AMD/Intel GPU)

**分離メモリ**:
- CPUとGPUが別々のメモリを持つ
- `.storageModeManaged` 使用時は明示的な同期が必要

```swift
// Intel Mac での注意点
let buffer = device.makeBuffer(length: size, options: .storageModeManaged)

// CPUで書き込み後
buffer.didModifyRange(0..<size)  // GPUに変更を通知

// GPU処理後、CPUで読む前
commandBuffer.addCompletedHandler { _ in
    // GPUの変更をCPUに反映（自動）
}
```

### Private バッファ（GPU専用）

中間結果など、CPUから読み書きしないデータに最適:

```swift
// GPU専用バッファ（最速）
let privateBuffer = device.makeBuffer(
    length: size,
    options: .storageModePrivate
)

// データを入れるには Blit コマンドを使う
let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
blitEncoder.copy(from: sharedBuffer, sourceOffset: 0,
                 to: privateBuffer, destinationOffset: 0, size: size)
blitEncoder.endEncoding()
```

### バッファ作成のコスト

バッファ作成は比較的重い処理。頻繁に作成・破棄せず、再利用を推奨:

```swift
// 悪い例: 毎フレームバッファを作成
func render() {
    let buffer = device.makeBuffer(...)  // 遅い！
    // ...
}

// 良い例: 一度作成して再利用
class Renderer {
    let buffer: MTLBuffer

    init(device: MTLDevice) {
        buffer = device.makeBuffer(...)!
    }

    func render() {
        // buffer を再利用
    }
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「Metalのバッファには複数のストレージモードがあります。Apple Siliconでは気にしなくていいことが多いのですが、理解しておくとパフォーマンスチューニングに役立ちます」

### 説明の流れ

1. **まずUMAの説明**
   「Apple Silicon（M1以降）ではCPUとGPUが同じメモリを使います。これがUnified Memory Architecture。CUDAのように`cudaMemcpy`でデータを転送する必要がないのはこのおかげです」

2. **ストレージモードの違いを実演**
   ```swift
   // 1. Shared（推奨）
   let sharedBuffer = device.makeBuffer(
       bytes: data, length: size, options: .storageModeShared
   )
   // CPU/GPU両方から直接アクセス可能

   // 2. Private（GPU専用）
   let privateBuffer = device.makeBuffer(
       length: size, options: .storageModePrivate
   )
   // GPUからのみアクセス、最速

   // 3. Managed（Intel Mac用）
   let managedBuffer = device.makeBuffer(
       length: size, options: .storageModeManaged
   )
   // CPU/GPU両方だが、同期が必要
   ```

3. **いつPrivateを使うか**
   「中間結果など、CPUから触る必要がないデータはPrivateにすると速くなることがあります。ただし、Sharedでも十分速いことが多いので、まずはSharedで書いて、プロファイリングして必要なら最適化しましょう」

4. **バッファの再利用を強調**
   「バッファ作成は重い処理です。毎フレーム作らず、一度作ったら使い回しましょう」

### 実践課題（オプション）

1. Shared と Private バッファでベンチマークを取り、速度差を測定
2. Intel Mac を持っている場合、Managed バッファの同期を試す

## クリア条件（オプション）

- [ ] `.storageModeShared` と `.storageModePrivate` の違いを説明できる
- [ ] Apple Silicon での Unified Memory の利点を説明できる
- [ ] バッファを再利用する理由を説明できる

## 補足情報

### パフォーマンス計測

```swift
// GPU処理時間の精密計測
let startTime = commandBuffer.gpuStartTime
let endTime = commandBuffer.gpuEndTime
let gpuTime = endTime - startTime  // ナノ秒
```

### 参考リンク

- [Setting Resource Storage Modes](https://developer.apple.com/documentation/metal/resource_fundamentals/setting_resource_storage_modes)
- [Choosing a Resource Storage Mode in macOS](https://developer.apple.com/documentation/metal/resource_fundamentals/choosing_a_resource_storage_mode_for_apple_gpus)
