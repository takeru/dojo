# トピック: Metalパフォーマンスチューニング

## メタ情報

- **ID**: performance-tips
- **難易度**: 上級
- **所要時間**: 15-20分（対話形式）/ 7分（読み物）
- **カテゴリ**: パフォーマンス最適化

## 前提知識

- Stage 6完了（行列演算）
- GPUの基本的な動作原理

## このトピックで学べること

- よくあるパフォーマンスボトルネック
- GPU時間の計測方法
- 最適化のベストプラクティス

## 関連ステージ

- Stage 6: 行列演算でニューラルネット風計算

## 要点（ドキュメント形式用）

### パフォーマンス計測

#### 基本的な計測

```swift
let start = CFAbsoluteTimeGetCurrent()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
let elapsed = CFAbsoluteTimeGetCurrent() - start
print("実行時間: \(elapsed * 1000) ms")
```

#### より正確な計測（GPU時間）

```swift
commandBuffer.addCompletedHandler { buffer in
    // GPU開始〜完了の時間（秒）
    let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
    print("GPU時間: \(gpuTime * 1000) ms")
}
```

### よくあるボトルネック

#### 1. バッファ作成のオーバーヘッド

```swift
// 悪い: 毎回バッファを作成
func process(data: [Float]) {
    let buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared)!
    // ...
}

// 良い: バッファを再利用
class Processor {
    let buffer: MTLBuffer

    init() {
        buffer = device.makeBuffer(length: maxSize, options: .storageModeShared)!
    }

    func process(data: [Float]) {
        memcpy(buffer.contents(), data, data.count * MemoryLayout<Float>.size)
        // ...
    }
}
```

#### 2. waitUntilCompleted の乱用

```swift
// 悪い: 毎回待機（GPU→CPU→GPUの往復）
commandBuffer1.commit()
commandBuffer1.waitUntilCompleted()  // 待機！
commandBuffer2.commit()
commandBuffer2.waitUntilCompleted()  // 待機！

// 良い: 最後にまとめて待機
commandBuffer1.commit()
commandBuffer2.commit()
commandBuffer2.waitUntilCompleted()  // 両方完了を待つ
```

#### 3. スレッドグループサイズの不適切な選択

```swift
// 悪い: 小さすぎるスレッドグループ
let threadgroupSize = MTLSize(width: 8, height: 1, depth: 1)

// 良い: 32の倍数（SIMD幅に合わせる）
let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
// または2Dの場合
let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)  // 256スレッド
```

#### 4. メモリアクセスパターン

```metal
// 悪い: 非連続アクセス
output[id * stride] = input[id * stride];  // ストライドアクセス

// 良い: 連続アクセス
output[id] = input[id];  // コアレスドアクセス
```

### 最適化チェックリスト

1. **バッファ管理**
   - [ ] バッファを再利用している
   - [ ] 適切なストレージモードを選択している
   - [ ] 不要なデータ転送を避けている

2. **コマンド実行**
   - [ ] 複数の処理を1つのコマンドバッファにまとめている
   - [ ] `waitUntilCompleted` を最小限にしている
   - [ ] 非同期実行を活用している

3. **スレッド構成**
   - [ ] スレッドグループサイズが32の倍数
   - [ ] 2D処理には2Dグリッドを使用
   - [ ] `dispatchThreads` を使用（非整数倍の処理）

4. **メモリアクセス**
   - [ ] 連続アクセスパターン
   - [ ] Threadgroupメモリを活用（繰り返しアクセス）
   - [ ] 分岐を最小限に

### プロファイリングツール

#### Instruments - GPU Trace

1. Xcode → Product → Profile (Cmd+I)
2. "Metal System Trace" を選択
3. アプリを実行して計測

#### GPU カウンター（コード内で取得）

```swift
// パイプラインの実行統計
let descriptor = MTLComputePassDescriptor()
if device.supportsCounterSampling(.atStageBoundary) {
    // カウンターサンプリングを設定
    // ...
}
```

### Apple Silicon 特有の最適化

#### Unified Memory の活用

```swift
// Apple Silicon では .storageModeShared が最適
// Intel Mac では .storageModeManaged が必要な場合も

if device.hasUnifiedMemory {
    // Apple Silicon
    options = .storageModeShared
} else {
    // Intel Mac
    options = .storageModeManaged
}
```

#### タイルベースレンダリング

Apple GPU はタイルベースのアーキテクチャ:
- レンダーパスの開始時に `loadAction: .clear` を使う
- 不要な load/store を避ける

## 対話形式の教え方ガイド（先生用）

### 導入

「プログラムは動いた。でも遅い。どこがボトルネックなのか、どう改善するのか。このトピックでは、Metalアプリのパフォーマンスを改善するコツを学びます」

### 説明の流れ

1. **まず計測**
   「最適化の鉄則: **測定せずに最適化するな**。まずどこが遅いか特定しましょう」

2. **よくある落とし穴を紹介**
   - 「毎回バッファを作ってませんか？」
   - 「`waitUntilCompleted` を乱発してませんか？」
   - 「スレッドグループサイズは32の倍数ですか？」

3. **実際に計測させる**
   ```swift
   let gpuStart = commandBuffer.gpuStartTime
   let gpuEnd = commandBuffer.gpuEndTime
   print("GPU時間: \((gpuEnd - gpuStart) * 1000) ms")
   ```

4. **Before/After で改善を実感**
   Stage 6 の行列積を最適化して速度比較

### 実践課題（オプション）

1. Stage 3 のベンチマークにGPU時間計測を追加
2. バッファ再利用版と毎回作成版で速度比較
3. 異なるスレッドグループサイズでベンチマーク

## クリア条件（オプション）

- [ ] GPU時間を正確に計測できる
- [ ] バッファ再利用の重要性を説明できる
- [ ] スレッドグループサイズが32の倍数であるべき理由を説明できる

## 補足情報

### 参考数値（Apple M1 Pro）

| 処理 | 典型的な時間 |
|------|-------------|
| バッファ作成 (1MB) | ~0.1ms |
| コマンドバッファ作成 | ~0.01ms |
| 100万要素の配列処理 | ~0.5ms |
| 512x512 行列積 | ~5-10ms |

### 参考リンク

- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/)
- [Optimizing GPU Performance](https://developer.apple.com/documentation/metal/gpu_debugging_and_profiling/optimizing_gpu_performance)
- [Instruments: GPU Trace](https://developer.apple.com/documentation/metal/gpu_debugging_and_profiling/using_gpu_counters_and_sampling)
