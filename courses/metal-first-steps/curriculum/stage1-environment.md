# Stage 1: Metal環境セットアップ

## 目標

このステージを完了すると、生徒は：
- 自分のMacでMetalが使えることを確認できる
- SwiftからMetalデバイスを取得し、GPU情報を表示できる
- Xcodeなしでコマンドラインからswiftコマンドでプログラムを実行できる

## 前提知識

- ターミナル（コマンドライン）の基本操作
- プログラミングの基本的な概念（変数、関数など）
- Swiftの知識は不要（このコースで学びながら進められます）

## 関連する発展トピック（サブクエスト）

このステージをクリアした後、以下のトピックで深く学べます：
- **metal-vs-cuda** - MetalとCUDAの違い、それぞれの特徴

## 教え方ガイド

### 導入（なぜこれを学ぶか）

「GPUプログラミングといえばNVIDIAのCUDA」と思っている人は多いかもしれません。でも、Macを持っているなら、Apple純正の**Metal**フレームワークを使って今すぐGPUプログラミングを始められます！

MetalはApple製デバイス（Mac、iPhone、iPad）のGPUを活用するための低レベルAPIです。特にApple Silicon（M1/M2/M3/M4チップ）では、CPUとGPUが同じメモリを共有する**Unified Memory**アーキテクチャにより、データ転送のオーバーヘッドが少なく、効率的なGPU計算が可能です。

このステージでは、まず環境を確認して、GPUの情報を取得するところから始めましょう。

### 説明の流れ

1. **Metalとは何か**
   - AppleのGPU用低レベルAPI
   - グラフィックス描画とコンピュート（汎用計算）の両方に使える
   - CUDAと違い、Xcodeなしでも`swift`コマンドで実行可能

2. **環境要件の確認**
   - macOS 10.11以降（Metal対応）
   - 推奨: Apple Silicon Mac（M1/M2/M3/M4）
   - Intel Macでも動作可能（AMD GPUまたはIntel統合GPU）

3. **最初のプログラム: GPU情報の取得**

   作業ディレクトリを作成：
   ```bash
   mkdir -p workspace/metal-first-steps
   cd workspace/metal-first-steps
   ```

   `check_metal.swift` を作成：
   ```swift
   import Metal

   // デフォルトのMetalデバイス（GPU）を取得
   guard let device = MTLCreateSystemDefaultDevice() else {
       print("Error: Metalが使えません。このMacはMetal非対応の可能性があります。")
       exit(1)
   }

   print("=== Metal GPU 情報 ===")
   print("GPU名: \(device.name)")
   print("レジストリID: \(device.registryID)")

   // Apple Silicon かどうか
   if device.hasUnifiedMemory {
       print("メモリ: Unified Memory（CPUとGPUでメモリ共有）")
   } else {
       print("メモリ: 専用VRAM")
   }

   // 推奨ワーキングセットサイズ
   let workingSetSize = device.recommendedMaxWorkingSetSize
   print("推奨最大メモリ: \(workingSetSize / 1024 / 1024) MB")

   // スレッドグループのサイズ上限
   let maxThreads = device.maxThreadsPerThreadgroup
   print("スレッドグループ最大サイズ: \(maxThreads.width) x \(maxThreads.height) x \(maxThreads.depth)")

   print("\nMetalの準備完了です！")
   ```

4. **実行方法**
   ```bash
   swift check_metal.swift
   ```

   出力例（Apple Silicon Macの場合）：
   ```
   === Metal GPU 情報 ===
   GPU名: Apple M1 Pro
   レジストリID: 4294969089
   メモリ: Unified Memory（CPUとGPUでメモリ共有）
   推奨最大メモリ: 10922 MB
   スレッドグループ最大サイズ: 1024 x 1024 x 1024

   Metalの準備完了です！
   ```

5. **コードの重要ポイント**
   - `import Metal`: Metalフレームワークを読み込み
   - `MTLCreateSystemDefaultDevice()`: システムのデフォルトGPUを取得
   - `device.name`: GPU名（Apple M1, AMD Radeon Pro 5500M など）
   - `device.hasUnifiedMemory`: Apple Siliconならtrue（CPUとGPU間でメモリ共有）
   - `device.maxThreadsPerThreadgroup`: 1つのスレッドグループの最大サイズ

### よくある間違い

- **「command not found: swift」**: Xcode Command Line Toolsがインストールされていない
  → `xcode-select --install` を実行
- **「Metalが使えません」と表示される**: 古いMacでMetal非対応、またはVMで実行している
- **ファイル名の拡張子が`.swift`でない**: swiftコンパイラがファイルを認識できない

## 演習課題

### 課題1: Metal環境の確認
上記のサンプルコードを実行し、自分のMacのGPU情報を確認してください。

### 課題2: GPU情報の追加表示
以下の情報も表示するようにコードを拡張してみてください：
- `device.supportsFamily(.apple7)` など、対応する機能ファミリーを調べる
- `device.maxBufferLength` で作成可能な最大バッファサイズを表示

### 課題3（発展）: 複数GPUの列挙
`MTLCopyAllDevices()` を使って、システム上のすべてのMetalデバイスを列挙してみてください（Intel Mac + eGPU環境などで複数のGPUが見える場合があります）。

## 評価基準

以下がすべて満たされたらステージクリア：

- [ ] `swift --version` がSwiftのバージョンを表示する
- [ ] サンプルコードが正しく実行でき、GPU名が表示される
- [ ] `hasUnifiedMemory` の値を確認し、自分のMacがUnified Memoryかどうか把握している
- [ ] `maxThreadsPerThreadgroup` の値を確認している

## ヒント集

### ヒント1（軽め）
まず `swift --version` でSwiftがインストールされているか確認しましょう。もしインストールされていなければ、`xcode-select --install` でXcode Command Line Toolsをインストールしてください。

### ヒント2（中程度）
サンプルコードを `check_metal.swift` というファイル名で保存し、以下のコマンドで実行します：
```bash
swift check_metal.swift
```

エラーが出た場合は、ファイル名の拡張子が `.swift` になっているか確認してください。

### ヒント3（具体的）
複数GPUの列挙は以下のようになります：
```swift
import Metal

let devices = MTLCopyAllDevices()
print("検出されたGPU数: \(devices.count)")
for (index, device) in devices.enumerated() {
    print("GPU \(index): \(device.name)")
}
```

Intel Mac + AMD GPUの環境では、Intel統合GPUとAMD外部GPUの両方が表示されることがあります。

## 補足・発展トピック

### なぜMetalか？

**CUDAとの比較**:
- CUDA: NVIDIA GPU専用、強力なエコシステム、科学計算・ML向け
- Metal: Apple製デバイス専用、iOS/macOS両対応、ゲーム・アプリ向け

**Metalの利点**:
- Apple Siliconでの効率的なUnified Memory
- Xcodeなしでコマンドラインから実行可能
- SwiftまたはObjective-Cで記述可能

### 参考リンク

- [Metal - Apple Developer Documentation](https://developer.apple.com/documentation/metal)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
