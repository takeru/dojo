# トピック: テクスチャ処理の基礎

## メタ情報

- **ID**: texture-processing
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 画像処理

## 前提知識

- Stage 5完了（画像処理の基本）

## このトピックで学べること

- MTLTexture の基本的な使い方
- バッファとテクスチャの違い
- フィルタリングとサンプリング

## 関連ステージ

- Stage 5: 画像処理（グレースケール変換）

## 要点（ドキュメント形式用）

### バッファ vs テクスチャ

| 特性 | バッファ | テクスチャ |
|------|----------|------------|
| データ構造 | 1次元配列 | 1D/2D/3D/Cube |
| アクセス | インデックス直接指定 | 座標で指定 |
| フィルタリング | なし | 線形/最近傍補間 |
| 境界処理 | 手動 | clamp/repeat/mirror |
| キャッシュ | 一般的 | 2D局所性に最適化 |

### テクスチャの作成

```swift
// テクスチャディスクリプタ
let descriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,    // 8bit RGBA
    width: width,
    height: height,
    mipmapped: false
)
descriptor.usage = [.shaderRead, .shaderWrite]
descriptor.storageMode = .shared  // Apple Silicon推奨

// テクスチャ作成
let texture = device.makeTexture(descriptor: descriptor)!

// データを書き込み
let region = MTLRegion(
    origin: MTLOrigin(x: 0, y: 0, z: 0),
    size: MTLSize(width: width, height: height, depth: 1)
)
texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData, bytesPerRow: width * 4)
```

### Shaderでのテクスチャアクセス

```metal
#include <metal_stdlib>
using namespace metal;

kernel void texture_example(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 pos [[thread_position_in_grid]]
) {
    // 読み取り
    float4 color = input.read(pos);

    // 処理
    float gray = dot(color.rgb, float3(0.299, 0.587, 0.114));

    // 書き込み
    output.write(float4(gray, gray, gray, color.a), pos);
}
```

### アクセスモード

```metal
texture2d<float, access::read>        // 読み取り専用
texture2d<float, access::write>       // 書き込み専用
texture2d<float, access::read_write>  // 読み書き両方
texture2d<float, access::sample>      // サンプリング可能（フィルタリング）
```

### サンプリング（補間あり）

```metal
// サンプラーを使った補間読み取り
kernel void sample_example(
    texture2d<float, access::sample> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    sampler s [[sampler(0)]],
    uint2 pos [[thread_position_in_grid]]
) {
    // 正規化座標（0.0〜1.0）でサンプリング
    float2 texSize = float2(input.get_width(), input.get_height());
    float2 uv = (float2(pos) + 0.5) / texSize;

    // 線形補間された値を取得
    float4 color = input.sample(s, uv);

    output.write(color, pos);
}
```

Swift側のサンプラー設定:
```swift
let samplerDescriptor = MTLSamplerDescriptor()
samplerDescriptor.minFilter = .linear      // 縮小時の補間
samplerDescriptor.magFilter = .linear      // 拡大時の補間
samplerDescriptor.sAddressMode = .clampToEdge  // 境界処理
samplerDescriptor.tAddressMode = .clampToEdge

let sampler = device.makeSamplerState(descriptor: samplerDescriptor)!
encoder.setSamplerState(sampler, index: 0)
```

### 畳み込みフィルタの例（ぼかし）

```metal
kernel void box_blur(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 pos [[thread_position_in_grid]]
) {
    // 3x3 ボックスフィルタ
    float4 sum = float4(0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint2 samplePos = uint2(
                clamp(int(pos.x) + dx, 0, int(input.get_width()) - 1),
                clamp(int(pos.y) + dy, 0, int(input.get_height()) - 1)
            );
            sum += input.read(samplePos);
        }
    }
    output.write(sum / 9.0, pos);
}
```

## 対話形式の教え方ガイド（先生用）

### 導入

「Stage 5 ではバッファを使って画像処理をしましたが、実はテクスチャを使う方が画像処理向きです。フィルタリングや境界処理が自動でできるんです」

### 説明の流れ

1. **バッファとテクスチャの違いを説明**
   「バッファは1次元の配列。テクスチャは2D/3Dの画像データ用で、GPUが画像処理に最適化されています」

2. **テクスチャアクセスの2種類**
   - `read(pos)`: 整数座標で直接読む
   - `sample(sampler, uv)`: 正規化座標で補間読み取り

3. **境界処理の自動化**
   「バッファだと `if (x < 0 || x >= width)` のチェックが必要でしたが、テクスチャなら `clampToEdge` で自動処理されます」

4. **実際に比較**
   Stage 5 のグレースケール処理をテクスチャ版で実装して比較

### 実践課題（オプション）

1. ぼかしフィルタをテクスチャで実装
2. エッジ検出（Sobelフィルタ）を実装
3. サンプラーを使って画像の拡大縮小を実装

## クリア条件（オプション）

- [ ] バッファとテクスチャの違いを説明できる
- [ ] `read` と `sample` の違いを説明できる
- [ ] 簡単な画像フィルタをテクスチャで実装できる

## 補足情報

### ピクセルフォーマット

```swift
.rgba8Unorm      // 8bit RGBA（0〜255 → 0.0〜1.0）
.rgba8Snorm      // 8bit RGBA（符号付き）
.rgba16Float     // 16bit 浮動小数点
.rgba32Float     // 32bit 浮動小数点（高精度）
.r8Unorm         // 8bit グレースケール
```

### 参考リンク

- [Working with Textures](https://developer.apple.com/documentation/metal/textures)
- [Creating and Sampling Textures](https://developer.apple.com/documentation/metal/textures/creating_and_sampling_textures)
