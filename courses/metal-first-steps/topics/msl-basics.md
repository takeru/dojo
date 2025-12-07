# トピック: Metal Shading Language入門

## メタ情報

- **ID**: msl-basics
- **難易度**: 初級
- **所要時間**: 10-15分（対話形式）/ 5分（読み物）
- **カテゴリ**: 言語機能

## 前提知識

- Stage 2完了（最初のCompute Shader）
- C言語の基本構文

## このトピックで学べること

- MSL (Metal Shading Language) の基本文法
- C++14ベースの構文と拡張
- よく使う属性（Attribute）の意味

## 関連ステージ

- Stage 2: 最初のCompute Shader

## 要点（ドキュメント形式用）

### MSLとは

Metal Shading Language (MSL) は、C++14をベースにした GPU シェーダ言語です。Metal専用で、Apple デバイスのGPU上で実行されるコードを記述します。

### 基本構文

```metal
#include <metal_stdlib>
using namespace metal;

// カーネル関数（GPU上で並列実行）
kernel void myKernel(
    device float* data [[buffer(0)]],      // GPUメモリへのポインタ
    constant float& scale [[buffer(1)]],   // 読み取り専用の定数
    uint id [[thread_position_in_grid]]    // スレッドID
) {
    data[id] = data[id] * scale;
}
```

### アドレス空間

| 空間 | 意味 | 用途 |
|-----|------|------|
| `device` | GPUメモリ | 読み書き可能なバッファ |
| `constant` | 定数メモリ | 読み取り専用、全スレッド共有 |
| `threadgroup` | 共有メモリ | スレッドグループ内で共有 |
| `thread` | プライベート | スレッドローカル変数 |

### 属性（Attribute）

```metal
[[buffer(N)]]              // バッファインデックス
[[texture(N)]]             // テクスチャインデックス
[[thread_position_in_grid]]       // グリッド内のスレッド位置
[[thread_position_in_threadgroup]] // グループ内のスレッド位置
[[threadgroup_position_in_grid]]   // グループの位置
[[threads_per_threadgroup]]        // グループ内のスレッド数
```

### 組み込み型

```metal
// スカラー型
float, half, int, uint, bool

// ベクトル型（2, 3, 4成分）
float2, float3, float4
int2, int3, int4
uint2, uint3, uint4

// 行列型
float2x2, float3x3, float4x4

// アクセス
float4 v = float4(1.0, 2.0, 3.0, 4.0);
float x = v.x;        // 1.0
float2 xy = v.xy;     // (1.0, 2.0)
float4 rgba = v.rgba; // swizzle
```

### 組み込み関数

```metal
// 数学関数
sin(x), cos(x), tan(x)
sqrt(x), pow(x, y), exp(x), log(x)
abs(x), min(a, b), max(a, b), clamp(x, min, max)

// ベクトル関数
dot(a, b)      // 内積
cross(a, b)    // 外積
length(v)      // ベクトル長
normalize(v)   // 正規化
distance(a, b) // 距離

// 補間
mix(a, b, t)   // 線形補間: a * (1-t) + b * t

// 同期
threadgroup_barrier(mem_flags::mem_threadgroup)
```

## 対話形式の教え方ガイド（先生用）

### 導入

「MSLはC++にとても似ていますが、GPU特有の機能がいくつか追加されています。最初は戸惑うかもしれませんが、パターンを覚えれば簡単です」

### 説明の流れ

1. **まず基本形を見せる**
   ```metal
   kernel void example(
       device float* data [[buffer(0)]],
       uint id [[thread_position_in_grid]]
   ) {
       data[id] *= 2.0;
   }
   ```

   「この3つの要素を覚えましょう：
   - `kernel` = GPU関数の印
   - `device float*` = GPUメモリのポインタ
   - `[[thread_position_in_grid]]` = このスレッドの番号」

2. **C++との違いを説明**
   - ポインタの前にアドレス空間（`device`, `constant`など）が必要
   - 引数に属性（`[[buffer(N)]]`など）を付ける
   - `printf` は使えない（デバッグは結果をバッファに書いて確認）

3. **ベクトル型を実演**
   ```metal
   float4 color = float4(1.0, 0.5, 0.0, 1.0);  // RGBA: オレンジ
   float r = color.r;    // 1.0
   float3 rgb = color.rgb;  // (1.0, 0.5, 0.0)

   // 計算も要素ごとに適用
   float4 doubled = color * 2.0;  // (2.0, 1.0, 0.0, 2.0)
   ```

4. **実際にコードを書かせる**
   「配列の各要素を2乗するシェーダを書いてみてください」

   答え:
   ```metal
   kernel void square(
       device float* data [[buffer(0)]],
       uint id [[thread_position_in_grid]]
   ) {
       float x = data[id];
       data[id] = x * x;
   }
   ```

### 実践課題（オプション）

1. ベクトルの正規化シェーダを書く
2. 2つの配列の内積を計算するシェーダを書く

## クリア条件（オプション）

- [ ] `device`, `constant`, `threadgroup` の違いを説明できる
- [ ] `[[buffer(N)]]` の意味を説明できる
- [ ] 簡単なカーネル関数を自分で書ける

## 補足情報

### C++14との主な違い

- 例外処理なし
- 仮想関数なし
- `new`/`delete` なし
- 標準ライブラリなし（`<metal_stdlib>` を使う）
- 再帰呼び出しは制限あり

### 参考リンク

- [Metal Shading Language Specification (PDF)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Shader Functions](https://developer.apple.com/documentation/metal/shader_libraries/metal_shading_language_functions)
