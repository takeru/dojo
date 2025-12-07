# トピック: Threadgroupメモリで高速化

## メタ情報

- **ID**: threadgroup-memory
- **難易度**: 上級
- **所要時間**: 15-20分（対話形式）/ 7分（読み物）
- **カテゴリ**: パフォーマンス最適化

## 前提知識

- Stage 4完了（スレッドグループの理解）
- Stage 6完了（行列演算の基本）

## このトピックで学べること

- Threadgroupメモリ（共有メモリ）の使い方
- タイリングによる行列積の高速化
- バリア同期の重要性

## 関連ステージ

- Stage 4: スレッドグループとディスパッチ
- Stage 6: 行列演算

## 要点（ドキュメント形式用）

### Threadgroupメモリとは

スレッドグループ内のスレッドが共有するオンチップメモリ:

| メモリ | 速度 | 容量 | スコープ |
|--------|------|------|----------|
| レジスタ | 最速 | 最小 | スレッド内 |
| **Threadgroup** | 高速 | 中 | スレッドグループ内 |
| Device (Global) | 低速 | 大 | 全スレッド |

### 基本的な使い方

```metal
kernel void example(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Threadgroupメモリを宣言（全スレッドで共有）
    threadgroup float shared_data[256];

    // グローバルメモリからThreadgroupメモリへロード
    shared_data[local_id] = input[id];

    // 同期: 全スレッドがロード完了を待つ
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroupメモリを使った計算（高速）
    float result = shared_data[local_id] + shared_data[(local_id + 1) % group_size];

    // 同期: 計算完了を待つ
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 結果をグローバルメモリへ書き戻す
    output[id] = result;
}
```

### バリア同期

`threadgroup_barrier()` は**必須**:
- これがないと、他のスレッドがまだデータを書いていない状態で読んでしまう
- データ競合（Race Condition）の原因

```metal
// 悪い例: バリアなし
shared_data[local_id] = input[id];
float val = shared_data[local_id + 1];  // 危険！まだ書かれていないかも

// 良い例: バリアあり
shared_data[local_id] = input[id];
threadgroup_barrier(mem_flags::mem_threadgroup);
float val = shared_data[local_id + 1];  // OK: 全スレッドが書き込み完了
```

### タイリングによる行列積高速化

ナイーブ実装では、各要素の計算で K 回のグローバルメモリアクセス:

```
C[i][j] = Σ A[i][k] * B[k][j]  // K回のメモリアクセス
```

タイリング実装では、タイルをThreadgroupメモリにロードして再利用:

```metal
#define TILE_SIZE 16

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 global_id [[thread_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    // Threadgroupメモリ（タイル）
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint row = global_id.y;
    uint col = global_id.x;

    float sum = 0.0f;

    // タイルごとに処理
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // タイルをロード
        uint aRow = row;
        uint aCol = t * TILE_SIZE + local_id.x;
        uint bRow = t * TILE_SIZE + local_id.y;
        uint bCol = col;

        tileA[local_id.y][local_id.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        tileB[local_id.y][local_id.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // タイル内で積和演算（高速メモリから読む）
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_id.y][k] * tileB[k][local_id.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 高速化の理由

- **メモリアクセス削減**: タイル内の要素は TILE_SIZE 回再利用
- **帯域幅効率**: グローバルメモリアクセスが 1/TILE_SIZE に
- **キャッシュ効率**: オンチップメモリはレイテンシが低い

## 対話形式の教え方ガイド（先生用）

### 導入

「GPUの計算は速いですが、メモリアクセスがボトルネックになることが多いです。Threadgroupメモリを使うと、頻繁にアクセスするデータをオンチップに置けて、劇的に速くなります」

### 説明の流れ

1. **メモリ階層を説明**
   ```
   CPU ←→ Device Memory ←→ Threadgroup Memory ←→ Registers
           (遅い・大)         (中速・中)          (最速・小)
   ```

2. **簡単な例で実演**
   「隣のスレッドの値を読みたい場合：」
   ```metal
   // 遅い: 毎回グローバルメモリから読む
   output[id] = input[id] + input[id + 1];

   // 速い: 一度Threadgroupメモリに載せて再利用
   threadgroup float shared[256];
   shared[local_id] = input[id];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   output[id] = shared[local_id] + shared[local_id + 1];
   ```

3. **バリアの重要性を強調**
   「`threadgroup_barrier()` を忘れると、まだ書かれていないデータを読んでしまいます。必ず入れましょう」

4. **行列積への応用**
   「行列積では、同じ要素を何度も読みます。タイリングでThreadgroupメモリに載せれば、グローバルメモリアクセスを大幅に減らせます」

### 実践課題（オプション）

1. Stage 6 の行列積をタイリング版に書き換えて、速度を比較
2. TILE_SIZE を 8, 16, 32 と変えて最適値を探す

## クリア条件（オプション）

- [ ] `threadgroup` アドレス空間の意味を説明できる
- [ ] `threadgroup_barrier()` が必要な理由を説明できる
- [ ] タイリングが高速化に効く理由を説明できる

## 補足情報

### Threadgroupメモリのサイズ上限

```swift
// デバイスの上限を確認
let maxMemory = device.maxThreadgroupMemoryLength
print("Threadgroupメモリ上限: \(maxMemory) bytes")
// Apple Silicon: 通常 32KB
```

### 参考リンク

- [Using Threadgroup Memory](https://developer.apple.com/documentation/metal/using_threadgroup_memory)
- [Optimizing Compute Shaders](https://developer.apple.com/documentation/metal/compute_passes/optimizing_compute_shaders)
