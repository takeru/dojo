# トピック: 自分のプログラムがどう動いてるか見てみよう（Nsight入門）

## メタ情報

- **ID**: visual-profiling
- **難易度**: 中級
- **所要時間**: 10-15分（対話形式）/ 3-5分（読み物）
- **カテゴリ**: デバッグ・プロファイリング

## 前提知識

- Stage 7完了（行列積などの実用的なプログラムを書ける）
- 基本的なCUDAプログラムが書ける

## このトピックで学べること

- Nsight Systemsの使い方
- プログラムのどこで時間がかかっているか可視化する方法
- ボトルネックの見つけ方

## 関連ステージ

- Stage 7: 行列積（プロファイリングの題材として最適）

## 要点（ドキュメント形式用）

Nsightは、CUDAプログラムの実行を可視化・分析するツールです。

### Nsight Systemsの起動

```bash
# コマンドラインでプロファイル取得
nsys profile --stats=true ./my_program

# GUI版の起動（結果を視覚的に見る）
nsight-sys
```

### プロファイル結果の見方

タイムライン表示で以下が可視化されます：
- CPU処理の時間
- GPU処理の時間
- メモリ転送（CPU⇔GPU）の時間
- カーネルの実行時間

### よくある発見

1. **メモリ転送が遅い**: 転送時間 >> 計算時間
2. **カーネルが遅い**: 最適化の余地あり
3. **GPU が遊んでいる**: CPU処理が長すぎる

### 簡単な使い方

```bash
# プログラムをプロファイル
nsys profile -o my_report ./my_program

# レポートを表示
nsys stats my_report.nsys-rep

# GUI で開く
nsight-sys my_report.nsys-rep
```

## 対話形式の教え方ガイド（先生用）

### 導入

「プログラムは動いてるけど、どこで時間がかかってるか分からないよね。Nsightを使えば、プログラムの実行を映画のように可視化できるんだ」

なぜこれを知っておくと便利か：
- ボトルネックが一目で分かる
- 最適化の方針が立てやすい
- GPU利用率が可視化される

### 説明の流れ

1. **簡単なプロファイルを取得**

   ```bash
   # 行列積プログラムをプロファイル
   nsys profile --stats=true ./matrix_mul
   ```

   実行後、統計情報が表示されます：
   ```
   [CUDA API]
   Time(%)   Total Time (ns)   Num Calls   Avg   Name
   -------   ---------------   ---------   ---   ----
   45.2%     1,234,567         2           ...   cudaMemcpy
   30.1%     823,456           1           ...   matrixMul (kernel)
   ...
   ```

2. **GUI で視覚化**

   ```bash
   nsys profile -o report ./matrix_mul
   nsight-sys report.nsys-rep
   ```

   タイムライン表示で、転送と計算が時系列で見える！

3. **ボトルネックの発見**

   「この例だと、メモリ転送が45%も占めてる。計算は30%。つまり、転送を減らせば速くなるってことだね」

4. **改善案を提示**

   - Pinned Memoryで転送を高速化
   - Streamsで転送と計算を並行実行
   - データをGPU上に残して転送を減らす

### 実践課題（オプション）

1. 自分の行列積プログラムをプロファイル
2. メモリ転送時間と計算時間の比率を確認
3. Nsight SystemsのGUIでタイムラインを見てみる

## クリア条件（オプション）

理解度チェック：
- [ ] Nsight Systemsでプロファイルを取得できる
- [ ] プロファイル結果から転送時間と計算時間を読み取れる
- [ ] ボトルネックを特定できる

## 補足情報

### Nsight ComputeとNsight Systemsの違い

| ツール | 用途 |
|--------|------|
| Nsight Systems | システム全体のタイムライン表示 |
| Nsight Compute | カーネル内部の詳細分析 |

まずはNsight Systemsで全体を把握し、必要ならNsight Computeで深掘りします。

### 便利なオプション

```bash
# CUDA API呼び出しをトレース
nsys profile --trace=cuda ./program

# より詳細な統計
nsys profile --stats=true --force-overwrite=true -o report ./program
```

### 参考リンク

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
