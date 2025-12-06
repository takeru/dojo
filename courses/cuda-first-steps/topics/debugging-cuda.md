# トピック: GPUのバグってどう見つけるの？エラー処理の基本

## メタ情報

- **ID**: debugging-cuda
- **難易度**: 初級
- **所要時間**: 8-10分
- **カテゴリ**: デバッグ

## 前提知識

- Stage 1完了

## このトピックで学べること

- CUDAのエラーチェック方法
- よくあるエラーとその対処法
- デバッグのコツ

## 関連ステージ

- すべてのステージ

## 要点（ドキュメント形式用）

CUDAのエラーは黙って失敗することが多いので、明示的にチェックが必要です。

### エラーチェックのマクロ

```cuda
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDAエラー: %s:%d, %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使い方
CUDA_CHECK(cudaMalloc(&d_arr, size));
CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));
```

### カーネルのエラーチェック

```cuda
kernel<<<grid, block>>>(args);

// カーネル起動のエラー
CUDA_CHECK(cudaGetLastError());

// カーネル実行完了を待つ
CUDA_CHECK(cudaDeviceSynchronize());
```

### よくあるエラー

| エラー | 原因 | 対処法 |
|--------|------|--------|
| `out of memory` | メモリ不足 | データサイズを減らす、cudaFree忘れチェック |
| `invalid argument` | 引数が不正 | NULLポインタ、サイズ0など |
| `invalid configuration` | ブロックサイズ超過 | 1024以下に |
| `illegal memory access` | 配列外アクセス | if (idx < n) チェック |

### printfデバッグ

```cuda
__global__ void kernel(int *arr, int n) {
    int idx = threadIdx.x;
    if (idx == 0) {  // スレッド0だけ表示
        printf("n = %d\n", n);
    }

    if (idx < n) {
        printf("idx=%d, arr[%d]=%d\n", idx, idx, arr[idx]);
    }
}
```

### cuda-memcheck

```bash
# メモリエラーをチェック
cuda-memcheck ./my_program
```

## 対話形式の教え方ガイド（先生用）

### 導入

「CUDAのエラーは分かりにくいけど、ちゃんとチェックすれば原因が分かるよ」

### 実践課題

1. エラーチェックマクロを追加
2. わざとエラーを起こしてメッセージを確認
3. cuda-memcheckを試す

## クリア条件

- [ ] cudaGetLastErrorを使える
- [ ] CUDA_CHECKマクロを書ける
- [ ] printf デバッグができる

## 補足情報

### 参考リンク

- [CUDA Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)
- [cuda-memcheck](https://docs.nvidia.com/cuda/cuda-memcheck/)
