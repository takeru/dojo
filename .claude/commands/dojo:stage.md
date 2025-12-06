---
description: "指定したステージに移動する（例: /stage 2）"
---

# ステージ切り替え

引数で指定されたステージに移動します。

## 使い方
```
/stage N
```
Nはステージ番号（1-9）

## 処理手順

1. **状態の確認**
   - `state/progress.json` から `current_course` を取得
   - ステージ番号が指定されているか確認（引数 $ARGUMENTS から取得）

2. **course.json からファイル名を取得**
   - `courses/{current_course}/course.json` を読み込む
   - `stages` 配列から指定されたステージ番号を探す
   - 該当ステージの `file` フィールドからファイル名を取得
   - `total_stages` で有効なステージ範囲を確認

   例：ステージ3の場合
   ```json
   {
     "stage": 3,
     "file": "stage3-variables.md",
     "title": "変数と型"
   }
   ```

3. **カリキュラム読み込み**
   - `courses/{current_course}/curriculum/{file}` を読み込む
   - ステージの目標、教え方、評価基準を把握

4. **進捗更新**
   - `state/progress.json` の `courses[current_course].current_stage` を更新
   - `last_activity` にタイムスタンプを記録

5. **ステージ開始の案内**
   - 現在のキャラクターの口調で
   - 新しいステージの目標を説明
   - 最初の課題を提示

## 注意事項

- 前のステージをスキップして進むことも可能（生徒の自由）
- ただし、前提知識が必要な場合は注意を促す
- キャラクターの口調を維持する

## ステージ一覧

ステージ一覧は `courses/{current_course}/course.json` の `stages` フィールドを参照してください。

$ARGUMENTS
