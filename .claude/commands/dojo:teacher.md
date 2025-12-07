---
description: "先生キャラクターを切り替える（例: /teacher ferris）"
---

# 先生キャラクター切り替え

引数で指定されたキャラクターに切り替えます。

## 使い方
```
/teacher NAME
```

## 利用可能なキャラクター

引数が指定されていない場合は、以下の手順で利用可能なキャラクター一覧を表示してください：

1. `characters/` ディレクトリをGlobツールで読み取る（`characters/*.md`）
2. 各キャラクターファイルを読み込んで、名前とタイプを抽出
3. 表形式で一覧表示
4. AskUserQuestionツールで選択肢を提示

## 処理手順

1. **引数の確認**
   - キャラクター名が指定されているか
   - 指定されていない場合は上記の一覧表示を実行

2. **キャラクターファイル読み込み**
   - `state/progress.json` から `current_course` を取得
   - 以下の順序で探索（最初に見つかったものを使用）：
     1. `courses/{current_course}/characters/NAME.md`（コース専用キャラ）
     2. `characters/NAME.md`（共通キャラ）
   - 口調、性格、教え方を把握

3. **進捗更新**
   - `state/progress.json` の `current_teacher` を更新

4. **切り替えの案内**
   - キャラクターファイルから読み取った口調・性格に基づいて自己紹介
   - 現在のステージの状況を確認

$ARGUMENTS
