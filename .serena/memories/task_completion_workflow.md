# タスク完了時のワークフロー

## 🎯 コード変更後の必須チェックリスト

### 1. コード品質チェック
```bash
# Ruffによるリンティング・フォーマット（必須）
uv run ruff check --fix .
uv run ruff format .

# 型チェック（推奨、型ヒント有りの場合）
# 現在は部分実装のため、将来的に導入
```

### 2. 機能テスト
#### 設定変更後
```bash
# 設定値の妥当性確認
python -c "import scripts.config as config; print(f'MODEL: {config.MODEL_ID}'); print(f'DORA: {config.USE_DORA}'); print(f'LR: {config.LR}')"
```

#### 訓練スクリプト変更後
```bash
# 短時間テスト実行（設定でEPOCHS=1、DATASET_SPLIT小さく）
python scripts/train.py

# 正常終了確認 → アダプター生成確認
ls -la ./out-llama-lora/adapter/
```

#### 推論スクリプト変更後
```bash
# 既存アダプターでの推論テスト
python scripts/infer.py "テスト用プロンプト"

# エラーハンドリング確認
python scripts/infer.py "test" --max_new_tokens 10
```

### 3. システム互換性確認
```bash
# M2 Max MPS対応確認
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

# デバイス自動検出テスト
python -c "
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f'Detected device: {device}')
"
```

## 📝 変更履歴の記録

### Git管理（推奨）
```bash
# 変更内容の確認
git status
git diff

# 意味のあるコミット
git add .
git commit -m "feat: DoRAランタイム最適化を追加

- LoraRuntimeConfigでephemeral_gpu_offloadを有効化
- MPSでのメモリ最適化処理を改善
- エラーハンドリングを強化"

# プッシュ前の最終確認
git log --oneline -5
```

### 設定変更の記録
```bash
# config.py変更時は影響範囲を確認
grep -r "config\." scripts/

# 重要な設定変更は設定ファイルにコメント追加
# 例: LR = 2e-5  # 以前2e-4から変更、安定性向上のため
```

## 🧪 本格運用前の統合テスト

### 完全ワークフローテスト
```bash
# フルパイプラインテスト
echo "=== ベースライン評価 ==="
python scripts/baseline_inference.py

echo "=== 訓練実行 ==="
python scripts/train.py

echo "=== ファインチューニング後推論 ==="
python scripts/infer.py "富士山の標高は？"

echo "=== アダプター統合 ==="
python scripts/merge.py

echo "=== 統合モデル確認 ==="
ls -la ./out-llama-lora/merged/

echo "=== テスト完了 ==="
```

### パフォーマンス確認
```bash
# メモリ使用量監視下での実行
/usr/bin/time -l python scripts/train.py

# 処理時間測定
time python scripts/infer.py "テストプロンプト"
```

## 🚨 問題発生時の対処

### 即座に実行すべき確認事項
```bash
# 1. 依存関係の問題
uv sync

# 2. 環境変数の確認
echo $PYTORCH_ENABLE_MPS_FALLBACK

# 3. 出力ディレクトリの権限
ls -la ./out-llama-lora/

# 4. ディスク容量確認
df -h .

# 5. プロセス確認
ps aux | grep python
```

### ロールバック手順
```bash
# Git履歴から復元
git log --oneline
git checkout HEAD~1 scripts/train.py

# 設定リセット
git checkout HEAD scripts/config.py
```

## ✅ 完了確認項目

### 必須チェックリスト
- [ ] Ruffによるコード品質チェック通過
- [ ] 最低1回の機能動作確認
- [ ] エラーメッセージの適切性確認
- [ ] M2 Max MPS環境での動作確認

### 推奨チェックリスト
- [ ] Git履歴への適切なコミット
- [ ] 設定変更のドキュメント更新
- [ ] パフォーマンス影響の確認
- [ ] 既存機能への影響評価

### 品質基準
- コードは既存スタイルに準拠
- エラーハンドリングが適切
- メモリ効率が維持されている
- M2 Max MPS対応が損なわれていない