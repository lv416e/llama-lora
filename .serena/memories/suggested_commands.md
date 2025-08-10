# 重要コマンド一覧

## 🚀 基本開発ワークフロー

### 1. 環境セットアップ
```bash
# プロジェクトディレクトリに移動
cd /path/to/llama-lora

# 依存関係のインストール
uv sync

# Hugging Face認証（初回のみ）
huggingface-cli login
```

### 2. 訓練・推論・マージワークフロー
```bash
# ① ベースライン評価（オプション）
python scripts/baseline_inference.py

# ② DoRA/LoRAファインチューニング実行
python scripts/train.py

# ③ ファインチューニング済みモデルでの推論テスト
python scripts/infer.py "富士山の標高は？"

# ④ アダプターをベースモデルに統合（スタンドアロン化）
python scripts/merge.py
```

## 🔧 開発・デバッグコマンド

### パッケージ管理（uv）
```bash
# 依存関係の同期
uv sync

# 新しい依存関係の追加
uv add package_name

# 依存関係の更新
uv lock --upgrade

# 仮想環境の確認
uv venv --python 3.12
```

### コード品質管理
```bash
# Ruffによるリンティング
uv run ruff check .

# Ruffによる自動修正
uv run ruff check --fix .

# Ruffによるフォーマット
uv run ruff format .
```

## 🍎 macOS (Darwin) 固有コマンド

### MPS環境設定
```bash
# MPS環境変数設定（必要に応じて）
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACCELERATE_USE_MPS=true

# システムリソース監視
top -o MEM    # メモリ使用量監視
activity_monitor  # GUI版リソースモニター
```

### ファイル・ディレクトリ操作
```bash
# ファインダーで開く
open .
open ./out-llama-lora

# ファイル検索
find . -name "*.py" -type f
find . -name "*lora*" -type d

# テキスト検索
grep -r "USE_DORA" scripts/
grep -r "MODEL_ID" . --include="*.py"
```

## 📊 推論コマンド詳細

### 基本推論
```bash
# 日本語プロンプト
python scripts/infer.py "日本語で簡潔に答えて。富士山の標高は？"

# 英語プロンプト
python scripts/infer.py "What is the height of Mount Fuji?"

# パラメータ付き推論
python scripts/infer.py "Your prompt here" --max_new_tokens 64 --temperature 0.7 --top_p 0.9
```

### ベースライン比較
```bash
# ベースモデルの性能確認
python scripts/baseline_inference.py

# ファインチューニング後の比較
python scripts/infer.py "同じプロンプト"
```

## 🔍 デバッグ・開発支援

### ログ・出力確認
```bash
# 訓練ログの確認
python scripts/train.py 2>&1 | tee training.log

# GPU使用量監視（CUDA環境）
nvidia-smi

# プロセス監視
ps aux | grep python
```

### Jupyter開発
```bash
# Jupyterの起動
uv run jupyter lab

# ノートブック実行
jupyter nbconvert --execute examples/tiny-llama-dora-test.ipynb
```

## ⚡ 緊急時・トラブルシューティング

### メモリ不足対応
```bash
# キャッシュクリア
python -c "import torch; torch.mps.empty_cache()"

# プロセス強制終了
pkill -f "python.*train.py"
```

### 設定リセット
```bash
# 出力ディレクトリクリア
rm -rf ./out-llama-lora

# uvキャッシュクリア
uv cache clean
```

## 🎯 頻用コマンド組み合わせ
```bash
# 完全なワークフロー実行
python scripts/train.py && python scripts/infer.py "テストプロンプト" && python scripts/merge.py

# 品質チェック後の実行
uv run ruff check --fix . && python scripts/train.py
```