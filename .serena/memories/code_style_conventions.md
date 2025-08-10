# コーディング規約・スタイルガイド

## 全般的なコード品質
- **モジュラー設計**: 機能別スクリプト分離（config.py で設定一元管理）
- **型ヒント**: 部分的に実装（引数の型指定あり）
- **docstring**: 重要関数には簡潔なdocstring（"""形式）
- **エラーハンドリング**: ファイル存在チェック、適切なエラーメッセージ

## インポート規約
```python
# 標準ライブラリ
import os
import warnings

# サードパーティライブラリ
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# プロジェクト内モジュール
import config
```

## 命名規則
### 定数（config.py）
```python
# 全て大文字、アンダースコア区切り
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PEFT_TARGET_MODULES = ["q_proj", "k_proj", ...]
BASE_OUTPUT_DIR = "./out-llama-lora"
```

### 変数・関数
```python
# snake_case形式
device = "cpu"
tok_ds = dataset.map(format_dataset, ...)
training_args = TrainingArguments(...)

# 関数名も snake_case
def format_dataset(e):
    return tok(s, truncation=True, ...)

def main():
    # メイン処理
```

## コード構造パターン
### メイン関数の構造
```python
def main():
    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    # --- セクション別処理 ---
    # 明確なコメントでセクション分割
```

### 設定パターン
```python
# デバイス依存設定の条件分岐
use_fp16 = device == "cuda"  # MPS対応

# 設定ファイルからの値参照
r=config.PEFT_R,
lora_alpha=config.PEFT_LORA_ALPHA,
```

## docstring形式
```python
def generate(model, tokenizer, prompt, max_new_tokens=128, ...):
    \"\"\"
    Generates text from a prompt using the given model and tokenizer.
    \"\"\"
    # 実装
```

## エラーハンドリング
```python
# ファイル存在チェック
if not os.path.isdir(config.ADAPTER_DIR):
    print(f"Error: Adapter directory not found at '{config.ADAPTER_DIR}'")
    print("Please run train.py to create an adapter first.")
    return

# 条件チェック
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
```

## コメント規約
- **セクション区切り**: `# --- Section Name ---`
- **インライン説明**: 重要な設定・判定に簡潔なコメント
- **設定説明**: `# Enable for CUDA`等、設定理由を明記

## 品質管理ツール
- **Ruff**: 自動リンティング・フォーマット
- **警告制御**: `warnings.filterwarnings("ignore", ...)`で適切に制御
- **printベースログ**: 進捗・状態表示（将来的にlogging推奨）