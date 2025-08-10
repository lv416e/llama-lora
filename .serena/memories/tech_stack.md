# 技術スタック詳細

## 言語・ランタイム
- **Python**: 3.12+ （.python-version で固定）
- **依存関係管理**: uv（高速なPythonパッケージマネージャー）

## 主要ライブラリ・バージョン
```toml
[project.dependencies]
- accelerate>=1.10.0      # 分散学習・最適化
- datasets>=4.0.0         # データセット管理
- jupyter>=1.1.1          # ノートブック環境
- peft>=0.17.0           # Parameter-Efficient Fine-Tuning（DoRA対応）
- ruff>=0.12.8           # 高速リンター・フォーマッター
- torch>=2.8.0           # PyTorch本体
- transformers>=4.55.0   # Hugging Face Transformers
```

## 機械学習フレームワーク構成
### Core ML Stack
- **PyTorch 2.8.0+**: テンソル計算・自動微分エンジン
- **Transformers 4.55.0**: 事前訓練済みモデル・トークナイザー
- **PEFT 0.17.0**: LoRA/DoRA実装（最新版でDoRA完全対応）
- **Datasets 4.0.0**: データセット処理・変換
- **Accelerate 1.10.0**: 分散学習・混合精度・デバイス管理

### 開発・品質管理
- **Ruff 0.12.8**: 高速リンター・フォーマッター（black/flake8代替）
- **Jupyter 1.1.1**: インタラクティブ開発・実験環境

## PEFT設定詳細
```python
# LoRA/DoRA設定
USE_DORA = True                    # DoRA有効化
PEFT_R = 16                       # ランク（低ランク行列の次元）
PEFT_LORA_ALPHA = 32              # スケーリングファクター
PEFT_LORA_DROPOUT = 0.05          # ドロップアウト率

# ターゲットモジュール（Llama専用設定）
PEFT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",    # アテンション層
    "gate_proj", "up_proj", "down_proj",        # MLP層
]
```

## システム要件・対応環境
- **プライマリ環境**: macOS (Darwin) - M2 Max MPS
- **対応デバイス**: CUDA、MPS、CPU（自動検出）
- **メモリ最適化**: gradient checkpointing、小バッチ設計
- **Gated Model対応**: Hugging Face認証必須（meta-llama モデル）

## パフォーマンス設定
- **バッチサイズ**: 1（メモリ制約対応）
- **勾配累積**: 8（実効バッチサイズ = 8）
- **シーケンス長**: 1024トークン
- **学習率**: 2e-5（保守的設定）
- **混合精度**: デバイス依存（CUDA: fp16/bf16, MPS: fp32）