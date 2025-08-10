# LoRA/DoRAアーキテクチャ実装詳細

## 🏗️ 実装アーキテクチャ概要

### コアコンポーネント構成
```
llama-lora 実装アーキテクチャ
├── 設定管理層 (config.py)
│   ├── モデル・データセット設定
│   ├── PEFT/LoRA パラメータ
│   └── 訓練ハイパーパラメータ
├── 訓練パイプライン (train.py)
│   ├── デバイス自動検出・最適化
│   ├── データ前処理・トークナイゼーション
│   ├── PEFTモデル構築・DoRA設定
│   └── Trainer統合・アーティファクト保存
├── 推論システム (infer.py)
│   ├── アダプター読み込み・結合
│   ├── 生成パラメータ制御
│   └── エラーハンドリング・結果出力
└── 統合・評価 (merge.py, baseline_inference.py)
    ├── アダプター統合・スタンドアロン化
    └── ベースライン比較・評価
```

## 🔬 DoRA (Weight-Decomposed Low-Rank Adaptation) 実装

### DoRA技術的詳細
- **標準LoRA拡張**: 重み分解による高精度適応
- **実装**: `use_dora=True` でPEFT 0.17.0の最新DoRA機能を活用
- **対象層**: Llamaアーキテクチャの全プロジェクション層に適用

### PEFT設定詳細
```python
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,           # 因果言語モデリング
    r=16,                                   # ランク（低ランク行列の次元）
    lora_alpha=32,                         # スケーリング係数（α）
    lora_dropout=0.05,                     # 正則化ドロップアウト
    target_modules=[
        # アテンション機構
        "q_proj",      # Query射影
        "k_proj",      # Key射影  
        "v_proj",      # Value射影
        "o_proj",      # Output射影
        
        # MLP（Feed-Forward）層
        "gate_proj",   # Gateメカニズム
        "up_proj",     # アップ射影
        "down_proj",   # ダウン射影
    ],
    use_dora=True,                         # DoRA有効化
)
```

## 🧠 Llama-3.2アーキテクチャ対応

### モデル特性
- **モデル**: meta-llama/Llama-3.2-1B-Instruct
- **パラメータ数**: 約1B（効率的なファインチューニング対象）
- **アーキテクチャ**: Transformer Decoder with RMSNorm, SwiGLU
- **認証**: Gated Model（Hugging Face認証必須）

### ターゲット層戦略
```
Llama-3.2 Layer Structure:
├── Attention Layers
│   ├── q_proj: Query変換（DoRA適用）
│   ├── k_proj: Key変換（DoRA適用）
│   ├── v_proj: Value変換（DoRA適用）
│   └── o_proj: Output変換（DoRA適用）
└── MLP Layers
    ├── gate_proj: SwiGLUゲート（DoRA適用）
    ├── up_proj: アップサンプリング（DoRA適用）
    └── down_proj: ダウンサンプリング（DoRA適用）
```

## ⚡ デバイス最適化アーキテクチャ

### M2 Max MPS対応
```python
# デバイス検出ロジック
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # M2 Max対応
    device = "mps"

# MPS特化設定
use_fp16 = device == "cuda"  # MPSでは混合精度無効化
training_args = TrainingArguments(
    fp16=use_fp16,           # CUDAのみ有効
    bf16=use_fp16,           # CUDAのみ有効
    optim="adamw_torch",     # MPS互換最適化器
)
```

### メモリ効率化戦略
```python
# 1. Gradient Checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()

# 2. 小バッチ + 勾配累積
per_device_train_batch_size=1    # 最小バッチ
gradient_accumulation_steps=8    # 実効バッチ = 8

# 3. シーケンス長制限
max_length=1024                  # メモリ制約対応
```

## 🔄 データフロー・パイプライン

### 1. 前処理パイプライン
```python
def format_dataset(example):
    # Alpaca形式テンプレート
    prompt = f"### Instruction:\n{example['instruction']}\n"
    prompt += f"### Input:\n{example['input']}\n"
    prompt += f"### Response:\n{example['output']}"
    
    # トークナイゼーション・パディング
    return tokenizer(prompt, 
                    truncation=True, 
                    max_length=1024, 
                    padding="max_length")
```

### 2. 訓練フロー
```
Data Loading → Tokenization → PEFT Wrapping → Training → Adapter Saving
     ↓              ↓             ↓             ↓            ↓
tatsu-lab/     Alpaca形式    DoRA適用済み   Trainer実行   ./out-llama-lora/
alpaca[1%]     テンプレート    PeftModel      メモリ最適化      adapter/
```

### 3. 推論フロー
```
Base Model → Adapter Loading → Inference → Generation
     ↓             ↓              ↓           ↓
Llama-3.2    ./adapter/    PeftModel    デコーディング
1B-Instruct  合成済み      .generate()   温度制御
```

## 🛠️ カスタマイゼーション・拡張ポイント

### パフォーマンスチューニング
```python
# ランク調整による精度・効率トレードオフ
PEFT_R = 8              # 高効率（少パラメータ）
PEFT_R = 16             # バランス（現在設定）
PEFT_R = 32             # 高精度（多パラメータ）

# 学習率スケジューリング
LR = 2e-5              # 現在：保守的
LR = 2e-4              # 以前：アグレッシブ
```

### 高度なDoRA機能
```python
# 将来的な拡張案
runtime_config = LoraRuntimeConfig(
    ephemeral_gpu_offload=True  # CUDA環境でのメモリ最適化
)
config = LoraConfig(
    use_dora=True,
    runtime_config=runtime_config
)
```

### 代替初期化手法
```python
# PiSSA初期化（高速収束）
config = LoraConfig(init_lora_weights="pissa")

# OLoRA初期化（安定性向上）  
config = LoraConfig(init_lora_weights="olora")
```