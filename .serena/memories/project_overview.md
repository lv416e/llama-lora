# llama-lora プロジェクト概要

## プロジェクトの目的
llama-loraは、Hugging Face Transformers とPEFTライブラリを使用して、Llamaモデル（特にLlama-3.2-1B-Instruct）をLoRA（Low-Rank Adaptation）およびDoRA（Weight-Decomposed Low-Rank Adaptation）で効率的にファインチューニングするためのプロジェクトです。

## 主な特徴
- **最新技術の採用**: DoRA対応（PEFT 0.17.0+）
- **M2 Max MPS完全対応**: Apple Silicon上での最適化
- **完全なワークフロー**: 訓練 → 推論 → マージ → 評価の全工程をカバー
- **メモリ効率**: gradient checkpointing、小バッチ + 勾配累積によるメモリ最適化
- **モジュラー設計**: 機能別にスクリプトが分離された保守しやすい構造

## 使用モデル・データセット
- **ベースモデル**: meta-llama/Llama-3.2-1B-Instruct（ゲート付きモデル）
- **データセット**: tatsu-lab/alpaca（1%サブセット、高速テスト用）
- **出力ディレクトリ**: ./out-llama-lora/ （アダプター、統合モデル、トークナイザー）

## プロジェクト構造
```
llama-lora/
├── pyproject.toml              # 依存関係・プロジェクト設定
├── README.md                   # 使用方法・セットアップ手順
├── uv.lock                     # 依存関係ロックファイル
├── scripts/                    # メインスクリプトディレクトリ
│   ├── config.py              # 設定一元管理
│   ├── train.py               # DoRA/LoRA訓練
│   ├── infer.py               # 推論・テスト
│   ├── merge.py               # アダプター統合
│   └── baseline_inference.py  # ベースモデル評価
└── examples/
    └── tiny-llama-dora-test.ipynb # 実験・検証用ノートブック
```

## 技術的ハイライト
- DoRAによる高精度ファインチューニング
- Llamaアーキテクチャ全プロジェクション層への適用
- M2 Max MPSでの混合精度最適化（fp16/bf16自動無効化）
- エラーハンドリングと存在チェック機能