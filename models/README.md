# Models Directory

This directory stores the fine-tuned model checkpoints for **VulnLocator**.

## Expected Directory Structure

```text
models/
└── codebert-baseline/
    ├── config.json
    ├── model.safetensors       (~498 MB)
    ├── tokenizer.json          (~3.5 MB)
    └── tokenizer_config.json
```

## How the Baseline Model Was Trained
The baseline model is fine-tuned from `microsoft/codebert-base` on the **PrimeVul** dataset using **Dual NVIDIA T4 GPUs** on Kaggle.
- Pre-tokenized `TensorDataset` to eliminate CPU bottlenecks.
- Multi-GPU distributed forward/backward passes using `torch.nn.DataParallel`.
- FP16 Automatic Mixed Precision (AMP).
- Inverse class-frequency weighted loss.
- 3 epochs yielding **0.855 ROC-AUC**.

See [src/train_kaggle_dual_t4.py](../src/train_kaggle_dual_t4.py) and [notebooks/kaggle_training_dual_t4.ipynb](../notebooks/kaggle_training_dual_t4.ipynb) for the exact training pipeline.
