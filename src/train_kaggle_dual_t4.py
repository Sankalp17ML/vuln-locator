"""
Kaggle Dual T4 Training Script for VulnLocator
===============================================
Fine-tunes microsoft/codebert-base for binary vulnerability classification
on the PrimeVul dataset using Dual NVIDIA T4 GPUs.

Key Optimizations:
1. Pre-tokenization: Eliminates CPU worker bottleneck by tokenizing all samples
   upfront into PyTorch TensorDatasets (drops epoch time from ~2 hours to ~40s).
2. Multi-GPU DataParallel: Distributes batches across 2x T4 GPUs (32GB VRAM total).
3. Mixed Precision (FP16): Uses torch.amp.autocast and GradScaler for fast compute
   and reduced memory footprint.
4. Class-Imbalance Handling: Computes inverse-frequency class weights for CrossEntropyLoss
   to handle PrimeVul's ~97% safe vs ~3% vulnerable distribution.
5. Dynamic Threshold Evaluation: Sweeps decision thresholds on validation split to
   maximize Vulnerable-class F1 score, then evaluates test split for ROC-AUC and PR metrics.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train CodeBERT on Kaggle Dual T4 GPUs")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/primevul_sample.csv",
        help="Path to PrimeVul sample CSV",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="Pretrained Hugging Face model identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/codebert-baseline",
        help="Directory to save fine-tuned model and tokenizer",
    )
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Global batch size across all GPUs",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pre_tokenize(tokenizer, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    """
    Pre-tokenizes all text samples into fixed-length tensors upfront.
    This eliminates the massive CPU bottleneck of tokenizing on-the-fly
    inside DataLoader worker loops.
    """
    print(f"Pre-tokenizing {len(texts)} samples (max_len={max_len})...")
    start = time.time()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    print(f"Pre-tokenization complete in {time.time() - start:.2f}s.")
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        # Mixed Precision forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    elapsed = time.time() - start_time
    avg_loss = total_loss / len(loader)
    return avg_loss, elapsed


def evaluate(model, loader, device, sweep_threshold=False):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    best_thresh = 0.5
    best_f1 = 0.0

    if sweep_threshold:
        # Sweep thresholds to maximize Vulnerable class F1
        for thresh in np.arange(0.05, 0.95, 0.02):
            preds = (all_probs >= thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

    final_preds = (all_probs >= best_thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, final_preds, average="binary", zero_division=0
    )

    metrics = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc": auc,
        "threshold": best_thresh,
    }
    return metrics, all_labels, final_preds, all_probs


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("VulnLocator - Kaggle Dual T4 Training Pipeline")
    print("=" * 60)

    # 1. Device and Multi-GPU setup
    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Available CUDA Devices: {device_count}")
    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 2. Data loading and stratified splits
    print(f"\nLoading dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=["func", "target"])
    df["target"] = df["target"].astype(int)

    label_counts = df["target"].value_counts().to_dict()
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: Safe (0): {label_counts.get(0, 0)}, Vulnerable (1): {label_counts.get(1, 0)}")
    vuln_pct = (label_counts.get(1, 0) / len(df)) * 100
    print(f"Vulnerable percentage: {vuln_pct:.2f}% (Extreme Class Imbalance)")

    # Stratified 70 / 15 / 15 split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["target"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["target"], random_state=args.seed
    )
    print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Tokenizer and Pre-tokenization
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_enc = pre_tokenize(tokenizer, train_df["func"].astype(str).tolist(), args.max_len)
    val_enc = pre_tokenize(tokenizer, val_df["func"].astype(str).tolist(), args.max_len)
    test_enc = pre_tokenize(tokenizer, test_df["func"].astype(str).tolist(), args.max_len)

    train_dataset = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(train_df["target"].values, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        val_enc["input_ids"],
        val_enc["attention_mask"],
        torch.tensor(val_df["target"].values, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        test_enc["input_ids"],
        test_enc["attention_mask"],
        torch.tensor(test_df["target"].values, dtype=torch.long),
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Model and Imbalanced Loss setup
    print(f"\nInitializing {args.model_name} for binary classification...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    # Wrap model with DataParallel if multiple GPUs are available
    if device_count > 1:
        print(f"Enabling torch.nn.DataParallel across {device_count} GPUs!")
        model = nn.DataParallel(model)

    # Inverse-frequency class weights
    class_counts = train_df["target"].value_counts().sort_index()
    weights = torch.tensor(
        [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32
    )
    weights = weights / weights.sum()
    print(f"Computed class loss weights: Safe={weights[0]:.4f}, Vulnerable={weights[1]:.4f}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # 5. Training loop
    print("\nStarting training loop...")
    best_val_auc = 0.0
    best_val_thresh = 0.5

    for epoch in range(1, args.epochs + 1):
        loss, epoch_time = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, sweep_threshold=True)

        print(
            f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {loss:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val P: {val_metrics['precision']:.3f} | "
            f"Val R: {val_metrics['recall']:.3f} | "
            f"Val F1: {val_metrics['f1']:.3f} | "
            f"Optimal Thresh: {val_metrics['threshold']:.2f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_thresh = val_metrics["threshold"]

    # 6. Evaluation on Test Set
    print("\n" + "=" * 60)
    print("Final Test Set Evaluation (Dual T4 Benchmark)")
    print("=" * 60)
    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, device, sweep_threshold=False
    )
    # Apply optimal validation threshold to test predictions
    test_preds_opt = (test_probs >= best_val_thresh).astype(int)

    print(f"Optimal Decision Threshold: {best_val_thresh:.2f}")
    print(f"ROC AUC: {test_metrics['auc']:.4f}")
    print("\nClassification Report (Test Split):")
    print(
        classification_report(
            test_labels,
            test_preds_opt,
            target_names=["Safe (0)", "Vulnerable (1)"],
            digits=4,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds_opt))

    # 7. Model Serialization
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving fine-tuned model artifacts to: {args.output_dir}")
    # Extract underlying model from DataParallel wrapper if present
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete! Model artifacts saved successfully.")


if __name__ == "__main__":
    main()
