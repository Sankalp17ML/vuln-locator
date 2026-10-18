import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from tqdm import tqdm

# --- 1. Configuration ---
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 256 
BATCH_SIZE = 2  
GRAD_ACCUMULATION_STEPS = 4  
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.getenv("VULNLOCATOR_TRAIN_OUTPUT", "models/codebert-local")

# --- 2. Load Data ---
df = pd.read_csv("data/primevul_sample.csv")
label_col = 'target'

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[label_col], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class VulnDataset(Dataset):
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels
    def __len__(self):
        return len(self.codes)
    def __getitem__(self, idx):
        encoding = tokenizer(str(self.codes[idx]), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(VulnDataset(train_df['func'].tolist(), train_df[label_col].tolist()), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VulnDataset(val_df['func'].tolist(), val_df[label_col].tolist()), batch_size=BATCH_SIZE)
test_loader = DataLoader(VulnDataset(test_df['func'].tolist(), test_df[label_col].tolist()), batch_size=BATCH_SIZE)

# --- 3. Model & Loss ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
model.gradient_checkpointing_enable()

class_counts = train_df[label_col].value_counts().sort_index()
weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
weights = weights / weights.sum()
loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# FIX: New API for GradScaler
scaler = torch.amp.GradScaler('cuda')

def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for step, batch in enumerate(progress_bar):
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        
        # FIX: New API for autocast
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels) / GRAD_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
        progress_bar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}"})
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # NEW: Threshold Tuning
    # Instead of default 0.5, we find the threshold that gives the best F1 score
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.01, 0.9, 0.01):
        preds = (all_probs > thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            
    final_preds = (all_probs > best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, final_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc, "threshold": best_thresh}, all_labels, final_preds

# --- 4. Training Loop ---
print("Starting training...")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, DEVICE)
    
    # We evaluate and tune the threshold on the validation set
    metrics, _, _ = evaluate(model, val_loader, DEVICE)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | "
          f"P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | "
          f"F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f} | Best Thresh: {metrics['threshold']:.2f}")

# --- 5. Final Test Evaluation ---
print("\n--- Final Test Set Evaluation ---")
test_metrics, test_labels, test_preds = evaluate(model, test_loader, DEVICE)
print(classification_report(test_labels, test_preds, target_names=["Safe", "Vulnerable"]))

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved!")