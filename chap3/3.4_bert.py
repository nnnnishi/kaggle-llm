#!/usr/bin/env python3
"""
3.4 BERT系統のモデル
『Kaggle ではじめる大規模言語モデル入門』第3章

DeBERTaを使ったテキスト分類
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
print("=" * 60)
print("3.4 BERT系統のモデル（DeBERTa）")
print("=" * 60)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# データディレクトリ
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

# ハイパーパラメータ
MODEL_NAME = 'microsoft/deberta-v3-small'  # 軽量版DeBERTa
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
N_SPLITS = 5
RANDOM_STATE = 42

print(f"\n⚙️ 設定:")
print(f"   Model: {MODEL_NAME}")
print(f"   Max Length: {MAX_LENGTH}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning Rate: {LEARNING_RATE}")

# =============================================================================
# データセットクラス
# =============================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # 0-indexed
        
        return item

# =============================================================================
# 学習関数
# =============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)

# =============================================================================
# メイン処理
# =============================================================================
def main():
    # データ読み込み
    print("\n📁 データ読み込み...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    train['Review Text'] = train['Review Text'].fillna('')
    test['Review Text'] = test['Review Text'].fillna('')
    
    print(f"   train: {train.shape}")
    print(f"   test:  {test.shape}")
    
    X = train['Review Text'].values
    y = train['Rating'].values
    X_test = test['Review Text'].values
    
    # トークナイザー
    print(f"\n🔤 トークナイザー読み込み: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # クロスバリデーション
    print("\n" + "=" * 60)
    print("🔄 クロスバリデーション")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros((len(test), 5))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold + 1}/{N_SPLITS} {'='*20}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Dataset & DataLoader
        train_dataset = ReviewDataset(X_train, y_train, tokenizer, MAX_LENGTH)
        val_dataset = ReviewDataset(X_val, y_val, tokenizer, MAX_LENGTH)
        test_dataset = ReviewDataset(X_test, None, tokenizer, MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # モデル
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=5
        ).to(device)
        
        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 学習
        best_f1 = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_preds = eval_epoch(model, val_loader, device)
            
            val_preds_class = val_preds + 1  # 1-indexed
            acc = accuracy_score(y_val, val_preds_class)
            f1 = f1_score(y_val, val_preds_class, average='macro')
            
            print(f"   Loss: {train_loss:.4f} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_val_preds = val_preds_class.copy()
                
                # テスト予測
                test_pred = eval_epoch(model, test_loader, device)
        
        # OOF予測保存
        oof_preds[val_idx] = best_val_preds
        
        # テスト予測（one-hot風に加算）
        for i, pred in enumerate(test_pred):
            test_preds[i, pred] += 1
        
        # スコア記録
        acc = accuracy_score(y_val, best_val_preds)
        f1 = f1_score(y_val, best_val_preds, average='macro')
        scores.append({'accuracy': acc, 'f1': f1})
        print(f"\n📊 Fold {fold + 1} Best: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
        
        # メモリ解放
        del model
        torch.cuda.empty_cache()
    
    # =============================================================================
    # 結果サマリー
    # =============================================================================
    print("\n" + "=" * 60)
    print("📊 結果サマリー")
    print("=" * 60)
    
    mean_acc = np.mean([s['accuracy'] for s in scores])
    mean_f1 = np.mean([s['f1'] for s in scores])
    std_acc = np.std([s['accuracy'] for s in scores])
    std_f1 = np.std([s['f1'] for s in scores])
    
    print(f"\n🎯 CV結果:")
    print(f"   Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"   Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    print(f"\n📋 Classification Report (OOF):")
    print(classification_report(y, oof_preds, digits=4))
    
    # テスト予測（多数決）
    test_class = np.argmax(test_preds, axis=1) + 1
    print(f"\nテスト予測の分布:")
    print(pd.Series(test_class).value_counts().sort_index())
    
    # 保存
    submission = pd.DataFrame({
        'id': range(len(test)),
        'Rating': test_class
    })
    submission.to_csv(f'{DATA_DIR}/submission_bert.csv', index=False)
    print(f"\n💾 予測結果を保存: {DATA_DIR}/submission_bert.csv")
    
    # OOF予測も保存
    np.save(f'{DATA_DIR}/oof_bert.npy', oof_preds)
    print(f"💾 OOF予測を保存: {DATA_DIR}/oof_bert.npy")
    
    print("\n" + "=" * 60)
    print("✅ 3.4 BERT 完了!")
    print("=" * 60)

if __name__ == '__main__':
    main()
