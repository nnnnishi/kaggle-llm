#!/usr/bin/env python3
"""
3.4.5 Kaggleへの架け橋
『Kaggle ではじめる大規模言語モデル入門』第3章

Kaggleコンペティション形式での推論と提出ファイル作成
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
print("=" * 60)
print("3.4.5 Kaggleへの架け橋")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")

# データディレクトリ
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

# 設定
MODEL_NAME = 'microsoft/deberta-v3-small'
MAX_LENGTH = 256
BATCH_SIZE = 32

# =============================================================================
# データセットクラス
# =============================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
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
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

# =============================================================================
# 推論関数
# =============================================================================
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

# =============================================================================
# メイン処理
# =============================================================================
def main():
    # テストデータ読み込み
    print("\n📁 テストデータ読み込み...")
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    test['Review Text'] = test['Review Text'].fillna('')
    print(f"   テストデータ: {test.shape}")
    
    # トークナイザー＆モデル
    print(f"\n🔤 モデル読み込み: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
    ).to(device)
    
    # 注意: 実際のKaggle提出では、学習済みモデルの重みをロードする必要があります
    # model.load_state_dict(torch.load('model_weights.pth'))
    print("   ⚠️ 注意: 事前学習済みモデルを使用（ファインチューニングなし）")
    
    # データセット＆データローダー
    test_dataset = ReviewDataset(test['Review Text'].values, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 推論
    print("\n🔮 推論実行中...")
    predictions, probabilities = predict(model, test_loader, device)
    
    # Rating への変換 (0-indexed → 1-indexed)
    ratings = predictions + 1
    
    # =============================================================================
    # 提出ファイル作成
    # =============================================================================
    print("\n" + "=" * 60)
    print("📤 提出ファイル作成")
    print("=" * 60)
    
    # 基本的な提出ファイル
    submission = pd.DataFrame({
        'id': range(len(test)),
        'Rating': ratings
    })
    
    submission_path = f'{DATA_DIR}/submission_kaggle.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n💾 保存: {submission_path}")
    
    # 予測分布の確認
    print(f"\n📊 予測分布:")
    print(pd.Series(ratings).value_counts().sort_index())
    
    # 確率値も保存（アンサンブル用）
    probs_df = pd.DataFrame(
        probabilities,
        columns=[f'prob_class_{i+1}' for i in range(5)]
    )
    probs_path = f'{DATA_DIR}/test_probabilities.csv'
    probs_df.to_csv(probs_path, index=False)
    print(f"💾 確率保存: {probs_path}")
    
    # =============================================================================
    # Kaggle提出のヒント
    # =============================================================================
    print("\n" + "=" * 60)
    print("💡 Kaggle提出のヒント")
    print("=" * 60)
    print("""
【提出手順】
1. submission_kaggle.csv をダウンロード
2. Kaggleコンペページで "Submit Predictions" をクリック
3. CSVファイルをアップロード

【スコア向上のヒント】
- 複数モデルのアンサンブル
- ハイパーパラメータチューニング
- データ拡張（同義語置換など）
- 疑似ラベリング
""")
    
    print("\n✅ 3.4.5 完了!")

if __name__ == '__main__':
    main()
