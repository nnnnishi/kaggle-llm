#!/usr/bin/env python3
"""
3.5 大規模言語モデル
『Kaggle ではじめる大規模言語モデル入門』第3章

Gemmaを使ったテキスト分類（プロンプトベース）
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
print("=" * 60)
print("3.5 大規模言語モデル（Gemma）")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# データディレクトリ
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

# 設定
MODEL_NAME = 'google/gemma-2-2b-jpn-it'  # 日本語対応Gemma
MAX_NEW_TOKENS = 16
BATCH_SIZE = 1  # LLMは1サンプルずつ処理

# サンプル数制限（デモ用）
SAMPLE_LIMIT = 100  # フルデータの場合はNoneに設定

# =============================================================================
# プロンプトテンプレート
# =============================================================================
PROMPT_TEMPLATE = """You are a helpful assistant that classifies product reviews.

Given the following product review, predict the rating from 1 to 5 stars.
- 1 star: Very negative review
- 2 stars: Negative review
- 3 stars: Neutral review
- 4 stars: Positive review
- 5 stars: Very positive review

Review: {review}

Respond with ONLY a single number from 1 to 5.
Rating:"""

# =============================================================================
# LLM推論関数
# =============================================================================
def predict_with_llm(model, tokenizer, texts, batch_size=1):
    """LLMを使って評価を予測"""
    predictions = []
    
    for text in tqdm(texts, desc='LLM Predicting'):
        # プロンプト作成
        prompt = PROMPT_TEMPLATE.format(review=text[:500])  # 長すぎるテキストは切り詰め
        
        # トークン化
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # 決定的生成
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # デコード
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Rating抽出
        rating = extract_rating(generated)
        predictions.append(rating)
    
    return np.array(predictions)

def extract_rating(text):
    """生成されたテキストからRatingを抽出"""
    # "Rating:" の後の数字を探す
    match = re.search(r'Rating:\s*(\d)', text)
    if match:
        rating = int(match.group(1))
        if 1 <= rating <= 5:
            return rating
    
    # 最後の数字を探す
    numbers = re.findall(r'\b([1-5])\b', text)
    if numbers:
        return int(numbers[-1])
    
    # デフォルト値
    return 3

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
    
    # サンプル制限（デモ用）
    if SAMPLE_LIMIT:
        print(f"\n⚠️ デモモード: {SAMPLE_LIMIT}サンプルのみ使用")
        train_sample = train.head(SAMPLE_LIMIT)
        test_sample = test.head(SAMPLE_LIMIT)
    else:
        train_sample = train
        test_sample = test
    
    # モデル読み込み
    print(f"\n🤖 モデル読み込み: {MODEL_NAME}")
    print("   （初回はモデルダウンロードに時間がかかります）")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # メモリ節約
            device_map='auto',
        )
        print("   ✅ モデル読み込み完了")
    except Exception as e:
        print(f"   ❌ モデル読み込みエラー: {e}")
        print("\n💡 Hugging Face トークンが必要な場合があります:")
        print("   1. https://huggingface.co/settings/tokens でトークン取得")
        print("   2. huggingface-cli login を実行")
        print("   3. Gemmaの利用規約に同意")
        return
    
    # =============================================================================
    # 検証データで精度確認
    # =============================================================================
    print("\n" + "=" * 60)
    print("🔍 検証（訓練データのサンプル）")
    print("=" * 60)
    
    val_texts = train_sample['Review Text'].values
    val_labels = train_sample['Rating'].values
    
    val_preds = predict_with_llm(model, tokenizer, val_texts)
    
    # 精度計算
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"\n📊 検証結果:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Macro F1: {f1:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(val_labels, val_preds, digits=4))
    
    # =============================================================================
    # テスト予測
    # =============================================================================
    print("\n" + "=" * 60)
    print("🔮 テスト予測")
    print("=" * 60)
    
    test_texts = test_sample['Review Text'].values
    test_preds = predict_with_llm(model, tokenizer, test_texts)
    
    print(f"\n📊 テスト予測分布:")
    print(pd.Series(test_preds).value_counts().sort_index())
    
    # 保存
    submission = pd.DataFrame({
        'id': range(len(test_preds)),
        'Rating': test_preds
    })
    submission_path = f'{DATA_DIR}/submission_llm.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n💾 保存: {submission_path}")
    
    # =============================================================================
    # 完了
    # =============================================================================
    print("\n" + "=" * 60)
    print("✅ 3.5 LLM 完了!")
    print("=" * 60)
    print(f"""
【補足】
- 完全なデータセットで実行する場合は SAMPLE_LIMIT = None に設定
- より大きなモデル（gemma-7b等）を使うと精度向上の可能性あり
- プロンプトの最適化でも精度向上可能

【次のステップ】
- 3.6: 複数モデルのアンサンブル
""")

if __name__ == '__main__':
    main()
