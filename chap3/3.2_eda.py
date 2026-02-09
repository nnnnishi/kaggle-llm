#!/usr/bin/env python3
"""
3.2 探索的データ解析（EDA）
『Kaggle ではじめる大規模言語モデル入門』第3章

atmaCup #17 データセットの探索的データ解析を行う
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 日本語フォント設定（Colab用）
plt.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# データの読み込み
# =============================================================================
print("=" * 60)
print("3.2 探索的データ解析（EDA）")
print("=" * 60)

# データディレクトリ（ローカル or Colab）
import os
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
clothing_master = pd.read_csv(f'{DATA_DIR}/clothing_master.csv')

print(f"\n📁 データサイズ:")
print(f"  train: {train.shape}")
print(f"  test:  {test.shape}")
print(f"  clothing_master: {clothing_master.shape}")

# =============================================================================
# 基本統計情報
# =============================================================================
print("\n" + "=" * 60)
print("📊 基本統計情報")
print("=" * 60)

print("\n--- train.info() ---")
print(train.dtypes)

print("\n--- 欠損値 ---")
print(train.isnull().sum())

print("\n--- train.describe() ---")
print(train.describe())

# =============================================================================
# ターゲット変数の分析（Rating）
# =============================================================================
print("\n" + "=" * 60)
print("⭐ ターゲット変数（Rating）の分布")
print("=" * 60)

print("\nRating の値ごとの件数:")
print(train['Rating'].value_counts().sort_index())

print("\nRating の比率:")
print(train['Rating'].value_counts(normalize=True).sort_index().round(3))

# =============================================================================
# テキストデータの分析
# =============================================================================
print("\n" + "=" * 60)
print("📝 テキストデータ（Review Text）の分析")
print("=" * 60)

# 欠損値
missing_text = train['Review Text'].isnull().sum()
print(f"\nReview Text の欠損値: {missing_text} ({missing_text/len(train)*100:.2f}%)")

# テキスト長
train['text_length'] = train['Review Text'].fillna('').apply(len)
train['word_count'] = train['Review Text'].fillna('').apply(lambda x: len(x.split()))

print(f"\nテキスト長（文字数）:")
print(train['text_length'].describe())

print(f"\n単語数:")
print(train['word_count'].describe())

# =============================================================================
# Rating と テキスト長の関係
# =============================================================================
print("\n" + "=" * 60)
print("📈 Rating と テキスト長の関係")
print("=" * 60)

rating_text_stats = train.groupby('Rating').agg({
    'text_length': 'mean',
    'word_count': 'mean'
}).round(2)
print(rating_text_stats)

# =============================================================================
# 年齢分布
# =============================================================================
print("\n" + "=" * 60)
print("👤 年齢（Age）の分布")
print("=" * 60)

print(train['Age'].describe())

# =============================================================================
# カテゴリ情報（clothing_master との結合）
# =============================================================================
print("\n" + "=" * 60)
print("👗 カテゴリ情報")
print("=" * 60)

train_with_category = train.merge(clothing_master, on='Clothing ID', how='left')

print("\nDivision Name:")
print(train_with_category['Division Name'].value_counts())

print("\nDepartment Name:")
print(train_with_category['Department Name'].value_counts())

print("\nClass Name:")
print(train_with_category['Class Name'].value_counts())

# =============================================================================
# サマリー
# =============================================================================
print("\n" + "=" * 60)
print("📋 EDA サマリー")
print("=" * 60)

print(f"""
【データ概要】
- 訓練データ: {len(train):,} 件
- テストデータ: {len(test):,} 件
- ターゲット: Rating (1-5の5段階評価)

【テキスト特徴】
- 平均テキスト長: {train['text_length'].mean():.0f} 文字
- 平均単語数: {train['word_count'].mean():.0f} 単語
- 欠損率: {missing_text/len(train)*100:.2f}%

【Rating 分布】
- Rating 5 が最も多い（正のバイアス）
- 低評価(1-2)は全体の約{(train['Rating'] <= 2).mean()*100:.1f}%

【次のステップ】
- 3.3: TF-IDF + 機械学習モデル
- 3.4: BERT系モデル
""")

print("\n✅ EDA 完了!")
