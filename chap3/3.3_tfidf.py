#!/usr/bin/env python3
"""
3.3 単語の頻度情報を用いたモデル（TF-IDF + ML）
『Kaggle ではじめる大規模言語モデル入門』第3章

TF-IDFで特徴量を作成し、LightGBMで分類
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
print("=" * 60)
print("3.3 TF-IDF + LightGBM")
print("=" * 60)

# データディレクトリ
import os
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

# パラメータ
N_SPLITS = 5
RANDOM_STATE = 42
MAX_FEATURES = 10000

# =============================================================================
# データ読み込み
# =============================================================================
print("\n📁 データ読み込み...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

# 欠損値を空文字で埋める
train['Review Text'] = train['Review Text'].fillna('')
test['Review Text'] = test['Review Text'].fillna('')

print(f"  train: {train.shape}")
print(f"  test:  {test.shape}")

# ターゲット変数
X = train['Review Text']
y = train['Rating']

print(f"\n🎯 ターゲット分布:")
print(y.value_counts().sort_index())

# =============================================================================
# TF-IDF 特徴量抽出
# =============================================================================
print("\n" + "=" * 60)
print("🔤 TF-IDF 特徴量抽出")
print("=" * 60)

# TF-IDFベクトライザー
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),  # uni-gram + bi-gram
    min_df=5,            # 最低5文書に出現
    max_df=0.9,          # 90%以上の文書に出現する単語は除外
    sublinear_tf=True,   # TFに対数スケール適用
)

# 訓練データでfitしてtransform
X_tfidf = tfidf.fit_transform(X)
X_test_tfidf = tfidf.transform(test['Review Text'])

print(f"  TF-IDF特徴量数: {X_tfidf.shape[1]}")
print(f"  訓練データ shape: {X_tfidf.shape}")
print(f"  テストデータ shape: {X_test_tfidf.shape}")

# =============================================================================
# LightGBM による分類（クロスバリデーション）
# =============================================================================
print("\n" + "=" * 60)
print("🌲 LightGBM クロスバリデーション")
print("=" * 60)

# LightGBMパラメータ
lgb_params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

# クロスバリデーション
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(train), 5))  # 5クラス分の確率
test_preds = np.zeros((len(test), 5))
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
    
    X_train, X_val = X_tfidf[train_idx], X_tfidf[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train - 1)  # 0-indexed
    val_data = lgb.Dataset(X_val, label=y_val - 1, reference=train_data)
    
    # 学習
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]
    )
    
    # 予測
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    
    # テストデータ予測
    test_preds += model.predict(X_test_tfidf) / N_SPLITS
    
    # スコア計算
    val_pred_class = np.argmax(val_pred, axis=1) + 1  # 1-indexed
    acc = accuracy_score(y_val, val_pred_class)
    f1 = f1_score(y_val, val_pred_class, average='macro')
    scores.append({'accuracy': acc, 'f1': f1})
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")

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
print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

# OOF予測の評価
oof_class = np.argmax(oof_preds, axis=1) + 1
print(f"\n📋 Classification Report (OOF):")
print(classification_report(y, oof_class, digits=4))

# =============================================================================
# テスト予測
# =============================================================================
print("\n" + "=" * 60)
print("📤 テスト予測")
print("=" * 60)

test_class = np.argmax(test_preds, axis=1) + 1
print(f"\nテスト予測の分布:")
print(pd.Series(test_class).value_counts().sort_index())

# 予測結果保存
submission = pd.DataFrame({
    'id': range(len(test)),
    'Rating': test_class
})
submission.to_csv(f'{DATA_DIR}/submission_tfidf.csv', index=False)
print(f"\n💾 予測結果を保存: {DATA_DIR}/submission_tfidf.csv")

# =============================================================================
# 完了
# =============================================================================
print("\n" + "=" * 60)
print("✅ 3.3 TF-IDF + LightGBM 完了!")
print("=" * 60)
print(f"""
【次のステップ】
- 3.4: BERT系モデルでさらに精度向上
- 3.5: LLMを使った分類
""")
