#!/usr/bin/env python3
"""
3.6 アンサンブル
『Kaggle ではじめる大規模言語モデル入門』第3章

複数モデルの予測を組み合わせて最終予測を作成
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
print("=" * 60)
print("3.6 アンサンブル")
print("=" * 60)

# データディレクトリ
if os.path.exists('/content/kaggle-llm/data'):
    DATA_DIR = '/content/kaggle-llm/data'
elif os.path.exists('/root/kaggle-llm/data'):
    DATA_DIR = '/root/kaggle-llm/data'
else:
    DATA_DIR = 'data'

# =============================================================================
# 予測ファイルの読み込み
# =============================================================================
print("\n📁 予測ファイル読み込み...")

predictions = {}
available_models = []

# TF-IDF予測
tfidf_path = f'{DATA_DIR}/submission_tfidf.csv'
if os.path.exists(tfidf_path):
    predictions['tfidf'] = pd.read_csv(tfidf_path)['Rating'].values
    available_models.append('tfidf')
    print(f"   ✅ TF-IDF: {tfidf_path}")
else:
    print(f"   ⚠️ TF-IDF: ファイルなし")

# BERT予測
bert_path = f'{DATA_DIR}/submission_bert.csv'
if os.path.exists(bert_path):
    predictions['bert'] = pd.read_csv(bert_path)['Rating'].values
    available_models.append('bert')
    print(f"   ✅ BERT: {bert_path}")
else:
    print(f"   ⚠️ BERT: ファイルなし")

# LLM予測
llm_path = f'{DATA_DIR}/submission_llm.csv'
if os.path.exists(llm_path):
    predictions['llm'] = pd.read_csv(llm_path)['Rating'].values
    available_models.append('llm')
    print(f"   ✅ LLM: {llm_path}")
else:
    print(f"   ⚠️ LLM: ファイルなし")

# Kaggle Bridge予測
kaggle_path = f'{DATA_DIR}/submission_kaggle.csv'
if os.path.exists(kaggle_path):
    predictions['kaggle'] = pd.read_csv(kaggle_path)['Rating'].values
    available_models.append('kaggle')
    print(f"   ✅ Kaggle: {kaggle_path}")
else:
    print(f"   ⚠️ Kaggle: ファイルなし")

print(f"\n📊 利用可能なモデル: {available_models}")

if len(available_models) < 2:
    print("\n⚠️ アンサンブルには最低2つのモデル予測が必要です")
    print("   先に 3.3, 3.4, 3.5 を実行してください")
    
    # 単一モデルの予測があれば、それを最終予測として使用
    if len(available_models) == 1:
        model_name = available_models[0]
        final_preds = predictions[model_name]
        print(f"\n   単一モデル ({model_name}) の予測を使用します")
    else:
        print("   予測ファイルがありません。終了します。")
        exit(1)
else:
    # =============================================================================
    # アンサンブル方法
    # =============================================================================
    print("\n" + "=" * 60)
    print("🔄 アンサンブル手法")
    print("=" * 60)
    
    # 予測を配列にまとめる
    pred_matrix = np.array([predictions[m] for m in available_models])
    n_samples = pred_matrix.shape[1]
    
    # --- 方法1: 多数決（Voting） ---
    print("\n📊 方法1: 多数決（Hard Voting）")
    voting_preds = []
    for i in range(n_samples):
        votes = pred_matrix[:, i]
        # 最頻値を取得（タイの場合は小さい方）
        unique, counts = np.unique(votes, return_counts=True)
        voting_preds.append(unique[np.argmax(counts)])
    voting_preds = np.array(voting_preds)
    print(f"   予測分布: {pd.Series(voting_preds).value_counts().sort_index().to_dict()}")
    
    # --- 方法2: 平均（Soft Voting風） ---
    print("\n📊 方法2: 平均")
    avg_preds = np.round(pred_matrix.mean(axis=0)).astype(int)
    avg_preds = np.clip(avg_preds, 1, 5)  # 1-5の範囲にクリップ
    print(f"   予測分布: {pd.Series(avg_preds).value_counts().sort_index().to_dict()}")
    
    # --- 方法3: 重み付き平均 ---
    print("\n📊 方法3: 重み付き平均")
    # モデルごとの重み（BERTやLLMを重視）
    weights = {
        'tfidf': 0.2,
        'bert': 0.4,
        'llm': 0.3,
        'kaggle': 0.1,
    }
    
    # 利用可能なモデルの重みを正規化
    available_weights = np.array([weights.get(m, 0.25) for m in available_models])
    available_weights = available_weights / available_weights.sum()
    
    print(f"   重み: {dict(zip(available_models, available_weights.round(3)))}")
    
    weighted_preds = np.zeros(n_samples)
    for i, model in enumerate(available_models):
        weighted_preds += predictions[model] * available_weights[i]
    weighted_preds = np.round(weighted_preds).astype(int)
    weighted_preds = np.clip(weighted_preds, 1, 5)
    print(f"   予測分布: {pd.Series(weighted_preds).value_counts().sort_index().to_dict()}")
    
    # デフォルトは重み付き平均
    final_preds = weighted_preds
    
    # =============================================================================
    # 各手法の一致度
    # =============================================================================
    print("\n" + "=" * 60)
    print("📏 アンサンブル手法間の一致度")
    print("=" * 60)
    
    print(f"   Voting vs Average: {(voting_preds == avg_preds).mean()*100:.1f}%")
    print(f"   Voting vs Weighted: {(voting_preds == weighted_preds).mean()*100:.1f}%")
    print(f"   Average vs Weighted: {(avg_preds == weighted_preds).mean()*100:.1f}%")

# =============================================================================
# 最終予測の保存
# =============================================================================
print("\n" + "=" * 60)
print("💾 最終予測の保存")
print("=" * 60)

# テストデータの読み込み（IDの確認用）
test = pd.read_csv(f'{DATA_DIR}/test.csv')

# 最終提出ファイル
submission = pd.DataFrame({
    'id': range(len(final_preds)),
    'Rating': final_preds
})

final_path = f'{DATA_DIR}/submission_ensemble.csv'
submission.to_csv(final_path, index=False)
print(f"\n✅ 最終予測を保存: {final_path}")

print(f"\n📊 最終予測分布:")
print(submission['Rating'].value_counts().sort_index())

# =============================================================================
# OOF予測があれば検証スコアを計算
# =============================================================================
oof_bert_path = f'{DATA_DIR}/oof_bert.npy'
if os.path.exists(oof_bert_path):
    print("\n" + "=" * 60)
    print("📈 OOF検証スコア（BERT）")
    print("=" * 60)
    
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    oof_preds = np.load(oof_bert_path)
    
    acc = accuracy_score(train['Rating'], oof_preds)
    f1 = f1_score(train['Rating'], oof_preds, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Macro F1: {f1:.4f}")

# =============================================================================
# 完了
# =============================================================================
print("\n" + "=" * 60)
print("✅ 3.6 アンサンブル 完了!")
print("=" * 60)
print(f"""
【生成されたファイル】
- {final_path}

【Kaggle提出】
1. submission_ensemble.csv をダウンロード
2. Kaggleコンペページで Submit

【さらなる改善案】
- Stacking: メタモデルで予測を統合
- Blending: 別の検証セットで重みを最適化
- 異なるモデルアーキテクチャの追加
""")
