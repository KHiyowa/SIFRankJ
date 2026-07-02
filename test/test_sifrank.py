#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIFRank 動作確認用テストスクリプト (CPUモード)
"""
import os
import sys
import socket
import psutil

# macOSでpsutil.net_connections()がPermissionErrorになるのを回避
def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

psutil.net_connections = lambda: []

# プロジェクトルートを追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import nltk
from embeddings.word_emb_elmo import WordEmbeddings
from embeddings.sent_emb_sif import SentEmbeddings
from model.method import SIFRank, SIFRank_plus
from stanfordcorenlp import StanfordCoreNLP

# NLTKデータダウンロード
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# パス設定
options_file = os.path.join(PROJECT_ROOT, "auxiliary_data", "elmo_2x4096_512_2048cnn_2xhighway_options.json")
weight_file = os.path.join(PROJECT_ROOT, "auxiliary_data", "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
stanford_corenlp_dir = os.path.join(PROJECT_ROOT, "stanford-corenlp-full-2018-02-27")

# model/から実行して相対パスが正しく解決されるようにする
os.chdir(os.path.join(PROJECT_ROOT, "model"))

# CPUモードで初期化 (cuda_device=-1)
print("Initializing ELMo (CPU mode)...")
ELMO = WordEmbeddings(options_file, weight_file, cuda_device=-1)
SIF = SentEmbeddings(ELMO, lamda=1.0)

print("Initializing StanfordCoreNLP...")
en_model = StanfordCoreNLP(stanford_corenlp_dir, quiet=True)

# テスト文章
text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach. The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line."

print("\n" + "="*60)
print("SIFRank テスト開始")
print("="*60)

# SIFRank
print("\n[SIFRank] 重要語抽出中...")
keyphrases = SIFRank(text, SIF, en_model, N=10, elmo_layers_weight=[0.0, 1.0, 0.0])
print(f"\n抽出された重要語 (Top 10):")
for i, (kp, score) in enumerate(keyphrases, 1):
    print(f"     {i}. {kp} (score: {score:.4f})")

# SIFRank+
print("\n[SIFRank+] 重要語抽出中...")
keyphrases_plus = SIFRank_plus(text, SIF, en_model, N=10, elmo_layers_weight=[0.0, 1.0, 0.0])
print(f"\n抽出された重要語 (Top 10):")
for i, (kp, score) in enumerate(keyphrases_plus, 1):
    print(f"     {i}. {kp} (score: {score:.4f})")

# SIFRank (Top 5)
print("\n[SIFRank (Top 5)] 重要語抽出中...")
keyphrases_5 = SIFRank(text, SIF, en_model, N=5, elmo_layers_weight=[0.0, 1.0, 0.0])
print(f"\n抽出された重要語 (Top 5):")
for i, (kp, score) in enumerate(keyphrases_5, 1):
    print(f"     {i}. {kp} (score: {score:.4f})")

print("\n" + "="*60)
print("テスト完了！環境構築成功しました。")
print("="*60)

# StanfordCoreNLPシャットダウン
en_model.close()
