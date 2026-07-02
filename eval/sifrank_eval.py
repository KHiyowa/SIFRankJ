#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIFRank evaluation on DUC-2001 dataset.
"""

import argparse
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import psutil
psutil.net_connections = lambda: []

import nltk
from embeddings.word_emb_elmo import WordEmbeddings
from embeddings.sent_emb_sif import SentEmbeddings
from model.method import SIFRank, SIFRank_plus
from stanfordcorenlp import StanfordCoreNLP


def unwrap_text(raw):
    lines = raw.splitlines()
    paragraphs = []
    current = []
    para_prefix = " " * 3
    for line in lines:
        if line.startswith(para_prefix):
            if current:
                paragraphs.append(" ".join(current).strip())
            current = [line.strip()]
        else:
            if line.strip():
                current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n".join(paragraphs)


def load_documents(txt_dir, limit=None):
    data = {}
    for fname in sorted(os.listdir(txt_dir)):
        if limit is not None and len(data) >= limit:
            break
        fpath = os.path.join(txt_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", errors="replace_with_space") as f:
            raw = f.read()
        text = unwrap_text(raw).lower()
        text = re.sub(r'[<>\[\]{}]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        data[fname] = text
    return data


def load_references(ref_path):
    with open(ref_path, "r") as f:
        raw = json.load(f)
    result = {}
    for doc_id, entries in raw.items():
        phrases = [entry[0] for entry in entries if entry]
        result[doc_id] = phrases
    return result


def get_prf(num_correct, num_extracted, num_standard):
    p = float(num_correct) / float(num_extracted) if num_extracted > 0 else 0.0
    r = float(num_correct) / float(num_standard) if num_standard > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def print_prf(p, r, f1, n):
    print(f"\n  N={n}")
    print(f"    P      = {p:.4f}")
    print(f"    R      = {r:.4f}")
    print(f"    F1     = {f1:.4f}")


def evaluate(rank_fn, data, labels, sif, en_model,
             elmo_layers_weight, n_list=(5, 10, 15), normalize=None):
    if normalize is None:
        porter = nltk.PorterStemmer()
        normalize = lambda phrase: " ".join(porter.stem(t) for t in phrase.split())

    counts = {n: 0 for n in n_list}
    extracts = {n: 0 for n in n_list}
    total_standard = 0

    for doc_id, text in data.items():
        if doc_id not in labels:
            print(f"       [skip] no reference for {doc_id}")
            continue
        print(f"       {doc_id}")
        kp_list = rank_fn(text, sif, en_model, N=max(n_list),
                           elmo_layers_weight=elmo_layers_weight)
        std_phrases = labels[doc_id]
        std_norm = set(normalize(p) for p in std_phrases)
        std_lower = [p.lower() for p in std_phrases]
        total_standard += len(std_phrases)

        for n in n_list:
            top_n = kp_list[:n]
            actual_n = len(top_n)
            extracts[n] += actual_n
            for kp, _ in top_n:
                if normalize(kp) in std_norm or kp.lower() in std_lower:
                    counts[n] += 1

    results = {}
    for n in n_list:
        p, r, f1 = get_prf(counts[n], extracts[n], total_standard)
        results[n] = (p, r, f1)
        print_prf(p, r, f1, n)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFRank evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N documents")
    parser.add_argument("--method", choices=["sifrank", "sifrank_plus"],
                        default="sifrank", help="Ranking method")
    parser.add_argument("--all", action="store_true",
                        help="Run both SIFRank and SIFRank+")
    args = parser.parse_args()

    time_start = time.time()

    TXT_DIR = os.path.expanduser(
        "~/Developer/ake-datasets/datasets/DUC-2001/src/txt")
    REF_FILE = os.path.expanduser(
        "~/Developer/ake-datasets/datasets/DUC-2001/references/test.reader.json")
    OPTIONS_FILE = os.path.join(
        PROJECT_ROOT, "auxiliary_data",
        "elmo_2x4096_512_2048cnn_2xhighway_options.json")
    WEIGHT_FILE = os.path.join(
        PROJECT_ROOT, "auxiliary_data",
        "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
    STANFORD_DIR = os.path.join(
        PROJECT_ROOT, "stanford-corenlp-full-2018-02-27")

    LAMDA = 1.0
    ELMO_LAYERS_WEIGHT = [1.0, 0.0, 0.0]

    for resource in ("corpora/wordnet", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.replace("corpora/", ""), quiet=True)

    print("Loading documents...")
    data = load_documents(TXT_DIR, limit=args.limit)
    print(f"   {len(data)} documents loaded")

    print("Loading references...")
    labels = load_references(REF_FILE)
    print(f"   {len(labels)} documents with references")

    print("Initializing ELMo (CPU mode)...")
    ELMO = WordEmbeddings(OPTIONS_FILE, WEIGHT_FILE, cuda_device=-1)
    SIF = SentEmbeddings(ELMO, lamda=LAMDA, database="Duc2001")

    print("Initializing StanfordCoreNLP...")
    en_model = StanfordCoreNLP(STANFORD_DIR, port=9999, quiet=True)

    if args.all:
        methods = [("SIFRank", SIFRank), ("SIFRank+", SIFRank_plus)]
    elif args.method == "sifrank_plus":
        methods = [("SIFRank+", SIFRank_plus)]
    else:
        methods = [("SIFRank", SIFRank)]

    for name, fn in methods:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {name}")
        print(f"{'=' * 60}")
        evaluate(fn, data, labels, SIF, en_model, ELMO_LAYERS_WEIGHT)

    en_model.close()
    elapsed = time.time() - time_start
    print(f"\nTotal time: {elapsed:.1f}s")
