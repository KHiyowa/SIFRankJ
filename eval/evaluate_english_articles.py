#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate English articles (DUC-2001) with SIFRank / SIFRank+.
Ported from evaluate_japanese_articles.py (yom07_exec branch),
using the English pipeline from sifrank_eval.py.

Supports sharding for parallel execution across separate processes.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import psutil
psutil.net_connections = lambda: []

import nltk
from embeddings.word_emb_elmo import WordEmbeddings
from embeddings.sent_emb_sif import SentEmbeddings
from model.method import SIFRank, SIFRank_plus
from stanfordcorenlp import StanfordCoreNLP


DEFAULT_CUTOFFS = ("1", "3", "5", "10", "15", "all")


# ---------------------------------------------------------------------------
# Text loading helpers (DUC-2001 specific)
# ---------------------------------------------------------------------------

def unwrap_text(raw):
    """Unwrap DUC-2001 hard-wrapped paragraphs into single-line paragraphs."""
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
    """Load DUC-2001 plain-text documents from *txt_dir*."""
    data = {}
    for fname in sorted(os.listdir(txt_dir)):
        if limit is not None and len(data) >= limit:
            break
        fpath = os.path.join(txt_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", errors="replace") as f:
            raw = f.read()
        text = unwrap_text(raw).lower()
        text = re.sub(r'[<>\[\]{}]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        data[fname] = text
    return data


def load_references(ref_path):
    """Load DUC-2001 reference keyphrases (test.reader.json)."""
    with open(ref_path, "r") as f:
        raw = json.load(f)
    result = {}
    for doc_id, entries in raw.items():
        phrases = [entry[0] for entry in entries if entry]
        result[doc_id] = phrases
    return result


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_porter = None

def normalize_phrase(text):
    """Porter-stem each token for evaluation matching."""
    global _porter
    if _porter is None:
        _porter = nltk.PorterStemmer()
    return " ".join(_porter.stem(t) for t in text.strip().lower().split())


# ---------------------------------------------------------------------------
# Metrics helpers (mirrors the Japanese version)
# ---------------------------------------------------------------------------

def prf(hit_count, predicted_count, gold_count):
    precision = float(hit_count) / float(predicted_count) if predicted_count else 0.0
    recall = float(hit_count) / float(gold_count) if gold_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def dedupe_predictions(keyphrases):
    """Deduplicate keyphrase list returned by SIFRank, preserving rank order."""
    seen = set()
    predictions = []
    for rank, (phrase, score) in enumerate(keyphrases, start=1):
        normalized = normalize_phrase(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        predictions.append({
            "rank": rank,
            "phrase": phrase,
            "normalized": normalized,
            "score": float(score),
        })
    return predictions


def evaluate_predictions(predictions, gold, cutoffs):
    """Per-article evaluation at each cutoff."""
    gold_norm = set(normalize_phrase(p) for p in gold)
    gold_lower = set(p.lower() for p in gold)
    metrics = {}
    for cutoff in cutoffs:
        if cutoff == "all":
            top_predictions = predictions
        else:
            top_predictions = predictions[:int(cutoff)]

        hits = set()
        for item in top_predictions:
            if item["normalized"] in gold_norm or item["phrase"].lower() in gold_lower:
                hits.add(item["normalized"])

        precision, recall, f1 = prf(len(hits), len(top_predictions), len(gold_norm))

        reciprocal_rank = 0.0
        first_hit_rank = None
        for i, prediction in enumerate(top_predictions, start=1):
            if prediction["normalized"] in gold_norm or prediction["phrase"].lower() in gold_lower:
                first_hit_rank = i
                reciprocal_rank = 1.0 / float(i)
                break

        metrics[cutoff] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rr": reciprocal_rank,
            "first_hit_rank": first_hit_rank,
            "hit_count": len(hits),
            "predicted_count": len(top_predictions),
            "gold_count": len(gold_norm),
            "hits": sorted(hits),
        }
    return metrics


def aggregate_article_metrics(article_results, cutoffs):
    """Aggregate per-article metrics into macro summary."""
    summary = {}
    article_count = len(article_results)
    for cutoff in cutoffs:
        hit_count = sum(r["metrics"][cutoff]["hit_count"] for r in article_results)
        predicted_count = sum(r["metrics"][cutoff]["predicted_count"] for r in article_results)
        gold_count = sum(r["metrics"][cutoff]["gold_count"] for r in article_results)
        precision, recall, f1 = prf(hit_count, predicted_count, gold_count)
        mrr = (
            sum(r["metrics"][cutoff]["rr"] for r in article_results) / float(article_count)
            if article_count else 0.0
        )
        summary[cutoff] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "hit_count": hit_count,
            "predicted_count": predicted_count,
            "gold_count": gold_count,
        }
    return summary


# ---------------------------------------------------------------------------
# CSV output (same schema as Japanese version)
# ---------------------------------------------------------------------------

def write_csv(article_results, summary, cutoffs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope", "doc_id", "cutoff",
        "precision", "recall", "f1",
        "rr", "mrr",
        "hit_count", "predicted_count", "gold_count",
        "first_hit_rank",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in article_results:
            for cutoff in cutoffs:
                m = result["metrics"][cutoff]
                writer.writerow({
                    "scope": "article",
                    "doc_id": result["doc_id"],
                    "cutoff": cutoff,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "rr": m["rr"],
                    "mrr": "",
                    "hit_count": m["hit_count"],
                    "predicted_count": m["predicted_count"],
                    "gold_count": m["gold_count"],
                    "first_hit_rank": m["first_hit_rank"],
                })
        for cutoff in cutoffs:
            m = summary[cutoff]
            writer.writerow({
                "scope": "summary",
                "doc_id": "",
                "cutoff": cutoff,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "rr": "",
                "mrr": m["mrr"],
                "hit_count": m["hit_count"],
                "predicted_count": m["predicted_count"],
                "gold_count": m["gold_count"],
                "first_hit_rank": "",
            })


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def parse_cutoffs(value):
    cutoffs = []
    for raw in value.split(","):
        c = raw.strip().lower()
        if not c:
            continue
        if c == "all":
            cutoffs.append(c)
        else:
            if int(c) < 1:
                raise ValueError("cutoffs must be positive integers or 'all'")
            cutoffs.append(c)
    return cutoffs


def shard_dict(data, shard_index, num_shards):
    """Return the shard_index-th slice of an ordered dict."""
    keys = list(data.keys())
    shard_keys = keys[shard_index::num_shards]
    return {k: data[k] for k in shard_keys}


def build_extractor(args):
    """Build ELMo + SIF + StanfordCoreNLP pipeline."""
    # SentEmbeddings uses relative paths like '../auxiliary_data/...'
    # so CWD must be a direct child of PROJECT_ROOT (e.g. eval/).
    eval_dir = os.path.join(PROJECT_ROOT, "eval")
    os.chdir(eval_dir)

    options_file = os.path.join(
        PROJECT_ROOT, "auxiliary_data",
        "elmo_2x4096_512_2048cnn_2xhighway_options.json")
    weight_file = os.path.join(
        PROJECT_ROOT, "auxiliary_data",
        "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
    stanford_dir = os.path.join(
        PROJECT_ROOT, "stanford-corenlp-full-2018-02-27")

    print("Initializing ELMo (CPU mode)...", flush=True)
    elmo = WordEmbeddings(options_file, weight_file, cuda_device=-1)
    sif = SentEmbeddings(elmo, lamda=args.lamda, database=args.database)

    print(f"Initializing StanfordCoreNLP (port={args.port})...", flush=True)
    en_model = StanfordCoreNLP(stanford_dir, port=args.port, quiet=True)

    return sif, en_model


def evaluate_articles(data, labels, sif, en_model, args):
    """Run evaluation over DUC-2001 documents."""
    cutoffs = parse_cutoffs(args.cutoffs)
    elmo_layers_weight = [float(x) for x in args.elmo_layers_weight.split(",")]
    article_results = []
    skipped = []

    rank_fn = SIFRank_plus if args.rank_method in ("sifrank_plus", "sifrank+") else SIFRank

    for doc_id, text in data.items():
        if not text.strip():
            skipped.append({"doc_id": doc_id, "reason": "empty text"})
            print(f"  [skip] empty text for {doc_id}", flush=True)
            continue

        if doc_id not in labels:
            skipped.append({"doc_id": doc_id, "reason": "no reference keyphrases"})
            print(f"  [skip] no reference for {doc_id}", flush=True)
            continue

        gold = labels[doc_id]

        if getattr(args, "filter_first_para", False):
            filtered_gold = []
            paragraphs = text.split('\n')
            if len(paragraphs) > 1:
                para1 = paragraphs[0].lower()
                para_rest = " ".join(paragraphs[1:]).lower()
                for p in gold:
                    p_lower = p.lower()
                    if p_lower in para1 and p_lower not in para_rest:
                        continue
                    filtered_gold.append(p)
            else:
                filtered_gold = gold
            
            gold = filtered_gold
            if not gold:
                skipped.append({"doc_id": doc_id, "reason": "all references filtered out by first-para rule"})
                print(f"  [skip] all references filtered out for {doc_id}", flush=True)
                continue

        if args.rank_method in ("sifrank_plus", "sifrank+"):
            kp_list = rank_fn(
                text, sif, en_model,
                N=args.max_rank,
                elmo_layers_weight=elmo_layers_weight,
                position_bias=args.position_bias,
            )
        else:
            kp_list = rank_fn(
                text, sif, en_model,
                N=args.max_rank,
                elmo_layers_weight=elmo_layers_weight,
            )

        predictions = dedupe_predictions(kp_list)
        metrics = evaluate_predictions(predictions, gold, cutoffs)
        article_results.append({
            "doc_id": doc_id,
            "gold": gold,
            "predictions": predictions,
            "metrics": metrics,
        })
        print(f"  evaluated {len(article_results)} articles: {doc_id}", flush=True)

    summary = aggregate_article_metrics(article_results, cutoffs)
    return {
        "settings": {
            "rank_method": args.rank_method,
            "database": args.database,
            "cutoffs": cutoffs,
            "max_rank": args.max_rank,
            "position_bias": args.position_bias,
            "elmo_layers_weight": elmo_layers_weight,
        },
        "summary": summary,
        "articles": article_results,
        "skipped": skipped,
        "article_count": len(article_results),
        "skipped_count": len(skipped),
    }


# ---------------------------------------------------------------------------
# Merge mode: combine shard outputs into a single result
# ---------------------------------------------------------------------------

def merge_shards(shard_files, cutoffs, output_path, csv_output=None):
    """Merge multiple shard JSON outputs into one combined result."""
    all_articles = []
    all_skipped = []
    settings = None
    for sf in shard_files:
        with open(sf, "r") as f:
            shard = json.load(f)
        all_articles.extend(shard["articles"])
        all_skipped.extend(shard.get("skipped", []))
        if settings is None:
            settings = shard["settings"]

    all_articles.sort(key=lambda r: r["doc_id"])
    summary = aggregate_article_metrics(all_articles, cutoffs)

    result = {
        "settings": settings,
        "summary": summary,
        "articles": all_articles,
        "skipped": all_skipped,
        "article_count": len(all_articles),
        "skipped_count": len(all_skipped),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if csv_output:
        write_csv(all_articles, summary, cutoffs, csv_output)

    print("\n" + "=" * 60)
    print("Summary (merged)")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {output_path}")
    if csv_output:
        print(f"wrote {csv_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate English articles (DUC-2001) with SIFRank/SIFRank+.")

    sub = parser.add_subparsers(dest="mode")

    # --- run mode (default) ---
    run_p = sub.add_parser("run", help="Run evaluation (default mode)")
    run_p.add_argument(
        "--txt-dir",
        default=os.path.expanduser(
            "~/Developer/ake-datasets/datasets/DUC-2001/src/txt"))
    run_p.add_argument(
        "--ref-file",
        default=os.path.expanduser(
            "~/Developer/ake-datasets/datasets/DUC-2001/references/test.reader.json"))
    run_p.add_argument("--output", default="eval_results/english_articles_sifrank.json")
    run_p.add_argument("--csv-output", default=None)
    run_p.add_argument("--rank-method", default="sifrank",
                       choices=["sifrank", "sifrank_plus", "sifrank+"])
    run_p.add_argument("--database", default="Duc2001")
    run_p.add_argument("--lamda", type=float, default=1.0)
    run_p.add_argument("--position-bias", type=float, default=3.4)
    run_p.add_argument("--elmo-layers-weight", default="1.0,0.0,0.0")
    run_p.add_argument("--cutoffs", default=",".join(DEFAULT_CUTOFFS))
    run_p.add_argument("--max-rank", type=int, default=1000)
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--port", type=int, default=9999,
                       help="StanfordCoreNLP server port.")
    run_p.add_argument("--shard-index", type=int, default=None,
                       help="Shard index (0-based). Use with --num-shards.")
    run_p.add_argument("--num-shards", type=int, default=None,
                       help="Total number of shards.")
    run_p.add_argument("--filter-first-para", action="store_true",
                       help="Exclude gold keyphrases that appear ONLY in the 1st paragraph.")

    # --- merge mode ---
    merge_p = sub.add_parser("merge", help="Merge shard outputs")
    merge_p.add_argument("shard_files", nargs="+",
                         help="Shard JSON files to merge.")
    merge_p.add_argument("--output", default="eval_results/english_articles_sifrank.json")
    merge_p.add_argument("--csv-output", default=None)
    merge_p.add_argument("--cutoffs", default=",".join(DEFAULT_CUTOFFS))

    return parser.parse_args()


def main():
    args = parse_args()

    # Default to 'run' mode if no subcommand given
    if args.mode is None:
        args.mode = "run"
        # Re-parse with run defaults
        sys.argv.insert(1, "run")
        args = parse_args()

    # Resolve all path arguments to absolute paths to prevent issues with CWD changes
    if hasattr(args, "output") and args.output:
        args.output = str(Path(args.output).resolve())
    if hasattr(args, "csv_output") and args.csv_output:
        args.csv_output = str(Path(args.csv_output).resolve())
    if hasattr(args, "txt_dir") and args.txt_dir:
        args.txt_dir = str(Path(args.txt_dir).resolve())
    if hasattr(args, "ref_file") and args.ref_file:
        args.ref_file = str(Path(args.ref_file).resolve())
    if hasattr(args, "shard_files") and args.shard_files:
        args.shard_files = [str(Path(sf).resolve()) for sf in args.shard_files]

    if args.mode == "merge":
        cutoffs = parse_cutoffs(args.cutoffs)
        merge_shards(args.shard_files, cutoffs, args.output, args.csv_output)
        return

    # --- run mode ---
    time_start = time.time()

    for resource in ("corpora/wordnet", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.replace("corpora/", ""), quiet=True)

    print("Loading documents...", flush=True)
    data = load_documents(args.txt_dir, limit=args.limit)
    print(f"  {len(data)} documents loaded", flush=True)

    # Apply sharding if requested
    if args.shard_index is not None and args.num_shards is not None:
        data = shard_dict(data, args.shard_index, args.num_shards)
        print(f"  shard {args.shard_index}/{args.num_shards}: {len(data)} documents", flush=True)

    print("Loading references...", flush=True)
    labels = load_references(args.ref_file)
    print(f"  {len(labels)} documents with references", flush=True)

    sif, en_model = build_extractor(args)
    result = evaluate_articles(data, labels, sif, en_model, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        cutoffs = parse_cutoffs(args.cutoffs)
        write_csv(result["articles"], result["summary"], cutoffs, args.csv_output)

    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"\nwrote {output_path}", flush=True)
    if args.csv_output:
        print(f"wrote {args.csv_output}", flush=True)

    try:
        en_model.close()
    except Exception:
        pass
    elapsed = time.time() - time_start
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
