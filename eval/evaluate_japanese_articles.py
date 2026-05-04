#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import pickle
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddings import sent_emb_sif, word_emb_elmo_manylangs
from model.method import extract_keyphrases
from model.nlp_adapters import GinzaNLPAdapter


DEFAULT_CUTOFFS = ("1", "3", "5", "10", "all")


def get_field(article, name, default=None):
    if isinstance(article, dict):
        return article.get(name, default)
    return getattr(article, name, default)


def normalize_phrase(text):
    return unicodedata.normalize("NFKC", str(text).strip())


def article_text(article):
    title = get_field(article, "title", "") or ""
    maintext = get_field(article, "maintext", "") or ""
    return str(title) + "\n" + str(maintext)


def article_id(article, fallback):
    return get_field(article, "article_id", fallback)


def raw_dollar_keywords(article):
    keywords = get_field(article, "keyword_with_dollar", None)
    if keywords is not None:
        return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]

    keywords = get_field(article, "keywords", []) or []
    dollar_keywords = []
    for keyword in keywords:
        keyword = str(keyword).strip()
        if "$" in keyword:
            dollar_keywords.append(keyword.lstrip("$").strip())
    return [keyword for keyword in dollar_keywords if keyword]


def gold_keywords_in_text(article, text):
    seen = set()
    gold = []
    for keyword in raw_dollar_keywords(article):
        if keyword not in text:
            continue
        normalized = normalize_phrase(keyword)
        if normalized in seen:
            continue
        seen.add(normalized)
        gold.append({"surface": keyword, "normalized": normalized})
    return gold


def dedupe_predictions(keyphrases):
    seen = set()
    predictions = []
    for rank, item in enumerate(keyphrases, start=1):
        phrase, score = item
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


def prf(hit_count, predicted_count, gold_count):
    precision = float(hit_count) / float(predicted_count) if predicted_count else 0.0
    recall = float(hit_count) / float(gold_count) if gold_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_predictions(predictions, gold, cutoffs):
    gold_set = {item["normalized"] for item in gold}
    metrics = {}
    for cutoff in cutoffs:
        if cutoff == "all":
            top_predictions = predictions
        else:
            top_predictions = predictions[:int(cutoff)]

        top_set = {item["normalized"] for item in top_predictions}
        hits = top_set & gold_set
        precision, recall, f1 = prf(len(hits), len(top_predictions), len(gold_set))

        reciprocal_rank = 0.0
        first_hit_rank = None
        for i, prediction in enumerate(top_predictions, start=1):
            if prediction["normalized"] in gold_set:
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
            "gold_count": len(gold_set),
            "hits": sorted(hits),
        }
    return metrics


def aggregate_article_metrics(article_results, cutoffs):
    summary = {}
    article_count = len(article_results)
    for cutoff in cutoffs:
        hit_count = sum(result["metrics"][cutoff]["hit_count"] for result in article_results)
        predicted_count = sum(result["metrics"][cutoff]["predicted_count"] for result in article_results)
        gold_count = sum(result["metrics"][cutoff]["gold_count"] for result in article_results)
        precision, recall, f1 = prf(hit_count, predicted_count, gold_count)
        mrr = (
            sum(result["metrics"][cutoff]["rr"] for result in article_results) / float(article_count)
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


def load_articles(path, article_format):
    path = Path(path)
    if article_format == "auto":
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            article_format = "jsonl"
        elif suffix == ".json":
            article_format = "json"
        else:
            article_format = "pickle"

    if article_format == "pickle":
        with path.open("rb") as f:
            return pickle.load(f)

    if article_format == "json":
        with path.open(encoding="utf-8") as f:
            articles = json.load(f)
        return articles

    if article_format == "jsonl":
        articles = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))
        return articles

    raise ValueError("Unsupported article format: " + article_format)


def parse_cutoffs(value):
    cutoffs = []
    for raw_cutoff in value.split(","):
        cutoff = raw_cutoff.strip().lower()
        if not cutoff:
            continue
        if cutoff == "all":
            cutoffs.append(cutoff)
            continue
        if int(cutoff) < 1:
            raise ValueError("cutoffs must be positive integers or all")
        cutoffs.append(cutoff)
    return cutoffs


def write_csv(article_results, summary, cutoffs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "article_id",
        "cutoff",
        "precision",
        "recall",
        "f1",
        "rr",
        "mrr",
        "hit_count",
        "predicted_count",
        "gold_count",
        "first_hit_rank",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in article_results:
            for cutoff in cutoffs:
                metrics = result["metrics"][cutoff]
                writer.writerow({
                    "scope": "article",
                    "article_id": result["article_id"],
                    "cutoff": cutoff,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "rr": metrics["rr"],
                    "mrr": "",
                    "hit_count": metrics["hit_count"],
                    "predicted_count": metrics["predicted_count"],
                    "gold_count": metrics["gold_count"],
                    "first_hit_rank": metrics["first_hit_rank"],
                })
        for cutoff in cutoffs:
            metrics = summary[cutoff]
            writer.writerow({
                "scope": "summary",
                "article_id": "",
                "cutoff": cutoff,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "rr": "",
                "mrr": metrics["mrr"],
                "hit_count": metrics["hit_count"],
                "predicted_count": metrics["predicted_count"],
                "gold_count": metrics["gold_count"],
                "first_hit_rank": "",
            })


def build_extractor(args):
    elmo = word_emb_elmo_manylangs.WordEmbeddings(
        model_dir=args.elmo_model_dir,
        batch_size=args.batch_size,
        cuda_device=args.cuda_device,
    )
    vocab_file = args.vocab_file or "auxiliary_data/ja_wikipedia_vocab_" + args.split_mode + ".txt"
    sif = sent_emb_sif.SentEmbeddings(
        elmo,
        weightfile_pretrain=vocab_file,
        lamda=args.lamda,
        database="",
    )
    ja_model = GinzaNLPAdapter(split_mode=args.split_mode)
    return sif, ja_model


def evaluate_articles(articles, sif, ja_model, args):
    cutoffs = parse_cutoffs(args.cutoffs)
    article_results = []
    skipped = []
    for index, article in enumerate(articles, start=1):
        text = article_text(article)
        gold = gold_keywords_in_text(article, text)
        current_id = article_id(article, index)
        if not gold:
            skipped.append({
                "article_id": current_id,
                "reason": "no keyword_with_dollar appears in article text",
            })
            continue

        keyphrases = extract_keyphrases(
            text,
            sif,
            ja_model,
            rank_method=args.rank_method,
            N=args.max_rank,
            if_DS=not args.no_ds,
            if_EA=not args.no_ea,
            position_bias=args.position_bias,
        )
        predictions = dedupe_predictions(keyphrases)
        metrics = evaluate_predictions(predictions, gold, cutoffs)
        article_results.append({
            "article_id": current_id,
            "title": get_field(article, "title", ""),
            "gold": gold,
            "predictions": predictions,
            "metrics": metrics,
        })
        print("evaluated " + str(len(article_results)) + " articles: " + str(current_id))

    summary = aggregate_article_metrics(article_results, cutoffs)
    return {
        "settings": {
            "rank_method": args.rank_method,
            "split_mode": args.split_mode,
            "vocab_file": args.vocab_file or "auxiliary_data/ja_wikipedia_vocab_" + args.split_mode + ".txt",
            "cutoffs": cutoffs,
            "max_rank": args.max_rank,
            "position_bias": args.position_bias,
            "if_DS": not args.no_ds,
            "if_EA": not args.no_ea,
        },
        "summary": summary,
        "articles": article_results,
        "skipped": skipped,
        "article_count": len(article_results),
        "skipped_count": len(skipped),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Japanese Article DTOs with SIFRank/SIFRank+.")
    parser.add_argument("articles", help="Pickle/JSON/JSONL file containing Article DTOs.")
    parser.add_argument("--articles-format", default="auto", choices=["auto", "pickle", "json", "jsonl"])
    parser.add_argument("--output", default="eval_results/japanese_articles_sifrank_plus.json")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--rank-method", default="sifrank_plus", choices=["sifrank", "sifrank_plus", "sifrank+"])
    parser.add_argument("--split-mode", default="A", choices=["A", "B", "C"])
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--elmo-model-dir", default="auxiliary_data/ja.model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--position-bias", type=float, default=3.4)
    parser.add_argument("--cutoffs", default=",".join(DEFAULT_CUTOFFS))
    parser.add_argument("--max-rank", type=int, default=1000, help="Number of SIFRank results to request; used as the all cutoff.")
    parser.add_argument("--no-ds", action="store_true")
    parser.add_argument("--no-ea", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    articles = load_articles(args.articles, args.articles_format)
    sif, ja_model = build_extractor(args)
    result = evaluate_articles(articles, sif, ja_model, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        cutoffs = parse_cutoffs(args.cutoffs)
        write_csv(result["articles"], result["summary"], cutoffs, args.csv_output)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("wrote " + str(output_path))
    if args.csv_output:
        print("wrote " + str(args.csv_output))


if __name__ == "__main__":
    main()
