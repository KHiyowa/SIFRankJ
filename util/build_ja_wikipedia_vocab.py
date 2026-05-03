#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import ginza
import spacy


DEFAULT_POS = {"NOUN", "PROPN", "ADJ", "NUM"}
TEXT_SUFFIXES = {".txt", ".text", ".json", ".jsonl"}


def iter_input_files(paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and (child.suffix.lower() in TEXT_SUFFIXES or child.suffix == ""):
                    yield child
        elif path.is_file():
            yield path


def iter_texts(path, encoding):
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"} or suffix == "":
        plain_lines = []
        with path.open(encoding=encoding, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    plain_lines.append(line)
                    continue
                if isinstance(obj, dict):
                    text = obj.get("text") or obj.get("content") or obj.get("body")
                    if text:
                        yield text
                elif isinstance(obj, str):
                    yield obj
        if plain_lines:
            yield "\n".join(plain_lines)
    else:
        yield path.read_text(encoding=encoding, errors="ignore")


def should_count_token(token, allowed_pos, stopwords, min_token_len):
    text = unicodedata.normalize("NFKC", token.text.strip())
    if len(text) < min_token_len:
        return False
    if token.is_space or token.is_punct:
        return False
    if allowed_pos and token.pos_ not in allowed_pos:
        return False
    if text in stopwords:
        return False
    return text


def build_counter(nlp, input_paths, allowed_pos, stopwords, min_token_len, encoding, max_docs=None):
    counter = Counter()
    docs_seen = 0
    for path in iter_input_files(input_paths):
        for text in iter_texts(path, encoding):
            doc = nlp(text)
            docs_seen += 1
            for token in doc:
                token_text = should_count_token(token, allowed_pos, stopwords, min_token_len)
                if token_text:
                    counter[token_text] += 1
            if max_docs is not None and docs_seen >= max_docs:
                return counter
    return counter


def write_counter(counter, output_path, min_count):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token, count in counter.most_common():
            if count < min_count:
                break
            f.write(token + " " + str(count) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build SIFRankJ word-frequency files from Japanese Wikipedia text.")
    parser.add_argument("inputs", nargs="+", help="Input text/JSONL files or directories, e.g. WikiExtractor output.")
    parser.add_argument("--output-dir", default="auxiliary_data", help="Directory for generated vocab files.")
    parser.add_argument("--output-prefix", default="ja_wikipedia_vocab", help="Output filename prefix.")
    parser.add_argument("--model", default="ja_ginza", help="spaCy/GiNZA model name.")
    parser.add_argument("--modes", nargs="+", default=["A", "B"], choices=["A", "B", "C"], help="Sudachi split modes.")
    parser.add_argument("--pos", nargs="*", default=sorted(DEFAULT_POS), help="UPOS tags to count. Use --pos with no values to count all POS.")
    parser.add_argument("--use-ginza-stopwords", action="store_true", help="Exclude ginza.STOP_WORDS.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum count to write.")
    parser.add_argument("--min-token-len", type=int, default=1, help="Minimum token length to count.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional document limit for smoke tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    allowed_pos = set(args.pos) if args.pos else None
    stopwords = {unicodedata.normalize("NFKC", word) for word in ginza.STOP_WORDS} if args.use_ginza_stopwords else set()
    output_dir = Path(args.output_dir)

    nlp = spacy.load(args.model)
    for mode in args.modes:
        ginza.set_split_mode(nlp, mode)
        counter = build_counter(
            nlp,
            args.inputs,
            allowed_pos,
            stopwords,
            args.min_token_len,
            args.encoding,
            max_docs=args.max_docs,
        )
        output_path = output_dir / (args.output_prefix + "_" + mode + ".txt")
        write_counter(counter, output_path, args.min_count)
        print("wrote " + str(output_path) + " (" + str(len(counter)) + " tokens)")


if __name__ == "__main__":
    main()
