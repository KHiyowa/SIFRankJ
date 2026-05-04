#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import unicodedata
from collections import Counter
from pathlib import Path


DEFAULT_POS = ["\u540d\u8a5e", "\u5f62\u5bb9\u8a5e", "\u6570\u8a5e"]


def iter_input_files(paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    yield child
        elif path.is_file():
            yield path


def parse_sudachi_line(line):
    line = line.rstrip("\n")
    if not line or line == "EOS":
        return None
    columns = line.split("\t")
    if len(columns) < 2:
        return None
    surface = columns[0].strip()
    features = columns[1].split(",")
    pos = features[0] if features else ""
    return surface, pos


def should_count(surface, pos, allowed_pos, stopwords, min_token_len):
    surface = unicodedata.normalize("NFKC", surface)
    if len(surface) < min_token_len:
        return False
    if allowed_pos and pos not in allowed_pos:
        return False
    if surface in stopwords:
        return False
    return surface


def build_counter(input_paths, allowed_pos, stopwords, min_token_len, encoding):
    counter = Counter()
    for path in iter_input_files(input_paths):
        with path.open(encoding=encoding, errors="ignore") as f:
            for line in f:
                parsed = parse_sudachi_line(line)
                if parsed is None:
                    continue
                surface, pos = parsed
                token = should_count(surface, pos, allowed_pos, stopwords, min_token_len)
                if token:
                    counter[token] += 1
    return counter


def write_counter(counter, output_path, min_count):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token, count in counter.most_common():
            if count < min_count:
                break
            f.write(token + " " + str(count) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Count Sudachi CLI token output into a SIFRankJ frequency file.")
    parser.add_argument("inputs", nargs="+", help="Sudachi tokenized files or directories.")
    parser.add_argument("--output", required=True, help="Output vocabulary file.")
    parser.add_argument("--pos", nargs="*", default=DEFAULT_POS, help="Japanese POS names to count. Use --pos with no values to count all POS.")
    parser.add_argument("--stopwords", default=None, help="Optional newline-separated stopword file.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum count to write.")
    parser.add_argument("--min-token-len", type=int, default=1, help="Minimum token length to count.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    return parser.parse_args()


def main():
    args = parse_args()
    allowed_pos = set(args.pos) if args.pos else None
    stopwords = set()
    if args.stopwords:
        with open(args.stopwords, encoding="utf-8") as f:
            stopwords = {unicodedata.normalize("NFKC", line.strip()) for line in f if line.strip()}
    counter = build_counter(args.inputs, allowed_pos, stopwords, args.min_token_len, args.encoding)
    write_counter(counter, Path(args.output), args.min_count)
    print("wrote " + args.output + " (" + str(len(counter)) + " tokens)")


if __name__ == "__main__":
    main()
