#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


DEFAULT_POS = ["\u540d\u8a5e", "\u5f62\u5bb9\u8a5e", "\u6570\u8a5e"]
WIKI_TEXT_FIELDS = ("text", "content", "body")


def iter_input_files(paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    yield child
        elif path.is_file():
            yield path


def iter_sudachi_input_texts(path, encoding, input_format):
    with path.open(encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if input_format in {"auto", "wikiextractor-json"}:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if input_format == "wikiextractor-json":
                        continue
                else:
                    text = None
                    if isinstance(obj, dict):
                        for field in WIKI_TEXT_FIELDS:
                            text = obj.get(field)
                            if text:
                                break
                    elif isinstance(obj, str):
                        text = obj
                    if text:
                        yield text
                    continue
            if input_format in {"auto", "wikiextractor-text"} and (line.startswith("<doc") or line.startswith("</doc")):
                continue
            yield line


def prepare_sudachi_input(input_path, output_path, encoding, input_format):
    with output_path.open("w", encoding="utf-8") as f:
        for text in iter_sudachi_input_texts(input_path, encoding, input_format):
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


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


def count_sudachi_output(path, allowed_pos, stopwords, min_token_len, encoding):
    counter = Counter()
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


def process_input_with_sudachi(input_path, java_cmd, jar_path, split_mode, output_path, allowed_pos, stopwords, min_token_len, encoding, input_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        prepared_input = Path(temp_dir) / (input_path.name + ".txt")
        prepare_sudachi_input(input_path, prepared_input, encoding, input_format)
        run_sudachi(java_cmd, jar_path, split_mode, prepared_input, output_path)
    return count_sudachi_output(output_path, allowed_pos, stopwords, min_token_len, encoding)


def run_sudachi(java_cmd, jar_path, split_mode, input_path, output_path):
    command = [java_cmd, "-jar", str(jar_path), "-m", split_mode, "-o", str(output_path), str(input_path)]
    subprocess.run(command, check=True)


def build_counter_with_sudachi(input_paths, allowed_pos, stopwords, min_token_len, encoding, input_format, java_cmd, jar_path, split_mode, keep_sudachi_output, sudachi_output_dir, workers):
    input_files = list(iter_input_files(input_paths))
    stopwords = frozenset(stopwords)
    if keep_sudachi_output:
        sudachi_output_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            (input_path, sudachi_output_dir / (str(i) + "_" + input_path.name + "." + split_mode + ".sudachi"))
            for i, input_path in enumerate(input_files, start=1)
        ]
        return run_sudachi_tasks(tasks, allowed_pos, stopwords, min_token_len, encoding, input_format, java_cmd, jar_path, split_mode, workers)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        tasks = [
            (input_path, temp_dir / (str(i) + "_" + input_path.name + "." + split_mode + ".sudachi"))
            for i, input_path in enumerate(input_files, start=1)
        ]
        counter = run_sudachi_tasks(tasks, allowed_pos, stopwords, min_token_len, encoding, input_format, java_cmd, jar_path, split_mode, workers)
    return counter


def run_sudachi_tasks(tasks, allowed_pos, stopwords, min_token_len, encoding, input_format, java_cmd, jar_path, split_mode, workers):
    counter = Counter()
    total = len(tasks)
    workers = max(1, workers)
    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            input_path, output_path = task
            print("[" + str(i) + "/" + str(total) + "] sudachi " + str(input_path))
            counter.update(process_input_with_sudachi(input_path, java_cmd, jar_path, split_mode, output_path, allowed_pos, stopwords, min_token_len, encoding, input_format))
        return counter

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for input_path, output_path in tasks:
            future = executor.submit(process_input_with_sudachi, input_path, java_cmd, jar_path, split_mode, output_path, allowed_pos, stopwords, min_token_len, encoding, input_format)
            futures[future] = input_path
        completed = 0
        for future in as_completed(futures):
            completed += 1
            input_path = futures[future]
            counter.update(future.result())
            print("[" + str(completed) + "/" + str(total) + "] counted " + str(input_path))
    return counter


def write_counter(counter, output_path, min_count):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token, count in counter.most_common():
            if count < min_count:
                break
            f.write(token + " " + str(count) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a SIFRankJ frequency file from text via Java Sudachi, or count existing Sudachi output.")
    parser.add_argument("inputs", nargs="+", help="Input text files/directories, or Sudachi tokenized files with --pretokenized.")
    parser.add_argument("--output", required=True, help="Output vocabulary file.")
    parser.add_argument("--java", default="java", help="Java executable.")
    parser.add_argument("--sudachi-jar", default="../sudachi/sudachi-0.7.5.jar", help="Path to Sudachi Java jar.")
    parser.add_argument("--mode", default="A", choices=["A", "B", "C"], help="Sudachi split mode.")
    parser.add_argument("--pretokenized", action="store_true", help="Treat inputs as existing Sudachi tokenized output.")
    parser.add_argument("--keep-sudachi-output", action="store_true", help="Keep intermediate Sudachi output files.")
    parser.add_argument("--sudachi-output-dir", default="data/sudachi", help="Directory for kept Sudachi output files.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel Java Sudachi worker processes.")
    parser.add_argument("--input-format", default="auto", choices=["auto", "plain", "wikiextractor-json", "wikiextractor-text"], help="Input text format before Java Sudachi tokenization.")
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
    if args.pretokenized:
        counter = build_counter(args.inputs, allowed_pos, stopwords, args.min_token_len, args.encoding)
    else:
        counter = build_counter_with_sudachi(
            args.inputs,
            allowed_pos,
            stopwords,
            args.min_token_len,
            args.encoding,
            args.input_format,
            args.java,
            Path(args.sudachi_jar),
            args.mode,
            args.keep_sudachi_output,
            Path(args.sudachi_output_dir),
            args.workers,
        )
    write_counter(counter, Path(args.output), args.min_count)
    print("wrote " + args.output + " (" + str(len(counter)) + " tokens)")


if __name__ == "__main__":
    main()
