#!/usr/bin/env bash
# Run 17 shards in parallel, then merge results.
set -euo pipefail

cd "$(dirname "$0")/.."

export JAVA_HOME=/opt/homebrew/opt/openjdk

WORKERS=17
BASE_PORT=9999
PYTHON=.env/bin/python
SCRIPT=eval/evaluate_english_articles.py
OUTDIR=eval_results/shards
METHOD=sifrank_plus
DATABASE=enwiki
FINAL_JSON=eval_results/english_articles_${METHOD}_filtered.json
FINAL_CSV=eval_results/english_articles_${METHOD}_filtered.csv

mkdir -p "$OUTDIR"

echo "=== Launching $WORKERS shards ==="

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
    port=$((BASE_PORT + i))
    out="$OUTDIR/shard_${i}.json"
    echo "  shard $i  port=$port  -> $out"
    $PYTHON "$SCRIPT" run \
        --shard-index "$i" \
        --num-shards "$WORKERS" \
        --port "$port" \
        --rank-method "$METHOD" \
        --database "$DATABASE" \
        --filter-first-para \
        --output "$out" \
        > "$OUTDIR/shard_${i}.log" 2>&1 &
    pids+=($!)
done

echo "=== Waiting for ${#pids[@]} processes ==="

failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    if wait "$pid"; then
        echo "  shard $i (pid=$pid) done"
    else
        echo "  shard $i (pid=$pid) FAILED (exit $?)"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "ERROR: $failed shards failed. Check logs in $OUTDIR/"
    exit 1
fi

echo "=== Merging shards ==="

shard_files=()
for i in $(seq 0 $((WORKERS - 1))); do
    shard_files+=("$OUTDIR/shard_${i}.json")
done

$PYTHON "$SCRIPT" merge \
    "${shard_files[@]}" \
    --output "$FINAL_JSON" \
    --csv-output "$FINAL_CSV"

echo "=== Done ==="
