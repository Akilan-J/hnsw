#!/usr/bin/env bash
# Reproduces every number in the project from scratch with one command:
#   ./scripts/bench.sh
# Synthetic data is generated from a fixed seed, so it needs no download. SIFT
# runs are included automatically once ./scripts/fetch_sift.sh has been run.
#
# Single-threaded measurements. On Apple Silicon the OS may schedule the JVM on
# an efficiency core; close other heavy work and expect a few % run-to-run noise.
set -euo pipefail
cd "$(dirname "$0")/.."

# Fixed heap so GC sizing doesn't vary between runs; AlwaysPreTouch faults the
# heap in up front so page faults don't land inside timed queries.
JAVA="java ${JAVA_OPTS:--Xms3g -Xmx3g -XX:+AlwaysPreTouch} -cp build"
BRUTE_QUERIES=${BRUTE_QUERIES:-200}

./scripts/build.sh
echo "java: $(java -version 2>&1 | head -1)"
echo "cpu : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"

SYNTH=data/synth-n100k-d128-l16
UNIFORM=data/uniform-n100k-d128
[ -f $SYNTH/base.fvecs ]   || $JAVA hnsw.tools.Generate --out $SYNTH --n 100000 --dim 128 --latent-dim 16 --clusters 64 --seed 42
[ -f $UNIFORM/base.fvecs ] || $JAVA hnsw.tools.Generate --out $UNIFORM --n 100000 --dim 128 --latent-dim 128 --clusters 1 --cluster-std 1 --noise 0 --seed 42

echo
echo "=== how hard is each dataset? (relative contrast near 1 = everything equidistant)"
$JAVA hnsw.tools.Stats --data $UNIFORM
$JAVA hnsw.tools.Stats --data $SYNTH
if [ -f data/sift1m/base.fvecs ]; then
    $JAVA hnsw.tools.Stats --data data/sift1m --base-limit 100000
    $JAVA hnsw.tools.Stats --data data/sift1m
fi

echo
echo "=== ground-truth correctness vs. shipped SIFT ground truth"
if [ -f data/sift1m/base.fvecs ]; then
    $JAVA hnsw.tools.VerifyGroundTruth --data data/sift1m
else
    echo "(skipped: no SIFT data - ./scripts/fetch_sift.sh)"
fi

echo
echo "=== brute-force baseline"
$JAVA hnsw.tools.Bench --data $SYNTH --index brute --queries $BRUTE_QUERIES
if [ -f data/sift1m/base.fvecs ]; then
    $JAVA hnsw.tools.Bench --data data/sift1m --base-limit 100000 --index brute --queries $BRUTE_QUERIES
    $JAVA hnsw.tools.Bench --data data/sift1m --index brute --queries $BRUTE_QUERIES
fi
