#!/usr/bin/env bash
# Reproduces every number in the project from scratch:
#   ./scripts/bench.sh            all stages (~1 hour with SIFT on an M2)
#   ./scripts/bench.sh stage4     one stage (stage1 .. stage4, or stage4a .. stage4d)
#   ./scripts/bench.sh plots      redraw results/ charts and tables from the CSVs
# Synthetic data is generated from a fixed seed, so it needs no download. SIFT
# runs need ./scripts/fetch_sift.sh first; stage 4 is SIFT-based and requires it.
#
# Stage 4 writes its results to results/ (CSV, SVG charts, markdown tables);
# those are committed, so the README's numbers can be traced to a run.
#
# Single-threaded measurements. On Apple Silicon the OS may schedule the JVM on
# an efficiency core; close other heavy work. Each point is the median of 3
# passes, and the tables show the min-max spread.
set -euo pipefail
cd "$(dirname "$0")/.."

# Fixed heap so GC sizing doesn't vary between runs; AlwaysPreTouch faults the
# heap in up front so page faults don't land inside timed queries.
JAVA="java ${JAVA_OPTS:--Xms3g -Xmx3g -XX:+AlwaysPreTouch} -cp build"
JAVA_1M="java ${JAVA_OPTS:--Xms4g -Xmx4g -XX:+AlwaysPreTouch} -cp build"
BRUTE_QUERIES=${BRUTE_QUERIES:-200}
SYNTH=data/synth-n100k-d128-l16
UNIFORM=data/uniform-n100k-d128
SIFT100K="--data data/sift1m --base-limit 100000"

have_sift() { [ -f data/sift1m/base.fvecs ]; }

synthetic_data() {
    [ -f $SYNTH/base.fvecs ]   || $JAVA hnsw.tools.Generate --out $SYNTH --n 100000 --dim 128 --latent-dim 16 --clusters 64 --seed 42
    [ -f $UNIFORM/base.fvecs ] || $JAVA hnsw.tools.Generate --out $UNIFORM --n 100000 --dim 128 --latent-dim 128 --clusters 1 --cluster-std 1 --noise 0 --seed 42
}

# ------------------------------------------------------------------ stage 1
stage1() {
    echo
    echo "=== how hard is each dataset? (relative contrast near 1 = everything equidistant)"
    $JAVA hnsw.tools.Stats --data $UNIFORM
    $JAVA hnsw.tools.Stats --data $SYNTH
    if have_sift; then
        $JAVA hnsw.tools.Stats $SIFT100K
        $JAVA hnsw.tools.Stats --data data/sift1m
    fi

    echo
    echo "=== ground-truth correctness vs. shipped SIFT ground truth"
    if have_sift; then
        $JAVA hnsw.tools.VerifyGroundTruth --data data/sift1m
    else
        echo "(skipped: no SIFT data - ./scripts/fetch_sift.sh)"
    fi

    echo
    echo "=== brute-force baseline"
    $JAVA hnsw.tools.Bench --data $SYNTH --index brute --queries $BRUTE_QUERIES
    if have_sift; then
        $JAVA hnsw.tools.Bench $SIFT100K --index brute --queries $BRUTE_QUERIES
        $JAVA hnsw.tools.Bench --data data/sift1m --index brute --queries $BRUTE_QUERIES --repeats 1
    fi
}

# ------------------------------------------------------------------ stage 2
# Flat NSW. Two graphs from the same code: degree capped at 2M with the
# "keep the nearest" rule, and uncapped. The difference between them is the
# setup for stage 3's neighbour-selection heuristic.
stage2() {
    echo
    echo "=== flat NSW: where greedy search gets stuck"
    for cap in 32 0; do
        $JAVA hnsw.tools.GreedyFailures --data $SYNTH --max-degree $cap
        echo
    done
    $JAVA hnsw.tools.GreedyFailures --data $UNIFORM --max-degree 0
    if have_sift; then
        for cap in 32 0; do
            echo
            $JAVA hnsw.tools.GreedyFailures $SIFT100K --max-degree $cap
        done
    fi

    echo
    echo "=== flat NSW: recall vs. throughput (brute force in stage 1 is the baseline)"
    for cap in 32 0; do
        $JAVA hnsw.tools.Bench --data $SYNTH --index nsw --queries 1000 --max-degree $cap
    done
    if have_sift; then
        for cap in 32 0; do
            $JAVA hnsw.tools.Bench $SIFT100K --index nsw --queries 1000 --max-degree $cap
        done
    fi
}

# ------------------------------------------------------------------ stage 3
# HNSW, plus the 2x2 ablation that separates what the hierarchy buys from what
# the neighbour-selection heuristic buys. --level-mult 0 puts every node on
# layer 0 (flat); --selection simple keeps the M nearest (stage 2's rule).
stage3() {
    echo
    echo "=== HNSW ablation: {flat, layered} x {simple, heuristic} selection"
    ablation() {
        for sel in simple heuristic; do
            $JAVA hnsw.tools.Bench "$@" --index hnsw --queries 1000 --level-mult 0 --selection $sel
            $JAVA hnsw.tools.Bench "$@" --index hnsw --queries 1000 --selection $sel
        done
    }
    ablation --data $SYNTH
    if have_sift; then
        ablation $SIFT100K
    fi
}

# ------------------------------------------------------------------ stage 4
# The measurement stage. Every run appends to a CSV in results/, and the plot
# tool turns those into the charts and tables the README uses.
EF=10,16,24,32,48,64,96,128,192,256,384,512
EF_WIDE=$EF,768,1024

stage4() {
    stage4a; stage4b; stage4c; stage4d; plots
}

# Every chart and table in results/, regenerated from whichever CSVs exist.
# Separate from the runs so a styling change never needs a re-measurement:
#   ./scripts/bench.sh plots
plots() {
    local P="$JAVA hnsw.tools.Plot"
    local sift="HNSW,flat + heuristic,flat NSW (stage 2)"
    if [ -f results/sift.csv ]; then
        $P tradeoff --csv results/sift.csv --where base_n=1000000 --series "$sift" \
            --title "SIFT1M: recall vs. throughput" \
            --subtitle "1M x 128-d vectors, k = 10, one thread on an Apple M2. Up and to the right is better; each point is one efSearch." \
            --svg results/sift1m-tradeoff.svg --table results/sift1m-tradeoff.md >/dev/null
        for t in 0.95 0.99; do
            $P at-recall --csv results/sift.csv --x base_n --x-label "vectors indexed" --target $t --series "$sift" \
                --title "SIFT: cost of recall@10 = $t as the dataset grows" \
                --subtitle "Distance computations per query - the machine-independent cost. Brute force is n." \
                --svg results/sift-scaling-$t.svg --table results/sift-scaling-$t.md >/dev/null
        done
        $P builds --csv results/sift.csv --series "$sift" --table results/sift-builds.md >/dev/null
    fi
    if [ -f results/dims.csv ]; then
        for t in 0.95 0.99; do
            $P at-recall --csv results/dims.csv --x latent-dim --x-label "latent (intrinsic) dimension" --target $t \
                --series "HNSW,flat + heuristic" --title "Cost of recall@10 = $t as intrinsic dimension rises" \
                --subtitle "Synthetic: 100k vectors of 128 floats from a latent space of the given dimension. Hollow = upper bound." \
                --svg results/dims-$t.svg --table results/dims-$t.md >/dev/null
        done
    fi
    if [ -f results/params.csv ]; then
        for t in 0.95 0.99; do
            $P at-recall --csv results/params.csv --where ef_construction=100 --x m --x-label M --target $t \
                --series HNSW --title M --table results/params-m-$t.md >/dev/null
            $P at-recall --csv results/params.csv --where m=16 --x ef_construction --x-label efConstruction \
                --target $t --series HNSW --title efConstruction --table results/params-efc-$t.md >/dev/null
        done
        $P builds --csv results/params.csv --series HNSW --table results/params-builds.md >/dev/null
    fi
    for latent in 16 128; do
        [ -f results/clusters-l$latent.csv ] || continue
        for t in 0.95 0.99; do
            $P at-recall --csv results/clusters-l$latent.csv --x clusters --x-label clusters --target $t \
                --series "HNSW,flat + heuristic" --title clusters --table results/clusters-l$latent-$t.md >/dev/null
        done
    done
    echo "plots and tables regenerated in results/"
}

environment() {
    mkdir -p results
    {
        echo "date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
        echo "java:   $(java -version 2>&1 | head -1)"
        echo "cpu:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
        echo "memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
        echo "os:     $(uname -sr)"
        echo "single-threaded; 1000 queries per point, median of 3 timed passes (up to 9 if they disagree by >10%); 1 s warm-up per point"
    } > results/ENVIRONMENT.txt
    cat results/ENVIRONMENT.txt
}

# 4a. The headline curve, and how each index scales with n. The same three
# indexes on 100k, 300k and 1M prefixes of SIFT1M.
stage4a() {
    if ! have_sift; then
        echo "stage 4a needs SIFT1M: ./scripts/fetch_sift.sh" >&2
        exit 1
    fi
    environment
    local series="HNSW,flat + heuristic,flat NSW (stage 2)"
    echo
    echo "=== 4a. SIFT: recall vs. throughput, at three dataset sizes"
    rm -f results/sift.csv
    for n in 100000 300000 1000000; do
        local J=$JAVA
        [ $n -ge 1000000 ] && J=$JAVA_1M
        $J hnsw.tools.Bench --data data/sift1m --base-limit $n --index hnsw --queries 1000 --ef-search $EF \
            --csv results/sift.csv --label "HNSW"
        $J hnsw.tools.Bench --data data/sift1m --base-limit $n --index hnsw --level-mult 0 --queries 1000 --ef-search $EF \
            --csv results/sift.csv --label "flat + heuristic"
        $J hnsw.tools.Bench --data data/sift1m --base-limit $n --index nsw --max-degree 0 --queries 1000 --ef-search $EF \
            --csv results/sift.csv --label "flat NSW (stage 2)"
    done
    $JAVA_1M hnsw.tools.Bench --data data/sift1m --index brute --queries 1000 --repeats 1 \
        --csv results/sift.csv --label "brute force"

}

# 4b. Dimensionality. Hold ambient dimension at 128 and raise the latent
# (intrinsic) dimension the data actually varies in. Prediction from stage 3:
# every index gets more expensive, and the hierarchy's advantage over a flat
# graph shrinks as the data loses its low-dimensional structure.
stage4b() {
    environment
    echo
    echo "=== 4b. intrinsic dimension: synthetic, 100k x 128-d, latent dimension 4 .. 128"
    rm -f results/dims.csv results/dims-contrast.txt
    for latent in 4 8 16 32 64 128; do
        local d=data/dims-l$latent
        [ -f $d/base.fvecs ] || $JAVA hnsw.tools.Generate --out $d --n 100000 --queries 1000 --dim 128 \
            --latent-dim $latent --clusters 64 --seed 42
        $JAVA hnsw.tools.Stats --data $d | tee -a results/dims-contrast.txt
        $JAVA hnsw.tools.Bench --data $d --index hnsw --queries 1000 --ef-search $EF_WIDE \
            --csv results/dims.csv --label "HNSW"
        $JAVA hnsw.tools.Bench --data $d --index hnsw --level-mult 0 --queries 1000 --ef-search $EF_WIDE \
            --csv results/dims.csv --label "flat + heuristic"
    done
}

# 4c. What M and efConstruction buy, one at a time from the defaults
# (M = 16, efConstruction = 100), on SIFT 100k.
stage4c() {
    have_sift || { echo "stage 4c needs SIFT1M" >&2; exit 1; }
    environment
    echo
    echo "=== 4c. HNSW parameters on SIFT 100k: M, then efConstruction"
    rm -f results/params.csv
    for m in 4 8 16 32 48; do
        $JAVA hnsw.tools.Bench $SIFT100K --index hnsw --m $m --queries 1000 --ef-search $EF \
            --csv results/params.csv --label "HNSW"
    done
    for efc in 50 200 400; do
        $JAVA hnsw.tools.Bench $SIFT100K --index hnsw --ef-construction $efc --queries 1000 --ef-search $EF \
            --csv results/params.csv --label "HNSW"
    done
}

# 4d. Cluster structure. 4b's prediction failed: the hierarchy helped 1.3-2.4x
# at every intrinsic dimension on the synthetic data, yet only ~5% on SIFT. The
# synthetic data has 64 well-separated clusters. Hypothesis: the upper layers
# earn their keep by jumping between clusters. Test: same generator, same
# latent dimension, 1 cluster vs 64 - only the clustering changes.
stage4d() {
    environment
    echo
    echo "=== 4d. cluster structure: 1 cluster vs 64, at latent dimension 16 and 128"
    for latent in 16 128; do
        rm -f results/clusters-l$latent.csv
        for clusters in 1 64; do
            local d=data/clusters$clusters-l$latent
            [ -f $d/base.fvecs ] || $JAVA hnsw.tools.Generate --out $d --n 100000 --queries 1000 --dim 128 \
                --latent-dim $latent --clusters $clusters --seed 42
            $JAVA hnsw.tools.Stats --data $d
            $JAVA hnsw.tools.Bench --data $d --index hnsw --queries 1000 --ef-search $EF_WIDE \
                --csv results/clusters-l$latent.csv --label "HNSW"
            $JAVA hnsw.tools.Bench --data $d --index hnsw --level-mult 0 --queries 1000 --ef-search $EF_WIDE \
                --csv results/clusters-l$latent.csv --label "flat + heuristic"
        done
    done
}

./scripts/build.sh
echo "java: $(java -version 2>&1 | head -1)"
echo "cpu : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
synthetic_data

case "${1:-all}" in
    all) stage1; stage2; stage3; stage4 ;;
    stage1|stage2|stage3|stage4) "$1" ;;
    stage4a|stage4b|stage4c|stage4d) "$1"; plots ;;
    plots) plots ;;
    *) echo "usage: $0 [all|stage1|stage2|stage3|stage4|stage4a..stage4d|plots]" >&2; exit 2 ;;
esac
