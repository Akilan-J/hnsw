package hnsw.bench;

import java.util.Arrays;
import java.util.Random;

import hnsw.index.KnnIndex;

/**
 * Measures one index on one query set: throughput, latency percentiles, and
 * recall@k against ground truth. Every index is measured by this same code, so
 * differences between them are differences between the indexes.
 *
 * <p>Single-threaded on purpose: one query at a time, back to back. That isolates
 * the algorithm's cost per query. In this setting QPS is simply 1 / mean latency.
 * The two only come apart under concurrency, which is a property of the
 * server around an index rather than of the index itself.
 */
public final class Benchmark {

    // Warm-up results are written here so the JIT can't prove them unused and
    // optimise the warm-up searches away. A volatile write can't be elided.
    private static volatile long blackhole;

    private Benchmark() {
    }

    /**
     * @param warmupQueries untimed searches run first so the JIT has compiled
     *                      the hot paths before the clock starts. They use base
     *                      vectors, not the timed queries: warming up on the exact
     *                      queries about to be timed would pre-load their graph
     *                      paths into CPU cache and flatter the numbers.
     */
    public static Result run(KnnIndex index, float[][] base, float[][] queries, int[][] groundTruth,
                             int k, int warmupQueries, long seed) {
        Random rng = new Random(seed);
        long sink = 0;
        for (int i = 0; i < warmupQueries; i++) {
            sink += index.search(base[rng.nextInt(base.length)], k)[0];
        }
        blackhole = sink;

        long[] latencies = new long[queries.length];
        int[][] results = new int[queries.length][];
        long distancesBefore = index.distanceComputations();
        long wallStart = System.nanoTime();
        for (int q = 0; q < queries.length; q++) {
            long t0 = System.nanoTime();
            results[q] = index.search(queries[q], k);
            latencies[q] = System.nanoTime() - t0;
        }
        long wallNanos = System.nanoTime() - wallStart;
        double distancesPerQuery = distancesBefore < 0 ? base.length // brute force: exactly n
                : (double) (index.distanceComputations() - distancesBefore) / queries.length;
        // Recall is scored after the clock stops so it never counts as search time.
        double recall = recallAtK(results, groundTruth, k);

        long[] sorted = latencies.clone();
        Arrays.sort(sorted);
        double meanMs = Arrays.stream(latencies).average().orElse(0) / 1e6;
        return new Result(index.describe(), queries.length, k, queries.length / (wallNanos / 1e9), meanMs,
                percentileMs(sorted, 50), percentileMs(sorted, 95), percentileMs(sorted, 99),
                sorted[sorted.length - 1] / 1e6, recall, distancesPerQuery);
    }

    /**
     * Fraction of the true k nearest neighbours that were returned, averaged over
     * queries. Ids only, no distances, so an index that returns a different
     * vector at exactly the same distance as the k-th true neighbour is marked
     * wrong even though it isn't. Not hypothetical: SIFT's integer coordinates
     * make 2% of top-100 positions exact ties. Stage 4 measures how much this
     * moves recall@10.
     */
    public static double recallAtK(int[][] results, int[][] groundTruth, int k) {
        double total = 0;
        for (int q = 0; q < results.length; q++) {
            int[] truth = groundTruth[q];
            int hits = 0;
            // k is small (10), so a nested scan beats building a hash set per query.
            for (int i = 0; i < Math.min(k, results[q].length); i++) {
                for (int j = 0; j < k; j++) {
                    if (results[q][i] == truth[j]) {
                        hits++;
                        break;
                    }
                }
            }
            total += (double) hits / k;
        }
        return total / results.length;
    }

    /** Nearest-rank percentile, same definition as the kvstore benchmark. */
    static double percentileMs(long[] sorted, double p) {
        int index = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))] / 1e6;
    }

    public record Result(String index, int queries, int k, double qps, double meanMs,
                         double p50Ms, double p95Ms, double p99Ms, double maxMs, double recall,
                         double distancesPerQuery) {

        public String pretty() {
            return String.format("%s%n  queries=%d k=%d%n  recall@%d : %.4f%n  throughput: %,.1f QPS%n"
                            + "  latency  : mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms max=%.3fms%n"
                            + "  distances: %,.0f per query",
                    index, queries, k, k, recall, qps, meanMs, p50Ms, p95Ms, p99Ms, maxMs, distancesPerQuery);
        }

        /** One line per run, for parameter sweeps. */
        public String row() {
            return String.format("  %-48s recall@%d=%.4f  %,9.1f QPS  p50=%7.3fms  p99=%7.3fms  %,9.0f dist/q",
                    index, k, recall, qps, p50Ms, p99Ms, distancesPerQuery);
        }

        public static String csvHeader() {
            return "index,queries,k,recall,qps,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,distances_per_query";
        }

        public String csv() {
            return String.format("\"%s\",%d,%d,%.5f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f",
                    index, queries, k, recall, qps, meanMs, p50Ms, p95Ms, p99Ms, maxMs, distancesPerQuery);
        }
    }
}
