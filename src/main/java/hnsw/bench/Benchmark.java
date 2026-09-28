package hnsw.bench;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import hnsw.distance.DistanceFunction;
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
     * @param warmupSeconds untimed searches first, for at least this long, so the
     *                      JIT has compiled the hot path for <i>this</i>
     *                      configuration before the clock starts. By time rather
     *                      than a query count, because a fixed count is too few for a
     *                      0.03 ms graph search and far too many for a 60 ms brute
     *                      force scan. Stage 3 showed the cost of too little: the
     *                      first point of every sweep had an inflated p99 and lower
     *                      QPS than the next, larger ef. Warm-up queries are base
     *                      vectors, not the timed queries, so the timed queries'
     *                      graph paths aren't pre-loaded into cache.
     * @param repeats       timed passes over the query set. The reported numbers
     *                      are from the pass with the median QPS, with the min and
     *                      max kept to show the run-to-run spread. The median
     *                      rather than the best: best-of-N reports the machine on its
     *                      luckiest run, and the mean lets one stall move the number.
     */
    public static Result run(KnnIndex index, float[][] base, float[][] queries, int[][] groundTruth,
                             DistanceFunction metric, int k, double warmupSeconds, int repeats, long seed) {
        Random rng = new Random(seed);
        long sink = 0;
        long warmupStart = System.nanoTime();
        for (int i = 0; i < 10 || System.nanoTime() - warmupStart < warmupSeconds * 1e9; i++) {
            sink += index.search(base[rng.nextInt(base.length)], k)[0];
        }
        blackhole = sink;

        Pass[] passes = new Pass[Math.max(1, repeats)];
        for (int r = 0; r < passes.length; r++) {
            passes[r] = timedPass(index, queries, k);
        }
        Pass[] byQps = passes.clone();
        Arrays.sort(byQps, Comparator.comparingDouble(Pass::qps));
        Pass median = byQps[byQps.length / 2];

        // The search is deterministic, so every pass returns the same results and
        // the same distance count; scoring one is scoring all of them.
        double distancesPerQuery = median.distances < 0 ? base.length // brute force: exactly n
                : (double) median.distances / queries.length;
        long[] sorted = median.latencies.clone();
        Arrays.sort(sorted);
        return new Result(index.describe(), queries.length, k, median.qps(), byQps[0].qps(),
                byQps[byQps.length - 1].qps(), passes.length,
                Arrays.stream(sorted).average().orElse(0) / 1e6,
                percentileMs(sorted, 50), percentileMs(sorted, 95), percentileMs(sorted, 99),
                sorted[sorted.length - 1] / 1e6,
                recallAtK(median.results, groundTruth, k),
                recallAtKWithTies(median.results, groundTruth, base, queries, metric, k),
                distancesPerQuery);
    }

    private record Pass(long[] latencies, int[][] results, long wallNanos, long distances) {
        double qps() {
            return latencies.length / (wallNanos / 1e9);
        }
    }

    private static Pass timedPass(KnnIndex index, float[][] queries, int k) {
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
        long distances = distancesBefore < 0 ? -1 : index.distanceComputations() - distancesBefore;
        return new Pass(latencies, results, wallNanos, distances);
    }

    /**
     * Fraction of the true k nearest neighbours that were returned, averaged over
     * queries, matching by id. Strict: a vector at exactly the same distance as the
     * k-th true neighbour, but a different id, counts as a miss.
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

    /**
     * Recall@k that accepts ties: a returned vector counts if it's no farther than
     * the k-th true neighbour. SIFT's integer coordinates make exact ties common
     * (2% of top-100 positions), and when the tie straddles rank k, id matching
     * marks a correct answer wrong. This is the convention ANN benchmark suites
     * use. It can only be >= strict recall, since every id match also passes the
     * distance test. The tolerance matches GroundTruth.verify: relative 1e-5, for
     * summation-order rounding.
     */
    public static double recallAtKWithTies(int[][] results, int[][] groundTruth, float[][] base,
                                           float[][] queries, DistanceFunction metric, int k) {
        double total = 0;
        for (int q = 0; q < results.length; q++) {
            float kth = metric.distance(queries[q], base[groundTruth[q][k - 1]]);
            float limit = kth + 1e-5f * Math.max(1f, Math.abs(kth));
            int hits = 0;
            for (int i = 0; i < Math.min(k, results[q].length); i++) {
                if (metric.distance(queries[q], base[results[q][i]]) <= limit) {
                    hits++;
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

    public record Result(String index, int queries, int k, double qps, double qpsMin, double qpsMax, int repeats,
                         double meanMs, double p50Ms, double p95Ms, double p99Ms, double maxMs,
                         double recall, double recallWithTies, double distancesPerQuery) {

        public String pretty() {
            return String.format("%s%n  queries=%d k=%d%n  recall@%d : %.4f (with ties: %.4f)%n"
                            + "  throughput: %,.1f QPS (median of %d; range %,.1f-%,.1f)%n"
                            + "  latency  : mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms max=%.3fms%n"
                            + "  distances: %,.0f per query",
                    index, queries, k, k, recall, recallWithTies, qps, repeats, qpsMin, qpsMax,
                    meanMs, p50Ms, p95Ms, p99Ms, maxMs, distancesPerQuery);
        }

        /** One line per run, for parameter sweeps. */
        public String row() {
            return String.format("  %-48s recall@%d=%.4f (ties %.4f)  %,9.1f QPS (±%2.0f%%)  p50=%7.3fms  p99=%7.3fms  %,9.0f dist/q",
                    index, k, recall, recallWithTies, qps, 50 * (qpsMax - qpsMin) / qps, p50Ms, p99Ms,
                    distancesPerQuery);
        }

        public static String csvHeader() {
            return "queries,k,recall,recall_ties,qps,qps_min,qps_max,repeats,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,"
                    + "distances_per_query";
        }

        public String csv() {
            return String.format("%d,%d,%.5f,%.5f,%.2f,%.2f,%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f",
                    queries, k, recall, recallWithTies, qps, qpsMin, qpsMax, repeats, meanMs, p50Ms, p95Ms,
                    p99Ms, maxMs, distancesPerQuery);
        }
    }
}
