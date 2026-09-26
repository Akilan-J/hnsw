package hnsw.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.IntStream;

import hnsw.data.Dataset;
import hnsw.data.VectorIO;
import hnsw.distance.DistanceFunction;
import hnsw.index.BruteForceIndex;

/**
 * Exact nearest neighbours for every query, computed once by brute force and
 * cached on disk as .ivecs (the same format SIFT ships its ground truth in).
 */
public final class GroundTruth {

    /** Depth stored on disk. Matches SIFT's shipped file; recall@10 uses a prefix. */
    public static final int STORED_K = 100;

    private GroundTruth() {
    }

    /**
     * Loads cached ground truth for this dataset configuration, computing and
     * saving it first if needed. One row per query in query.fvecs.
     */
    public static int[][] loadOrCompute(Dataset data) throws IOException {
        Path path = data.groundTruthPath(STORED_K);
        if (!Files.exists(path)) {
            System.out.printf("computing ground truth: %,d queries x %,d base vectors (%s), %d threads%n",
                    data.queries.length, data.base.length, data.metric.name(),
                    Runtime.getRuntime().availableProcessors());
            long start = System.nanoTime();
            int[][] gt = compute(data.base, data.queries, data.metric, STORED_K);
            System.out.printf("ground truth computed in %.1fs -> %s%n", (System.nanoTime() - start) / 1e9, path);
            VectorIO.writeIvecs(path, gt);
        }
        int[][] gt = VectorIO.readIvecs(path, 0);
        if (gt.length != data.queries.length || gt[0].length < STORED_K) {
            throw new IOException(path + " is " + gt.length + " x " + gt[0].length + ", expected "
                    + data.queries.length + " x " + STORED_K + " - stale cache? delete it to recompute");
        }
        return gt;
    }

    /**
     * Brute-force top-k for every query, in parallel across queries.
     *
     * <p>Queries are independent and the index is read-only, so this is
     * embarrassingly parallel. A parallel stream runs it on the common fork-join
     * pool. Each task writes only its own slot of {@code out}, so there is no
     * shared mutable state, and the stream's terminal operation happens-before
     * this method returns, so the caller sees every slot filled in.
     *
     * <p>Because TopK breaks ties by id, the result does not depend on how the
     * work was split across threads.
     */
    public static int[][] compute(float[][] base, float[][] queries, DistanceFunction metric, int k) {
        BruteForceIndex index = new BruteForceIndex(base, metric);
        int[][] out = new int[queries.length][];
        IntStream.range(0, queries.length).parallel().forEach(q -> out[q] = index.search(queries[q], k));
        return out;
    }

    /**
     * Compares our ground truth against a reference (e.g. the file shipped with
     * SIFT) position by position, and tells real errors apart from tie-breaking
     * differences.
     *
     * <p>If two vectors are exactly equidistant from a query, either order is
     * correct, and the reference used its own tie rule, not ours. So a
     * position where the ids differ counts as an error only if the two ids are
     * at different distances.
     *
     * <p>For SIFT this check is exact. Its components are integers in [0, 255],
     * so every squared difference and every partial sum is an integer below
     * 128 * 255^2 = 8.3M, under the 2^24 = 16.7M limit where float stops
     * representing every integer exactly. So the distance comes out bit-identical
     * whatever order it's summed in. For general float data, allow a relative
     * tolerance for summation-order rounding.
     */
    public static Verification verify(float[][] base, float[][] queries, DistanceFunction metric,
                                      int[][] ours, int[][] reference, int k) {
        long exact = 0, ties = 0, errors = 0;
        String firstError = null;
        for (int q = 0; q < queries.length; q++) {
            for (int i = 0; i < k; i++) {
                int a = ours[q][i];
                int b = reference[q][i];
                if (a == b) {
                    exact++;
                    continue;
                }
                float da = metric.distance(queries[q], base[a]);
                float db = metric.distance(queries[q], base[b]);
                if (Math.abs(da - db) <= 1e-5f * Math.max(1f, Math.abs(db))) {
                    ties++;
                } else {
                    errors++;
                    if (firstError == null) {
                        firstError = String.format("query %d rank %d: ours id %d (d=%s), reference id %d (d=%s)",
                                q, i, a, da, b, db);
                    }
                }
            }
        }
        return new Verification(exact, ties, errors, firstError);
    }

    public record Verification(long exactMatches, long tieSwaps, long errors, String firstError) {
        public boolean ok() {
            return errors == 0;
        }
    }
}
