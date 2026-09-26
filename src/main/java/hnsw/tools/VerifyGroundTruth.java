package hnsw.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import hnsw.bench.GroundTruth;
import hnsw.data.Dataset;
import hnsw.data.VectorIO;
import hnsw.distance.DistanceFunction;

/**
 * Checks our brute force against ground truth computed by someone else - the
 * file SIFT ships with. Recall is only as trustworthy as the ground truth it's
 * scored against, so this is the one check that doesn't rely on our own code
 * being right.
 *
 * <p>usage: VerifyGroundTruth --data DIR [--metric l2]
 * Exits non-zero on any real mismatch.
 */
public final class VerifyGroundTruth {

    public static void main(String[] argv) throws IOException {
        Args args = new Args(argv);
        Path dir = Path.of(args.required("data"));
        DistanceFunction metric = DistanceFunction.byName(args.str("metric", "l2"));
        args.done();

        // Full base set: the shipped file is for the whole dataset, not a prefix.
        Dataset data = Dataset.load(dir, metric, 0);
        Path shipped = data.shippedGroundTruthPath();
        if (!Files.exists(shipped)) {
            System.err.println(dir + " has no shipped groundtruth.ivecs to verify against");
            System.exit(2);
        }
        int[][] reference = VectorIO.readIvecs(shipped, 0);
        int[][] ours = GroundTruth.loadOrCompute(data);
        int k = Math.min(reference[0].length, GroundTruth.STORED_K);

        GroundTruth.Verification v = GroundTruth.verify(data.base, data.queries, metric, ours, reference, k);
        System.out.printf("%s: %,d queries x top-%d = %,d positions%n", data.name, data.queries.length, k,
                (long) data.queries.length * k);
        System.out.printf("  identical ids        : %,d%n", v.exactMatches());
        System.out.printf("  equal-distance swaps : %,d  (ties broken differently - both correct)%n", v.tieSwaps());
        System.out.printf("  real mismatches      : %,d%n", v.errors());
        if (!v.ok()) {
            System.out.println("  first mismatch: " + v.firstError());
            System.exit(1);
        }
        System.out.println("  OK - brute force agrees with the shipped ground truth");
    }
}
