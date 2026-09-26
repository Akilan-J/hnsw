package hnsw.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

import hnsw.bench.Benchmark;
import hnsw.bench.GroundTruth;
import hnsw.data.Dataset;
import hnsw.distance.DistanceFunction;
import hnsw.index.BruteForceIndex;
import hnsw.index.KnnIndex;

/**
 * Loads a dataset, builds an index, and benchmarks it against ground truth.
 *
 * <p>usage: Bench --data DIR [--metric l2] [--index brute] [--k 10]
 * [--base-limit N] [--queries N] [--warmup 50] [--seed 1] [--csv FILE]
 */
public final class Bench {

    public static void main(String[] argv) throws IOException {
        Args args = new Args(argv);
        Path dir = Path.of(args.required("data"));
        DistanceFunction metric = DistanceFunction.byName(args.str("metric", "l2"));
        String indexType = args.str("index", "brute");
        int k = args.integer("k", 10);
        int baseLimit = args.integer("base-limit", 0);
        int queryLimit = args.integer("queries", 0);
        int warmup = args.integer("warmup", 50);
        long seed = args.longValue("seed", 1);
        String csv = args.str("csv", null);
        args.done();
        if (k > GroundTruth.STORED_K) {
            throw new IllegalArgumentException("k=" + k + " exceeds stored ground truth depth " + GroundTruth.STORED_K);
        }

        long loadStart = System.nanoTime();
        Dataset data = Dataset.load(dir, metric, baseLimit);
        System.out.printf("loaded %s: %,d base x %d dims, %,d queries (%.1fs)%n",
                data.name, data.base.length, data.dim(), data.queries.length, (System.nanoTime() - loadStart) / 1e9);
        int[][] gt = GroundTruth.loadOrCompute(data);

        // Time a prefix of the query set. Brute force on SIFT1M is slow enough that
        // timing all 10k queries would take many minutes to say the same thing.
        int nq = queryLimit > 0 ? Math.min(queryLimit, data.queries.length) : data.queries.length;
        float[][] queries = Arrays.copyOf(data.queries, nq);

        long buildStart = System.nanoTime();
        KnnIndex index = switch (indexType) {
            case "brute" -> new BruteForceIndex(data.base, metric);
            default -> throw new IllegalArgumentException("unknown index '" + indexType + "'");
        };
        double buildSeconds = (System.nanoTime() - buildStart) / 1e9;
        System.out.printf("built %s in %.2fs%n", index.describe(), buildSeconds);

        Benchmark.Result result = Benchmark.run(index, data.base, queries, gt, k, warmup, seed);
        System.out.println(result.pretty());

        if (csv != null) {
            Path csvPath = Path.of(csv);
            if (csvPath.getParent() != null) {
                Files.createDirectories(csvPath.getParent());
            }
            if (!Files.exists(csvPath)) {
                Files.writeString(csvPath, "dataset,base_n," + Benchmark.Result.csvHeader() + "\n");
            }
            Files.writeString(csvPath, data.name + "," + data.base.length + "," + result.csv() + "\n",
                    StandardOpenOption.APPEND);
        }
    }
}
