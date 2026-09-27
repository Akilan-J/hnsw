package hnsw.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import hnsw.bench.Benchmark;
import hnsw.bench.GroundTruth;
import hnsw.data.Dataset;
import hnsw.distance.DistanceFunction;
import hnsw.index.BruteForceIndex;
import hnsw.index.FlatNswIndex;

/**
 * Loads a dataset, builds an index, and benchmarks it against ground truth.
 *
 * <p>usage: Bench --data DIR [--metric l2] [--index brute|nsw] [--k 10]
 * [--base-limit N] [--queries N] [--warmup 50] [--seed 1] [--csv FILE]
 * <br>nsw options: [--m 16] [--ef-construction 100] [--max-degree 32 (0 = unbounded)]
 * [--ef-search 10,20,40,80,160]
 *
 * <p>A graph index is built once and then benchmarked at every efSearch in the
 * list: build is the expensive part, and ef only affects search.
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
        int m = args.integer("m", 16);
        int efConstruction = args.integer("ef-construction", 100);
        int maxDegree = args.integer("max-degree", 2 * m);
        int[] efSearch = Arrays.stream(args.str("ef-search", "10,20,40,80,160").split(","))
                .map(String::trim).mapToInt(Integer::parseInt).toArray();
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

        List<Benchmark.Result> results = new ArrayList<>();
        switch (indexType) {
            case "brute" -> {
                BruteForceIndex index = new BruteForceIndex(data.base, metric);
                Benchmark.Result r = Benchmark.run(index, data.base, queries, gt, k, warmup, seed);
                System.out.println(r.pretty());
                results.add(r);
            }
            case "nsw" -> {
                FlatNswIndex index = new FlatNswIndex(data.base, metric, m, efConstruction, maxDegree);
                double seconds = build(index, data.base.length);
                printGraphStats(index, seconds);
                for (int ef : efSearch) {
                    index.setEfSearch(ef);
                    Benchmark.Result r = Benchmark.run(index, data.base, queries, gt, k, warmup, seed);
                    System.out.println(r.row());
                    results.add(r);
                }
            }
            default -> throw new IllegalArgumentException("unknown index '" + indexType + "'");
        }

        if (csv != null) {
            Path csvPath = Path.of(csv);
            if (csvPath.getParent() != null) {
                Files.createDirectories(csvPath.getParent());
            }
            if (!Files.exists(csvPath)) {
                Files.writeString(csvPath, "dataset,base_n," + Benchmark.Result.csvHeader() + "\n");
            }
            for (Benchmark.Result r : results) {
                Files.writeString(csvPath, data.name + "," + data.base.length + "," + r.csv() + "\n",
                        StandardOpenOption.APPEND);
            }
        }
    }

    private static double build(FlatNswIndex index, int n) {
        long start = System.nanoTime();
        int step = Math.max(1, n / 10);
        for (int i = 0; index.insertNext(); i++) {
            if ((i + 1) % step == 0 && n >= 200_000) {
                System.out.printf("  inserted %,d / %,d (%.0fs)%n", i + 1, n, (System.nanoTime() - start) / 1e9);
            }
        }
        return (System.nanoTime() - start) / 1e9;
    }

    private static void printGraphStats(FlatNswIndex index, double buildSeconds) {
        int n = index.size();
        long edges = 0;
        int maxDeg = 0;
        for (int i = 0; i < n; i++) {
            edges += index.degree(i);
            maxDeg = Math.max(maxDeg, index.degree(i));
        }
        long unreachable = Arrays.stream(index.hopsFrom(index.entryPoint())).filter(h -> h < 0).count();
        System.out.printf("built %s in %.1fs (%,.0f inserts/s)%n", index.describe(), buildSeconds, n / buildSeconds);
        System.out.printf("  graph: %,d directed edges, mean out-degree %.1f, max %d, %.1f MB, "
                        + "%,d nodes unreachable from entry%n",
                edges, (double) edges / n, maxDeg, index.graphBytes() / 1e6, unreachable);
    }
}
