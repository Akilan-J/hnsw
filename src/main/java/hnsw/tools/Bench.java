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
import hnsw.index.GraphIndex;
import hnsw.index.HnswIndex;

/**
 * Loads a dataset, builds an index, and benchmarks it against ground truth.
 *
 * <p>usage: Bench --data DIR [--metric l2] [--index brute|nsw|hnsw] [--k 10]
 * [--base-limit N] [--queries N] [--warmup-seconds 1] [--repeats 3] [--seed 1]
 * [--csv FILE --label SERIES]
 * <br>graph options: [--m 16] [--ef-construction 100] [--ef-search 10,20,40,80,160]
 * <br>nsw only: [--max-degree 32 (0 = unbounded)]
 * <br>hnsw only: [--selection heuristic|simple] [--level-mult 1/ln(m); 0 = flat] [--level-seed 42]
 *
 * <p>A graph index is built once and then benchmarked at every efSearch in the
 * list: build is the expensive part, and ef only affects search.
 *
 * <p>With --csv, one row per measurement is appended, carrying everything the
 * plot tool needs as its own column (label, dataset parameters, index
 * parameters, build cost, memory), so plots never parse strings back apart.
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
        double warmupSeconds = args.decimal("warmup-seconds", 1.0);
        int repeats = args.integer("repeats", 3);
        long seed = args.longValue("seed", 1);
        String csv = args.str("csv", null);
        String label = args.str("label", indexType);
        int m = args.integer("m", 16);
        int efConstruction = args.integer("ef-construction", 100);
        int maxDegree = args.integer("max-degree", 2 * m);
        HnswIndex.Selection selection = HnswIndex.Selection.valueOf(args.str("selection", "heuristic").toUpperCase());
        double levelMult = args.decimal("level-mult", HnswIndex.defaultLevelMultiplier(m));
        long levelSeed = args.longValue("level-seed", 42);
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

        List<String> rows = new ArrayList<>();
        String datasetColumns = String.join(",", csvField(label), csvField(data.name),
                Integer.toString(data.base.length), Integer.toString(data.dim()), csvField(datasetParams(dir)));
        switch (indexType) {
            case "brute" -> {
                BruteForceIndex index = new BruteForceIndex(data.base, metric);
                Benchmark.Result r = Benchmark.run(index, data.base, queries, gt, metric, k, warmupSeconds, repeats, seed);
                System.out.println(r.pretty());
                rows.add(String.join(",", datasetColumns, csvField(index.describe()), "", "", "", "0", "0", "0", r.csv()));
            }
            case "nsw", "hnsw" -> {
                long heapBefore = usedHeapAfterGc();
                GraphIndex index = indexType.equals("nsw")
                        ? new FlatNswIndex(data.base, metric, m, efConstruction, maxDegree)
                        : new HnswIndex(data.base, metric, m, efConstruction, selection, levelMult, levelSeed);
                double seconds = build(index, data.base.length);
                double heapMb = (usedHeapAfterGc() - heapBefore) / 1e6;
                double estimatedMb = index.graphBytes() / 1e6;
                System.out.printf("built %s in %.1fs (%,.0f inserts/s)%n%s%n", index.describe(), seconds,
                        data.base.length / seconds, index.graphStats());
                System.out.printf("  memory: graph estimate %.1f MB, measured heap growth %.1f MB "
                        + "(vectors: %.1f MB)%n", estimatedMb, heapMb, 4.0 * data.base.length * data.dim() / 1e6);
                for (int ef : efSearch) {
                    index.setEfSearch(ef);
                    Benchmark.Result r = Benchmark.run(index, data.base, queries, gt, metric, k, warmupSeconds, repeats, seed);
                    System.out.println(r.row());
                    rows.add(String.join(",", datasetColumns, csvField(index.describe()), Integer.toString(m),
                            Integer.toString(efConstruction), Integer.toString(ef), String.format("%.2f", seconds),
                            String.format("%.2f", estimatedMb), String.format("%.2f", heapMb), r.csv()));
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
                Files.writeString(csvPath, "label,dataset,base_n,dim,params,index,m,ef_construction,ef,"
                        + "build_s,graph_mb_est,graph_mb_heap," + Benchmark.Result.csvHeader() + "\n");
            }
            Files.writeString(csvPath, String.join("\n", rows) + "\n", StandardOpenOption.APPEND);
        }
    }

    /** The generator's params.txt, if this is a synthetic dataset. */
    private static String datasetParams(Path dir) throws IOException {
        Path p = dir.resolve("params.txt");
        return Files.exists(p) ? Files.readString(p).trim() : "";
    }

    private static String csvField(String s) {
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    /**
     * Heap in use after asking for full collections, so the number is live data
     * rather than garbage that happens not to be collected yet. System.gc() is a
     * request, not a command, so treat the result as approximate - it's here to
     * sanity-check graphBytes(), which is an estimate from array sizes.
     */
    private static long usedHeapAfterGc() {
        Runtime rt = Runtime.getRuntime();
        for (int i = 0; i < 3; i++) {
            rt.gc();
        }
        return rt.totalMemory() - rt.freeMemory();
    }

    private static double build(GraphIndex index, int n) {
        long start = System.nanoTime();
        int step = Math.max(1, n / 10);
        for (int i = 0; index.insertNext(); i++) {
            if ((i + 1) % step == 0 && n >= 200_000) {
                System.out.printf("  inserted %,d / %,d (%.0fs)%n", i + 1, n, (System.nanoTime() - start) / 1e9);
            }
        }
        return (System.nanoTime() - start) / 1e9;
    }
}
