package hnsw;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import hnsw.bench.Benchmark;
import hnsw.bench.GroundTruth;
import hnsw.data.Dataset;
import hnsw.data.VectorIO;
import hnsw.distance.CosineDistance;
import hnsw.distance.SquaredL2;
import hnsw.index.BruteForceIndex;
import hnsw.index.IndexTests;
import hnsw.index.TopK;

/**
 * Plain-main test runner - no JUnit, same as the rest of the project. Checks use
 * an explicit {@link #check} rather than Java's {@code assert}, because asserts
 * are disabled unless the JVM is started with -ea, and a test suite that
 * silently passes when run the wrong way is worse than none.
 *
 * <p>Run with ./scripts/test.sh
 */
public final class Tests {

    private static int passed, failed, skipped;

    public static void main(String[] args) throws Exception {
        run("squared L2 matches a double-precision reference, incl. dims not divisible by 4", Tests::squaredL2);
        run("cosine distance on normalized vectors matches 1 - cos", Tests::cosine);
        run("TopK equals a full sort under heavy ties, ties broken by id", Tests::topKMatchesSort);
        run("TopK with k larger than the number of candidates", Tests::topKSmallInput);
        run("brute force on a hand-checked 2-D example", Tests::bruteForceByHand);
        run("fvecs and ivecs round-trip bit-exactly", Tests::roundTrip);
        run("truncated or corrupt fvecs files are rejected", Tests::rejectsCorruptFiles);
        run("parallel ground truth equals serial brute force", Tests::parallelGroundTruth);
        run("verify() separates tie swaps from real errors", Tests::verifyClassifiesTies);
        run("recall@k scoring", Tests::recallScoring);
        run("brute force matches shipped SIFT1M ground truth (first 100 queries)", Tests::siftGroundTruth);

        // stage 2: flat NSW graph
        run("CandidateQueue pops in (distance, id) order, across growth", IndexTests::candidateQueueOrder);
        run("VisitedSet forgets everything on clear(), and only then", IndexTests::visitedSet);
        run("NSW graph invariants: no self-loops, no duplicates, cap respected", IndexTests::graphInvariants);
        run("unbounded NSW: every link has its reverse, all nodes reachable", IndexTests::unboundedIsSymmetric);
        run("NSW with ef >= n is exact (whole graph explored)", IndexTests::fullBeamIsExact);
        run("greedy (ef=1) always stops at a local minimum", IndexTests::greedyStopsAtLocalMinimum);
        run("NSW build and search are deterministic", IndexTests::deterministic);

        // stage 3: HNSW
        run("heuristic keeps one link per direction (worked example)", IndexTests::heuristicWorkedExample);
        run("HNSW invariants: caps per layer, links stay on their layer", IndexTests::hnswGraphInvariants);
        run("level distribution: 1/M of nodes per layer up", IndexTests::levelDistribution);
        run("level multiplier 0 gives a flat graph", IndexTests::zeroLevelMultiplierIsFlat);
        run("HNSW with ef >= n is exact", IndexTests::hnswExactAtFullEf);
        run("HNSW recall@10 >= 0.95 on clustered data", IndexTests::hnswRecallSmoke);
        run("HNSW build is deterministic for a fixed seed", IndexTests::hnswDeterministic);
        run("HNSW with one layer + simple selection builds stage 2's graph exactly", IndexTests::hnswWithSwitchesOffIsStage2);

        System.out.printf("%n%d passed, %d failed, %d skipped%n", passed, failed, skipped);
        System.exit(failed == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------------ tests

    static void squaredL2() {
        Random rng = new Random(1);
        for (int dim : new int[]{1, 2, 3, 4, 5, 7, 128, 131}) {
            float[] a = randomVector(rng, dim), b = randomVector(rng, dim);
            double ref = 0;
            for (int i = 0; i < dim; i++) {
                double d = (double) a[i] - b[i];
                ref += d * d;
            }
            float got = SquaredL2.INSTANCE.distance(a, b);
            check(Math.abs(got - ref) <= 1e-5 * Math.max(1, ref), "dim " + dim + ": got " + got + " want " + ref);
        }
        check(SquaredL2.INSTANCE.distance(new float[]{1, 2, 3}, new float[]{1, 2, 3}) == 0f, "self-distance is 0");
        expectThrows(() -> SquaredL2.INSTANCE.distance(new float[3], new float[4]), "dimension mismatch");
    }

    static void cosine() {
        Random rng = new Random(2);
        for (int dim : new int[]{3, 100, 129}) {
            float[] a = randomVector(rng, dim), b = randomVector(rng, dim);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < dim; i++) {
                dot += (double) a[i] * b[i];
                na += (double) a[i] * a[i];
                nb += (double) b[i] * b[i];
            }
            double ref = 1 - dot / Math.sqrt(na * nb);
            CosineDistance.normalizeInPlace(a);
            CosineDistance.normalizeInPlace(b);
            float got = CosineDistance.INSTANCE.distance(a, b);
            check(Math.abs(got - ref) < 1e-5, "dim " + dim + ": got " + got + " want " + ref);
            check(Math.abs(CosineDistance.INSTANCE.distance(a, a)) < 1e-6, "self-distance ~0");
        }
        expectThrows(() -> CosineDistance.normalizeInPlace(new float[4]), "zero vector");
    }

    static void topKMatchesSort() {
        Random rng = new Random(3);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + rng.nextInt(500);
            int k = 1 + rng.nextInt(20);
            // Distances drawn from only 8 values, so almost every comparison is a tie
            // and the id tie-break is exercised hard.
            float[] dist = new float[n];
            for (int i = 0; i < n; i++) {
                dist[i] = rng.nextInt(8);
            }
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> dist[i]).thenComparingInt(i -> i));
            int[] expected = new int[Math.min(k, n)];
            for (int i = 0; i < expected.length; i++) {
                expected[i] = order[i];
            }

            // Offer in a shuffled order: the result must not depend on scan order.
            int[] perm = shuffledRange(n, rng);
            TopK top = new TopK(Math.min(k, n));
            for (int id : perm) {
                top.offer(id, dist[id]);
            }
            int[] got = top.drainAscending();
            check(Arrays.equals(expected, got), "n=" + n + " k=" + k + "\n want " + Arrays.toString(expected)
                    + "\n got  " + Arrays.toString(got));
        }
    }

    static void topKSmallInput() {
        TopK top = new TopK(10);
        top.offer(7, 3f);
        top.offer(2, 1f);
        top.offer(5, 2f);
        check(!top.isFull() && top.size() == 3, "size 3, not full");
        check(Arrays.equals(new int[]{2, 5, 7}, top.drainAscending()), "ascending order");
        check(top.size() == 0, "drain empties the heap");
        expectThrows(() -> new TopK(0), "k=0");
    }

    static void bruteForceByHand() {
        float[][] points = {{0, 0}, {1, 0}, {0, 2}, {3, 3}, {-1, -1}};
        // Squared distances from (0.1, 0): .01, .81, 4.01, 17.41, 2.21
        BruteForceIndex index = new BruteForceIndex(points, SquaredL2.INSTANCE);
        float[] q = {0.1f, 0};
        check(Arrays.equals(new int[]{0, 1, 4, 2, 3}, index.search(q, 5)), "full ranking");
        check(Arrays.equals(new int[]{0, 1, 4}, index.search(q, 3)), "top 3");
        check(Arrays.equals(new int[]{0, 1, 4, 2, 3}, index.search(q, 50)), "k > n returns all n");
    }

    static void roundTrip() throws IOException {
        Path dir = Files.createTempDirectory("hnsw-test");
        Random rng = new Random(4);
        float[][] f = new float[37][13];
        for (float[] row : f) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (float) (rng.nextGaussian() * 1e3);
            }
        }
        f[0][0] = -0.0f; // sign of zero must survive too
        int[][] iv = new int[37][5];
        for (int[] row : iv) {
            for (int j = 0; j < row.length; j++) {
                row[j] = rng.nextInt();
            }
        }
        Path fp = dir.resolve("x.fvecs"), ip = dir.resolve("x.ivecs");
        VectorIO.writeFvecs(fp, f);
        VectorIO.writeIvecs(ip, iv);
        check(Files.size(fp) == 37L * (4 + 4 * 13), "fvecs size");
        float[][] f2 = VectorIO.readFvecs(fp, 0);
        for (int i = 0; i < f.length; i++) {
            for (int j = 0; j < f[i].length; j++) {
                check(Float.floatToRawIntBits(f[i][j]) == Float.floatToRawIntBits(f2[i][j]), "float bits " + i + "," + j);
            }
        }
        check(Arrays.deepEquals(iv, VectorIO.readIvecs(ip, 0)), "ivecs contents");
        check(VectorIO.readFvecs(fp, 10).length == 10, "limit reads a prefix");
        check(!Files.exists(dir.resolve("x.fvecs.tmp")), "temp file renamed away");
    }

    static void rejectsCorruptFiles() throws IOException {
        Path dir = Files.createTempDirectory("hnsw-test");
        Path p = dir.resolve("t.fvecs");
        VectorIO.writeFvecs(p, new float[10][8]);

        // A download that stopped 3 bytes short.
        try (FileChannel ch = FileChannel.open(p, StandardOpenOption.WRITE)) {
            ch.truncate(Files.size(p) - 3);
        }
        expectIOException(() -> VectorIO.readFvecs(p, 0), "truncated file");

        // Right size, but one record's dimension header is wrong.
        VectorIO.writeFvecs(p, new float[10][8]);
        byte[] bytes = Files.readAllBytes(p);
        bytes[5 * (4 + 32)] = 9; // header of vector 5: dim 8 -> 9
        Files.write(p, bytes);
        expectIOException(() -> VectorIO.readFvecs(p, 0), "bad record header");

        // A NaN component.
        float[][] withNaN = new float[3][4];
        withNaN[1][2] = Float.NaN;
        VectorIO.writeFvecs(p, withNaN);
        expectIOException(() -> VectorIO.readFvecs(p, 0), "NaN component");
    }

    static void parallelGroundTruth() {
        Random rng = new Random(5);
        float[][] base = new float[3000][16];
        for (float[] v : base) {
            for (int j = 0; j < v.length; j++) {
                v[j] = rng.nextInt(4); // small integer grid: lots of exact ties
            }
        }
        float[][] queries = new float[200][16];
        for (float[] v : queries) {
            for (int j = 0; j < v.length; j++) {
                v[j] = rng.nextInt(4);
            }
        }
        int[][] parallel = GroundTruth.compute(base, queries, SquaredL2.INSTANCE, 20);
        BruteForceIndex serial = new BruteForceIndex(base, SquaredL2.INSTANCE);
        for (int q = 0; q < queries.length; q++) {
            check(Arrays.equals(serial.search(queries[q], 20), parallel[q]), "query " + q);
        }
    }

    static void verifyClassifiesTies() {
        float[][] base = {{1, 0}, {0, 1}, {5, 5}};
        float[][] queries = {{0, 0}};
        // Ids 0 and 1 are both at distance 1 from the origin: swapping them is a tie.
        int[][] ours = {{0, 1}};
        GroundTruth.Verification tie = GroundTruth.verify(base, queries, SquaredL2.INSTANCE, ours, new int[][]{{1, 0}}, 2);
        check(tie.ok() && tie.tieSwaps() == 2 && tie.exactMatches() == 0, "swap of equidistant ids is not an error");
        GroundTruth.Verification err = GroundTruth.verify(base, queries, SquaredL2.INSTANCE, ours, new int[][]{{0, 2}}, 2);
        check(!err.ok() && err.errors() == 1 && err.exactMatches() == 1, "different distance is an error");
    }

    static void recallScoring() {
        int[][] truth = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        check(Benchmark.recallAtK(new int[][]{{1, 2, 3, 4}, {8, 7, 6, 5}}, truth, 4) == 1.0, "order-insensitive");
        check(Benchmark.recallAtK(new int[][]{{1, 2, 9, 9}, {0, 0, 0, 0}}, truth, 4) == 0.25, "partial");
        // Ground truth is stored 100 deep but recall@k looks only at its first k.
        check(Benchmark.recallAtK(new int[][]{{3, 4}}, new int[][]{{1, 2, 3, 4}}, 2) == 0.0, "only top-k of truth counts");
    }

    /**
     * The one test that doesn't trust our own code: someone else's ground truth.
     * 100 queries x 1M vectors keeps it to a few seconds; the full 10k-query check
     * is VerifyGroundTruth, run by bench.sh.
     */
    static void siftGroundTruth() throws IOException {
        Path dir = Path.of("data/sift1m");
        if (!Files.exists(dir.resolve("base.fvecs"))) {
            throw new Skip("data/sift1m not downloaded - run ./scripts/fetch_sift.sh");
        }
        Dataset data = Dataset.load(dir, SquaredL2.INSTANCE, 0);
        float[][] queries = Arrays.copyOf(data.queries, 100);
        int[][] reference = VectorIO.readIvecs(data.shippedGroundTruthPath(), 100);
        int[][] ours = GroundTruth.compute(data.base, queries, data.metric, 100);
        GroundTruth.Verification v = GroundTruth.verify(data.base, queries, data.metric, ours, reference, 100);
        check(v.ok(), v.errors() + " mismatches, first: " + v.firstError());
        System.out.printf("    (%,d identical, %,d tie swaps)%n", v.exactMatches(), v.tieSwaps());
    }

    // ---------------------------------------------------------------- harness

    interface Body {
        void run() throws Exception;
    }

    static final class Skip extends RuntimeException {
        private static final long serialVersionUID = 1L;

        Skip(String why) {
            super(why);
        }
    }

    static void run(String name, Body body) {
        try {
            body.run();
            passed++;
            System.out.println("  PASS: " + name);
        } catch (Skip s) {
            skipped++;
            System.out.println("  SKIP: " + name + " - " + s.getMessage());
        } catch (Throwable t) {
            failed++;
            System.out.println("  FAIL: " + name + "\n    " + t);
        }
    }

    static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    static void expectThrows(Runnable r, String what) {
        try {
            r.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("expected IllegalArgumentException: " + what);
    }

    interface IoBody {
        void run() throws IOException;
    }

    static void expectIOException(IoBody r, String what) {
        try {
            r.run();
        } catch (IOException expected) {
            return;
        }
        throw new AssertionError("expected IOException: " + what);
    }

    static float[] randomVector(Random rng, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) rng.nextGaussian();
        }
        return v;
    }

    static int[] shuffledRange(int n, Random rng) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
