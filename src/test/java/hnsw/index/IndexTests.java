package hnsw.index;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import hnsw.distance.SquaredL2;

/**
 * Tests for the graph index and its package-private parts. Lives in hnsw.index
 * so it can reach CandidateQueue and VisitedSet without widening their
 * visibility just for testing. Registered in hnsw.Tests, which owns the harness.
 */
public final class IndexTests {

    private IndexTests() {
    }

    public static void candidateQueueOrder() {
        Random rng = new Random(11);
        CandidateQueue q = new CandidateQueue(1); // forces several doublings
        for (int trial = 0; trial < 50; trial++) {
            q.clear();
            int n = 1 + rng.nextInt(300);
            float[][] pairs = new float[n][];
            for (int i = 0; i < n; i++) {
                float d = rng.nextInt(10); // heavy ties
                pairs[i] = new float[]{d, i};
                q.push(i, d);
            }
            Arrays.sort(pairs, (a, b) -> a[0] != b[0] ? Float.compare(a[0], b[0]) : Float.compare(a[1], b[1]));
            for (float[] p : pairs) {
                check(!q.isEmpty(), "queue ran dry early");
                check(q.peekId() == (int) p[1] && q.peekDist() == p[0],
                        "expected id " + (int) p[1] + " got " + q.peekId());
                q.pop();
            }
            check(q.isEmpty(), "queue should be empty");
        }
    }

    public static void visitedSet() {
        VisitedSet v = new VisitedSet(5);
        check(v.add(3), "fresh set: 3 is new");
        check(!v.add(3), "3 is now visited");
        check(v.add(4), "4 is new");
        v.clear();
        check(v.add(3) && v.add(4), "clear forgets everything");
        check(!v.add(3), "and remembers again after");
    }

    public static void graphInvariants() {
        for (int cap : new int[]{0, 8, 12}) {
            FlatNswIndex index = build(randomVectors(2000, 16, 12), 6, 30, cap);
            for (int u = 0; u < index.size(); u++) {
                Set<Integer> seen = new HashSet<>();
                check(cap == 0 || index.degree(u) <= cap, "node " + u + " exceeds cap " + cap);
                for (int j = 0; j < index.degree(u); j++) {
                    int v = index.neighbor(u, j);
                    check(v != u, "self-loop at " + u);
                    check(v >= 0 && v < index.size(), "link to nonexistent node " + v);
                    check(seen.add(v), "duplicate link " + u + " -> " + v);
                }
            }
        }
    }

    public static void unboundedIsSymmetric() {
        FlatNswIndex index = build(randomVectors(2000, 16, 13), 6, 30, 0);
        for (int u = 0; u < index.size(); u++) {
            for (int j = 0; j < index.degree(u); j++) {
                int v = index.neighbor(u, j);
                boolean back = false;
                for (int i = 0; i < index.degree(v) && !back; i++) {
                    back = index.neighbor(v, i) == u;
                }
                check(back, "link " + u + " -> " + v + " has no reverse");
            }
        }
        // Every node links to earlier ones and gets a link back, so by induction
        // everything is reachable from node 0. Pruning is what breaks this.
        check(Arrays.stream(index.hopsFrom(0)).allMatch(h -> h >= 0), "unreachable node in unbounded graph");
    }

    public static void fullBeamIsExact() {
        float[][] data = randomVectors(1500, 8, 14);
        FlatNswIndex index = build(data, 5, 20, 0);
        index.setEfSearch(data.length);
        BruteForceIndex exact = new BruteForceIndex(data, SquaredL2.INSTANCE);
        float[][] queries = randomVectors(50, 8, 15);
        for (float[] q : queries) {
            check(Arrays.equals(exact.search(q, 10), index.search(q, 10)), "ef=n should match brute force");
        }
    }

    public static void greedyStopsAtLocalMinimum() {
        float[][] data = randomVectors(3000, 16, 16);
        FlatNswIndex index = build(data, 6, 30, 12);
        index.setEfSearch(1);
        for (float[] q : randomVectors(200, 16, 17)) {
            int got = index.search(q, 1)[0];
            float d = SquaredL2.INSTANCE.distance(q, data[got]);
            for (int j = 0; j < index.degree(got); j++) {
                check(SquaredL2.INSTANCE.distance(q, data[index.neighbor(got, j)]) >= d,
                        "greedy stopped at " + got + " but neighbour " + index.neighbor(got, j) + " is closer");
            }
        }
    }

    public static void deterministic() {
        float[][] data = randomVectors(1500, 16, 18);
        FlatNswIndex a = build(data, 6, 30, 12);
        FlatNswIndex b = build(data, 6, 30, 12);
        for (int u = 0; u < data.length; u++) {
            check(a.degree(u) == b.degree(u), "degree differs at " + u);
            for (int j = 0; j < a.degree(u); j++) {
                check(a.neighbor(u, j) == b.neighbor(u, j), "links differ at " + u);
            }
        }
        // Same index, same query, twice: the visited set must reset between searches.
        float[] q = randomVectors(1, 16, 19)[0];
        a.setEfSearch(20);
        check(Arrays.equals(a.search(q, 10), a.search(q, 10)), "repeat search differs");
    }

    // ------------------------------------------------------------------ stage 3

    /**
     * The worked example for the heuristic. Base q at the origin; candidates,
     * nearest first, with squared distances to q:
     * <pre>
     *   0: a = ( 1.0, 0.0)   1.00
     *   1: b = ( 1.1, 0.1)   1.22   right next to a, same direction
     *   2: c = ( 0.0, 1.5)   2.25   a different direction
     *   3: d = (-2.0, 0.0)   4.00   the opposite direction, farthest of all
     * </pre>
     * Simple keeps the 3 nearest: a, b, c - two links pointing the same way.
     * The heuristic keeps a; drops b, because b is 0.02 from a but 1.22 from q
     * (reach b via a); keeps c (3.25 from a, farther than its 2.25 from q); and
     * keeps d even though it's the farthest candidate, because neither a (9.0
     * away) nor c (6.25) is closer to d than q is (4.0). Result: a, c, d - three
     * directions instead of two.
     */
    public static void heuristicWorkedExample() {
        float[][] v = {{1f, 0f}, {1.1f, 0.1f}, {0f, 1.5f}, {-2f, 0f}};
        int[] ids = {0, 1, 2, 3};
        float[] dists = {1f, 1.22f, 2.25f, 4f};
        HnswIndex heuristic = new HnswIndex(v, SquaredL2.INSTANCE, 2, 2, HnswIndex.Selection.HEURISTIC, 0, 1);
        HnswIndex simple = new HnswIndex(v, SquaredL2.INSTANCE, 2, 2, HnswIndex.Selection.SIMPLE, 0, 1);
        check(Arrays.equals(new int[]{0, 2, 3}, heuristic.selectNeighbors(ids, dists, 3)),
                "heuristic: " + Arrays.toString(heuristic.selectNeighbors(ids, dists, 3)));
        check(Arrays.equals(new int[]{0, 1, 2}, simple.selectNeighbors(ids, dists, 3)), "simple keeps nearest 3");
        // The heuristic may return fewer than asked: here only 3 directions exist.
        check(heuristic.selectNeighbors(ids, dists, 4).length == 3, "b stays redundant even with room for 4");
    }

    public static void hnswGraphInvariants() {
        for (HnswIndex.Selection sel : HnswIndex.Selection.values()) {
            HnswIndex index = buildHnsw(randomVectors(3000, 16, 21), 6, 30, sel, HnswIndex.defaultLevelMultiplier(6));
            check(index.level(index.entryPoint()) == index.topLevel(), "entry point must be on the top layer");
            for (int u = 0; u < index.size(); u++) {
                for (int layer = 0; layer <= index.level(u); layer++) {
                    int cap = layer == 0 ? 12 : 6;
                    check(index.degree(u, layer) <= cap, sel + ": node " + u + " layer " + layer + " over cap");
                    Set<Integer> seen = new HashSet<>();
                    for (int j = 0; j < index.degree(u, layer); j++) {
                        int v = index.neighbor(u, layer, j);
                        check(v != u, "self-loop");
                        check(seen.add(v), "duplicate link");
                        // A link on layer l must point to a node that exists on layer l.
                        check(index.level(v) >= layer, "link to node " + v + " which isn't on layer " + layer);
                    }
                }
            }
        }
    }

    /** With mL = 1/ln(M), P(level >= 1) = 1/M. 20k draws: 2,500 expected, sd ~47. */
    public static void levelDistribution() {
        int m = 8;
        HnswIndex index = buildHnsw(randomVectors(20_000, 4, 22), m, 16, HnswIndex.Selection.HEURISTIC,
                HnswIndex.defaultLevelMultiplier(m));
        int[] atLeast = new int[8];
        for (int i = 0; i < index.size(); i++) {
            for (int l = 0; l <= Math.min(index.level(i), 7); l++) {
                atLeast[l]++;
            }
        }
        check(Math.abs(atLeast[1] - 2500) < 250, "level>=1 count " + atLeast[1] + ", expected ~2500");
        check(Math.abs(atLeast[2] - 312) < 90, "level>=2 count " + atLeast[2] + ", expected ~312");
    }

    public static void zeroLevelMultiplierIsFlat() {
        HnswIndex index = buildHnsw(randomVectors(2000, 8, 23), 6, 30, HnswIndex.Selection.HEURISTIC, 0);
        check(index.topLevel() == 0, "mL = 0 must give a single layer");
        for (int i = 0; i < index.size(); i++) {
            check(index.level(i) == 0, "node " + i + " above layer 0");
        }
    }

    public static void hnswExactAtFullEf() {
        float[][] data = randomVectors(1500, 8, 24);
        HnswIndex index = buildHnsw(data, 6, 30, HnswIndex.Selection.HEURISTIC, HnswIndex.defaultLevelMultiplier(6));
        check(Arrays.stream(index.layer0HopsFromEntry()).allMatch(h -> h >= 0), "layer 0 not fully reachable");
        index.setEfSearch(data.length);
        BruteForceIndex exact = new BruteForceIndex(data, SquaredL2.INSTANCE);
        for (float[] q : randomVectors(50, 8, 25)) {
            check(Arrays.equals(exact.search(q, 10), index.search(q, 10)), "ef=n should match brute force");
        }
    }

    /** Smoke test on clustered data: a sane configuration must reach high recall. */
    public static void hnswRecallSmoke() {
        hnsw.data.SyntheticClusters model = new hnsw.data.SyntheticClusters(32, 8, 20, 0.35, 0.02, 26);
        float[][] data = model.sample(5000, 27);
        float[][] queries = model.sample(200, 28);
        HnswIndex index = buildHnsw(data, 8, 64, HnswIndex.Selection.HEURISTIC, HnswIndex.defaultLevelMultiplier(8));
        index.setEfSearch(64);
        BruteForceIndex exact = new BruteForceIndex(data, SquaredL2.INSTANCE);
        int hits = 0;
        for (float[] q : queries) {
            int[] truth = exact.search(q, 10);
            for (int id : index.search(q, 10)) {
                for (int t : truth) {
                    if (t == id) {
                        hits++;
                    }
                }
            }
        }
        double recall = hits / (10.0 * queries.length);
        check(recall >= 0.95, "recall@10 = " + recall);
    }

    /**
     * With both switches off - one layer, keep-the-nearest selection - HNSW is
     * stage 2's capped flat graph. Checking that they build the identical graph
     * means every ablation comparison changes exactly one thing.
     */
    public static void hnswWithSwitchesOffIsStage2() {
        float[][] data = randomVectors(2000, 16, 30);
        HnswIndex h = buildHnsw(data, 6, 30, HnswIndex.Selection.SIMPLE, 0);
        FlatNswIndex f = build(data, 6, 30, 12);
        for (int u = 0; u < data.length; u++) {
            check(h.degree(u, 0) == f.degree(u), "degree differs at " + u);
            for (int j = 0; j < f.degree(u); j++) {
                check(h.neighbor(u, 0, j) == f.neighbor(u, j), "link differs at node " + u);
            }
        }
    }

    public static void hnswDeterministic() {
        float[][] data = randomVectors(2000, 16, 29);
        double mL = HnswIndex.defaultLevelMultiplier(6);
        HnswIndex a = buildHnsw(data, 6, 30, HnswIndex.Selection.HEURISTIC, mL);
        HnswIndex b = buildHnsw(data, 6, 30, HnswIndex.Selection.HEURISTIC, mL);
        check(a.entryPoint() == b.entryPoint() && a.topLevel() == b.topLevel(), "entry point differs");
        for (int u = 0; u < data.length; u++) {
            check(a.level(u) == b.level(u), "level differs at " + u);
            for (int layer = 0; layer <= a.level(u); layer++) {
                check(a.degree(u, layer) == b.degree(u, layer), "degree differs");
                for (int j = 0; j < a.degree(u, layer); j++) {
                    check(a.neighbor(u, layer, j) == b.neighbor(u, layer, j), "links differ");
                }
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    static HnswIndex buildHnsw(float[][] data, int m, int efc, HnswIndex.Selection sel, double mL) {
        HnswIndex index = new HnswIndex(data, SquaredL2.INSTANCE, m, efc, sel, mL, 42);
        while (index.insertNext()) {
            // each call inserts one vector
        }
        return index;
    }

    static FlatNswIndex build(float[][] data, int m, int efc, int cap) {
        FlatNswIndex index = new FlatNswIndex(data, SquaredL2.INSTANCE, m, efc, cap);
        index.insertAll();
        return index;
    }

    static float[][] randomVectors(int n, int dim, long seed) {
        Random rng = new Random(seed);
        float[][] out = new float[n][dim];
        for (float[] v : out) {
            for (int i = 0; i < dim; i++) {
                v[i] = (float) rng.nextGaussian();
            }
        }
        return out;
    }

    static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
