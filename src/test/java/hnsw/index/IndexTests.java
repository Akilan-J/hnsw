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

    // ------------------------------------------------------------------ helpers

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
