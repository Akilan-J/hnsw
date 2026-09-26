package hnsw.distance;

/**
 * Squared Euclidean distance. The square root is monotonic, so skipping it never
 * changes which vector is nearer - and this is the hottest function in the project,
 * called once per base vector per brute-force query.
 */
public final class SquaredL2 implements DistanceFunction {

    public static final SquaredL2 INSTANCE = new SquaredL2();

    private SquaredL2() {
    }

    @Override
    public float distance(float[] a, float[] b) {
        int n = a.length;
        // Without this, a longer b would silently be compared on a's prefix only.
        if (b.length != n) {
            throw new IllegalArgumentException("dimension mismatch: " + n + " vs " + b.length);
        }
        // Four independent running sums instead of one. With a single sum every add
        // has to wait for the previous add to finish, so the loop runs at the
        // *latency* of a float add rather than its throughput. The JIT won't split
        // the sum for us: float addition isn't associative, so reordering it changes
        // the last bits of the result, and Java forbids that unless the code asks.
        // This code asks. The price is that the result can differ in the last bit
        // from a naive left-to-right sum - harmless, because ground truth and every
        // index here use this same function.
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i <= n - 4; i += 4) {
            float d0 = a[i] - b[i];
            float d1 = a[i + 1] - b[i + 1];
            float d2 = a[i + 2] - b[i + 2];
            float d3 = a[i + 3] - b[i + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        for (; i < n; i++) { // dimensions that aren't a multiple of 4
            float d = a[i] - b[i];
            s0 += d * d;
        }
        return (s0 + s1) + (s2 + s3);
    }

    @Override
    public String name() {
        return "l2";
    }
}
