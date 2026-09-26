package hnsw.distance;

/**
 * Cosine distance, 1 - cos(a, b), in the range [0, 2].
 *
 * <p>Only correct on unit vectors: for those cos(a, b) is just the dot product.
 * The alternative - computing |a| and |b| inside every call - triples the
 * arithmetic in the hottest loop, for norms that never change. So the norms are
 * paid once per vector at load time instead (see {@link #normalizeInPlace}).
 */
public final class CosineDistance implements DistanceFunction {

    public static final CosineDistance INSTANCE = new CosineDistance();

    private CosineDistance() {
    }

    @Override
    public float distance(float[] a, float[] b) {
        int n = a.length;
        if (b.length != n) {
            throw new IllegalArgumentException("dimension mismatch: " + n + " vs " + b.length);
        }
        // Same four-accumulator trick as SquaredL2, for the same reason.
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i <= n - 4; i += 4) {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        for (; i < n; i++) {
            s0 += a[i] * b[i];
        }
        return 1f - ((s0 + s1) + (s2 + s3));
    }

    @Override
    public String name() {
        return "cosine";
    }

    @Override
    public boolean requiresUnitVectors() {
        return true;
    }

    /**
     * Scales v to unit length. The norm is accumulated in double because it runs
     * once per vector, so precision is free here.
     */
    public static void normalizeInPlace(float[] v) {
        double sumSquares = 0;
        for (float x : v) {
            sumSquares += (double) x * x;
        }
        if (sumSquares == 0) {
            // A zero vector has no direction, so cosine similarity is undefined.
            // Refusing is better than quietly giving it distance 1 to everything.
            throw new IllegalArgumentException("cannot normalize a zero vector for cosine distance");
        }
        double inv = 1.0 / Math.sqrt(sumSquares);
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (v[i] * inv);
        }
    }
}
