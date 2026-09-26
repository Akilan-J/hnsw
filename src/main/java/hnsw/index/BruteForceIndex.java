package hnsw.index;

import hnsw.distance.DistanceFunction;

/**
 * Exact k-NN: compute the distance to every stored vector, keep the best k.
 *
 * <p>This is the ground truth every recall number is scored against, so it is
 * deliberately the simplest possible correct thing. No early exit, no
 * cleverness: the only way an approximate index can beat it is by skipping
 * vectors, and the only way to know what skipping cost is to have this.
 *
 * <p>Search is read-only and allocates its own TopK, so it is safe to call from
 * many threads at once - which is how ground truth gets computed in parallel.
 */
public final class BruteForceIndex implements KnnIndex {

    private final float[][] vectors;
    private final DistanceFunction metric;

    public BruteForceIndex(float[][] vectors, DistanceFunction metric) {
        this.vectors = vectors;
        this.metric = metric;
    }

    @Override
    public int[] search(float[] query, int k) {
        TopK top = new TopK(Math.min(k, vectors.length));
        for (int i = 0; i < vectors.length; i++) {
            top.offer(i, metric.distance(query, vectors[i]));
        }
        return top.drainAscending();
    }

    @Override
    public String describe() {
        return "brute-force(n=" + vectors.length + ", " + metric.name() + ")";
    }
}
