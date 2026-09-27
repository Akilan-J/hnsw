package hnsw.index;

/**
 * Anything that answers "which k stored vectors are nearest to this query". The
 * benchmark harness only knows this interface, so brute force, the flat graph and
 * HNSW are measured by exactly the same code.
 */
public interface KnnIndex {

    /**
     * Ids of the (approximately, for graph indexes) k nearest stored vectors,
     * nearest first. Returns fewer than k only if fewer than k vectors exist.
     */
    int[] search(float[] query, int k);

    /**
     * Cumulative number of distance evaluations, or -1 if the index doesn't
     * count them. Brute force doesn't bother: it is always exactly n per query.
     */
    default long distanceComputations() {
        return -1;
    }

    /** Human-readable name and parameters, for benchmark output. */
    String describe();
}
