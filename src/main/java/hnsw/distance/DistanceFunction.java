package hnsw.distance;

/**
 * A dissimilarity between two vectors: smaller means closer.
 *
 * <p>Nothing in this project ever adds distances together or reads their absolute
 * value - brute force ranks them, and the graph indexes only ask "is this one
 * smaller than that one". So an implementation may return any value that is
 * monotonic in the true distance. {@link SquaredL2} relies on exactly that to skip
 * the square root.
 */
public interface DistanceFunction {

    float distance(float[] a, float[] b);

    /** Short name used on the command line and in cache file names. */
    String name();

    /**
     * True if this function is only correct on unit-length vectors, in which case
     * {@code Dataset} normalizes everything once at load time.
     */
    default boolean requiresUnitVectors() {
        return false;
    }

    static DistanceFunction byName(String name) {
        return switch (name) {
            case "l2" -> SquaredL2.INSTANCE;
            case "cosine" -> CosineDistance.INSTANCE;
            default -> throw new IllegalArgumentException("unknown metric '" + name + "' (expected l2 or cosine)");
        };
    }
}
