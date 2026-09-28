package hnsw.index;

/**
 * Search results with their distances, nearest first. The distances are kept
 * because HNSW's neighbour selection compares them rather than recomputing.
 */
public record Neighbors(int[] ids, float[] distances) {

    public int size() {
        return ids.length;
    }
}
