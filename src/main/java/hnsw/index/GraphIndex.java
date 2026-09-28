package hnsw.index;

/**
 * What the benchmark needs from a graph index beyond searching: build it one
 * insert at a time (so progress can be reported on long builds), turn the
 * search-time beam width, and describe the graph that came out.
 */
public interface GraphIndex extends KnnIndex {

    /** Inserts the next vector in array order; false once all are in. */
    boolean insertNext();

    void setEfSearch(int ef);

    /** Multi-line summary of the built graph: degrees, memory, reachability. */
    String graphStats();

    /** Estimated bytes held by the graph structure, excluding the shared vectors. */
    long graphBytes();
}
