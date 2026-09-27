package hnsw.index;

import java.util.Arrays;

/**
 * "Have I already computed the distance to this node during this search?"
 *
 * <p>The obvious version is a fresh {@code boolean[n]} per search, but zeroing
 * 1M entries on every query is O(n) work - the very cost the graph exists to
 * avoid. A {@code HashSet<Integer>} is O(visited) but boxes every id. Instead
 * each slot stores the number of the search that last marked it, and starting
 * a new search just increments that number, so every old mark is instantly
 * stale. Clearing is O(1) and a lookup is one array read.
 *
 * <p>Cost: 4 bytes per node, held for the life of the index, and one instance
 * per searching thread - this makes the index single-threaded for search.
 */
final class VisitedSet {

    private final int[] marks;
    private int epoch;

    VisitedSet(int capacity) {
        marks = new int[capacity];
        epoch = 1; // marks start at 0, so nothing counts as visited yet
    }

    /** Forget everything. O(1) except once every 2^31 searches. */
    void clear() {
        epoch++;
        if (epoch == Integer.MAX_VALUE) {
            // The counter is about to wrap, and a wrapped epoch could collide with
            // a stale mark. Pay for one real clear instead.
            Arrays.fill(marks, 0);
            epoch = 1;
        }
    }

    /** Marks id as visited; returns false if it already was. */
    boolean add(int id) {
        if (marks[id] == epoch) {
            return false;
        }
        marks[id] = epoch;
        return true;
    }
}
