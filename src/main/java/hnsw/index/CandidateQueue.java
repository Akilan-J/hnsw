package hnsw.index;

import java.util.Arrays;

/**
 * Unbounded min-heap of (distance, id): the frontier of a graph search, always
 * handing back the closest node not yet expanded.
 *
 * <p>It's the mirror image of {@link TopK}. TopK is a max-heap because it asks
 * "what's the worst result I'm keeping?"; this is a min-heap because the search
 * asks "what's the best node I haven't explored yet?". The two together are the
 * whole search loop.
 *
 * <p>Unbounded because anything that improved the result set is worth
 * exploring, and the result set's bound already limits how much gets pushed.
 * The index reuses one instance across searches ({@link #clear()} is O(1)), so
 * after warm-up it never allocates.
 */
final class CandidateQueue {

    private int[] ids;
    private float[] dists;
    private int size;

    CandidateQueue(int initialCapacity) {
        ids = new int[Math.max(1, initialCapacity)];
        dists = new float[ids.length];
    }

    void push(int id, float dist) {
        if (size == ids.length) {
            ids = Arrays.copyOf(ids, size * 2);
            dists = Arrays.copyOf(dists, size * 2);
        }
        int i = size++;
        // Sift up: a min-heap moves a child above any parent it sorts before.
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (!TopK.before(dist, id, dists[parent], ids[parent])) {
                break;
            }
            ids[i] = ids[parent];
            dists[i] = dists[parent];
            i = parent;
        }
        ids[i] = id;
        dists[i] = dist;
    }

    boolean isEmpty() {
        return size == 0;
    }

    int peekId() {
        return ids[0];
    }

    float peekDist() {
        return dists[0];
    }

    void pop() {
        size--;
        if (size == 0) {
            return;
        }
        int id = ids[size];
        float dist = dists[size];
        int i = 0;
        while (true) {
            int child = 2 * i + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size && TopK.before(dists[child + 1], ids[child + 1], dists[child], ids[child])) {
                child++;
            }
            if (!TopK.before(dists[child], ids[child], dist, id)) {
                break;
            }
            ids[i] = ids[child];
            dists[i] = dists[child];
            i = child;
        }
        ids[i] = id;
        dists[i] = dist;
    }

    void clear() {
        size = 0;
    }
}
