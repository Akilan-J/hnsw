package hnsw.index;

/**
 * Keeps the k smallest (distance, id) pairs seen so far, as a bounded max-heap.
 *
 * <p>Why a max-heap for the <i>smallest</i> k: the question asked on every
 * candidate is "is this better than the worst one I'm keeping?", and a max-heap
 * keeps that worst one at the root, answered in O(1). Replacing it costs
 * O(log k). A full scan of n vectors is O(n log k) time and O(k) memory.
 * Alternatives: sort all n distances (O(n log n) time, O(n) memory per query), or
 * quickselect (O(n) average, still O(n) memory). With k = 10 and n = 1M, log k is
 * tiny, and in practice almost no candidates make it past the O(1) root check.
 * HNSW's ef-bounded result set in stage 3 is this same structure.
 *
 * <p><b>Ties are broken by id</b>, so the order is total: (d1, id1) comes before
 * (d2, id2) iff d1 &lt; d2, or d1 == d2 and id1 &lt; id2. This matters because
 * this class produces ground truth. Without a tie rule, which of two
 * equidistant vectors made the cut would depend on scan order, and a
 * multi-threaded ground-truth run could disagree with a single-threaded one.
 *
 * <p>Parallel primitive arrays rather than a {@code PriorityQueue<Neighbor>}: no
 * boxing and no object per entry. That doesn't matter much here, but it matters
 * in the graph search, where this is on the hot path.
 */
public final class TopK {

    private final int capacity;
    private final int[] ids;
    private final float[] dists;
    private int size;

    public TopK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        this.capacity = k;
        this.ids = new int[k];
        this.dists = new float[k];
    }

    /** Offers a candidate; returns true if it was kept. */
    public boolean offer(int id, float dist) {
        if (size < capacity) {
            ids[size] = id;
            dists[size] = dist;
            siftUp(size++);
            return true;
        }
        if (!before(dist, id, dists[0], ids[0])) {
            return false; // no better than the worst we're keeping - the common case
        }
        ids[0] = id;
        dists[0] = dist;
        siftDown(0);
        return true;
    }

    public int size() {
        return size;
    }

    public boolean isFull() {
        return size == capacity;
    }

    /** Distance of the worst entry kept. Only meaningful when size() > 0. */
    public float worstDistance() {
        return dists[0];
    }

    /**
     * Empties the heap and returns its ids, nearest first. Repeatedly popping the
     * max and filling the output from the back is heapsort's second phase.
     */
    public int[] drainAscending() {
        int[] out = new int[size];
        while (size > 0) {
            out[size - 1] = ids[0];
            size--;
            ids[0] = ids[size];
            dists[0] = dists[size];
            siftDown(0);
        }
        return out;
    }

    /** The total order: true if (d1, id1) sorts strictly before (d2, id2). */
    static boolean before(float d1, int id1, float d2, int id2) {
        return d1 < d2 || (d1 == d2 && id1 < id2);
    }

    private void siftUp(int i) {
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            // Max-heap: a child that sorts after its parent must move up.
            if (!before(dists[parent], ids[parent], dists[i], ids[i])) {
                return;
            }
            swap(i, parent);
            i = parent;
        }
    }

    private void siftDown(int i) {
        while (true) {
            int left = 2 * i + 1;
            if (left >= size) {
                return;
            }
            int right = left + 1;
            int largest = left;
            if (right < size && before(dists[left], ids[left], dists[right], ids[right])) {
                largest = right;
            }
            if (!before(dists[i], ids[i], dists[largest], ids[largest])) {
                return;
            }
            swap(i, largest);
            i = largest;
        }
    }

    private void swap(int a, int b) {
        int id = ids[a];
        ids[a] = ids[b];
        ids[b] = id;
        float d = dists[a];
        dists[a] = dists[b];
        dists[b] = d;
    }
}
