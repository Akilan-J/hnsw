package hnsw.index;

import java.util.Arrays;

import hnsw.distance.DistanceFunction;

/**
 * Navigable small world graph, one layer, no hierarchy (Malkov et al., 2014).
 *
 * <p><b>Idea.</b> Store each vector as a node linked to some of its near
 * neighbours. To search, start somewhere and keep moving to whichever neighbour
 * is closer to the query. If the graph is "navigable", those steps lead to the
 * query's neighbourhood after visiting a tiny fraction of the nodes.
 *
 * <p><b>Construction</b> is incremental: insert vectors one at a time, use the
 * search itself to find the new vector's M nearest already-inserted nodes, and
 * link both ways. Where do the long jumps a small world needs come from? Nobody
 * adds them on purpose. The first few nodes are inserted into an almost empty
 * graph, so their "nearest" neighbours are far away. Those early links become
 * the highways. Later nodes land in a crowded graph and get short, local links.
 * So navigability depends on insertion order being random with respect to
 * where the vectors are - insert one cluster at a time and the early highways
 * all stay inside the first cluster.
 *
 * <p><b>Degree cap.</b> With {@code maxDegree = 0} lists grow without bound, as
 * in the original NSW. With a cap, a full node that gains a new link keeps only
 * its maxDegree <i>nearest</i> neighbours. That bounds memory, but note which
 * links it drops: the longest ones, i.e. exactly the early highways. Stage 3's
 * neighbour-selection heuristic exists to fix this.
 *
 * <p>Not thread-safe: search reuses one visited set and one candidate queue.
 */
public final class FlatNswIndex implements GraphIndex {

    private final float[][] vectors;
    private final DistanceFunction metric;
    private final int m;
    private final int efConstruction;
    private final int maxDegree;

    // Adjacency lists. neighbors[i][0 .. degree[i]) are node i's out-links. Links
    // are directed: with a cap, i -> j can survive while j -> i was pruned.
    private final int[][] neighbors;
    private final int[] degree;
    private int size;

    private final VisitedSet visited;
    private final CandidateQueue candidates;
    private int efSearch = 10;

    // Cumulative counters, read by the benchmark before and after a run. Distance
    // evaluations are the machine-independent cost of a search: QPS depends on
    // the CPU, but "computed 1.3% of the distances brute force does" doesn't.
    private long distanceCount;
    private long expansionCount;

    /** Entry point for every search: the first node inserted. */
    private static final int ENTRY = 0;

    /**
     * @param m              links created from each new node
     * @param efConstruction beam width when searching for a new node's neighbours
     * @param maxDegree      cap on any node's out-degree, or 0 for unbounded
     */
    public FlatNswIndex(float[][] vectors, DistanceFunction metric, int m, int efConstruction, int maxDegree) {
        if (m < 1 || efConstruction < m || (maxDegree != 0 && maxDegree < m)) {
            throw new IllegalArgumentException("need m >= 1, efConstruction >= m, maxDegree = 0 or >= m; got m="
                    + m + " efConstruction=" + efConstruction + " maxDegree=" + maxDegree);
        }
        this.vectors = vectors;
        this.metric = metric;
        this.m = m;
        this.efConstruction = efConstruction;
        this.maxDegree = maxDegree;
        this.neighbors = new int[vectors.length][];
        this.degree = new int[vectors.length];
        this.visited = new VisitedSet(vectors.length);
        this.candidates = new CandidateQueue(256);
    }

    @Override
    public boolean insertNext() {
        if (size == vectors.length) {
            return false;
        }
        int id = size;
        neighbors[id] = new int[maxDegree > 0 ? maxDegree : 2 * m];
        if (id > 0) {
            // Search the graph as it stands - nodes 0..id-1 - before adding id to it.
            int[] nearest = beamSearch(vectors[id], efConstruction);
            int links = Math.min(m, nearest.length);
            for (int j = 0; j < links; j++) {
                addLink(id, nearest[j]);
                addLink(nearest[j], id);
            }
        }
        size++;
        return true;
    }

    public void insertAll() {
        for (int i = size; i < vectors.length; i++) {
            insertNext();
        }
    }

    @Override
    public void setEfSearch(int ef) {
        if (ef < 1) {
            throw new IllegalArgumentException("ef must be >= 1");
        }
        this.efSearch = ef;
    }

    @Override
    public int[] search(float[] query, int k) {
        // The beam can't be narrower than the number of results wanted.
        int[] found = beamSearch(query, Math.max(efSearch, k));
        return found.length <= k ? found : Arrays.copyOf(found, k);
    }

    /**
     * Best-first search with a result set bounded at ef.
     *
     * <p>Keep two heaps: {@code candidates}, nodes seen but not yet expanded
     * (closest first), and {@code results}, the best ef nodes seen so far (worst
     * on top). Repeatedly expand the closest candidate: compute the distance to
     * each unvisited neighbour, and keep any that beat the current worst result,
     * both as a result and as a candidate worth expanding later.
     *
     * <p>With ef = 1 this is plain greedy hill-climbing: always move to the best
     * neighbour, stop when no neighbour is closer. Stopping there is exactly a
     * <i>local minimum</i> - a node closer to the query than all its neighbours,
     * which need not be the global nearest. A larger ef keeps ef nodes alive at
     * once, so the search can continue through a neighbour that is slightly
     * worse than the best so far. That is how it gets out of shallow local minima.
     */
    private int[] beamSearch(float[] query, int ef) {
        visited.clear();
        candidates.clear();
        TopK results = new TopK(ef);

        float d0 = distance(query, ENTRY);
        visited.add(ENTRY);
        candidates.push(ENTRY, d0);
        results.offer(ENTRY, d0);

        while (!candidates.isEmpty()) {
            int current = candidates.peekId();
            float currentDist = candidates.peekDist();
            candidates.pop();
            // The stopping rule, and the only place an approximation enters. The
            // closest unexpanded node is already farther than our worst result, so
            // we bet that expanding it - or anything farther - won't find something
            // better. It's a bet, not a proof: a far node can have a near
            // neighbour. That bet is what makes the search sublinear, and ef
            // controls how conservative it is.
            if (results.isFull() && currentDist > results.worstDistance()) {
                break;
            }
            expansionCount++;
            int[] links = neighbors[current];
            for (int j = 0, deg = degree[current]; j < deg; j++) {
                int next = links[j];
                if (!visited.add(next)) {
                    continue;
                }
                float d = distance(query, next);
                if (results.offer(next, d)) {
                    candidates.push(next, d);
                }
            }
        }
        return results.drainAscending();
    }

    /** Adds a directed link from -> to, pruning from's list if it's at the cap. */
    private void addLink(int from, int to) {
        int[] links = neighbors[from];
        int deg = degree[from];
        if (deg < links.length) {
            links[deg] = to;
            degree[from] = deg + 1;
            return;
        }
        if (maxDegree == 0) {
            // Unbounded: grow by doubling, so appends are amortised O(1).
            links = Arrays.copyOf(links, links.length * 2);
            links[deg] = to;
            neighbors[from] = links;
            degree[from] = deg + 1;
            return;
        }
        // At the cap: keep the maxDegree nearest of the old links plus the new one.
        // The new link itself may be the one that loses.
        TopK keep = new TopK(maxDegree);
        float[] base = vectors[from];
        for (int j = 0; j < deg; j++) {
            keep.offer(links[j], distance(base, links[j]));
        }
        keep.offer(to, distance(base, to));
        int[] kept = keep.drainAscending();
        System.arraycopy(kept, 0, links, 0, kept.length);
        degree[from] = kept.length;
    }

    /** Every distance the index computes goes through here, so the counter is exact. */
    private float distance(float[] query, int node) {
        distanceCount++;
        return metric.distance(query, vectors[node]);
    }

    // ------------------------------------------------------------ inspection

    public int size() {
        return size;
    }

    public int entryPoint() {
        return ENTRY;
    }

    public int degree(int node) {
        return degree[node];
    }

    public int neighbor(int node, int i) {
        return neighbors[node][i];
    }

    @Override
    public long distanceComputations() {
        return distanceCount;
    }

    public long expansions() {
        return expansionCount;
    }

    /**
     * Hop distance from source to every node by BFS over out-links; -1 where
     * unreachable. A node the entry point can't reach can never be returned by any
     * search, whatever ef is - a different failure from a local minimum.
     */
    public int[] hopsFrom(int source) {
        int[] hops = new int[size];
        Arrays.fill(hops, -1);
        int[] queue = new int[size];
        int head = 0, tail = 0;
        hops[source] = 0;
        queue[tail++] = source;
        while (head < tail) {
            int u = queue[head++];
            for (int j = 0; j < degree[u]; j++) {
                int v = neighbors[u][j];
                if (hops[v] < 0) {
                    hops[v] = hops[u] + 1;
                    queue[tail++] = v;
                }
            }
        }
        return hops;
    }

    /**
     * Bytes held by the graph structure alone (the vectors are shared with
     * brute force, so they aren't counted): each adjacency array's capacity plus
     * its 16-byte header, the degree array, and the visited-set marks.
     */
    @Override
    public long graphBytes() {
        long bytes = 0;
        for (int i = 0; i < size; i++) {
            bytes += 16 + 4L * neighbors[i].length;
        }
        return bytes + 4L * degree.length + 4L * vectors.length;
    }

    @Override
    public String graphStats() {
        long edges = 0;
        int maxDeg = 0;
        for (int i = 0; i < size; i++) {
            edges += degree[i];
            maxDeg = Math.max(maxDeg, degree[i]);
        }
        long unreachable = Arrays.stream(hopsFrom(ENTRY)).filter(h -> h < 0).count();
        return String.format("  %,d directed edges, mean out-degree %.1f, max %d; %.1f MB graph, "
                + "%,d nodes unreachable from entry", edges, (double) edges / size, maxDeg, graphBytes() / 1e6, unreachable);
    }

    @Override
    public String describe() {
        return String.format("flat-nsw(m=%d, efC=%d, maxDeg=%s, ef=%d)",
                m, efConstruction, maxDegree == 0 ? "inf" : Integer.toString(maxDegree), efSearch);
    }
}
