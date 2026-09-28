package hnsw.index;

import java.util.Arrays;
import java.util.Random;

import hnsw.distance.DistanceFunction;

/**
 * Hierarchical navigable small world graph (Malkov and Yashunin, 2016).
 *
 * <p><b>The structure.</b> Every vector is a node on layer 0. Each node is also
 * given a random top layer, and it appears on every layer from 0 up to that one.
 * The chance of reaching layer l shrinks geometrically - with the default level
 * multiplier, each layer holds about 1/M of the nodes of the layer below. So
 * layer 0 is dense and its links are short, while the top layer holds a handful
 * of nodes whose links span the whole dataset. Each layer is a navigable graph
 * like stage 2's, over just the nodes that reach it.
 *
 * <p><b>Why that's a skip list.</b> A skip list is a sorted linked list with
 * sparser "express" lists stacked on top, each keeping a random 1/p of the one
 * below. Search runs along the top list until the next step would overshoot,
 * drops a level, and repeats. Each level costs O(1) expected steps and there are
 * O(log n) levels. HNSW is the same idea with "sorted list" replaced by
 * "proximity graph": greedy search on a sparse upper layer covers large
 * distances in a few hops, and each layer down refines the position at a finer
 * scale. What it buys over stage 2's flat graph: the long links live on the
 * upper layers by construction, instead of surviving by accident of insertion
 * order - so every layer can cap its degree without losing them.
 *
 * <p><b>Search.</b> Starting at the global entry point (a node on the top
 * layer), greedy-descend: on each upper layer find the single closest node with
 * ef = 1, then use it as the starting point one layer down. On layer 0, run the
 * full ef-bounded beam search from stage 2.
 *
 * <p><b>Insert.</b> Pick the new node's level. Greedy-descend like a search
 * down to that level. Then, on each layer from there to 0, beam-search with
 * efConstruction, choose M neighbours from what was found (see
 * {@link #selectNeighbors}), and link both ways. A neighbour whose list
 * overflows its cap - M on upper layers, 2M on layer 0 - re-selects which of its
 * links to keep, using the same rule.
 *
 * <p><b>Ablation switches.</b> A level multiplier of 0 puts every node on layer
 * 0: that's a flat graph. Selection SIMPLE keeps the M nearest instead of
 * running the heuristic. Together they separate what the hierarchy buys from
 * what the heuristic buys.
 *
 * <p>Not thread-safe; see {@link VisitedSet}.
 */
public final class HnswIndex implements GraphIndex {

    public enum Selection {
        /** Keep the M nearest candidates. Stage 2's rule. */
        SIMPLE,
        /** Keep candidates that are closer to the base than to anything already kept. */
        HEURISTIC
    }

    private final float[][] vectors;
    private final DistanceFunction metric;
    private final int m;
    private final int maxDegreeUpper;
    private final int maxDegreeLayer0;
    private final int efConstruction;
    private final double levelMultiplier;
    private final Selection selection;
    private final Random levelRng;

    private final int[] levels;
    // links[node][layer] holds the node's out-links on that layer, and
    // degrees[node][layer] how many of the slots are used. Most nodes only reach
    // layer 0 (15 in 16 with M = 16), so this is a short array for nearly all.
    private final int[][][] links;
    private final int[][] degrees;
    private int size;
    private int entryPoint = -1;
    private int topLevel = -1;

    private final VisitedSet visited;
    private final CandidateQueue candidates;
    private int efSearch = 10;
    private long distanceCount;

    /**
     * @param m               links chosen per node per layer on insert. Upper
     *                        layers cap degree at m, layer 0 at 2m.
     * @param efConstruction  beam width when searching for a new node's neighbours
     * @param levelMultiplier mL in level = floor(-ln(U) * mL); 0 = flat graph.
     *                        Use {@link #defaultLevelMultiplier}.
     * @param seed            seed for level assignment, so builds are reproducible
     */
    public HnswIndex(float[][] vectors, DistanceFunction metric, int m, int efConstruction,
                     Selection selection, double levelMultiplier, long seed) {
        if (m < 2 || efConstruction < m || levelMultiplier < 0) {
            throw new IllegalArgumentException("need m >= 2, efConstruction >= m, levelMultiplier >= 0; got m="
                    + m + " efConstruction=" + efConstruction + " levelMultiplier=" + levelMultiplier);
        }
        this.vectors = vectors;
        this.metric = metric;
        this.m = m;
        this.maxDegreeUpper = m;
        // Layer 0 holds every node and is where the final, fine-grained search
        // happens, so it gets more links than the sparse layers above. 2M is the
        // paper's recommendation; lower hurts recall, higher costs memory and
        // makes each expansion more expensive.
        this.maxDegreeLayer0 = 2 * m;
        this.efConstruction = efConstruction;
        this.levelMultiplier = levelMultiplier;
        this.selection = selection;
        this.levelRng = new Random(seed);
        this.levels = new int[vectors.length];
        this.links = new int[vectors.length][][];
        this.degrees = new int[vectors.length][];
        this.visited = new VisitedSet(vectors.length);
        this.candidates = new CandidateQueue(256);
    }

    /**
     * mL = 1 / ln(M). Then P(level >= l) = e^(-l / mL) = M^-l, so each layer keeps
     * a 1/M fraction of the one below, and there are about log_M(n) layers.
     * With M links per node and 1/M of the nodes promoted, a node's links on
     * layer l+1 reach about as far as the M-hop neighbourhood on layer l -
     * each layer up zooms out by roughly one "neighbourhood". The paper chose this
     * value by experiment; it isn't derived as optimal.
     */
    public static double defaultLevelMultiplier(int m) {
        return 1.0 / Math.log(m);
    }

    // ---------------------------------------------------------------- insert

    @Override
    public boolean insertNext() {
        if (size == vectors.length) {
            return false;
        }
        int id = size;
        int level = randomLevel();
        levels[id] = level;
        links[id] = new int[level + 1][];
        degrees[id] = new int[level + 1];
        for (int layer = 0; layer <= level; layer++) {
            links[id][layer] = new int[maxDegree(layer)];
        }
        if (entryPoint < 0) {
            entryPoint = id;
            topLevel = level;
            size++;
            return true;
        }

        float[] v = vectors[id];
        // Above the new node's own top layer it needs no links, just a good
        // starting point for the layers below, so a greedy walk is enough.
        int ep = entryPoint;
        for (int layer = topLevel; layer > level; layer--) {
            ep = searchLayer(v, new int[]{ep}, 1, layer).ids()[0];
        }
        // On its own layers, search wider (efConstruction) since the candidates
        // found are what the neighbours get chosen from. Everything found on one
        // layer seeds the search on the next: those nodes are all on the lower
        // layer too, and a wide start is cheaper than rediscovering them.
        int[] entries = {ep};
        for (int layer = Math.min(level, topLevel); layer >= 0; layer--) {
            Neighbors found = searchLayer(v, entries, efConstruction, layer);
            int[] chosen = selectNeighbors(found.ids(), found.distances(), m);
            for (int neighbor : chosen) {
                addLink(id, neighbor, layer);
                addLink(neighbor, id, layer);
            }
            entries = found.ids();
        }
        if (level > topLevel) {
            // The new node reaches higher than anything before it, so it becomes
            // the one node every search starts from.
            entryPoint = id;
            topLevel = level;
        }
        size++;
        return true;
    }

    /**
     * floor(-ln(U) * mL) with U uniform in (0, 1]: an exponential variable
     * rounded down, i.e. a geometric distribution over levels. It's random rather
     * than "every M-th node" so that the promoted nodes are spread evenly over
     * the data no matter what order it arrives in.
     */
    private int randomLevel() {
        double u = 1.0 - levelRng.nextDouble(); // (0, 1]: ln(0) would be -infinity
        return (int) (-Math.log(u) * levelMultiplier);
    }

    private int maxDegree(int layer) {
        return layer == 0 ? maxDegreeLayer0 : maxDegreeUpper;
    }

    /** Adds from -> to on a layer, re-selecting from's links if that overflows its cap. */
    private void addLink(int from, int to, int layer) {
        int[] list = links[from][layer];
        int deg = degrees[from][layer];
        if (deg < list.length) {
            list[deg] = to;
            degrees[from][layer] = deg + 1;
            return;
        }
        // Full. Rank the old links plus the new one by distance from `from` and
        // re-run neighbour selection over them, as if choosing from scratch.
        TopK ranked = new TopK(deg + 1);
        float[] base = vectors[from];
        for (int j = 0; j < deg; j++) {
            ranked.offer(list[j], distance(base, list[j]));
        }
        ranked.offer(to, distance(base, to));
        Neighbors sorted = ranked.drainSorted();
        int[] kept = selectNeighbors(sorted.ids(), sorted.distances(), list.length);
        System.arraycopy(kept, 0, list, 0, kept.length);
        degrees[from][layer] = kept.length;
    }

    /**
     * Chooses up to {@code max} links for a base node from candidates sorted by
     * distance to it.
     *
     * <p>SIMPLE keeps the nearest ones. The trouble is where they are: if the
     * base sits at the edge of a dense cluster, its M nearest are all in that
     * cluster, pointing the same way. Every link leads somewhere the others
     * already lead, and nothing leads out. Stage 2's capped graph showed the
     * cost: nodes cut off entirely, and a recall ceiling on clustered data.
     *
     * <p>HEURISTIC walks the candidates nearest-first and keeps one only if it's
     * closer to the base than to every candidate already kept. If a kept node r
     * is closer to candidate c than the base is, then a search at the base can
     * reach c by going through r, so the direct link to c is redundant - it adds
     * degree but no new direction. What's left is one link per direction: the
     * nearest node in each cluster around the base, including clusters much
     * farther away than the M-th nearest point. Those are the long links, kept on
     * purpose instead of by accident.
     *
     * <p>This is the pruning rule of the relative neighbourhood graph (Toussaint,
     * 1980), applied only to the candidate list. The full RNG contains the
     * minimum spanning tree and so is always connected, which is the intuition
     * for why this rule keeps the graph connected where "nearest M" doesn't.
     * Applied to a limited candidate list, that becomes a tendency, not a guarantee.
     *
     * <p>It may keep fewer than {@code max} links. That's intended: a node in the
     * middle of a uniform region needs fewer directions than one between clusters.
     * The cost is one distance per (candidate, kept node) pair - up to
     * candidates x max evaluations per selection, paid at build time only.
     */
    int[] selectNeighbors(int[] candidateIds, float[] candidateDists, int max) {
        int n = Math.min(candidateIds.length, candidateDists.length);
        if (selection == Selection.SIMPLE) {
            return Arrays.copyOf(candidateIds, Math.min(max, n));
        }
        int[] kept = new int[Math.min(max, n)];
        int count = 0;
        for (int i = 0; i < n && count < kept.length; i++) {
            int c = candidateIds[i];
            float[] cv = vectors[c];
            boolean newDirection = true;
            for (int j = 0; j < count; j++) {
                // <=, not <: an exact duplicate of a kept node (distance 0) is
                // always redundant. SIFT has duplicate vectors.
                if (distance(cv, kept[j]) <= candidateDists[i]) {
                    newDirection = false;
                    break;
                }
            }
            if (newDirection) {
                kept[count++] = c;
            }
        }
        return count == kept.length ? kept : Arrays.copyOf(kept, count);
    }

    // ---------------------------------------------------------------- search

    public void setEfSearch(int ef) {
        if (ef < 1) {
            throw new IllegalArgumentException("ef must be >= 1");
        }
        this.efSearch = ef;
    }

    @Override
    public int[] search(float[] query, int k) {
        if (size == 0) {
            return new int[0];
        }
        int ep = entryPoint;
        for (int layer = topLevel; layer > 0; layer--) {
            ep = searchLayer(query, new int[]{ep}, 1, layer).ids()[0];
        }
        int[] found = searchLayer(query, new int[]{ep}, Math.max(efSearch, k), 0).ids();
        return found.length <= k ? found : Arrays.copyOf(found, k);
    }

    /**
     * Stage 2's beam search, restricted to one layer's links and seeded with
     * any number of entry points. Same stopping rule, same bet: stop once the
     * closest unexpanded node is farther than the worst of the ef results.
     */
    private Neighbors searchLayer(float[] query, int[] entries, int ef, int layer) {
        visited.clear();
        candidates.clear();
        TopK results = new TopK(ef);
        for (int e : entries) {
            if (visited.add(e)) {
                float d = distance(query, e);
                candidates.push(e, d);
                results.offer(e, d);
            }
        }
        while (!candidates.isEmpty()) {
            int current = candidates.peekId();
            float currentDist = candidates.peekDist();
            candidates.pop();
            if (results.isFull() && currentDist > results.worstDistance()) {
                break;
            }
            int[] list = links[current][layer];
            for (int j = 0, deg = degrees[current][layer]; j < deg; j++) {
                int next = list[j];
                if (!visited.add(next)) {
                    continue;
                }
                float d = distance(query, next);
                if (results.offer(next, d)) {
                    candidates.push(next, d);
                }
            }
        }
        return results.drainSorted();
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
        return entryPoint;
    }

    public int topLevel() {
        return topLevel;
    }

    public int level(int node) {
        return levels[node];
    }

    public int degree(int node, int layer) {
        return degrees[node][layer];
    }

    public int neighbor(int node, int layer, int i) {
        return links[node][layer][i];
    }

    @Override
    public long distanceComputations() {
        return distanceCount;
    }

    /** BFS over layer-0 links from the entry point; -1 where unreachable. */
    public int[] layer0HopsFromEntry() {
        int[] hops = new int[size];
        Arrays.fill(hops, -1);
        int[] queue = new int[size];
        int head = 0, tail = 0;
        hops[entryPoint] = 0;
        queue[tail++] = entryPoint;
        while (head < tail) {
            int u = queue[head++];
            for (int j = 0; j < degrees[u][0]; j++) {
                int v = links[u][0][j];
                if (hops[v] < 0) {
                    hops[v] = hops[u] + 1;
                    queue[tail++] = v;
                }
            }
        }
        return hops;
    }

    /**
     * Estimated bytes held by the graph (not the vectors, which every index
     * shares): per node, the per-layer array of lists, each list at its full
     * capacity, the degree counts, each with a 16-byte array header and 4-byte
     * references (compressed oops, the JVM default below 32 GB of heap); plus
     * the level array and the visited-set marks.
     */
    public long graphBytes() {
        long bytes = 0;
        for (int i = 0; i < size; i++) {
            int layers = levels[i] + 1;
            bytes += 16 + 4L * layers;            // links[i]
            bytes += 16 + 4L * layers;            // degrees[i]
            for (int layer = 0; layer < layers; layer++) {
                bytes += 16 + 4L * maxDegree(layer);
            }
        }
        return bytes + 4L * levels.length + 4L * vectors.length;
    }

    @Override
    public String graphStats() {
        StringBuilder sb = new StringBuilder();
        int[] perLevel = new int[topLevel + 1];
        for (int i = 0; i < size; i++) {
            for (int l = 0; l <= levels[i]; l++) {
                perLevel[l]++;
            }
        }
        long unreachable = Arrays.stream(layer0HopsFromEntry()).filter(h -> h < 0).count();
        sb.append(String.format("  %d layers, entry point node %d; %.1f MB graph, %,d nodes unreachable on layer 0%n",
                topLevel + 1, entryPoint, graphBytes() / 1e6, unreachable));
        for (int l = topLevel; l >= 0; l--) {
            long edges = 0;
            int max = 0;
            for (int i = 0; i < size; i++) {
                if (levels[i] >= l) {
                    edges += degrees[i][l];
                    max = Math.max(max, degrees[i][l]);
                }
            }
            sb.append(String.format("  layer %d: %,9d nodes, mean out-degree %5.1f, max %d (cap %d)%n",
                    l, perLevel[l], perLevel[l] == 0 ? 0 : (double) edges / perLevel[l], max, maxDegree(l)));
        }
        return sb.toString().stripTrailing();
    }

    @Override
    public String describe() {
        return String.format("hnsw(m=%d, efC=%d, %s, mL=%.3f, ef=%d)",
                m, efConstruction, selection.name().toLowerCase(), levelMultiplier, efSearch);
    }
}
