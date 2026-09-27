package hnsw.tools;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import hnsw.bench.GroundTruth;
import hnsw.data.Dataset;
import hnsw.distance.DistanceFunction;
import hnsw.distance.SquaredL2;
import hnsw.index.FlatNswIndex;

/**
 * Shows where greedy search on a flat graph goes wrong, and by how much.
 *
 * <p>Runs pure greedy search (ef = 1, looking for the single nearest neighbour)
 * from the entry point and sorts every query into one of two outcomes:
 * <ul>
 * <li><b>found</b>: it stopped at the true nearest neighbour (or one exactly as
 * close).</li>
 * <li><b>local minimum</b>: it stopped at a node closer to the query than every
 * one of that node's neighbours, but not the nearest overall. The tool checks
 * that claim directly instead of trusting it, then measures how bad the stop
 * was: how far down the true ranking the node sits, how much farther it is,
 * and how many graph hops away the right answer actually was.</li>
 * </ul>
 * Then it widens the beam (ef = 2, 4, 8, ...) to show how much of the failure a
 * wider search recovers without changing the graph at all.
 *
 * <p>usage: GreedyFailures --data DIR [--metric l2] [--base-limit N] [--queries 1000]
 * [--m 16] [--ef-construction 100] [--max-degree 32 (0 = unbounded)]
 */
public final class GreedyFailures {

    public static void main(String[] argv) throws IOException {
        Args args = new Args(argv);
        Path dir = Path.of(args.required("data"));
        DistanceFunction metric = DistanceFunction.byName(args.str("metric", "l2"));
        int baseLimit = args.integer("base-limit", 0);
        int nq = args.integer("queries", 1000);
        int m = args.integer("m", 16);
        int efConstruction = args.integer("ef-construction", 100);
        int maxDegree = args.integer("max-degree", 2 * m);
        args.done();

        Dataset data = Dataset.load(dir, metric, baseLimit);
        int[][] gt = GroundTruth.loadOrCompute(data);
        nq = Math.min(nq, data.queries.length);
        FlatNswIndex index = new FlatNswIndex(data.base, metric, m, efConstruction, maxDegree);
        index.insertAll();
        long unreachable = Arrays.stream(index.hopsFrom(index.entryPoint())).filter(h -> h < 0).count();
        System.out.printf("%s, %,d base vectors: %s%n  %,d nodes unreachable from the entry point%n",
                data.name, data.base.length, index.describe(), unreachable);

        index.setEfSearch(1);
        int found = 0, stuck = 0, notActuallyMinimum = 0, beyondTop100 = 0, trueNnUnreachable = 0;
        double[] ranks = new double[nq], ratios = new double[nq], hopsAway = new double[nq];
        int rankCount = 0, hopCount = 0;
        long expansionsBefore = index.expansions();
        for (int q = 0; q < nq; q++) {
            float[] query = data.queries[q];
            int got = index.search(query, 1)[0];
            int truth = gt[q][0];
            float dGot = metric.distance(query, data.base[got]);
            float dTruth = metric.distance(query, data.base[truth]);
            if (dGot == dTruth) {
                found++;
                continue;
            }
            stuck++;
            // Verify the definition rather than assume it: no out-neighbour of the
            // stopping node may be closer to the query than the node itself.
            for (int j = 0; j < index.degree(got); j++) {
                if (metric.distance(query, data.base[index.neighbor(got, j)]) < dGot) {
                    notActuallyMinimum++;
                    break;
                }
            }
            int rank = indexOf(gt[q], got);
            if (rank < 0) {
                beyondTop100++;
            } else {
                ranks[rankCount++] = rank + 1; // 1-based: rank 2 = second nearest
            }
            ratios[stuck - 1] = metric == SquaredL2.INSTANCE ? Math.sqrt(dGot / dTruth) : dGot / dTruth;
            int hops = index.hopsFrom(got)[truth];
            if (hops < 0) {
                trueNnUnreachable++;
            } else {
                hopsAway[hopCount++] = hops;
            }
        }
        double meanPath = (double) (index.expansions() - expansionsBefore) / nq;

        System.out.printf("%ngreedy search (ef=1) for the single nearest neighbour, %d queries, mean path %.1f hops:%n",
                nq, meanPath);
        System.out.printf("  found the true nearest neighbour : %5.1f%%%n", 100.0 * found / nq);
        System.out.printf("  stopped in a local minimum       : %5.1f%%  (%d of them failed the local-minimum check)%n",
                100.0 * stuck / nq, notActuallyMinimum);
        if (stuck > 0) {
            // Split, because a percentile over only the in-top-100 cases would hide
            // how many stops weren't even close.
            System.out.printf("    stopped outside the true top 100: %d of %d%n", beyondTop100, stuck);
            System.out.printf("    rank, for the %d inside top 100 : p50 %s, p90 %s%n",
                    rankCount, pct(ranks, rankCount, 50), pct(ranks, rankCount, 90));
            System.out.printf("    distance vs. true NN          : p50 %sx, p90 %sx%n",
                    pct(ratios, stuck, 50), pct(ratios, stuck, 90));
            System.out.printf("    hops from there to the true NN: p50 %s, p90 %s, unreachable in %d cases%n",
                    pct(hopsAway, hopCount, 50), pct(hopsAway, hopCount, 90), trueNnUnreachable);
        }

        System.out.println("\nwidening the beam on the same graph (recall@1 = fraction that found the true NN):");
        for (int ef = 1; ef <= 64; ef *= 2) {
            index.setEfSearch(ef);
            long distBefore = index.distanceComputations();
            int hits = 0;
            for (int q = 0; q < nq; q++) {
                int got = index.search(data.queries[q], 1)[0];
                if (metric.distance(data.queries[q], data.base[got])
                        == metric.distance(data.queries[q], data.base[gt[q][0]])) {
                    hits++;
                }
            }
            System.out.printf("  ef=%-3d recall@1=%.3f  %,6.0f distances/query%n",
                    ef, (double) hits / nq, (double) (index.distanceComputations() - distBefore) / nq);
        }
    }

    private static int indexOf(int[] a, int v) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == v) {
                return i;
            }
        }
        return -1;
    }

    private static String pct(double[] values, int n, double p) {
        if (n == 0) {
            return "-";
        }
        double[] sorted = Arrays.copyOf(values, n);
        Arrays.sort(sorted);
        double v = sorted[Math.max(0, (int) Math.ceil(p / 100 * n) - 1)];
        return v == Math.rint(v) ? String.format("%.0f", v) : String.format("%.2f", v);
    }
}
