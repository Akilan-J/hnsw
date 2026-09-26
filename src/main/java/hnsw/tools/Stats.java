package hnsw.tools;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import hnsw.data.Dataset;
import hnsw.distance.DistanceFunction;
import hnsw.distance.SquaredL2;

/**
 * Measures how hard a dataset is for nearest-neighbour search, which makes the
 * claim "random vectors are all roughly equidistant" a number instead of an
 * assertion.
 *
 * <p>For a sample of queries it reports:
 * <ul>
 * <li><b>relative contrast</b> = mean distance to the base set / distance to the
 * nearest neighbour. Near 1 means the nearest neighbour is barely nearer than an
 * average point, so "nearest" carries little information and any index that
 * returns something vaguely close scores well on distance but gets recall
 * wrong.</li>
 * <li><b>d10 / d1</b> = distance to the 10th neighbour / distance to the 1st. Near
 * 1 means the top 10 sit on one thin shell around the query.</li>
 * </ul>
 * Distances are true Euclidean here (the square root taken), because ratios of
 * squared distances would exaggerate the contrast.
 *
 * <p>usage: Stats --data DIR [--metric l2] [--base-limit N] [--sample 100]
 */
public final class Stats {

    public static void main(String[] argv) throws IOException {
        Args args = new Args(argv);
        Path dir = Path.of(args.required("data"));
        DistanceFunction metric = DistanceFunction.byName(args.str("metric", "l2"));
        int baseLimit = args.integer("base-limit", 0);
        int sample = args.integer("sample", 100);
        args.done();

        Dataset data = Dataset.load(dir, metric, baseLimit);
        boolean sqrt = metric == SquaredL2.INSTANCE;
        int nq = Math.min(sample, data.queries.length);
        double sumContrast = 0, sumD10OverD1 = 0;
        int used = 0;
        float[] d = new float[data.base.length];
        for (int q = 0; q < nq; q++) {
            double mean = 0;
            for (int i = 0; i < d.length; i++) {
                float v = metric.distance(data.queries[q], data.base[i]);
                d[i] = sqrt ? (float) Math.sqrt(v) : v;
                mean += d[i];
            }
            mean /= d.length;
            // A full sort is fine for a one-off diagnostic over a small sample.
            Arrays.sort(d);
            if (d[0] == 0) {
                continue; // query duplicates a base vector; contrast is infinite, skip it
            }
            sumContrast += mean / d[0];
            sumD10OverD1 += d[Math.min(9, d.length - 1)] / d[0];
            used++;
        }
        System.out.printf("%-28s n=%,d dim=%d  relative contrast=%.3f  d10/d1=%.3f  (over %d queries)%n",
                data.name, data.base.length, data.dim(), sumContrast / used, sumD10OverD1 / used, used);
    }
}
