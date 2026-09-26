package hnsw.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import hnsw.data.SyntheticClusters;
import hnsw.data.VectorIO;

/**
 * Writes a synthetic clustered dataset in the same layout as a downloaded one
 * (base.fvecs + query.fvecs), so everything downstream has one loading path.
 *
 * <p>usage: Generate --out DIR [--n 100000] [--queries 1000] [--dim 128]
 * [--latent-dim 16] [--clusters 64] [--cluster-std 0.35] [--noise 0.02] [--seed 42]
 *
 * <p>{@code --clusters 1 --latent-dim 128} gives one isotropic Gaussian blob,
 * i.e. "random vectors" - the control case that shows why random data is a
 * misleading benchmark.
 */
public final class Generate {

    public static void main(String[] argv) throws IOException {
        Args args = new Args(argv);
        Path out = Path.of(args.required("out"));
        int n = args.integer("n", 100_000);
        int nq = args.integer("queries", 1_000);
        int dim = args.integer("dim", 128);
        int latent = args.integer("latent-dim", 16);
        int clusters = args.integer("clusters", 64);
        double clusterStd = args.decimal("cluster-std", 0.35);
        double noise = args.decimal("noise", 0.02);
        long seed = args.longValue("seed", 42);
        args.done();

        Files.createDirectories(out);
        // Ground-truth cache names don't encode the vectors themselves, so any
        // cached answer for the old contents of this directory is now wrong.
        try (Stream<Path> stale = Files.list(out)) {
            for (Path p : (Iterable<Path>) stale.filter(p -> p.getFileName().toString().startsWith("gt-"))::iterator) {
                Files.delete(p);
                System.out.println("deleted stale " + p);
            }
        }

        SyntheticClusters model = new SyntheticClusters(dim, latent, clusters, clusterStd, noise, seed);
        // Separate seeds so base and queries are independent draws from one model.
        VectorIO.writeFvecs(out.resolve("base.fvecs"), model.sample(n, seed + 1));
        VectorIO.writeFvecs(out.resolve("query.fvecs"), model.sample(nq, seed + 2));
        String params = String.format("n=%d queries=%d dim=%d latent-dim=%d clusters=%d cluster-std=%s noise=%s seed=%d%n",
                n, nq, dim, latent, clusters, clusterStd, noise, seed);
        Files.writeString(out.resolve("params.txt"), params);
        System.out.print("wrote " + out + ": " + params);
    }
}
