package hnsw.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Turns benchmark CSVs into SVG charts plus a markdown table of the same numbers.
 * Plain string-building, no plotting library - the project has no dependencies,
 * and an SVG is just XML.
 *
 * <p>Two charts:
 * <ul>
 * <li>{@code tradeoff}: recall@k (x) against queries per second (y, log scale),
 * one line per series, each point one efSearch. This is the standard way ANN
 * indexes are compared: no index has "a" speed, only a speed at a given recall.
 * Brute-force rows are drawn as a single reference point.</li>
 * <li>{@code at-recall}: for each series and each value of a parameter (a
 * dataset parameter like latent-dim, or a CSV column like m), the cost of
 * reaching a target recall, interpolated between the two efSearch points that
 * bracket it. Cost is distances per query by default: it measures the
 * algorithm rather than the CPU.</li>
 * </ul>
 *
 * <p>Styling follows a fixed chart spec: up to three series in a validated,
 * colourblind-safe order, told apart by marker shape as well as colour; text
 * always in ink colours, never series colours; hairline grid; light and dark
 * variants chosen by the viewer's colour scheme. Every point has a hover title,
 * and the markdown table is the accessible twin of the chart.
 *
 * <p>usage:
 * <br>Plot tradeoff --csv FILE [--dataset NAME] [--series "A,B,C"] [--recall recall|recall_ties]
 * --title T [--subtitle S] [--svg OUT] [--table OUT]
 * <br>Plot at-recall --csv FILE --x KEY [--y distances_per_query] --target 0.95 [--series ...]
 * --title T [--subtitle S] [--x-label L] [--svg OUT] [--table OUT]
 * <br>Plot builds --csv FILE [--series ...] [--table OUT]   (build time and memory, table only)
 * <br>All modes take [--where col=value,...] to filter rows.
 */
public final class Plot {

    private static final int W = 760, H = 460;
    private static final int LEFT = 72, RIGHT = 28, TOP = 92, BOTTOM = 56;
    private static final String[] MARKERS = {"circle", "square", "triangle"};

    public static void main(String[] argv) throws IOException {
        if (argv.length == 0) {
            throw new IllegalArgumentException("usage: Plot tradeoff|at-recall --csv FILE ...");
        }
        String mode = argv[0];
        Args args = new Args(Arrays.copyOfRange(argv, 1, argv.length));
        List<Map<String, String>> rows = readCsv(Path.of(args.required("csv")));
        // --where col=value[,col=value...] keeps only matching rows, e.g.
        // base_n=1000000 to pick one prefix size out of a scaling run.
        String where = args.str("where", null);
        if (where != null) {
            for (String cond : where.split(",")) {
                String[] kv = cond.split("=", 2);
                rows.removeIf(r -> !kv[1].equals(r.get(kv[0])));
            }
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("no rows left after --where " + where);
        }
        String recallCol = args.str("recall", "recall");
        String title = mode.equals("builds") ? "" : args.required("title");
        String subtitle = args.str("subtitle", "");
        String svgOut = args.str("svg", null);
        String tableOut = args.str("table", null);
        List<String> seriesOrder = seriesOrder(rows, args.str("series", null));

        String svg;
        String table;
        switch (mode) {
            case "tradeoff" -> {
                args.done();
                svg = tradeoff(rows, seriesOrder, recallCol, title, subtitle);
                table = tradeoffTable(rows, seriesOrder, recallCol);
            }
            case "at-recall" -> {
                String xKey = args.required("x");
                String yKey = args.str("y", "distances_per_query");
                double target = args.decimal("target", 0.95);
                String xLabel = args.str("x-label", xKey);
                args.done();
                Map<String, TreeMap<Double, Double>> points = atRecall(rows, seriesOrder, xKey, yKey, recallCol, target);
                svg = atRecallChart(points, seriesOrder, xLabel, yLabel(yKey), title, subtitle, target);
                table = atRecallTable(points, seriesOrder, xLabel, yLabel(yKey), target);
            }
            case "builds" -> {
                args.done();
                svg = null;
                table = buildsTable(rows, seriesOrder);
            }
            default -> throw new IllegalArgumentException("unknown chart '" + mode + "'");
        }
        if (svgOut != null && svg != null) {
            Files.writeString(Path.of(svgOut), svg);
        }
        if (tableOut != null) {
            Files.writeString(Path.of(tableOut), table);
        }
        System.out.print(table);
    }

    // ------------------------------------------------------------ tradeoff

    private static String tradeoff(List<Map<String, String>> rows, List<String> order, String recallCol,
                                   String title, String subtitle) {
        List<Map<String, String>> graph = new ArrayList<>(), brute = new ArrayList<>();
        for (Map<String, String> r : rows) {
            (r.get("index").startsWith("brute") ? brute : graph).add(r);
        }
        double minRecall = rows.stream().mapToDouble(r -> num(r, recallCol)).min().orElse(0);
        double x0 = Math.min(0.9, Math.floor(minRecall * 20) / 20);
        double x1 = 1.0 + (1.0 - x0) * 0.02; // a little air so recall-1.0 marks aren't cut by the frame
        double yMin = rows.stream().mapToDouble(r -> num(r, "qps")).min().orElse(1);
        double yMax = rows.stream().mapToDouble(r -> num(r, "qps")).max().orElse(10);
        Axis x = Axis.linear(x0, x1, LEFT, W - RIGHT);
        Axis y = Axis.log(yMin, yMax, H - BOTTOM, TOP);

        Svg s = new Svg(title, subtitle);
        s.gridAndAxes(x, y, "recall@10 (" + (recallCol.equals("recall") ? "strict" : "ties accepted") + ")",
                "queries per second (log scale)");
        for (int i = 0; i < order.size(); i++) {
            String series = order.get(i);
            List<double[]> pts = new ArrayList<>();
            List<String> tips = new ArrayList<>();
            graph.stream().filter(r -> r.get("label").equals(series))
                    .sorted(Comparator.comparingDouble(r -> num(r, "ef")))
                    .forEach(r -> {
                        pts.add(new double[]{x.map(num(r, recallCol)), y.map(num(r, "qps"))});
                        tips.add(String.format(Locale.ROOT, "%s, ef=%s: recall %.4f, %,.0f QPS, %,.0f distances/query",
                                series, r.get("ef"), num(r, recallCol), num(r, "qps"), num(r, "distances_per_query")));
                    });
            s.series(i, pts, tips);
        }
        for (Map<String, String> r : brute) {
            double px = x.map(num(r, recallCol)), py = y.map(num(r, "qps"));
            s.referencePoint(px, py, String.format(Locale.ROOT, "brute force: %,.1f QPS", num(r, "qps")),
                    String.format(Locale.ROOT, "brute force (exact): recall %.4f, %,.1f QPS", num(r, recallCol), num(r, "qps")));
        }
        s.legend(order);
        return s.finish();
    }

    private static String tradeoffTable(List<Map<String, String>> rows, List<String> order, String recallCol) {
        StringBuilder t = new StringBuilder("| series | ef | recall@10 | recall@10 (ties) | QPS (min-max) | p50 ms | p99 ms | distances/query |\n"
                + "|---|---:|---:|---:|---:|---:|---:|---:|\n");
        List<String> all = new ArrayList<>(order);
        rows.stream().map(r -> r.get("label")).distinct().filter(l -> !all.contains(l)).forEach(all::add);
        for (String series : all) {
            rows.stream().filter(r -> r.get("label").equals(series))
                    .sorted(Comparator.comparingDouble(r -> r.get("ef").isEmpty() ? 0 : num(r, "ef")))
                    .forEach(r -> t.append(String.format(Locale.ROOT, "| %s | %s | %.4f | %.4f | %,.0f (%,.0f-%,.0f) | %.3f | %.3f | %,.0f |%n",
                            series, r.get("ef").isEmpty() ? "-" : r.get("ef"), num(r, "recall"), num(r, "recall_ties"),
                            num(r, "qps"), num(r, "qps_min"), num(r, "qps_max"), num(r, "p50_ms"), num(r, "p99_ms"),
                            num(r, "distances_per_query"))));
        }
        return t.toString();
    }

    // ------------------------------------------------------------ builds

    /** One row per built index (the CSV repeats build columns on every ef row). */
    private static String buildsTable(List<Map<String, String>> rows, List<String> order) {
        StringBuilder t = new StringBuilder("| series | vectors | m | efConstruction | build s | inserts/s "
                + "| graph MB (estimate) | graph MB (measured heap) | bytes/vector |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        Map<String, Map<String, String>> seen = new LinkedHashMap<>();
        for (Map<String, String> r : rows) {
            if (!r.get("index").startsWith("brute")) {
                seen.putIfAbsent(r.get("label") + "|" + r.get("base_n") + "|" + r.get("m") + "|" + r.get("ef_construction"), r);
            }
        }
        List<Map<String, String>> builds = new ArrayList<>(seen.values());
        builds.sort(Comparator.<Map<String, String>>comparingInt(r -> order.indexOf(r.get("label")))
                .thenComparingDouble(r -> num(r, "base_n")).thenComparingDouble(r -> num(r, "m"))
                .thenComparingDouble(r -> num(r, "ef_construction")));
        for (Map<String, String> r : builds) {
            double n = num(r, "base_n");
            t.append(String.format(Locale.ROOT, "| %s | %,.0f | %s | %s | %.1f | %,.0f | %.1f | %.1f | %.0f |%n",
                    r.get("label"), n, r.get("m"), r.get("ef_construction"), num(r, "build_s"), n / num(r, "build_s"),
                    num(r, "graph_mb_est"), num(r, "graph_mb_heap"), num(r, "graph_mb_est") * 1e6 / n));
        }
        return t.toString();
    }

    // ------------------------------------------------------------ at-recall

    /**
     * series -> (x -> y at the target recall). NaN when the series never reached
     * the target in the ef range measured. Linear interpolation in recall between
     * the last point below the target and the first at or above it. If even the
     * smallest ef was already above target, that point's cost is an upper bound,
     * stored negated so the table can flag it.
     */
    private static Map<String, TreeMap<Double, Double>> atRecall(List<Map<String, String>> rows, List<String> order,
                                                                 String xKey, String yKey, String recallCol, double target) {
        Map<String, Map<Double, List<Map<String, String>>>> grouped = new LinkedHashMap<>();
        for (Map<String, String> r : rows) {
            if (r.get("index").startsWith("brute")) {
                continue;
            }
            grouped.computeIfAbsent(r.get("label"), k -> new TreeMap<>())
                    .computeIfAbsent(xValue(r, xKey), k -> new ArrayList<>()).add(r);
        }
        Map<String, TreeMap<Double, Double>> out = new LinkedHashMap<>();
        for (String series : order) {
            TreeMap<Double, Double> line = new TreeMap<>();
            grouped.getOrDefault(series, Map.of()).forEach((xv, group) -> {
                group.sort(Comparator.comparingDouble(r -> num(r, "ef")));
                double y = Double.NaN;
                for (int i = 0; i < group.size(); i++) {
                    double rc = num(group.get(i), recallCol);
                    if (rc >= target) {
                        if (i == 0) {
                            // Already past the target at the smallest ef measured, so
                            // this cost is an upper bound. Stored negative to flag it;
                            // the chart plots |y|, the table prints "<= y".
                            y = -num(group.get(0), yKey);
                        } else {
                            double r0 = num(group.get(i - 1), recallCol);
                            double y0 = num(group.get(i - 1), yKey), y1 = num(group.get(i), yKey);
                            y = y0 + (y1 - y0) * (target - r0) / (rc - r0);
                        }
                        break;
                    }
                }
                line.put(xv, y);
            });
            out.put(series, line);
        }
        return out;
    }

    private static String atRecallChart(Map<String, TreeMap<Double, Double>> points, List<String> order, String xLabel,
                                        String yLabel, String title, String subtitle, double target) {
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE, yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        for (TreeMap<Double, Double> line : points.values()) {
            for (Map.Entry<Double, Double> e : line.entrySet()) {
                xMin = Math.min(xMin, e.getKey());
                xMax = Math.max(xMax, e.getKey());
                if (!e.getValue().isNaN()) {
                    yMin = Math.min(yMin, Math.abs(e.getValue()));
                    yMax = Math.max(yMax, Math.abs(e.getValue()));
                }
            }
        }
        Axis x = Axis.log2(xMin, xMax, LEFT, W - RIGHT, points.values().stream()
                .flatMap(l -> l.keySet().stream()).distinct().sorted().mapToDouble(d -> d).toArray());
        Axis y = Axis.log(yMin, yMax, H - BOTTOM, TOP);
        Svg s = new Svg(title, subtitle);
        s.gridAndAxes(x, y, xLabel + " (log scale)", yLabel + " at recall@10 = " + target + " (log scale)");
        for (int i = 0; i < order.size(); i++) {
            String series = order.get(i);
            List<double[]> pts = new ArrayList<>();
            List<String> tips = new ArrayList<>();
            for (Map.Entry<Double, Double> e : points.getOrDefault(series, new TreeMap<>()).entrySet()) {
                if (!e.getValue().isNaN()) {
                    double v = Math.abs(e.getValue());
                    pts.add(new double[]{x.map(e.getKey()), y.map(v)});
                    tips.add(String.format(Locale.ROOT, "%s, %s=%s: %s%,.0f", series, xLabel, fmtX(e.getKey()),
                            e.getValue() < 0 ? "at most " : "", v));
                }
            }
            s.series(i, pts, tips);
        }
        s.legend(order);
        return s.finish();
    }

    private static String atRecallTable(Map<String, TreeMap<Double, Double>> points, List<String> order,
                                        String xLabel, String yLabel, double target) {
        TreeMap<Double, Boolean> xs = new TreeMap<>();
        points.values().forEach(l -> l.keySet().forEach(k -> xs.put(k, true)));
        StringBuilder t = new StringBuilder("| " + xLabel + " |");
        for (String s : order) {
            t.append(" ").append(s).append(" |");
        }
        t.append("\n|---:|").append("---:|".repeat(order.size())).append("\n");
        for (double xv : xs.keySet()) {
            t.append("| ").append(fmtX(xv)).append(" |");
            for (String s : order) {
                Double v = points.getOrDefault(s, new TreeMap<>()).get(xv);
                t.append(v == null ? " - |" : v.isNaN() ? " not reached |"
                        : v < 0 ? String.format(Locale.ROOT, " &le; %,.0f |", -v) : String.format(Locale.ROOT, " %,.0f |", v));
            }
            t.append("\n");
        }
        t.append("\n").append(yLabel).append(" needed to reach recall@10 = ").append(target)
                .append(", interpolated between efSearch points. &le; means even the smallest ef measured was past the\n")
                .append("target, so that cost is an upper bound; \"not reached\" means the largest ef measured fell short.\n");
        return t.toString();
    }

    /** x from a CSV column if there is one, else from the dataset's generator params ("key=value ..."). */
    private static double xValue(Map<String, String> r, String key) {
        if (r.containsKey(key) && !r.get(key).isEmpty()) {
            return Double.parseDouble(r.get(key));
        }
        for (String kv : r.get("params").split("\\s+")) {
            if (kv.startsWith(key + "=")) {
                return Double.parseDouble(kv.substring(key.length() + 1));
            }
        }
        throw new IllegalArgumentException("no column or dataset param '" + key + "' in row for " + r.get("dataset"));
    }

    private static String yLabel(String yKey) {
        return switch (yKey) {
            case "distances_per_query" -> "distances per query";
            case "qps" -> "queries per second";
            default -> yKey;
        };
    }

    // ------------------------------------------------------------ plumbing

    private static List<String> seriesOrder(List<Map<String, String>> rows, String explicit) {
        List<String> order = new ArrayList<>();
        if (explicit != null) {
            order.addAll(Arrays.asList(explicit.split(",")));
        } else {
            rows.stream().filter(r -> !r.get("index").startsWith("brute")).map(r -> r.get("label"))
                    .distinct().forEach(order::add);
        }
        // The palette is validated for three series shown together. A fourth
        // colour would fail the colourblind-separation check against its
        // neighbours, so more series means splitting into another chart.
        if (order.size() > 3) {
            throw new IllegalArgumentException("at most 3 series per chart, got " + order);
        }
        return order;
    }

    private static double num(Map<String, String> r, String key) {
        String v = r.get(key);
        if (v == null) {
            throw new IllegalArgumentException("no column '" + key + "'");
        }
        return Double.parseDouble(v);
    }

    private static String fmtX(double v) {
        return v == Math.rint(v) ? Long.toString((long) v) : Double.toString(v);
    }

    /** RFC 4180-ish: fields may be double-quoted, with "" for a literal quote. */
    static List<Map<String, String>> readCsv(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        List<String> header = splitCsvLine(lines.get(0));
        List<Map<String, String>> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) {
                continue;
            }
            List<String> fields = splitCsvLine(lines.get(i));
            if (fields.size() != header.size()) {
                throw new IOException(path + ":" + (i + 1) + ": " + fields.size() + " fields, header has " + header.size());
            }
            Map<String, String> row = new HashMap<>();
            for (int j = 0; j < header.size(); j++) {
                row.put(header.get(j), fields.get(j));
            }
            rows.add(row);
        }
        return rows;
    }

    private static List<String> splitCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quoted) {
                if (c == '"' && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    cur.append('"');
                    i++;
                } else if (c == '"') {
                    quoted = false;
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        out.add(cur.toString());
        return out;
    }

    // ------------------------------------------------------------ axes

    /** Maps data values to pixels, and knows where its ticks go. */
    private record Axis(double lo, double hi, double p0, double p1, boolean log, double[] ticks, boolean base2) {

        static Axis linear(double lo, double hi, double p0, double p1) {
            double step = (hi - lo) > 0.3 ? 0.1 : 0.05;
            List<Double> t = new ArrayList<>();
            for (double v = Math.ceil(lo / step - 1e-9) * step; v <= Math.min(hi, 1.0) + 1e-9; v += step) {
                t.add(Math.round(v * 1000) / 1000.0);
            }
            return new Axis(lo, hi, p0, p1, false, t.stream().mapToDouble(d -> d).toArray(), false);
        }

        /** Log10 axis padded out to the 1-2-5 ticks around the data. */
        static Axis log(double min, double max, double p0, double p1) {
            List<Double> t = new ArrayList<>();
            for (int e = (int) Math.floor(Math.log10(min)); e <= (int) Math.ceil(Math.log10(max)); e++) {
                for (int m : new int[]{1, 2, 5}) {
                    t.add(m * Math.pow(10, e));
                }
            }
            double lo = t.stream().filter(v -> v <= min).max(Double::compare).orElse(min);
            double hi = t.stream().filter(v -> v >= max).min(Double::compare).orElse(max);
            return new Axis(lo, hi, p0, p1, true,
                    t.stream().filter(v -> v >= lo && v <= hi).mapToDouble(d -> d).toArray(), false);
        }

        /** Log2 axis whose ticks are exactly the x values measured. */
        static Axis log2(double min, double max, double p0, double p1, double[] ticks) {
            double pad = 0.15; // in doublings, so end points aren't on the frame
            return new Axis(min / Math.pow(2, pad), max * Math.pow(2, pad), p0, p1, true, ticks, true);
        }

        double map(double v) {
            double f = log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
            return p0 + f * (p1 - p0);
        }

        String label(double v) {
            if (base2) {
                return fmtX(v);
            }
            if (!log) {
                return String.format(Locale.ROOT, "%.2f", v);
            }
            if (v >= 1e6) {
                return fmtX(v / 1e6) + "M";
            }
            if (v >= 1e3) {
                return fmtX(v / 1e3) + "k";
            }
            return fmtX(v);
        }
    }

    // ------------------------------------------------------------ svg

    /** Accumulates SVG markup. Colours are CSS custom properties, so dark mode is one override block. */
    private static final class Svg {
        private final StringBuilder b = new StringBuilder();
        private final StringBuilder marks = new StringBuilder();

        Svg(String title, String subtitle) {
            b.append(String.format(Locale.ROOT,
                    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" "
                            + "role=\"img\" aria-labelledby=\"t d\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">%n",
                    W, H, W, H));
            b.append("<title id=\"t\">").append(esc(title)).append("</title>\n");
            b.append("<desc id=\"d\">").append(esc(subtitle)).append("</desc>\n");
            // Palette: the first three categorical slots, validated as a set for
            // colourblind separation in both modes. Ink roles for all text.
            b.append("""
                    <style>
                      svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
                            --grid:#e1e0d9; --axis:#c3c2b7; --s0:#2a78d6; --s1:#eb6834; --s2:#1baf7a; }
                      @media (prefers-color-scheme: dark) {
                        svg { --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
                              --grid:#2c2c2a; --axis:#383835; --s0:#3987e5; --s1:#d95926; --s2:#199e70; }
                      }
                      .bg { fill:var(--surface); }
                      .title { fill:var(--ink); font-size:16px; font-weight:600; }
                      .sub { fill:var(--ink-2); font-size:12px; }
                      .tick { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
                      .axis-label { fill:var(--ink-2); font-size:12px; }
                      .legend { fill:var(--ink); font-size:12px; }
                      .grid { stroke:var(--grid); stroke-width:1; }
                      .axis { stroke:var(--axis); stroke-width:1; }
                      .ref { fill:var(--muted); stroke:var(--surface); stroke-width:2; }
                      .ref-label { fill:var(--ink-2); font-size:11px; }
                    """);
            for (int i = 0; i < 3; i++) {
                b.append(String.format("  .l%d { fill:none; stroke:var(--s%d); stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }%n", i, i));
                b.append(String.format("  .m%d { fill:var(--s%d); stroke:var(--surface); stroke-width:2; }%n", i, i));
            }
            b.append("</style>\n");
            b.append(String.format("<rect class=\"bg\" width=\"%d\" height=\"%d\" rx=\"8\"/>%n", W, H));
            b.append(String.format("<text class=\"title\" x=\"%d\" y=\"28\">%s</text>%n", LEFT - 48, esc(title)));
            b.append(String.format("<text class=\"sub\" x=\"%d\" y=\"48\">%s</text>%n", LEFT - 48, esc(subtitle)));
        }

        void gridAndAxes(Axis x, Axis y, String xLabel, String yLabel) {
            for (double v : y.ticks) {
                double py = y.map(v);
                b.append(String.format(Locale.ROOT, "<line class=\"grid\" x1=\"%d\" x2=\"%d\" y1=\"%.1f\" y2=\"%.1f\"/>%n", LEFT, W - RIGHT, py, py));
                b.append(String.format(Locale.ROOT, "<text class=\"tick\" x=\"%d\" y=\"%.1f\" text-anchor=\"end\" dominant-baseline=\"middle\">%s</text>%n",
                        LEFT - 8, py, y.label(v)));
            }
            for (double v : x.ticks) {
                double px = x.map(v);
                b.append(String.format(Locale.ROOT, "<line class=\"grid\" x1=\"%.1f\" x2=\"%.1f\" y1=\"%d\" y2=\"%d\"/>%n", px, px, TOP, H - BOTTOM));
                b.append(String.format(Locale.ROOT, "<text class=\"tick\" x=\"%.1f\" y=\"%d\" text-anchor=\"middle\">%s</text>%n",
                        px, H - BOTTOM + 18, x.label(v)));
            }
            b.append(String.format("<line class=\"axis\" x1=\"%d\" x2=\"%d\" y1=\"%d\" y2=\"%d\"/>%n", LEFT, W - RIGHT, H - BOTTOM, H - BOTTOM));
            b.append(String.format("<text class=\"axis-label\" x=\"%d\" y=\"%d\" text-anchor=\"middle\">%s</text>%n",
                    (LEFT + W - RIGHT) / 2, H - 14, esc(xLabel)));
            b.append(String.format("<text class=\"axis-label\" transform=\"translate(18 %d) rotate(-90)\" text-anchor=\"middle\">%s</text>%n",
                    (TOP + H - BOTTOM) / 2, esc(yLabel)));
        }

        void series(int slot, List<double[]> pts, List<String> tips) {
            if (pts.isEmpty()) {
                return;
            }
            StringBuilder path = new StringBuilder();
            for (double[] p : pts) {
                path.append(path.length() == 0 ? "M" : " L").append(String.format(Locale.ROOT, "%.1f %.1f", p[0], p[1]));
            }
            marks.append(String.format("<path class=\"l%d\" d=\"%s\"/>%n", slot, path));
            for (int i = 0; i < pts.size(); i++) {
                marks.append(marker(slot, pts.get(i)[0], pts.get(i)[1], tips.get(i)));
            }
        }

        void referencePoint(double px, double py, String label, String tip) {
            marks.append(String.format(Locale.ROOT, "<g><title>%s</title><rect class=\"ref\" x=\"%.1f\" y=\"%.1f\" width=\"10\" height=\"10\" transform=\"rotate(45 %.1f %.1f)\"/></g>%n",
                    esc(tip), px - 5, py - 5, px, py));
            marks.append(String.format(Locale.ROOT, "<text class=\"ref-label\" x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\">%s</text>%n",
                    px - 10, py - 10, esc(label)));
        }

        /** Legend row under the subtitle: a line-and-marker key in the series colour, the name in ink. */
        void legend(List<String> order) {
            double x = LEFT - 48;
            for (int i = 0; i < order.size(); i++) {
                b.append(String.format(Locale.ROOT, "<line class=\"l%d\" x1=\"%.1f\" x2=\"%.1f\" y1=\"70\" y2=\"70\"/>%n", i, x, x + 24));
                b.append(marker(i, x + 12, 70, order.get(i)));
                b.append(String.format(Locale.ROOT, "<text class=\"legend\" x=\"%.1f\" y=\"70\" dominant-baseline=\"middle\">%s</text>%n",
                        x + 32, esc(order.get(i))));
                x += 32 + 7.0 * order.get(i).length() + 28; // approximate text width at 12px
            }
        }

        /** Shape per slot as a second channel besides colour; a surface-coloured ring keeps overlaps legible. */
        private static String marker(int slot, double x, double y, String tip) {
            String shape = switch (MARKERS[slot]) {
                case "square" -> String.format(Locale.ROOT, "<rect class=\"m%d\" x=\"%.1f\" y=\"%.1f\" width=\"9\" height=\"9\"/>", slot, x - 4.5, y - 4.5);
                case "triangle" -> String.format(Locale.ROOT, "<path class=\"m%d\" d=\"M%.1f %.1f L%.1f %.1f L%.1f %.1f Z\"/>",
                        slot, x, y - 6, x + 5.5, y + 4, x - 5.5, y + 4);
                default -> String.format(Locale.ROOT, "<circle class=\"m%d\" cx=\"%.1f\" cy=\"%.1f\" r=\"4.5\"/>", slot, x, y);
            };
            return "<g><title>" + esc(tip) + "</title>" + shape + "</g>\n";
        }

        String finish() {
            return b + marks.toString() + "</svg>\n";
        }

        private static String esc(String s) {
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
        }
    }
}
