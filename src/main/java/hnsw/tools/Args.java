package hnsw.tools;

import java.util.HashMap;
import java.util.Map;

/**
 * Minimal {@code --key value} parser. Each getter consumes its key, and
 * {@link #done()} rejects anything left over, so a typo like {@code --querys 100}
 * fails loudly instead of quietly running with the default - which in a
 * benchmark means publishing a number for a configuration you didn't ask for.
 */
final class Args {

    private final Map<String, String> values = new HashMap<>();

    Args(String[] argv) {
        for (int i = 0; i < argv.length; i++) {
            if (!argv[i].startsWith("--")) {
                throw new IllegalArgumentException("unexpected argument: " + argv[i]);
            }
            String key = argv[i].substring(2);
            boolean hasValue = i + 1 < argv.length && !argv[i + 1].startsWith("--");
            values.put(key, hasValue ? argv[++i] : "true");
        }
    }

    String str(String key, String def) {
        String v = values.remove(key);
        return v == null ? def : v;
    }

    String required(String key) {
        String v = values.remove(key);
        if (v == null) {
            throw new IllegalArgumentException("missing required --" + key);
        }
        return v;
    }

    int integer(String key, int def) {
        String v = values.remove(key);
        return v == null ? def : Integer.parseInt(v.replace("_", ""));
    }

    long longValue(String key, long def) {
        String v = values.remove(key);
        return v == null ? def : Long.parseLong(v.replace("_", ""));
    }

    double decimal(String key, double def) {
        String v = values.remove(key);
        return v == null ? def : Double.parseDouble(v);
    }

    boolean flag(String key) {
        return values.remove(key) != null;
    }

    void done() {
        if (!values.isEmpty()) {
            throw new IllegalArgumentException("unknown option(s): --" + String.join(", --", values.keySet()));
        }
    }
}
