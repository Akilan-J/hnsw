#!/usr/bin/env bash
# Unit-level checks. The SIFT cross-check runs only if data/sift1m exists
# (./scripts/fetch_sift.sh); otherwise it is reported as skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/build.sh
java -Xmx2g -cp build hnsw.Tests
