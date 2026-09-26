#!/usr/bin/env bash
# Unit-level checks. The SIFT cross-check runs only if data/siftsmall exists
# (./scripts/fetch_sift.sh small); otherwise it is reported as skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/build.sh
java -cp build hnsw.Tests
