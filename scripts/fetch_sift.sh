#!/usr/bin/env bash
# Downloads the TEXMEX SIFT datasets (Jegou et al., INRIA) into data/ and
# renames them to the layout Dataset.load expects:
#   data/<name>/base.fvecs  query.fvecs  groundtruth.ivecs
#
#   ./scripts/fetch_sift.sh small   siftsmall: 10k base, 100 queries (~5 MB)
#   ./scripts/fetch_sift.sh full    sift1m:    1M base, 10k queries (~160 MB download)
#   ./scripts/fetch_sift.sh         both
#
# The server is FTP-only. If it's unreachable, the same files are mirrored on
# several ANN benchmark sites; anything in .fvecs/.ivecs form works.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_URL=ftp://ftp.irisa.fr/local/texmex/corpus

fetch() {
    local archive=$1 prefix=$2 dest=$3
    if [ -f "data/$dest/base.fvecs" ]; then
        echo "data/$dest already present, skipping"
        return
    fi
    mkdir -p data/tmp
    echo "downloading $archive ..."
    # -C - resumes a partial download instead of starting over.
    curl --fail -C - -o "data/tmp/$archive" "$BASE_URL/$archive"
    tar -xzf "data/tmp/$archive" -C data/tmp
    mkdir -p "data/$dest"
    mv "data/tmp/$prefix/${prefix}_base.fvecs"        "data/$dest/base.fvecs"
    mv "data/tmp/$prefix/${prefix}_query.fvecs"       "data/$dest/query.fvecs"
    mv "data/tmp/$prefix/${prefix}_groundtruth.ivecs" "data/$dest/groundtruth.ivecs"
    rm -rf "data/tmp/$prefix" "data/tmp/$archive"
    echo "data/$dest ready"
}

case "${1:-both}" in
    small) fetch siftsmall.tar.gz siftsmall siftsmall ;;
    full)  fetch sift.tar.gz sift sift1m ;;
    both)  fetch siftsmall.tar.gz siftsmall siftsmall; fetch sift.tar.gz sift sift1m ;;
    *) echo "usage: $0 [small|full|both]"; exit 2 ;;
esac
rmdir data/tmp 2>/dev/null || true
