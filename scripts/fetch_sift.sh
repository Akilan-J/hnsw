#!/usr/bin/env bash
# Downloads SIFT1M (Jegou et al., INRIA TEXMEX corpus) into data/sift1m/ in the
# layout Dataset.load expects: base.fvecs, query.fvecs, groundtruth.ivecs.
#
# Source: the canonical host is ftp://ftp.irisa.fr/local/texmex/corpus/, but
# FTP's separate data connection is blocked on many networks (the control
# connection answers, then zero bytes arrive). So this pulls the same files
# over HTTPS from a Hugging Face mirror, pinned to one commit so the bytes can't
# change underneath the benchmarks, and checks each file's SHA-256.
#
# What the hash does and doesn't prove: it proves we got exactly the bytes the
# pinned mirror commit holds. It can't prove the mirror matches INRIA's
# originals. That's covered by the exact file sizes (1M x 516 bytes, etc.) and
# by VerifyGroundTruth, which recomputes the answers from base+query and checks
# them against the shipped groundtruth.ivecs.
set -euo pipefail
cd "$(dirname "$0")/.."

REV=bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b
URL=https://huggingface.co/datasets/qbo-odp/sift1m/resolve/$REV
DEST=data/sift1m

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch() {
    local remote=$1 local_name=$2 expected=$3
    local out="$DEST/$local_name"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$expected" ]; then
        echo "$out already present and verified"
        return
    fi
    echo "downloading $remote ..."
    # -C - resumes a partial download; -L follows the redirect to the CDN.
    curl --fail -L -C - -o "$out.part" "$URL/$remote"
    local got
    got=$(sha256 "$out.part")
    if [ "$got" != "$expected" ]; then
        echo "SHA-256 mismatch for $remote: got $got, expected $expected" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
    echo "$out ok"
}

mkdir -p "$DEST"
fetch sift_query.fvecs       query.fvecs       f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc
fetch sift_groundtruth.ivecs groundtruth.ivecs 2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f
fetch sift_base.fvecs        base.fvecs        21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816
echo "$DEST ready"
