#!/bin/bash
# Download SIFT1M (TEXMEX corpus) into data/sift/ and verify file sizes.
# Run on an Euler LOGIN node (compute nodes may not have internet):
#   bash scripts/get_sift1m.sh
# Override the source with SIFT_URL=... if the FTP server is unreachable.
# VERIFY_ONLY=1 checks the files without downloading (used by Slurm jobs).
set -euo pipefail

SIFT_URL="${SIFT_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
DEST="$DATA/sift"

# Expected sizes in bytes: count * (4 + 4 * dim).
declare -A EXPECTED=(
    [sift_base.fvecs]=516000000          # 1,000,000 x 128 floats
    [sift_query.fvecs]=5160000           # 10,000 x 128 floats
    [sift_groundtruth.ivecs]=4040000     # 10,000 x 100 ints
)

file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

verify() {
    local ok=0
    for f in "${!EXPECTED[@]}"; do
        if [[ ! -f "$DEST/$f" ]]; then
            echo "missing: $DEST/$f"; ok=1
        elif [[ "$(file_size "$DEST/$f")" != "${EXPECTED[$f]}" ]]; then
            echo "wrong size: $DEST/$f ($(file_size "$DEST/$f") bytes, expected ${EXPECTED[$f]})"; ok=1
        fi
    done
    return $ok
}

if verify >/dev/null 2>&1; then
    echo "SIFT1M already present and verified in $DEST"
    exit 0
fi

if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
    verify >&2 || true
    echo "SIFT1M missing; run 'bash scripts/get_sift1m.sh' on a login node first" >&2
    exit 1
fi

mkdir -p "$DATA"
echo "Downloading $SIFT_URL ..."
curl -fL --retry 3 -o "$DATA/sift.tar.gz" "$SIFT_URL"
tar -xzf "$DATA/sift.tar.gz" -C "$DATA"
rm -f "$DATA/sift.tar.gz"

if verify; then
    echo "SIFT1M ready in $DEST"
else
    echo "SIFT1M verification failed" >&2
    exit 1
fi
