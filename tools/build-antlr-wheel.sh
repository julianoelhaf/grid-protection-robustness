#!/usr/bin/env bash
set -euo pipefail
destination=$(realpath -m "$1")
root=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$destination"
archive=antlr4-python3-runtime-4.9.3.tar.gz
curl --fail --silent --show-error --location \
    "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/$archive" \
    -o "$scratch/$archive"
printf '%s  %s\n' f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b "$scratch/$archive" | sha256sum --check
grep -E '^(packaging|setuptools|wheel)==' "$root/requirements.lock" > "$scratch/build.lock"
python -m pip install --require-hashes --only-binary=:all: -r "$scratch/build.lock"
export SOURCE_DATE_EPOCH=315532800 TZ=UTC LC_ALL=C PYTHONHASHSEED=0
umask 022
python -m pip wheel --no-cache-dir --no-index --no-build-isolation --no-deps \
    --wheel-dir "$destination" "$scratch/$archive"
