#!/usr/bin/env bash
set -euo pipefail
run=$(realpath -m "$1")
throttle=${ARRAY_THROTTLE:-4}
[[ $throttle =~ ^[1-9][0-9]*$ ]] || { echo "Invalid ARRAY_THROTTLE" >&2; exit 2; }
mkdir -p "$run/logs"
name=gr-eusipco-$(printf '%s' "$run" | sha256sum | cut -c1-16)
script=$(realpath "$(dirname "$0")/case.sbatch")
cli="$(command -v grid-protection-robustness)"

# Serialize submitters, and query expanded queue rows to include pending array tasks.
exec 9>"$run/.submit.lock"
flock 9
queued=$(squeue --noheader --array --user "$USER" --name "$name" --format '%K')
active=$(printf '%s\n' "$queued" | awk '/^[0-9]+$/ { print $1 }' | paste -sd, -)
ids=$("$cli" eusipco remaining --run "$run" --active "$active" --array)
if [[ -z $ids ]]; then
    printf 'No incomplete cases are available for submission.\n'
    exit 0
fi
sbatch --parsable --job-name="$name" --array="$ids%$throttle" \
    --output="$run/logs/%A_%a.out" "$script" "$cli" "$run"
