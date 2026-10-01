#!/usr/bin/env bash
# Generate finite configs using the existing app. Write completed run IDs to a new manifest.
# generate.sh DATA.txt CONFIG.toml [CONFIG.toml ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 2 ]] || { echo 'usage: generate.sh DATA.txt CONFIG.toml ...' >&2; exit 2; }
manifest=$1; shift
[[ ! -e $manifest ]] || { echo "Refusing to overwrite $manifest" >&2; exit 2; }
mkdir -p "$(dirname "$manifest")" experiments/card-outcomes/logs
# Reject unbounded jobs before starting any work.
.venv/bin/python - "$@" <<'PY'
import sys,tomllib
for file in sys.argv[1:]:
    r=tomllib.load(open(file,'rb'))['run']
    assert r['groups']>0, f'{file}: finite groups required'
PY
: > "$manifest"
for config in "$@"; do
  id=$(.venv/bin/python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["run"]["id"])' "$config")
  log="experiments/card-outcomes/logs/$id.log"
  [[ ! -e $log ]] || { echo "Existing log $log; choose a new run ID" >&2; exit 2; }
  echo "START $id; log: $log"
  ./apps/card_marginals/run.sh "$config" > "$log" 2>&1
  .venv/bin/python - "$id" >> "$manifest" <<'PY'
import glob,json,sys
paths=glob.glob('runs/schema=card_marginals_v1/date=*/id='+sys.argv[1]+'/run.json')
assert len(paths)==1, paths
r=json.load(open(paths[0])); assert r['status']=='done',r['status']
print(r['run_id'])
PY
  echo "DONE $id"
done
echo "Data manifest: $manifest; generation complete, ready to train."
