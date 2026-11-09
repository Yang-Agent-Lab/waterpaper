#!/bin/bash
set -eu
cd ._data_org_072b561
export PYTHONPATH=src:.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=GPU-88cf4f45-24fa-5682-5ecc-0a9156e12a8d
P=./.venv/bin/python
ROOT=._inputs/data_org_v1
OPTIONS=(--source-manifest "$ROOT/source_manifest.json" --path-map-json "$ROOT/target_path_map_072b561.json")
$P - <<'PY'
import json,sys,locale
from pathlib import Path
catalog=Path('._inputs/data_org_v1/fixed/china_event_catalog_v1.csv')
with catalog.open() as f:encoding=f.encoding;lines=sum(1 for _ in f)
assert sys.flags.utf8_mode==1
Path('./external-home/data_org_utf8_runtime_072b561.json').write_text(json.dumps(dict(PYTHONUTF8=1,PYTHONIOENCODING='utf-8',python_utf8_mode=sys.flags.utf8_mode,locale_encoding=locale.getencoding(),actual_catalog_open_encoding=encoding,catalog_lines=lines,source_code='072b561',scientific_conditions_changed=False),indent=2))
PY
PHASE=GPU_smoke_resume
trap 'STATUS=$?; printf "{\"exit_code\":%s,\"last_phase\":\"%s\",\"commit\":\"072b561\",\"PYTHONUTF8\":1}\n" "$STATUS" "$PHASE" > ./external-home/data_org_pipeline_utf8_072b561_completion.json' EXIT
printf '{"state":"running","phase":"GPU_smoke_resume","commit":"072b561","PYTHONUTF8":1}\n' > ./external-home/data_org_pipeline_utf8_072b561_progress.json
$P src/run_versioned.py --run-dir results/experiments/data_org_target_GPU_smoke_resume_utf8_072b561_20261013 -- $P -m data_organization_v1.runner --mode smoke --output results/experiments/data_org_target_GPU_smoke_072b561_20261013/artifacts "${OPTIONS[@]}" --q-only-dir "$ROOT/Q_only"
PHASE=formal
printf '{"state":"running","phase":"formal","commit":"072b561","PYTHONUTF8":1}\n' > ./external-home/data_org_pipeline_utf8_072b561_progress.json
$P src/run_versioned.py --run-dir results/experiments/data_org_formal_072b561_20261013 -- $P -m data_organization_v1.runner --mode formal --output results/experiments/data_org_formal_072b561_20261013/artifacts "${OPTIONS[@]}" --q-only-dir "$ROOT/Q_only"
PHASE=complete
printf '{"state":"complete","phase":"complete","commit":"072b561","PYTHONUTF8":1}\n' > ./external-home/data_org_pipeline_utf8_072b561_progress.json
