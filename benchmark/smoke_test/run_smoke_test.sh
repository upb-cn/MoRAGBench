#!/bin/bash
# MoRAGBench smoke test: build + install the cli app, run all index methods on one dataset on the phone,
# then compare with the August numbers. Run from anywhere:  bash benchmark/smoke_test/run_smoke_test.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LOG="$HERE/smoke_test.log"
exec > >(tee "$LOG") 2>&1
echo "=== MoRAGBench smoke test — $(date) ==="

# --- tools ---
ADB="${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools/adb"
[ -x "$ADB" ] || ADB="$(command -v adb || true)"
[ -x "$ADB" ] || { echo "ERROR: adb not found"; exit 1; }
export PATH="$(dirname "$ADB"):$PATH"
if [ -z "$JAVA_HOME" ] && [ -d "/Applications/Android Studio.app/Contents/jbr/Contents/Home" ]; then
  export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
fi
PY="$(command -v python3.11 || true)"
[ -n "$PY" ] || { echo "ERROR: python3.11 not found (needed by numpy 2.4 / pandas 3)"; exit 1; }

# --- phone ---
SERIAL="$("$ADB" devices | awk 'NR>1 && $2=="device"{print $1; exit}')"
[ -n "$SERIAL" ] || { echo "ERROR: no phone found. Check USB cable + 'Allow USB debugging' on the phone."; "$ADB" devices; exit 1; }
echo "Phone: $SERIAL ($("$ADB" -s "$SERIAL" shell getprop ro.product.model | tr -d '\r'))"

REPEATS="${REPEATS:-3}"          # query runs per method (each loads the saved index), e.g. REPEATS=5
COOLDOWN="${COOLDOWN:-20}"       # minimum seconds between runs
MAX_TEMP="${MAX_TEMP:-34.0}"     # wait until the battery is at or below this (C) before each run
MAX_WAIT="${MAX_WAIT:-300}"      # but never wait longer than this (s) for one cool-down
A() { "$ADB" -s "$SERIAL" "$@"; }
batt_temp() { A shell dumpsys battery | awk '/temperature/{printf "%.1f", $2/10; exit}'; }
cool_down() {   # wait until the phone is cool enough, so every run starts from similar conditions
  local waited=0 t
  t=$(batt_temp)
  while awk -v t="$t" -v m="$MAX_TEMP" 'BEGIN{exit !(t>m)}'; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "  (still $t C after ${MAX_WAIT}s - starting anyway)"; break; fi
    echo "  cooling down: battery $t C > $MAX_TEMP C, waiting..."; sleep 30; waited=$((waited+30)); t=$(batt_temp)
  done
  echo "  battery $t C - starting"
}

# --- phone conditions check (warnings only) ---
echo; echo "=== Phone conditions ==="
PS=$(A shell settings get global low_power | tr -d '\r')
[ "$PS" = "1" ] && echo "WARNING: Power saving is ON - turn it off for fair numbers." || echo "Power saving: off"
A shell dumpsys battery | grep -E 'AC powered|USB powered|level|temperature' | sed 's/^ */  /'
echo "  CPU cores (max MHz): $(for i in $(seq 0 $(( $(A shell nproc | tr -d '\r') - 1 ))); do printf "cpu%s=%s " $i $(( $(A shell cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq | tr -d '\r') / 1000 )); done)"
A shell cat /proc/meminfo | awk '/MemTotal|MemAvailable/{printf "  %s %.1f GB\n",$1,$2/1048576}'
# keep the screen on while plugged into USB for the duration of the test, restore afterwards
A shell svc power stayon usb; A shell input keyevent KEYCODE_WAKEUP
trap 'A shell svc power stayon false >/dev/null 2>&1' EXIT
echo "Screen will stay on during the test (restored when the script ends)."

# --- 1. build + install cli app ---
echo; echo "=== [1/4] Building and installing the cli app (first build can take several minutes) ==="
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "SKIP_BUILD=1 -> using the app already on the phone"
else
  cd "$ROOT/android"; chmod +x gradlew
  ./gradlew :cli:installDebug
  echo "Letting the phone settle for 60 s after install (Android optimises new apps in the background)..."
  sleep 60
fi

# --- 2. python env (separate from your old 3.9 venv) ---
echo; echo "=== [2/4] Python environment ==="
VENV="$ROOT/.venv"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt" python-dotenv

# --- 3. benchmarks ---
echo; echo "=== [3/4] Running benchmarks on the phone ==="
cd "$ROOT/client"
DATASET="${DATASET:-fashion_mnist}"   # a folder in configs/, e.g. DATASET=ny_times
[ -d "$HERE/configs/$DATASET" ] || { echo "ERROR: no configs for dataset '$DATASET' in $HERE/configs/"; exit 1; }
METHODS="${METHODS:-flat ivf hnsw ivfpq binary binaryivf}"   # e.g. METHODS="flat hnsw" to test fewer
THREADS="${THREADS:-1}"          # FAISS CPU threads; 0 = all cores (old behaviour)
FG_UI="${FG_UI:-1}"              # 1 = show the app's benchmark screen during runs (fast cores); 0 = old behaviour
export MORAG_FOREGROUND_UI="$FG_UI"
PIN="${PIN:-none}"               # core type: fastest | mid | slowest | none (Android overrides pinning on most phones)
RUN_TAG="$(date +%Y%m%d_%H%M)_${DATASET}_t${THREADS}_fg${FG_UI}"
OUT="$HERE/results/$RUN_TAG"
mkdir -p "$OUT/configs"
# The seed is part of the index cache key (for HNSW/Flat it changes nothing else; for IVF-type
# methods it picks the training sample). A fresh seed per session means phase 1 always builds a
# new index, and phase 2 loads exactly that index.
SESSION_SEED="${SESSION_SEED:-$(( $(date +%s) % 1000000 ))}"
for m in $METHODS; do   # copy configs with the thread setting, so the folder records exactly what ran
  "$VENV/bin/python" -c "import json,sys; c=json.load(open(sys.argv[1])); t=int(sys.argv[3]); p=sys.argv[4]; c['faiss']['num_threads']=t if t>0 else None; c['faiss']['cpu_pin']=None if p=='none' else p; c['faiss']['use_cache']=True; c['ann_dataset']['seed']=int(sys.argv[5]); json.dump(c,open(sys.argv[2],'w'),indent=2)" \
      "$HERE/configs/$DATASET/${DATASET}_$m.json" "$OUT/configs/${DATASET}_$m.json" "$THREADS" "$PIN" "$SESSION_SEED"
done
echo "Results folder: $OUT  (threads: $THREADS, benchmark screen: $FG_UI, seed $SESSION_SEED)"

# Phase 1 - SETUP: build each index once and save it. Measures setup time, build memory, file size.
# (Its query numbers are not used: a long build heats the fastest core, so queries straight after
#  would run on a slowed core - for HNSW on ny_times about 40% slower.)
echo; echo "=== Phase 1: build each index once (setup cost) ==="
for m in $METHODS; do
  echo; echo "----- build : $DATASET / $m -----"
  cool_down
  A shell input keyevent KEYCODE_WAKEUP
  "$VENV/bin/python" main.py --config "$OUT/configs/${DATASET}_$m.json" \
      --output_path "$OUT/${DATASET}_${m}_build" --device "$SERIAL"
done

# Phase 2 - QUERIES: load the saved index (under a second) on a cooled phone and measure searches.
# Repeats only redo the queries, so every method is measured at full core speed.
echo; echo "=== Phase 2: query runs on the saved indexes ($REPEATS per method, interleaved, start at <= $MAX_TEMP C) ==="
for (( r=1; r<=REPEATS; r++ )); do
  for m in $METHODS; do
    echo; echo "----- queries $r/$REPEATS : $DATASET / $m -----"
    cool_down
    A shell input keyevent KEYCODE_WAKEUP
    "$VENV/bin/python" main.py --config "$OUT/configs/${DATASET}_$m.json" \
        --output_path "$OUT/${DATASET}_${m}_run$r" --device "$SERIAL"
    sleep "$COOLDOWN"
  done
done

# --- cache check: a different ef_search must reuse the cached HNSW index and still apply its own ef_search ---
if [ "${CACHE_CHECK:-0}" = "1" ]; then   # phase 2 already proves loading; kept as an option
  echo; echo "=== Cache check (HNSW ef_search 40 builds + saves, ef_search 80 must load it) ==="
  # Give the check its own cache key, so it can never pick up an index saved earlier
  # (deleting the app's cache from adb is not reliable on newer Android).
  # The seed is part of the cache key; for HNSW it changes nothing else.
  CHECK_SEED=$(( $(date +%s) % 1000000 ))
  echo "  using fresh cache key (seed $CHECK_SEED)"
  for ef in 40 80; do
    "$VENV/bin/python" -c "import json,sys; c=json.load(open(sys.argv[1])); c['faiss']['use_cache']=True; c['faiss']['config']['ef_search']=int(sys.argv[3]); c['ann_dataset']['seed']=int(sys.argv[4]); json.dump(c,open(sys.argv[2],'w'),indent=2)" \
        "$OUT/configs/${DATASET}_hnsw.json" "$OUT/configs/cache_check_ef$ef.json" "$ef" "$CHECK_SEED"
    cool_down
    "$VENV/bin/python" main.py --config "$OUT/configs/cache_check_ef$ef.json" --output_path "$OUT/cache_check_ef$ef" --device "$SERIAL"
  done
fi

# --- 4. compare ---
echo; echo "=== [4/4] Summary ==="
"$VENV/bin/python" "$HERE/compare.py" "$OUT"
echo; echo "Done. Full log: $LOG"
