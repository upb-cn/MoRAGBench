#!/bin/bash
# MoRAGBench smoke test: build + install the cli app, run fashion_mnist Flat/IVF/HNSW on the phone,
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

REPEATS="${REPEATS:-3}"          # runs per method, e.g.  REPEATS=5 bash ...
COOLDOWN="${COOLDOWN:-30}"       # seconds to wait between runs
A() { "$ADB" -s "$SERIAL" "$@"; }

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
THREADS="${THREADS:-1}"          # FAISS CPU threads; 0 = all cores (old behaviour)
PIN="${PIN:-mid}"                # core type: fastest | mid | slowest | none
RUN_TAG="$(date +%Y%m%d_%H%M)_t${THREADS}_${PIN}"
OUT="$HERE/results/$RUN_TAG"
mkdir -p "$OUT/configs"
for m in flat ivf hnsw; do   # copy configs with the thread setting, so the folder records exactly what ran
  "$VENV/bin/python" -c "import json,sys; c=json.load(open(sys.argv[1])); t=int(sys.argv[3]); p=sys.argv[4]; c['faiss']['num_threads']=t if t>0 else None; c['faiss']['cpu_pin']=None if p=='none' else p; json.dump(c,open(sys.argv[2],'w'),indent=2)" \
      "$HERE/configs/fashion_mnist_$m.json" "$OUT/configs/fashion_mnist_$m.json" "$THREADS" "$PIN"
done
echo "Results folder: $OUT  ($REPEATS runs per method, methods interleaved, FAISS threads: $THREADS, core type: $PIN)"
for r in $(seq 1 "$REPEATS"); do
  for m in flat ivf hnsw; do
    echo; echo "----- run $r/$REPEATS : fashion_mnist / $m  (battery temp $(A shell dumpsys battery | awk '/temperature/{print $2/10}') C) -----"
    A shell input keyevent KEYCODE_WAKEUP
    "$VENV/bin/python" main.py --config "$OUT/configs/fashion_mnist_$m.json" \
        --output_path "$OUT/fashion_mnist_${m}_run$r" --device "$SERIAL"
    sleep "$COOLDOWN"
  done
done

# --- 4. compare ---
echo; echo "=== [4/4] Summary ==="
"$VENV/bin/python" "$HERE/compare.py" "$OUT"
echo; echo "Done. Full log: $LOG"
