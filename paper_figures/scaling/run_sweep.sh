#!/bin/zsh
# Reproducible driver for the Fig 1D scaling sweep (EFGP vs SGPR-49/1024 vs SKI).
#
# Recommended use: run on a FRESHLY-REBOOTED laptop with no other apps open, so every fit sees the
# same quiet machine (this is what makes the timings a fair, comparable comparison and gives the
# memory-heavy baselines (SKI@250k ~11.7GB, SGPR-1024@500k ~14GB) enough free RAM to complete).
#
#   zsh paper_figures/scaling/run_sweep.sh            # full fresh run (wipes scaling_data.json)
#   RESUME_ONLY=1 zsh paper_figures/scaling/run_sweep.sh   # keep existing results, run only missing
#
# Guards: RSS cap (14GB) + 1s poll kill a worker before a system OOM; the swap-growth ceiling is
# OFF by default (it measured system-wide swap and falsely OOM'd fits that actually fit). Self-heals
# if the driver dies. Writes results after every fit and regenerates Fig 1 at the end.
set -u
REPO=${0:A:h}/../..            # repo root (paper_figures/scaling/run_sweep.sh -> repo)
REPO=${REPO:A}
PY=$HOME/myenv/bin/python
OUT=$REPO/paper_figures/scaling/scaling_data.json
LOG=$REPO/paper_figures/scaling/sweep.log
cd $REPO
: > "$LOG"

if [[ "${RESUME_ONLY:-0}" != "1" ]]; then
  rm -f "$OUT"                 # fresh run
fi

# uniform settings for EVERY fit; swap-ceiling OFF (rely on RSS cap + fast poll).
# RSS cap 15GB (of 17): high enough that the memory-heaviest fit that can still fit on a quiesced
# laptop (SGPR-1024 @500k ~14GB) gets a real attempt; the driver kills a worker cleanly above it
# (before a system OOM), and run_phase self-heals if a fit ever crashes the driver.
export BENCH_RESUME=1 BENCH_RSS_LIMIT_GB=15 BENCH_POLL_S=1 BENCH_LOAD_GATE=1.5 BENCH_COOLDOWN_S=15

missing () {  # $1=methods csv  $2=sizes csv  -> exit 1 if any (method,T) missing from OUT
  $PY - "$OUT" "$1" "$2" <<'PYEOF'
import json,sys,os
out,methods,sizes=sys.argv[1],sys.argv[2].split(","),[int(x) for x in sys.argv[3].split(",")]
have=set()
if os.path.exists(out):
    d=json.load(open(out)); have={(r["method"],r["T"]) for r in d["results"] if r.get("eval")=="test"}
miss=[(m,t) for m in methods for t in sizes if (m,t) not in have]
print("MISSING",miss); sys.exit(1 if miss else 0)
PYEOF
}

force_oom () {  # record any still-missing target as oom (only if the driver repeatedly crashed on it)
  $PY - "$OUT" "$1" "$2" <<'PYEOF'
import json,sys,os
out,methods,sizes=sys.argv[1],sys.argv[2].split(","),[int(x) for x in sys.argv[3].split(",")]
d=json.load(open(out)) if os.path.exists(out) else {"config":{},"results":[]}
res=d["results"]; have={(r["method"],r["T"]) for r in res if r.get("eval")=="test"}
for m in methods:
  for t in sizes:
    if (m,t) not in have:
      res.append({"method":m,"T":t,"status":"oom","eval":"test","oom_reason":"driver_crash_forced",
                  "note":"forced oom after repeated driver failure"}); print("FORCED_OOM",m,t)
json.dump({"config":d.get("config",{}),"results":res},open(out,"w"),indent=2)
PYEOF
}

run_phase () {  # $1=methods $2=sizes $3=min_free_gb $4=gate_wait_s $5=label
  export BENCH_METHODS=$1 BENCH_SIZES=$2 BENCH_MIN_FREE_GB=$3 BENCH_GATE_MAX_WAIT_S=$4
  echo "=== PHASE $5 : methods=$1 sizes=$2  $(date) ===" >> $LOG
  local a
  for a in 1 2; do
    $PY $REPO/paper_figures/scaling/run_benchmark.py >> $LOG 2>&1
    if missing "$1" "$2" >> $LOG 2>&1; then echo "phase $5 complete ($(date))" >> $LOG; return; fi
    echo "phase $5 attempt $a incomplete (driver died?); retry in 30s" >> $LOG; sleep 30
  done
  force_oom "$1" "$2" >> $LOG 2>&1
}

echo "############ SWEEP START $(date) ############" >> $LOG
# priority order: fast/valuable first, heaviest last
run_phase efgp     "10000,100000,250000,500000,1000000" 2.5 300 "efgp-all"
run_phase sgpr49   "10000,100000,250000,500000,1000000" 2.5 300 "sgpr49-all"
run_phase sgpr1024 "10000,100000,250000"                3   600 "sgpr1024-light"
run_phase ski      "10000,100000,250000"                3   600 "ski-through-250k"
run_phase sgpr1024 "500000"                             3   600 "sgpr1024-500k"
run_phase "efgp,sgpr49,sgpr1024,ski" "10000,100000,250000,500000,1000000" 2.5 300 "fill-dropouts"

echo "############ ALL PHASES DONE $(date) ############" >> $LOG
$PY $REPO/paper_figures/fig1_overview.py >> $LOG 2>&1 && echo "FIGURE_REGENERATED" >> $LOG
$PY - "$OUT" >> $LOG 2>&1 <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); res=[r for r in d["results"] if r.get("eval")=="test"]
order={"efgp":0,"sgpr49":1,"sgpr1024":2,"ski":3}; res.sort(key=lambda r:(order.get(r["method"],9),r["T"]))
print("\nFINAL RESULTS:")
for r in res:
    if r.get("status")=="ok":
        _e=r.get("rmse", r.get("nrmse"))
        print(f"  {r['method']:9} {r['T']:>9,}  ok    t={r['time']:.2f}s  rmse={_e:.5f}  "
              f"peakRSS={r.get('peak_rss_gb',0):.1f}GB  minAvail={r.get('min_avail_gb')}  "
              f"swapDelta={r.get('swap_delta_mb')}MB")
    else:
        print(f"  {r['method']:9} {r['T']:>9,}  {r.get('status'):<7} {r.get('oom_reason') or r.get('note','')}")
PYEOF
echo "SWEEP_COMPLETE $(date)" >> $LOG
