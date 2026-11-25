#!/usr/bin/env bash
# Patient auto-resume for the benchmark across a flaky VPN. EACH iteration first
# waits (up to ~2h) for the remote to be reachable, then kills stale GPU python and
# relaunches the resume-aware run (which skips completed pairs). Loops until the JSON
# reports ok=True. Survives long, frequent downtimes.
OPTS="-o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o IdentityAgent=none -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519"
H="virgile@192.168.1.168"
PY='C:\Users\Virgile\latent_mas\venv\Scripts\python.exe'
SCRIPT='C:\Users\Virgile\latent_mas\run_m6_benchmark.py'
JSON='C:\Users\Virgile\latent_mas\out\m6_benchmark.json'
alive() { ssh $OPTS "$H" 'echo A' 2>/dev/null | grep -q A; }
readfield() { ssh $OPTS "$H" "powershell -NoProfile -Command \"try { (Get-Content $JSON -Raw | ConvertFrom-Json).$1 } catch { 'none' }\"" 2>/dev/null | tr -d '\r' | tr -d ' '; }

for i in $(seq 1 30); do
  # patient reconnect wait (up to ~2h this iteration)
  got=0
  for k in $(seq 1 120); do
    if alive; then got=1; break; fi
    sleep 60
  done
  if [ $got -eq 0 ]; then echo "[w] remote down >2h at iter $i; stopping ($(date +%H:%M:%S))"; break; fi
  ok=$(readfield ok); cnt=$(readfield pairs.Count)
  echo "[w] iter $i: reachable, ok=$ok pairs=$cnt ($(date +%H:%M:%S))"
  if [ "$ok" = "True" ]; then echo "[w] COMPLETE"; break; fi
  ssh $OPTS "$H" 'powershell -NoProfile -Command "Get-Process python* -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id }"' 2>/dev/null
  sleep 3
  echo "[w] launching run ($(date +%H:%M:%S))"
  ssh $OPTS "$H" "$PY -u $SCRIPT" 2>&1 | tail -4
  echo "[w] run ended ($(date +%H:%M:%S))"
done
echo "===== final m6_benchmark.json ====="
ssh $OPTS "$H" "powershell -NoProfile -Command \"Get-Content $JSON -Raw\"" 2>/dev/null
echo "[w] patient watcher done ($(date +%H:%M:%S))"
