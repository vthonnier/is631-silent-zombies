#!/usr/bin/env bash
# Auto-restart driver for the resume-aware benchmark. Relaunches the run whenever
# the remote is reachable (killing any stale GPU python first) until the JSON
# reports ok=True. Each (re)launch resumes from the last saved pair, so VPN drops
# cost a restart, not progress. Dumps the final JSON at the end.
OPTS="-o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o IdentityAgent=none -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519"
H="virgile@192.168.1.168"
PY='C:\Users\Virgile\latent_mas\venv\Scripts\python.exe'
SCRIPT='C:\Users\Virgile\latent_mas\run_m6_benchmark.py'
JSON='C:\Users\Virgile\latent_mas\out\m6_benchmark.json'
readfield() { ssh $OPTS "$H" "powershell -NoProfile -Command \"try { (Get-Content $JSON -Raw | ConvertFrom-Json).$1 } catch { 'none' }\"" 2>/dev/null | tr -d '\r' | tr -d ' '; }

for i in $(seq 1 20); do
  r=0
  for j in $(seq 1 40); do
    if ssh $OPTS "$H" 'echo A' 2>/dev/null | grep -q A; then r=1; break; fi
    sleep 15
  done
  if [ $r -eq 0 ]; then echo "[w] remote unreachable, stopping"; break; fi
  ok=$(readfield ok); cnt=$(readfield pairs.Count)
  echo "[w] iter $i: ok=$ok pairs=$cnt ($(date +%H:%M:%S))"
  if [ "$ok" = "True" ]; then echo "[w] COMPLETE"; break; fi
  ssh $OPTS "$H" 'powershell -NoProfile -Command "Get-Process python* -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id }"' 2>/dev/null
  sleep 3
  echo "[w] launching run ($(date +%H:%M:%S))"
  ssh $OPTS "$H" "$PY -u $SCRIPT" 2>&1 | tail -4
  echo "[w] run ended ($(date +%H:%M:%S))"
done
echo "===== final m6_benchmark.json ====="
ssh $OPTS "$H" "powershell -NoProfile -Command \"Get-Content $JSON -Raw\"" 2>/dev/null
echo "[w] done"
