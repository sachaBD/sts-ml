# usage: waitr.sh r  -> waits up to ~9.5min for round r finish; prints state
r=$1; R=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r$(printf %02d $r)
last=bench-policy.done; [ $((r%3)) = 0 ] && last=compare.done
for i in $(seq 1 57); do [ -e $R/$last ] && { echo ROUNDDONE; exit; }; kill -0 $(cat experiments/champ-oracle-exit/exit-c.pid) 2>/dev/null || { echo DRIVERDEAD; exit; }; sleep 10; done
echo "waiting $(date -u +%T) newest: $(ls -t $R 2>/dev/null | head -1) $(find runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r* -newermt '-20 minutes' 2>/dev/null | head -1)"
