# waits per round; appends LOG row; exits on alert (rc2), driver death (rc1), or after a real-bench round (rc0, prints round)
cd /home/sborowsk/project/sts_combat_rl; E=experiments/champ-oracle-exit
r=${1:-10}
while [ $r -le 18 ]; do
  R=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r$(printf %02d $r)
  last=bench-policy.done; [ $((r%3)) = 0 ] && last=compare.done
  while [ ! -e $R/$last ]; do kill -0 $(cat $E/exit-c.pid) 2>/dev/null || { echo DRIVERDEAD r$r; exit 1; }; sleep 20; done
  .venv/bin/python $E/rowc.py $r > /tmp/c2_r$r; head -1 /tmp/c2_r$r >> $E/LOG.md
  P=$(grep -o 'bench-policy [0-9]*/409=[0-9.]*' /tmp/c2_r$r | grep -o '=[0-9.]*' | tr -d =); O=$(grep -o 'bench-oracle [0-9]*/409=[0-9.]*' /tmp/c2_r$r | grep -o '=[0-9.]*' | tr -d =)
  A=$(awk -v p=$P -v o=$O 'BEGIN{print (p<.22||o<.37)?"REG":"ok"}'); grep -q 'ALERTS: \[.*[a-z]' /tmp/c2_r$r && A="$A+$(grep ALERTS /tmp/c2_r$r)"
  echo "r$r $A"; [ "${A#ok}" != "$A" ] && [ "$A" = ok ] || { cat /tmp/c2_r$r; exit 2; }
  [ $((r%3)) = 0 ] && { cat /tmp/c2_r$r; exit 0; }
  r=$((r+1))
done
