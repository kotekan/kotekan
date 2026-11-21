#!/bin/bash
# Fold the GuoWang beam-dump reductions into the 1202 satellite library and commit it when it
# changes, so the branch history records when each satellite was found, went quiet or came back.
# Cron, on the gnss VM.
#
# Each run:
#   1. copies gw_reduce's reduce_log.jsonl from each of SOURCES (cf06 and cf05's old loop, frozen;
#      corrcal, the 11-07 catch-up; cf05's loop since); a failed
#      copy keeps the previous one
#   2. guowang_db.py update with those logs and the published db_proposals.jsonl
#   3. commits the library alone (git commit --only) on kv/gw1202 in the gw1202-wt worktree
#
# Nothing is pushed. Log: /var/tmp/gnss-logs/guowang_db_sync.log
set -u
WT=/home/kvand/gnss/gw1202-wt
DB=python/scripts/gnss/guowang/guowang_1202_db.json
C=/var/tmp/gnss-guowang-db
PY=/home/kvand/gnss/venv/bin/python3
BRANCH=kv/gw1202
now() { date -u +%FT%TZ; }

mkdir -p "$C"
exec 9>"$C/lock"
flock -n 9 || { echo "$(now) another run holds the lock"; exit 0; }

# a new F-engine start goes into the reducer's frame0 list, so its windows are placed in time
"$PY" "$WT/scripts/gnss/site/gw_frame0_sync.py" 2>&1 | sed "s/^/$(now) frame0: /"

# host[:dir]; dir defaults to /var/tmp/kvand-gw1202. cf05:/var/tmp/kvand-gw1202b is the cf05 loop
# since the 11-07 catch-up (gw_catchup.py --follow, its own namespace).
SOURCES="cf06 cf05 corrcal cf05:/var/tmp/kvand-gw1202b"
for src in $SOURCES; do
    h=${src%%:*}; d=/var/tmp/kvand-gw1202; tag=$h
    if [ "$src" != "$h" ]; then d=${src#*:}; tag=${h}_$(basename "$d"); fi
    if timeout 120 scp -q -o BatchMode=yes -o ConnectTimeout=20 \
        "$h:$d/reduce_log.jsonl" "$C/reduce_log_$tag.jsonl.tmp"; then
        mv "$C/reduce_log_$tag.jsonl.tmp" "$C/reduce_log_$tag.jsonl"
    else
        echo "$(now) $src: copy failed, using the previous copy"
    fi
done

cd "$WT" || exit 1
if [ "$(git symbolic-ref --short HEAD 2>/dev/null)" != "$BRANCH" ]; then
    echo "$(now) $WT is not on $BRANCH: nothing done"
    exit 1
fi

logs=()
for src in $SOURCES; do
    h=${src%%:*}; tag=$h
    [ "$src" != "$h" ] && tag=${h}_$(basename "${src#*:}")
    [ -f "$C/reduce_log_$tag.jsonl" ] && logs+=(--reduce-log "$C/reduce_log_$tag.jsonl")
done
"$PY" python/scripts/gnss/guowang/guowang_db.py --db "$DB" update "${logs[@]}" \
    --proposals /home/kvand/gnss/builds/gw1202/db_proposals.jsonl \
    --message-out "$C/msg.txt" > "$C/update.out" 2>&1 || {
    echo "$(now) update failed:"; cat "$C/update.out"; exit 1; }

if git diff --quiet -- "$DB"; then
    echo "$(now) no change ($(head -1 "$C/update.out"))"
elif git commit -q --only -F "$C/msg.txt" -- "$DB"; then
    echo "$(now) committed $(git rev-parse --short HEAD): $(head -1 "$C/msg.txt")"
else
    echo "$(now) commit failed"
    exit 1
fi
