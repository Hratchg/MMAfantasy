#!/usr/bin/env bash
# Common MMAfantasy corpus queries in one shot. Host has no psql; goes via the
# mmafantasy-db-1 container. Requires DOCKER_HOST set to the colima socket.
set -euo pipefail
: "${DOCKER_HOST:=unix:///Users/hratchghanime/.colima/default/docker.sock}"
export DOCKER_HOST
PSQL=(docker exec mmafantasy-db-1 psql -U ufc -d ufc_prediction -tA)

echo "=== corpus counts ==="
"${PSQL[@]}" -c "select 'fighters',count(*) from fighters union all select 'fights',count(*) from fights union all select 'fight_odds',count(*) from fight_odds union all select 'events',count(*) from events union all select 'round_stats',count(*) from round_stats;"

echo "=== event sources + inflation factor (all-source vs ufcstats) ==="
"${PSQL[@]}" -c "select source, count(*) from events group by source order by 2 desc;"
"${PSQL[@]}" -c "with fs as (select e.source from fights f join events e on f.event_id=e.id where f.winner_id is not null) select (select count(*) from fs) as all_source_fights, (select count(*) from fs where source='ufcstats') as ufcstats_fights, round((select count(*) from fs)::numeric/nullif((select count(*) from fs where source='ufcstats'),0),3) as inflation_factor;"

echo "=== odds coverage: new (>=2026-05-01) vs old ==="
"${PSQL[@]}" -c "select e.date >= '2026-05-01' as new_event, count(distinct f.id) fights, count(distinct fo.fight_id) fights_with_odds from fights f join events e on f.event_id=e.id left join fight_odds fo on fo.fight_id=f.id group by 1 order by 1;"

echo "=== latest 8 events ==="
"${PSQL[@]}" -c "select date, name from events order by date desc limit 8;"
