#!/usr/bin/env bash
#
# run_integration_suite.sh
#
# Level 1 of the testing ladder: the PAID tier, run deliberately.
#
# For every exercise this script goes into the directory, syncs dependencies,
# brings up any services the exercise depends on, runs the exercise's own
# walkthrough (uv run python main.py walk) with real model calls so the
# actual ANSWERS are produced and captured, then runs the pytest suite with
# every opt-in paid gate enabled, stops the services again, and prints one
# report at the end.
#
# This costs real money by design. Each run is a few cents per exercise that
# calls models; 03 is the most expensive on a cold database because it
# indexes its corpus. The script asks once before spending unless --yes.
#
# Ground rules, matching the rest of the repository:
#   - every exercise is a standalone application: its .env stays where it is
#     and is read in place; nothing is centralised
#   - directory names are hardcoded; a missing folder is reported, never
#     silently skipped
#   - containers are STOPPED afterwards, never removed, and volumes are
#     always kept - they hold indexes that cost money to build
#   - a failing exercise is data, not a reason to stop the sweep
#
# Output: full transcript in logs/integration_run_YYYY-MM-DD_HHMMSS.log at
# the project root (the logs folder is created if absent). The screen shows
# progress, the tail of each walkthrough's answers, and the final table.
#
# Options:
#     --dry-run           show the plan, run nothing, spend nothing
#     --only 03,07        run just these exercises (by number)
#     --skip 03           skip these exercises
#     --timeout 900       seconds allowed per step (default 900)
#     --no-sync           assume dependencies are installed, skip uv sync
#     --keep-services     leave services running afterwards (for debugging)
#     --yes               do not ask for spend confirmation

# Deliberately no 'set -e'. A failing exercise is data, not a reason to stop.
set -uo pipefail

TIMEOUT_SECONDS=900
DRY_RUN="no"
DO_SYNC="yes"
KEEP_SERVICES="no"
ASSUME_YES="no"
ONLY=""
SKIP=""

START_EPOCH=$(date +%s)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN="yes"; shift ;;
        --no-sync)        DO_SYNC="no"; shift ;;
        --keep-services)  KEEP_SERVICES="yes"; shift ;;
        --yes)            ASSUME_YES="yes"; shift ;;
        --only)           ONLY="${2:-}"; shift 2 ;;
        --skip)           SKIP="${2:-}"; shift 2 ;;
        --timeout)        TIMEOUT_SECONDS="${2:-900}"; shift 2 ;;
        -h|--help)        sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                printf 'Unknown option: %s (try --help)\n' "$1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging. The transcript lives in logs/ at the project root, timestamped so
# runs never overwrite each other and results can be compared over time.
# ---------------------------------------------------------------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/integration_run_$(date '+%Y-%m-%d_%H%M%S').log"
: > "$LOG"

log() {
    printf '%s\n' "$*" | tee -a "$LOG"
}

log_only() {
    printf '%s\n' "$*" >> "$LOG"
}

rule() {
    log "------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Cleanup. Runs on normal exit and on interrupt. Stop only - containers and
# volumes are never removed by this script.
# ---------------------------------------------------------------------------
STARTED_COMPOSE_IN=""
# Space-separated list: this suite can start more than one long-lived Neo4j
# container (03's graph container and 13's temporal container).
STARTED_NEO4J_CONTAINERS=""

cleanup() {
    local exit_code=$?
    if [ -n "$STARTED_COMPOSE_IN" ] && [ "$KEEP_SERVICES" = "no" ]; then
        printf '\nStopping Docker services in %s (containers and volumes kept)...\n' \
            "$STARTED_COMPOSE_IN" | tee -a "$LOG"
        ( cd "$STARTED_COMPOSE_IN" && docker compose stop ) >> "$LOG" 2>&1
        STARTED_COMPOSE_IN=""
    fi
    if [ -n "$STARTED_NEO4J_CONTAINERS" ] && [ "$KEEP_SERVICES" = "no" ]; then
        # Stop, never remove: the containers and their volumes belong to the
        # exercises' own setup scripts and may hold paid-for indexes.
        for started_container in $STARTED_NEO4J_CONTAINERS; do
            printf '\nStopping Neo4j container %s (container and volume kept)...\n' \
                "$started_container" | tee -a "$LOG"
            docker stop "$started_container" >> "$LOG" 2>&1
        done
        STARTED_NEO4J_CONTAINERS=""
    fi
    exit $exit_code
}
trap cleanup EXIT
trap 'printf "\nInterrupted.\n" | tee -a "$LOG"; exit 130' INT TERM

# ---------------------------------------------------------------------------
# uv runs each project in its own venv; unexport any inherited one.
# ---------------------------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
    log "Note: deactivating inherited virtualenv ${VIRTUAL_ENV} for child processes"
    unset VIRTUAL_ENV
fi
unset PYTHONHOME 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Paid gates. This is the point of Level 1: every opt-in expensive test in
# the repository runs. When a new exercise grows a gate, export it here so
# this script stays the single switch for the paid tier.
# ---------------------------------------------------------------------------
export GRAPHRAG_RUN_INDEX=1
export TEMPORAL_RUN_INDEX=1

# ---------------------------------------------------------------------------
# The exercises, named explicitly, with their walkthrough commands. Nothing
# is discovered; a renamed or missing folder is reported, not dropped.
# Every exercise reads its own .env in place - that independence is the
# point of the repository, so nothing here touches or centralises env files.
# ---------------------------------------------------------------------------
EXERCISES=(
    "01_stale_index_replay"
    "02_hybrid_rerank_retrieval"
    "03_graphrag_fruit_graph"
    "04_rag_eval_ragas"
    "05_pii_erasure_replay"
    "06_injection_defence_llm_guard"
    "07_model_routing_litellm"
    "08_bounded_agent_langgraph"
    "09_mcp_tools_and_scan"
    "10_observability_langfuse_otel"
    "11_semantic_chunking_two_llms"
    "12_disk_ann_diskannpy"
    "13_temporal_bitemporal_graphiti"
)

# Service needs per exercise, mirrored from run_all_exercises.sh:
#   01 - Postgres via its own docker-compose.yml (service name: db)
#   03 - the long-lived Neo4j container its setup script manages
# The names below must match 03's setup script. Never removed here.
NEO4J_CONTAINER="neo4j-fruit-graph"
NEO4J_VOLUME="neo4j-fruit-graph-data"
NEO4J_IMAGE="neo4j:5.26"

# 13_temporal_bitemporal_graphiti has its OWN Neo4j container on shifted
# ports (7475/7688), managed by its setup script. Same rules: reuse, start
# if stopped, create if absent, stop only, never remove.
TEMPORAL_CONTAINER="neo4j-temporal-graph"
TEMPORAL_VOLUME="neo4j-temporal-graph-data"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
log "=================================================================="
log " Level 1 integration suite - PAID tier, real model calls"
log " Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log " Root:    $(pwd)"
log " Log:     $LOG"
log "=================================================================="
log ""

FATAL="no"

if ! command -v uv >/dev/null 2>&1; then
    log "FATAL: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    FATAL="yes"
else
    log "uv        $(uv --version 2>/dev/null | head -1)"
fi

DOCKER_OK="no"
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        DOCKER_OK="yes"
        log "docker    available and running"
    else
        log "docker    installed but the daemon is not responding"
        log "          exercises needing services (01, 03) will be skipped, not failed"
    fi
else
    log "docker    not installed - exercises needing services (01, 03) will be skipped"
fi

MISSING_COUNT=0
PRESENT_COUNT=0
for exercise in "${EXERCISES[@]}"; do
    if [ -d "$exercise" ]; then
        PRESENT_COUNT=$((PRESENT_COUNT + 1))
    else
        MISSING_COUNT=$((MISSING_COUNT + 1))
        log "missing   $exercise"
    fi
done
if [ "$PRESENT_COUNT" -eq 0 ]; then
    log ""
    log "FATAL: none of the named exercises exist here. Is this the project root?"
    FATAL="yes"
fi

if [ "$FATAL" = "yes" ]; then
    log ""
    log "Cannot continue."
    exit 2
fi

log ""
log "This run makes REAL model calls in every exercise that has a key in its"
log ".env: full walkthroughs plus the paid tests (GRAPHRAG_RUN_INDEX=1 and"
log "TEMPORAL_RUN_INDEX=1 are exported). Expect a few cents per exercise; 03"
log "and 13 cost the most when their databases are empty, and only cents"
log "when their indexes already exist."

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
wanted() {
    local name="$1"
    local number="${name%%_*}"

    if [ -n "$ONLY" ]; then
        case ",${ONLY}," in
            *",${number},"*) ;;
            *) return 1 ;;
        esac
    fi
    if [ -n "$SKIP" ]; then
        case ",${SKIP}," in
            *",${number},"*) return 1 ;;
        esac
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "yes" ]; then
    log ""
    log "Plan (dry run - nothing will be executed, nothing will be spent):"
    for exercise in "${EXERCISES[@]}"; do
        if ! wanted "$exercise"; then
            log "  filtered   $exercise"
            continue
        fi
        if [ ! -d "$exercise" ]; then
            log "  MISSING    $exercise"
            continue
        fi
        services=""
        if [ "$exercise" = "01_stale_index_replay" ]; then
            services=" [Postgres via compose]"
        fi
        if [ "$exercise" = "03_graphrag_fruit_graph" ]; then
            services=" [Neo4j container: $NEO4J_CONTAINER]"
        fi
        if [ "$exercise" = "13_temporal_bitemporal_graphiti" ]; then
            services=" [Neo4j container: $TEMPORAL_CONTAINER]"
        fi
        log "  would run  $exercise: walkthrough + paid tests$services"
    done
    log ""
    log "Dry run complete. Nothing was changed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Spend confirmation. One question, once, before any tokens burn.
# ---------------------------------------------------------------------------
if [ "$ASSUME_YES" != "yes" ]; then
    printf 'Proceed and spend real tokens? [y/N] '
    read -r answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) log "Aborted before spending anything."; exit 0 ;;
    esac
fi

# ---------------------------------------------------------------------------
# Results. Parallel arrays keyed by index.
# ---------------------------------------------------------------------------
RESULT_NAME=()
RESULT_STATUS=()
RESULT_DETAIL=()
RESULT_SECONDS=()

record() {
    RESULT_NAME+=("$1")
    RESULT_STATUS+=("$2")
    RESULT_DETAIL+=("$3")
    RESULT_SECONDS+=("$4")
}

# Append a captured output file to the log; echo its tail to the screen.
report_output() {
    local label="$1"
    local file="$2"
    local screen_lines="${3:-25}"

    log_only ""
    log_only "----- $label (full output) -----"
    cat "$file" >> "$LOG"
    log_only "----- end $label -----"

    if [ -s "$file" ]; then
        log ""
        log "  last $screen_lines lines of $label:"
        tail -n "$screen_lines" "$file" | sed 's/^/    /'
    fi
}

summarise_pytest() {
    local file="$1"
    local summary
    summary="$(grep -oE '[0-9]+ (passed|failed|skipped|error|errors|xfailed|deselected)' "$file" \
               | tr '\n' ',' | sed 's/,$//; s/,/, /g')"
    if [ -z "$summary" ]; then
        summary="no test summary found"
    fi
    printf '%s' "$summary"
}

# ---------------------------------------------------------------------------
# Run one exercise: sync, services up, WALKTHROUGH (the real answers), paid
# pytest, services stopped, verdict.
# ---------------------------------------------------------------------------
run_exercise() {
    local exercise="$1"
    local started ended elapsed
    local workdir="$exercise"
    local out
    out="$(mktemp)"

    started=$(date +%s)

    rule
    log "[$exercise]"

    # -- preflight for this exercise -------------------------------------
    if [ ! -d "$workdir" ]; then
        log "  MISSING - folder does not exist"
        record "$exercise" "MISSING" "folder not found on disk" "0"
        rm -f "$out"; return
    fi
    if [ ! -f "$workdir/pyproject.toml" ]; then
        log "  SKIP - no pyproject.toml"
        record "$exercise" "SKIP" "no pyproject.toml" "0"
        rm -f "$out"; return
    fi
    if [ ! -f "$workdir/main.py" ]; then
        log "  SKIP - no main.py to run a walkthrough from"
        record "$exercise" "SKIP" "no main.py" "0"
        rm -f "$out"; return
    fi

    # -- dependencies ----------------------------------------------------
    if [ "$DO_SYNC" = "yes" ]; then
        log "  syncing dependencies..."
        local sync_rc=0
        ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" uv sync ) > "$out" 2>&1 || sync_rc=$?
        if [ "$sync_rc" -ne 0 ]; then
            log "  ERROR - uv sync failed (exit $sync_rc). Dropping this exercise."
            report_output "uv sync" "$out"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "uv sync failed (exit $sync_rc)" "$elapsed"
            rm -f "$out"; return
        fi
        log_only "----- uv sync output -----"
        cat "$out" >> "$LOG"
    else
        log "  skipping uv sync (--no-sync)"
    fi

    # -- services: 01 Postgres via compose -------------------------------
    local uses_compose="no"
    if [ "$exercise" = "01_stale_index_replay" ]; then
        uses_compose="yes"
        if [ "$DOCKER_OK" != "yes" ]; then
            log "  SKIP - needs Postgres but docker is unavailable"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "SKIP" "docker unavailable, needs Postgres" "$elapsed"
            rm -f "$out"; return
        fi
        log "  starting Postgres (compose)..."
        local compose_rc=0
        ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
            docker compose up -d --wait db ) > "$out" 2>&1 || compose_rc=$?
        if [ "$compose_rc" -ne 0 ]; then
            log "  ERROR - could not start Postgres (exit $compose_rc). Dropping this exercise."
            report_output "docker compose up" "$out"
            # Stop whatever half-started; containers and volumes are kept.
            ( cd "$workdir" && docker compose stop ) >> "$LOG" 2>&1
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "docker compose up failed (exit $compose_rc)" "$elapsed"
            rm -f "$out"; return
        fi
        STARTED_COMPOSE_IN="$workdir"
        log_only "----- docker compose up output -----"
        cat "$out" >> "$LOG"
        log "  Postgres healthy"
    fi

    # -- services: 03 Neo4j container ------------------------------------
    # Mirrors run_all_exercises.sh: reuse the long-lived container, start it
    # if stopped, create it if absent, stop (never remove) afterwards only
    # if this run started it.
    local started_neo4j="no"
    local exercise_neo4j=""
    if [ "$exercise" = "03_graphrag_fruit_graph" ]; then
        if [ "$DOCKER_OK" != "yes" ]; then
            log "  SKIP - needs the Neo4j container but docker is unavailable"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "SKIP" "docker unavailable, needs Neo4j" "$elapsed"
            rm -f "$out"; return
        fi

        # The password comes from the exercise .env, which is always present.
        local neo4j_password
        neo4j_password="$(grep -E '^NEO4J_PASSWORD=' "$workdir/.env" | head -1 | cut -d= -f2-)"

        if docker ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
            log "  Neo4j container already running - it will be left running afterwards"
        elif docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
            log "  starting existing Neo4j container..."
            docker start "$NEO4J_CONTAINER" > "$out" 2>&1
            started_neo4j="yes"
        else
            log "  creating Neo4j container ($NEO4J_IMAGE, GDS + APOC, volume $NEO4J_VOLUME)..."
            docker run -d \
                --name "$NEO4J_CONTAINER" \
                -p 7474:7474 -p 7687:7687 \
                -e NEO4J_AUTH="neo4j/$neo4j_password" \
                -e NEO4J_PLUGINS='["graph-data-science","apoc"]' \
                -e NEO4J_dbms_security_procedures_unrestricted='gds.*,apoc.*' \
                -v "$NEO4J_VOLUME:/data" \
                "$NEO4J_IMAGE" > "$out" 2>&1
            started_neo4j="yes"
        fi
        exercise_neo4j="$NEO4J_CONTAINER"
        if [ "$started_neo4j" = "yes" ]; then
            STARTED_NEO4J_CONTAINERS="$STARTED_NEO4J_CONTAINERS $NEO4J_CONTAINER"
        fi

        log "  waiting for Neo4j to answer on bolt..."
        local neo4j_ready="no"
        local attempt=0
        while [ "$attempt" -lt 60 ]; do
            if docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$neo4j_password" \
                    "RETURN 1" >/dev/null 2>&1; then
                neo4j_ready="yes"
                break
            fi
            sleep 3
            attempt=$((attempt + 1))
        done
        if [ "$neo4j_ready" != "yes" ]; then
            log "  ERROR - Neo4j did not become ready (docker logs $NEO4J_CONTAINER)."
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "Neo4j container did not become ready" "$elapsed"
            rm -f "$out"; return
        fi
        if ! docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$neo4j_password" \
                "RETURN gds.version(), apoc.version()" >/dev/null 2>&1; then
            log "  ERROR - GDS and/or APOC missing. Recreate the container:"
            log "            docker rm -f $NEO4J_CONTAINER   (volume is kept)"
            log "          then rerun 03's setup script or this suite."
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "Neo4j container missing GDS/APOC" "$elapsed"
            rm -f "$out"; return
        fi
        log "  Neo4j ready (GDS + APOC present)"
    fi

    # -- services: 13 temporal Neo4j container ----------------------------
    # Same reuse rules as 03, different container: 13's own setup script
    # manages neo4j-temporal-graph on ports 7475/7688 (APOC only; graphiti
    # needs no plugin check beyond bolt answering).
    if [ "$exercise" = "13_temporal_bitemporal_graphiti" ]; then
        if [ "$DOCKER_OK" != "yes" ]; then
            log "  SKIP - needs the temporal Neo4j container but docker is unavailable"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "SKIP" "docker unavailable, needs Neo4j" "$elapsed"
            rm -f "$out"; return
        fi

        # The password comes from the exercise .env, which is always present.
        local temporal_password
        temporal_password="$(grep -E '^NEO4J_PASSWORD=' "$workdir/.env" | head -1 | cut -d= -f2-)"

        if docker ps --format '{{.Names}}' | grep -qx "$TEMPORAL_CONTAINER"; then
            log "  temporal Neo4j container already running - left running afterwards"
        elif docker ps -a --format '{{.Names}}' | grep -qx "$TEMPORAL_CONTAINER"; then
            log "  starting existing temporal Neo4j container..."
            docker start "$TEMPORAL_CONTAINER" > "$out" 2>&1
            started_neo4j="yes"
        else
            log "  creating temporal Neo4j container ($NEO4J_IMAGE, APOC, volume $TEMPORAL_VOLUME)..."
            docker run -d \
                --name "$TEMPORAL_CONTAINER" \
                -p 7475:7474 -p 7688:7687 \
                -e NEO4J_AUTH="neo4j/$temporal_password" \
                -e NEO4J_PLUGINS='["apoc"]' \
                -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
                -v "$TEMPORAL_VOLUME:/data" \
                "$NEO4J_IMAGE" > "$out" 2>&1
            started_neo4j="yes"
        fi
        exercise_neo4j="$TEMPORAL_CONTAINER"
        if [ "$started_neo4j" = "yes" ]; then
            STARTED_NEO4J_CONTAINERS="$STARTED_NEO4J_CONTAINERS $TEMPORAL_CONTAINER"
        fi

        log "  waiting for temporal Neo4j to answer on bolt..."
        local temporal_ready="no"
        local temporal_attempt=0
        while [ "$temporal_attempt" -lt 60 ]; do
            if docker exec "$TEMPORAL_CONTAINER" cypher-shell -u neo4j -p "$temporal_password" \
                    "RETURN 1" >/dev/null 2>&1; then
                temporal_ready="yes"
                break
            fi
            sleep 3
            temporal_attempt=$((temporal_attempt + 1))
        done
        if [ "$temporal_ready" != "yes" ]; then
            log "  ERROR - temporal Neo4j did not become ready (docker logs $TEMPORAL_CONTAINER)."
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "temporal Neo4j container did not become ready" "$elapsed"
            rm -f "$out"; return
        fi
        log "  temporal Neo4j ready"
    fi

    # -- the walkthrough: real tokens, real answers ----------------------
    # This is the integration test proper: the application runs end to end
    # exactly as a person would run it, and its answers land in the log.
    log "  running walkthrough (real model calls)..."
    local walk_rc=0
    ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
        uv run python main.py walk ) > "$out" 2>&1 || walk_rc=$?

    report_output "walkthrough" "$out" 30

    local walk_detail="walk ok"
    if [ "$walk_rc" -eq 124 ] || [ "$walk_rc" -eq 137 ]; then
        walk_detail="walk timed out after ${TIMEOUT_SECONDS}s"
    elif [ "$walk_rc" -ne 0 ]; then
        walk_detail="walk failed (exit $walk_rc)"
    fi

    # -- the paid tests ---------------------------------------------------
    log "  running tests (paid gates enabled)..."
    local test_rc=0
    ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
        uv run pytest -q --tb=short ) > "$out" 2>&1 || test_rc=$?

    local counts
    counts="$(summarise_pytest "$out")"

    log_only ""
    log_only "----- pytest output -----"
    cat "$out" >> "$LOG"
    log_only "----- end pytest output -----"

    local test_detail="$counts"
    if [ "$test_rc" -eq 124 ] || [ "$test_rc" -eq 137 ]; then
        test_detail="tests timed out after ${TIMEOUT_SECONDS}s"
    fi

    # -- stop services again ----------------------------------------------
    # Stop, never 'down': containers and volumes are always kept.
    if [ "$uses_compose" = "yes" ] && [ "$KEEP_SERVICES" = "no" ]; then
        log "  stopping Postgres (containers and volumes kept)..."
        ( cd "$workdir" && docker compose stop ) >> "$LOG" 2>&1
        STARTED_COMPOSE_IN=""
    fi
    if [ "$started_neo4j" = "yes" ] && [ "$KEEP_SERVICES" = "no" ]; then
        # (The cleanup trap tolerates an already-stopped container, so the
        # list needs no pruning here.)
        log "  stopping Neo4j container $exercise_neo4j (container and volume kept)..."
        docker stop "$exercise_neo4j" >> "$LOG" 2>&1
    fi

    ended=$(date +%s); elapsed=$((ended - started))

    # -- verdict: both halves must succeed --------------------------------
    if [ "$walk_rc" -eq 0 ] && [ "$test_rc" -eq 0 ]; then
        log "  PASS  ($walk_detail; $counts) in ${elapsed}s"
        record "$exercise" "PASS" "$walk_detail; $counts" "$elapsed"
    elif [ "$walk_rc" -ne 0 ] && [ "$test_rc" -eq 0 ]; then
        log "  FAIL  ($walk_detail; tests: $counts) in ${elapsed}s"
        record "$exercise" "FAIL" "$walk_detail; tests: $counts" "$elapsed"
    elif [ "$walk_rc" -eq 0 ]; then
        log "  FAIL  (walk ok; tests: $test_detail, exit $test_rc) in ${elapsed}s"
        report_output "pytest" "$out"
        record "$exercise" "FAIL" "walk ok; tests: $test_detail (exit $test_rc)" "$elapsed"
    else
        log "  FAIL  ($walk_detail; tests: $test_detail, exit $test_rc) in ${elapsed}s"
        report_output "pytest" "$out"
        record "$exercise" "FAIL" "$walk_detail; tests: $test_detail" "$elapsed"
    fi

    rm -f "$out"
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for exercise in "${EXERCISES[@]}"; do
    if wanted "$exercise"; then
        run_exercise "$exercise"
    else
        log ""
        log "[$exercise] filtered out by --only/--skip"
        record "$exercise" "FILTERED" "excluded by command line" "0"
    fi
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
TOTAL_SECONDS=$(( $(date +%s) - START_EPOCH ))

passed=0; failed=0; skipped=0; errored=0; filtered=0; missing=0

log ""
log "=================================================================="
log " RESULTS - Level 1 integration (paid tier)"
log "=================================================================="
log ""
printf '%-34s %-9s %-8s %s\n' "EXERCISE" "STATUS" "TIME" "DETAIL" | tee -a "$LOG"
printf '%-34s %-9s %-8s %s\n' "--------" "------" "----" "------" | tee -a "$LOG"

index=0
while [ "$index" -lt "${#RESULT_NAME[@]}" ]; do
    name="${RESULT_NAME[$index]}"
    status="${RESULT_STATUS[$index]}"
    detail="${RESULT_DETAIL[$index]}"
    seconds="${RESULT_SECONDS[$index]}"

    printf '%-34s %-9s %-8s %s\n' "$name" "$status" "${seconds}s" "$detail" | tee -a "$LOG"

    case "$status" in
        PASS)     passed=$((passed + 1)) ;;
        FAIL)     failed=$((failed + 1)) ;;
        SKIP)     skipped=$((skipped + 1)) ;;
        ERROR)    errored=$((errored + 1)) ;;
        FILTERED) filtered=$((filtered + 1)) ;;
        MISSING)  missing=$((missing + 1)) ;;
    esac
    index=$((index + 1))
done

log ""
log "  passed   $passed"
log "  failed   $failed"
log "  errored  $errored"
log "  skipped  $skipped"
if [ "$missing" -gt 0 ]; then
    log "  missing  $missing"
fi
if [ "$filtered" -gt 0 ]; then
    log "  filtered $filtered"
fi
log ""
log "  total time ${TOTAL_SECONDS}s"
log "  full transcript with every answer: $LOG"
log ""

if [ "$failed" -eq 0 ] && [ "$errored" -eq 0 ] && [ "$missing" -eq 0 ]; then
    log "  Everything that could run, ran clean - answers are in the log."
    log ""
    exit 0
fi

log "  Problems in:"
index=0
while [ "$index" -lt "${#RESULT_NAME[@]}" ]; do
    case "${RESULT_STATUS[$index]}" in
        FAIL|ERROR|MISSING)
            log "    ${RESULT_NAME[$index]}  -  ${RESULT_DETAIL[$index]}"
            log "        grep -A40 '\\[${RESULT_NAME[$index]}\\]' $LOG"
            ;;
    esac
    index=$((index + 1))
done
log ""
exit 1
