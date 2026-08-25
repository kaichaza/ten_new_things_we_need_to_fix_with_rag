#!/usr/bin/env bash
#
# run_all_exercises.sh
#
# Runs every exercise in the repository as its own isolated uv project, brings
# up any Docker services an exercise depends on, runs its pytest suite, stops
# the services again, and prints one report at the end.
#
# Getting it onto the machine and running it:
#
#     cp /mnt/c/Users/kaich/Downloads/run_all_exercises.sh .
#     chmod +x run_all_exercises.sh
#     ./run_all_exercises.sh
#
# Nothing is left RUNNING afterwards, but nothing is ever deleted either:
# containers are stopped, not removed, and volumes are always kept. These
# containers hold vector indexes and graphs that cost real money to build.
# A failure in one exercise is recorded and the run moves on to the next
# one; it never aborts the whole sweep.
#
# Options:
#     --dry-run           show the plan, run nothing
#     --only 03,07        run just these exercises (by number)
#     --skip 03           skip these exercises
#     --timeout 900       seconds allowed per exercise (default 600)
#     --no-sync           assume dependencies are installed, skip uv sync
#     --docker-build      additionally verify each Dockerfile builds
#     --keep-services     leave Docker services up afterwards (for debugging)
#
# Full transcript is written to test_run.log in the project root; the previous
# run is kept as test_run.log.1.

# Deliberately no 'set -e'. A failing exercise is data, not a reason to stop.
set -uo pipefail

LOG="test_run.log"
LOG_PREVIOUS="test_run.log.1"
TIMEOUT_SECONDS=600
DRY_RUN="no"
DO_SYNC="yes"
DOCKER_BUILD="no"
KEEP_SERVICES="no"
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
        --docker-build)   DOCKER_BUILD="yes"; shift ;;
        --keep-services)  KEEP_SERVICES="yes"; shift ;;
        --only)           ONLY="${2:-}"; shift 2 ;;
        --skip)           SKIP="${2:-}"; shift 2 ;;
        --timeout)        TIMEOUT_SECONDS="${2:-600}"; shift 2 ;;
        -h|--help)        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                printf 'Unknown option: %s (try --help)\n' "$1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging. Everything goes to the screen and to the log file, so the two never
# disagree. Long command output is captured to a temp file, appended to the log
# in full, and summarised on screen.
# ---------------------------------------------------------------------------
if [ -f "$LOG" ]; then
    mv -f "$LOG" "$LOG_PREVIOUS"
fi
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
# Cleanup. Runs on normal exit and on interrupt, so Ctrl-C does not strand a
# Postgres container.
# ---------------------------------------------------------------------------
STARTED_COMPOSE_IN=""
STARTED_NEO4J_CONTAINER=""

cleanup() {
    local exit_code=$?
    if [ -n "$STARTED_COMPOSE_IN" ] && [ "$KEEP_SERVICES" = "no" ]; then
        # Stop, never 'down': these containers hold built indexes and other
        # expensive state, so containers and volumes are always kept.
        printf '\nStopping Docker services in %s (containers and volumes kept)...\n' \
            "$STARTED_COMPOSE_IN" | tee -a "$LOG"
        ( cd "$STARTED_COMPOSE_IN" && docker compose stop ) >> "$LOG" 2>&1
        STARTED_COMPOSE_IN=""
    fi
    if [ -n "$STARTED_NEO4J_CONTAINER" ] && [ "$KEEP_SERVICES" = "no" ]; then
        # Same rule: stop only. The container and its volume belong to the
        # exercise's own setup script and may hold a paid-for index.
        printf '\nStopping Neo4j container %s (container and volume kept)...\n' \
            "$STARTED_NEO4J_CONTAINER" | tee -a "$LOG"
        docker stop "$STARTED_NEO4J_CONTAINER" >> "$LOG" 2>&1
        STARTED_NEO4J_CONTAINER=""
    fi
    exit $exit_code
}
trap cleanup EXIT
trap 'printf "\nInterrupted.\n" | tee -a "$LOG"; exit 130' INT TERM

# ---------------------------------------------------------------------------
# uv creates and manages a venv per project and 'uv run' executes inside it
# without activation. If the caller happens to have a venv active, unexport it
# so each exercise gets its own interpreter rather than inheriting one.
# ---------------------------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
    log "Note: deactivating inherited virtualenv ${VIRTUAL_ENV} for child processes"
    unset VIRTUAL_ENV
fi
unset PYTHONHOME 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
log "=================================================================="
log " Exercise test sweep"
log " Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log " Root:    $(pwd)"
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
        log "          exercises needing services will be skipped, not failed"
    fi
else
    log "docker    not installed - exercises needing services will be skipped"
fi

# ---------------------------------------------------------------------------
# The exercises, named explicitly. Nothing is discovered, so a renamed or
# missing folder is reported rather than silently dropped from the sweep.
# Edit this list when the repository changes.
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
)

# 03_graphrag_fruit_graph does not use docker compose; it talks to the same
# long-lived Neo4j container its own setup script manages (GDS + APOC plugins,
# data in a named volume). The names here must match that setup script. This
# sweep may start or create the container, and may stop it again if it was
# the one that started it, but it NEVER removes the container or the volume:
# the volume can hold an index that cost real money to build.
NEO4J_CONTAINER="neo4j-fruit-graph"
NEO4J_VOLUME="neo4j-fruit-graph-data"
NEO4J_IMAGE="neo4j:5.26"

log "exercises expected: ${#EXERCISES[@]}"

# Every named folder must be present. Absences are counted here and reported
# in the results table, not skipped over quietly.
MISSING=()
PRESENT_COUNT=0
for exercise in "${EXERCISES[@]}"; do
    if [ -d "$exercise" ]; then
        PRESENT_COUNT=$((PRESENT_COUNT + 1))
    else
        MISSING+=("$exercise")
    fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    log ""
    log "WARNING: ${#MISSING[@]} named exercise(s) not found on disk:"
    for exercise in "${MISSING[@]}"; do
        log "  missing  $exercise"
    done
fi

if [ "$PRESENT_COUNT" -eq 0 ]; then
    log ""
    log "FATAL: none of the named exercises exist here. Is this the project root?"
    FATAL="yes"
fi

# The reverse check: a numbered folder on disk that is not in the list above
# would otherwise never be tested, and nobody would notice.
for candidate in [0-9][0-9]_*; do
    case "$candidate" in
        *.bak.*) continue ;;
    esac
    [ -d "$candidate" ] || continue
    known="no"
    for exercise in "${EXERCISES[@]}"; do
        if [ "$candidate" = "$exercise" ]; then
            known="yes"
            break
        fi
    done
    if [ "$known" = "no" ]; then
        log ""
        log "WARNING: $candidate exists but is not in the list in this script."
        log "         It will NOT be tested. Add it to EXERCISES if it should be."
    fi
done

if [ "$FATAL" = "yes" ]; then
    log ""
    log "Cannot continue."
    exit 2
fi

# A live key means the suites will make real, billable model calls. Say so
# once, loudly, rather than surprising anyone.
KEY_PRESENT="no"
for exercise in "${EXERCISES[@]}"; do
    if [ -f "$exercise/.env" ] && grep -qE '^OPENAI_API_KEY=.+' "$exercise/.env" 2>/dev/null; then
        if ! grep -qE '^OPENAI_API_KEY=sk-your-key-here\s*$' "$exercise/.env"; then
            KEY_PRESENT="yes"
            break
        fi
    fi
done

log ""
if [ "$KEY_PRESENT" = "yes" ]; then
    log "NOTE: a real OPENAI_API_KEY was found. Suites that need one will make"
    log "      billable calls. 03_graphrag_fruit_graph only runs its expensive"
    log "      indexing test when GRAPHRAG_RUN_INDEX=1 is also set; without it"
    log "      that test skips, and the free Neo4j checks still run."
else
    log "NOTE: no real OPENAI_API_KEY found. Suites needing one will report as"
    log "      skipped, which is the designed behaviour rather than a failure."
fi

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
# Results, kept at root level. Parallel arrays keyed by index.
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

# Append a captured output file to the log, and echo its tail to the screen.
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

# Pull the counts out of pytest's summary line.
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
# Run one exercise.
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

    if [ ! -d "$workdir/tests" ]; then
        log "  SKIP - no tests/ directory"
        record "$exercise" "SKIP" "no tests directory" "0"
        rm -f "$out"; return
    fi

    # .env is scaffolded rather than treated as an error: the suites are
    # written to skip cleanly when there is no key.
    if [ ! -f "$workdir/.env" ] && [ -f "$workdir/.env.example" ]; then
        cp "$workdir/.env.example" "$workdir/.env"
        log "  created .env from .env.example (no key - key-dependent tests will skip)"
    fi

    local python_pin
    python_pin="$(grep -oE 'requires-python[^"]*"[^"]+' "$workdir/pyproject.toml" \
                  | grep -oE '3\.[0-9]+' | head -1)"
    if [ -n "$python_pin" ]; then
        log "  python pin: $python_pin"
    fi

    # -- dependencies ----------------------------------------------------
    if [ "$DO_SYNC" = "yes" ]; then
        log "  syncing dependencies..."
        local sync_rc=0
        ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" uv sync ) > "$out" 2>&1 || sync_rc=$?
        if [ "$sync_rc" -ne 0 ]; then
            if [ "$sync_rc" -eq 124 ] || [ "$sync_rc" -eq 137 ]; then
                log "  ERROR - uv sync timed out after ${TIMEOUT_SECONDS}s. Dropping this exercise."
            else
                log "  ERROR - uv sync failed (exit $sync_rc). Dropping this exercise."
            fi
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

    # -- Docker services this exercise depends on ------------------------
    local uses_compose="no"
    if [ -f "$workdir/docker-compose.yml" ] || [ -f "$workdir/compose.yml" ]; then
        uses_compose="yes"
    fi

    if [ "$uses_compose" = "yes" ]; then
        if [ "$DOCKER_OK" != "yes" ]; then
            log "  SKIP - needs Docker services but the daemon is unavailable"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "SKIP" "docker unavailable, needs services" "$elapsed"
            rm -f "$out"; return
        fi

        log "  starting Docker services (compose)..."
        # Only the dependency service is started, not the app container: the
        # app is what pytest is about to run locally.
        local compose_rc=0
        ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
            docker compose up -d --wait db ) > "$out" 2>&1 || compose_rc=$?
        if [ "$compose_rc" -ne 0 ]; then
            log "  ERROR - could not start Docker services (exit $compose_rc). Dropping this exercise."
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
        log "  services healthy"
    fi

    # -- Neo4j container for the graphrag exercise ------------------------
    # 03_graphrag_fruit_graph has no compose file on purpose: its tests talk
    # to the long-lived container the exercise's setup script manages, so an
    # index built with real model calls survives between sweeps. Reuse that
    # container here. Running: leave it alone before and after. Stopped:
    # start it, stop it again after the tests. Absent: create it exactly the
    # way the setup script does (both plugins), and stop - not remove - it
    # afterwards.
    local started_neo4j="no"
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
        if [ "$started_neo4j" = "yes" ]; then
            STARTED_NEO4J_CONTAINER="$NEO4J_CONTAINER"
        fi

        # Wait for bolt. A first boot downloads the plugin jars, so this can
        # legitimately take a couple of minutes.
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
            log "  ERROR - Neo4j did not become ready. Dropping this exercise."
            log "          inspect with: docker logs $NEO4J_CONTAINER"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "Neo4j container did not become ready" "$elapsed"
            rm -f "$out"; return
        fi

        # The pipeline needs BOTH plugins: GDS for Leiden, APOC for the graph
        # writer and the entity resolver. A container created before APOC was
        # added would make the tests fail with confusing procedure-not-found
        # errors, so check up front and give recreation instructions instead.
        if ! docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$neo4j_password" \
                "RETURN gds.version(), apoc.version()" >/dev/null 2>&1; then
            log "  ERROR - GDS and/or APOC is missing from the Neo4j container."
            log "          Recreate it with both plugins and rerun:"
            log "            docker rm -f $NEO4J_CONTAINER"
            log "          (the data volume is kept; rerun the exercise setup"
            log "          script, or simply rerun this sweep, to recreate it)"
            ended=$(date +%s); elapsed=$((ended - started))
            record "$exercise" "ERROR" "Neo4j container missing GDS/APOC" "$elapsed"
            rm -f "$out"; return
        fi
        log "  Neo4j ready (GDS + APOC present)"
    fi

    # -- optional image build check --------------------------------------
    if [ "$DOCKER_BUILD" = "yes" ] && [ -f "$workdir/Dockerfile" ]; then
        if [ "$DOCKER_OK" != "yes" ]; then
            log "  (skipping image build - docker unavailable)"
        else
            log "  building image..."
            if ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
                    docker build -q -t "exercise_${exercise}" . ) > "$out" 2>&1; then
                log "  image builds OK"
                log_only "----- docker build output -----"; cat "$out" >> "$LOG"
            else
                log "  WARNING - image build failed (tests will still run)"
                report_output "docker build" "$out" 15
            fi
        fi
    fi

    # -- the tests -------------------------------------------------------
    log "  running tests..."
    local test_rc=0
    ( cd "$workdir" && timeout "${TIMEOUT_SECONDS}s" \
        uv run pytest -q --tb=short ) > "$out" 2>&1 || test_rc=$?

    local counts
    counts="$(summarise_pytest "$out")"

    log_only ""
    log_only "----- pytest output -----"
    cat "$out" >> "$LOG"
    log_only "----- end pytest output -----"

    # -- stop services again ----------------------------------------------
    # Stop, never 'down': every service container here carries built state
    # (LanceDB indexes, graphs) that is expensive to rebuild. Containers and
    # volumes are always kept; only the running state is wound back.
    if [ "$uses_compose" = "yes" ] && [ "$KEEP_SERVICES" = "no" ]; then
        log "  stopping Docker services (containers and volumes kept)..."
        ( cd "$workdir" && docker compose stop ) >> "$LOG" 2>&1
        STARTED_COMPOSE_IN=""
    fi

    if [ "$started_neo4j" = "yes" ] && [ "$KEEP_SERVICES" = "no" ]; then
        # This sweep started (or created) the container, so put it back to
        # stopped. The container and its data volume are kept.
        log "  stopping Neo4j container (container and volume kept)..."
        docker stop "$NEO4J_CONTAINER" >> "$LOG" 2>&1
        STARTED_NEO4J_CONTAINER=""
    fi

    ended=$(date +%s); elapsed=$((ended - started))

    if [ "$test_rc" -eq 0 ]; then
        log "  PASS  ($counts) in ${elapsed}s"
        record "$exercise" "PASS" "$counts" "$elapsed"
    elif [ "$test_rc" -eq 5 ]; then
        # pytest exit 5 means it collected nothing.
        log "  SKIP  (no tests collected) in ${elapsed}s"
        record "$exercise" "SKIP" "no tests collected" "$elapsed"
    elif [ "$test_rc" -eq 124 ] || [ "$test_rc" -eq 137 ]; then
        log "  FAIL  (timed out after ${TIMEOUT_SECONDS}s)"
        report_output "pytest" "$out"
        record "$exercise" "FAIL" "timed out after ${TIMEOUT_SECONDS}s" "$elapsed"
    else
        log "  FAIL  ($counts, exit $test_rc) in ${elapsed}s"
        report_output "pytest" "$out"
        record "$exercise" "FAIL" "$counts (exit $test_rc)" "$elapsed"
    fi

    rm -f "$out"
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
log ""
if [ "$DRY_RUN" = "yes" ]; then
    log "Plan (dry run - nothing will be executed):"
    for exercise in "${EXERCISES[@]}"; do
        if wanted "$exercise"; then
            if [ ! -d "$exercise" ]; then
                log "  MISSING    $exercise"
                continue
            fi
            services=""
            if [ -f "$exercise/docker-compose.yml" ]; then
                services=" [needs Docker services]"
            fi
            if [ "$exercise" = "03_graphrag_fruit_graph" ]; then
                services=" [needs Neo4j container: $NEO4J_CONTAINER]"
            fi
            log "  would run  $exercise$services"
        else
            log "  filtered   $exercise"
        fi
    done
    log ""
    log "Dry run complete. Nothing was changed."
    exit 0
fi

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
log " RESULTS"
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
log "  full transcript: $LOG"
if [ -f "$LOG_PREVIOUS" ]; then
    log "  previous run:    $LOG_PREVIOUS"
fi
log ""

if [ "$failed" -eq 0 ] && [ "$errored" -eq 0 ] && [ "$missing" -eq 0 ]; then
    log "  Everything that could run, ran clean."
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
