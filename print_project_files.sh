#!/usr/bin/env bash
#
# print_project_files.sh
#
# Walks the current directory tree and writes every text file into a single
# flat file, project_contents.txt, with each file's path shown above its
# contents. Binary files, lock files and generated caches are recorded by name
# only so the output stays readable.
#
# Run it from the project root:
#     ./print_project_files.sh
#
# The only terminal output is a confirmation line and the resulting file size.
# Everything else - what was skipped and why - is recorded inside the output
# file itself, so nothing is silently dropped.

set -euo pipefail

OUTPUT_FILE="project_contents.txt"
SCRIPT_NAME="$(basename "$0")"

# Text files larger than this are recorded by name and size only. Generated
# data files can be text and enormous, and one of them would swamp the output.
MAX_TEXT_BYTES=$((1024 * 1024))

# ---------------------------------------------------------------------------
# Directories pruned entirely. Add to this list as the project grows.
# ---------------------------------------------------------------------------
PRUNE_DIRS=(
    ".git"
    "__pycache__"
    ".venv"
    "venv"
    "logs"
    "data"
    "*.lance"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
    ".ipynb_checkpoints"
    ".idea"
    ".vscode"
    "node_modules"
    "dist"
    "build"
    ".eggs"
    ".tox"
    ".cache"
    "htmlcov"
    "*.egg-info"
    ".lancedb"
    "lancedb"
    ".uv"
)

# ---------------------------------------------------------------------------
# Files recorded by name only. Lock files are deterministic noise and can run
# to tens of thousands of lines.
# ---------------------------------------------------------------------------
SKIP_BY_NAME=(
    "uv.lock"
    "poetry.lock"
    "Pipfile.lock"
    "package-lock.json"
    "yarn.lock"
    "pnpm-lock.yaml"
    ".DS_Store"
    "Thumbs.db"
)

SKIP_BY_GLOB=(
    "*.pyc"
    "*.pyo"
    "*.pyd"
    "*.so"
    "*.o"
    "*.a"
    "*.class"
    "*.lance"
    "*.log"
)

# ---------------------------------------------------------------------------
# Files whose contents are deliberately withheld. A .env holds live API keys,
# and the whole point of this output file is that it gets pasted somewhere
# else. .env.example is not in this list and will be included in full.
# ---------------------------------------------------------------------------
SECRET_BY_NAME=(
    ".env"
    ".env.local"
    ".env.production"
    ".netrc"
    "credentials.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Return 0 if the first argument matches any of the remaining arguments,
# treating the patterns as globs so both exact names and wildcards work.
matches_any() {
    local candidate="$1"
    shift
    local pattern
    for pattern in "$@"; do
        # shellcheck disable=SC2053
        if [[ "$candidate" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

# Decide whether a file is binary. `file` reports "binary" as the encoding for
# anything that is not decodable text, which covers images, archives, compiled
# objects and index files without needing a list of extensions.
is_binary() {
    local path="$1"
    local encoding
    encoding="$(file --brief --mime-encoding -- "$path" 2>/dev/null || echo "binary")"
    if [ "$encoding" = "binary" ]; then
        return 0
    fi
    return 1
}

file_size_bytes() {
    stat --printf="%s" -- "$1"
}

# ---------------------------------------------------------------------------
# Build the find expression that prunes the excluded directories.
# ---------------------------------------------------------------------------
prune_expression=()
for directory_name in "${PRUNE_DIRS[@]}"; do
    if [ ${#prune_expression[@]} -gt 0 ]; then
        prune_expression+=( "-o" )
    fi
    prune_expression+=( "-name" "$directory_name" )
done

# ---------------------------------------------------------------------------
# Collect the file list first, so the manifest at the top of the output can be
# written before any of the contents.
# ---------------------------------------------------------------------------
declare -a ALL_FILES=()

while IFS= read -r -d '' found_path; do
    relative_path="${found_path#./}"

    # Never include the output file or this script.
    if [ "$relative_path" = "$OUTPUT_FILE" ]; then
        continue
    fi
    if [ "$relative_path" = "$SCRIPT_NAME" ]; then
        continue
    fi

    ALL_FILES+=( "$relative_path" )
done < <(find . -type d \( "${prune_expression[@]}" \) -prune -o -type f -print0 | sort -z)

# ---------------------------------------------------------------------------
# Write the output. Everything goes through a single redirect so the file is
# written once rather than reopened for every append.
# ---------------------------------------------------------------------------
included_count=0
binary_count=0
skipped_count=0
secret_count=0
oversize_count=0

{
    printf 'PROJECT CONTENTS\n'
    printf 'Root: %s\n' "$(pwd)"
    printf 'Generated: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'Files found: %d\n' "${#ALL_FILES[@]}"
    printf '\n'
    printf 'Excluded directories: %s\n' "${PRUNE_DIRS[*]}"
    printf 'Lock and cache files are listed by name only.\n'
    printf 'Binary files are listed by name and size only.\n'
    printf 'Secret files have their contents withheld.\n'
    printf '\n'

    printf 'MANIFEST\n'
    for relative_path in "${ALL_FILES[@]}"; do
        printf '  %s\n' "$relative_path"
    done
    printf '\n'

    # -----------------------------------------------------------------------
    # File contents. Each entry is bracketed by greppable markers so the file
    # can be split apart again, or read reliably by a tool.
    # -----------------------------------------------------------------------
    for relative_path in "${ALL_FILES[@]}"; do
        base_name="$(basename "$relative_path")"
        size_bytes="$(file_size_bytes "$relative_path")"

        printf '### FILE: %s\n' "$relative_path"

        if matches_any "$base_name" "${SECRET_BY_NAME[@]}"; then
            printf '[contents withheld: this file holds credentials]\n'
            secret_count=$((secret_count + 1))

        elif matches_any "$base_name" "${SKIP_BY_NAME[@]}"; then
            printf '[skipped: lock or metadata file, %s bytes]\n' "$size_bytes"
            skipped_count=$((skipped_count + 1))

        elif matches_any "$base_name" "${SKIP_BY_GLOB[@]}"; then
            printf '[skipped: generated or compiled artefact, %s bytes]\n' "$size_bytes"
            skipped_count=$((skipped_count + 1))

        elif is_binary "$relative_path"; then
            printf '[binary file, %s bytes]\n' "$size_bytes"
            binary_count=$((binary_count + 1))

        elif [ "$size_bytes" -gt "$MAX_TEXT_BYTES" ]; then
            printf '[text file too large to include, %s bytes]\n' "$size_bytes"
            oversize_count=$((oversize_count + 1))

        else
            cat -- "$relative_path"
            # Guarantee a newline before the end marker even when the source
            # file does not end with one.
            if [ -s "$relative_path" ]; then
                last_byte="$(tail --bytes=1 -- "$relative_path")"
                if [ -n "$last_byte" ]; then
                    printf '\n'
                fi
            fi
            included_count=$((included_count + 1))
        fi

        printf '### END FILE: %s\n\n' "$relative_path"
    done

    printf '### SUMMARY\n'
    printf 'Included in full : %d\n' "$included_count"
    printf 'Binary, name only: %d\n' "$binary_count"
    printf 'Skipped          : %d\n' "$skipped_count"
    printf 'Secrets withheld : %d\n' "$secret_count"
    printf 'Oversize         : %d\n' "$oversize_count"

} > "$OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Report. Two lines, nothing else.
# ---------------------------------------------------------------------------
output_bytes="$(file_size_bytes "$OUTPUT_FILE")"
output_megabytes="$(awk -v bytes="$output_bytes" 'BEGIN { printf "%.2f", bytes / 1048576 }')"

printf 'Built %s\n' "$OUTPUT_FILE"
printf 'Size: %s MB (%s bytes)\n' "$output_megabytes" "$output_bytes"
