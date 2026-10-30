#!/usr/bin/env bash
# Downloads the nabia_python_bridge wheels published on GitHub into ../dist, replacing the wheels there
# (needs the GitHub CLI, logged in with access to the repository).
#
#   scripts/fetch_wheels.sh                 # the newest successful workflow run, else the newest release
#   scripts/fetch_wheels.sh run [<run-id>]  # a workflow run's wheel artifacts (default: newest successful)
#   scripts/fetch_wheels.sh release [<tag>] # the wheels attached to a release (default: newest)
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${NABIA_REPO:-frannuca/nabia}"
wheels="$(dirname "$project")/dist"
source="${1:-auto}"
ref="${2:-}"

command -v gh >/dev/null || { echo "the GitHub CLI (gh) is required: https://cli.github.com" >&2; exit 1; }

latest_run() {
    gh run list --repo "$repo" --workflow wheels.yml --status success --limit 20 \
        --json databaseId --jq '.[].databaseId' |
    while read -r id; do
        # only runs that uploaded wheels (one wheels-<os>-<arch> artifact per platform)
        if gh api "repos/$repo/actions/runs/$id/artifacts" --jq '.artifacts[].name' | grep -q '^wheels-'; then
            echo "$id"; break
        fi
    done
}

rm -rf "$wheels.tmp"
mkdir -p "$wheels.tmp"
case "$source" in
    auto|run)
        id="${ref:-$(latest_run)}"
        if [[ -n "$id" ]]; then
            echo "workflow run $id: artifacts wheels-*"
            gh run download "$id" --repo "$repo" --pattern 'wheels-*' -D "$wheels.tmp"
            # one folder per artifact: keep only the wheel files, side by side
            find "$wheels.tmp" -mindepth 2 -name '*.whl' -exec mv {} "$wheels.tmp" \;
            find "$wheels.tmp" -mindepth 1 -type d -exec rm -rf {} +
        elif [[ "$source" == auto ]]; then
            echo "no successful workflow run with wheels; using the newest release"
            gh release download --repo "$repo" --pattern '*.whl' -D "$wheels.tmp"
        else
            echo "no successful workflow run with wheels" >&2; exit 1
        fi ;;
    release)
        echo "release ${ref:-(newest)}"
        gh release download ${ref:+"$ref"} --repo "$repo" --pattern '*.whl' -D "$wheels.tmp" ;;
    *)
        echo "usage: $0 [run [<run-id>] | release [<tag>]]" >&2; exit 2 ;;
esac

mkdir -p "$wheels"
rm -f "$wheels"/nabia_python_bridge-*.whl
mv "$wheels.tmp"/*.whl "$wheels"/ && rm -rf "$wheels.tmp"
ls -1 "$wheels"/nabia_python_bridge-*.whl
