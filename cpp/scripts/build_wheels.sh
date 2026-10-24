#!/usr/bin/env bash
# Builds one nabia_python_bridge wheel per Python version for this machine, and tests each one
# in a clean virtual environment. uv (https://docs.astral.sh/uv/) downloads missing interpreters.
#
#   cpp/scripts/build_wheels.sh                          # Python 3.10 to 3.14
#   cpp/scripts/build_wheels.sh 3.12 3.13                # only these versions
#   NABIA_WHEEL_TEST=0 cpp/scripts/build_wheels.sh       # build without testing
#   NABIA_WHEEL_BUILD_TYPE=RelWithDebInfo cpp/scripts/build_wheels.sh 3.14   # with debug symbols
#
# Wheels go to dist/ at the repository root (NABIA_WHEEL_DIR overrides it), where the Python
# projects install them from (python/scripts/setup_env.sh, webui/scripts/setup_env.sh). The tests
# are the bridge tests of the Python project (python/tests/bridge). On macOS the wheels target
# MACOSX_DEPLOYMENT_TARGET, 13.0 unless already set. Wheels for Linux and Windows are built by
# cibuildwheel (configured in pyproject.toml, run by .github/workflows/wheels.yml).
set -euo pipefail

cpp="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(dirname "$cpp")"
tests="$root/python/tests/bridge"
build_type="${NABIA_WHEEL_BUILD_TYPE:-Release}"
out="${NABIA_WHEEL_DIR:-$root/dist}"
versions=("$@")
[[ ${#versions[@]} -gt 0 ]] || versions=(3.10 3.11 3.12 3.13 3.14)

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
# Use an installed interpreter (Homebrew, python.org...) before a uv-managed one.
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-system}"
[[ "$(uname)" == Darwin ]] && export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

mkdir -p "$out"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

for v in "${versions[@]}"; do
    echo "=== Python $v ==="
    # Finds the interpreter, downloading it with uv when none is installed.
    py="$(uv python find "$v" 2>/dev/null || { uv python install "$v" >&2 && uv python find "$v"; })"
    # A pre-release (an old uv may only know a 3.x alpha) has no numpy wheels and the wrong ABI.
    "$py" -c 'import sys; sys.exit(sys.version_info.releaselevel != "final")' || {
        echo "$py is a pre-release Python; update uv (uv self update) or install Python $v" >&2; exit 1; }
    echo "$py ($("$py" -c 'import platform; print(platform.python_version())'))"
    # Isolated build: uv installs scikit-build-core and pybind11 for this interpreter.
    uv build --wheel --python "$py" --out-dir "$work/$v" -C cmake.build-type="$build_type" "$cpp"
    wheel="$(ls "$work/$v"/*.whl)"

    if [[ "${NABIA_WHEEL_TEST:-1}" != 0 ]]; then
        # Install in a fresh environment and run the tests from outside the source tree,
        # so they import the installed package.
        uv venv --quiet --python "$py" "$work/venv-$v"
        uv pip install --quiet --python "$work/venv-$v/bin/python" "$wheel[test]"
        (cd "$work" && "$work/venv-$v/bin/python" -m pytest -q -p no:cacheprovider "$tests")
    fi
    cp "$wheel" "$out/"
done

echo "=== wheels in $out ==="
ls -1 "$out"/*.whl
