<#
.SYNOPSIS
    Sets up nabia_python\.venv for the Python project on Windows (the interpreter to select in PyCharm).

.DESCRIPTION
    The Windows counterpart of setup_env.sh:
      1. installs the nabia_python_bridge wheel built by the C++ project (..\dist, see
         cpp\scripts\build_wheels.ps1) - reinstalled every time, so a rebuilt wheel is picked up;
      2. installs this nabia package in editable mode, with the market data and test extras (add llm for --live).

.PARAMETER Python
    Python version used to create .venv when it does not exist (py launcher), e.g. 3.12. Default: 3.

.PARAMETER PythonExe
    Python executable used to create .venv instead of the py launcher (e.g. in CI).

.PARAMETER WheelDir
    Folder with the wheels. Default: dist\ at the repository root.

.EXAMPLE
    nabia_python\scripts\setup_env.ps1
.EXAMPLE
    nabia_python\scripts\setup_env.ps1 -Python 3.12
#>
[CmdletBinding()]
param(
    [string] $Python = "3",
    [string] $WheelDir = "",
    [string] $PythonExe = ""
)
$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $PSScriptRoot
if (-not $WheelDir) { $WheelDir = Join-Path (Split-Path -Parent $project) "dist" }
$venv = Join-Path $project ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    if ($PythonExe) {
        & $PythonExe -m venv $venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py "-$Python" -m venv $venv
    } else {
        python -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw "could not create $venv with Python $Python" }
}
& $py -m pip install --quiet --upgrade pip

# pip picks the wheel matching this interpreter and platform among those in $WheelDir.
& $py -m pip install --force-reinstall --no-deps --no-index --find-links $WheelDir nabia_python_bridge
if ($LASTEXITCODE -ne 0) {
    throw "no nabia_python_bridge wheel for $(& $py -V) in ${WheelDir}: run cpp\scripts\build_wheels.ps1 first"
}
& $py -m pip install --quiet --find-links $WheelDir -e "$project[marketdata,test]"
if ($LASTEXITCODE -ne 0) { throw "could not install the nabia package" }

& $py -c "import nabia, nabia_python_bridge as nb; print('nabia', nabia.__file__); print('nabia_python_bridge', nb.__version__, nb.__file__)"
