<#
.SYNOPSIS
    Builds nabia_python_bridge wheels for Windows (win_amd64 by default) with cibuildwheel, and tests
    each one with the bridge tests in cpp\bindings\tests.

.DESCRIPTION
    The Windows counterpart of build_wheels.sh. cibuildwheel downloads every CPython it needs, builds
    the wheel from this C++ project (cpp/pyproject.toml, scikit-build-core, MSVC), installs it in a
    clean environment and runs the tests. The wheels end up in dist\ at the repository root.

    Needs, once:
      - Visual Studio 2022 Build Tools with the "Desktop development with C++" workload (C++20);
      - Git (CMake fetches Eigen and iir1);
      - Python 3.10 or newer with the py launcher (python.org installer), to run cibuildwheel.
    CMake is optional: scikit-build-core downloads it when it is not installed.

.PARAMETER Versions
    CPython versions to build, e.g. 3.12, 3.13. Default: 3.10 to 3.14.

.PARAMETER Arch
    AMD64 (Intel and AMD 64-bit, the default) or ARM64 (on an ARM machine).

.PARAMETER OutDir
    Where the wheels go. Default: dist\ at the repository root.

.PARAMETER NoTest
    Build without running the tests on each wheel.

.PARAMETER PythonExe
    Python used to run cibuildwheel. Default: the py launcher (py -3), else python on the PATH.

.EXAMPLE
    cpp\scripts\build_wheels.ps1
.EXAMPLE
    cpp\scripts\build_wheels.ps1 -Versions 3.12,3.13
.EXAMPLE
    cpp\scripts\build_wheels.ps1 -Versions 3.12 -NoTest
#>
[CmdletBinding()]
param(
    [string[]] $Versions = @("3.10", "3.11", "3.12", "3.13", "3.14"),
    [ValidateSet("AMD64", "ARM64")] [string] $Arch = "AMD64",
    [string] $OutDir = "",
    [switch] $NoTest,
    [string] $PythonExe = ""
)
$ErrorActionPreference = "Stop"

$cpp = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $cpp
if (-not $OutDir) { $OutDir = Join-Path $root "dist" }
$CibuildwheelVersion = "4.3.4"

# --- Prerequisites ----------------------------------------------------------------------------
$hostArgs = @()
if ($PythonExe) {
    $hostExe = $PythonExe
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $hostExe = "py"
    $hostArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $hostExe = "python"
} else {
    throw "Python was not found: install Python 3.10+ from python.org."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found: install Git for Windows (CMake fetches Eigen and iir1 with it)."
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $component = if ($Arch -eq "ARM64") { "Microsoft.VisualStudio.Component.VC.Tools.ARM64" } else { "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" }
    $vs = & $vswhere -latest -products * -requires $component -property installationPath
    if (-not $vs) {
        throw "Visual Studio has no C++ compiler: add the 'Desktop development with C++' workload."
    }
    Write-Host "MSVC: $vs"
} else {
    Write-Warning "vswhere.exe not found; the build fails if Visual Studio 2022 C++ tools are missing."
}

# --- cibuildwheel, in its own virtual environment ---------------------------------------------
$tools = Join-Path ([System.IO.Path]::GetTempPath()) "nabia-cibuildwheel"
if (-not (Test-Path (Join-Path $tools "Scripts\python.exe"))) {
    & $hostExe @hostArgs -m venv $tools
    if ($LASTEXITCODE -ne 0) { throw "could not create $tools" }
}
& (Join-Path $tools "Scripts\python.exe") -m pip install --quiet --upgrade pip "cibuildwheel==$CibuildwheelVersion"
if ($LASTEXITCODE -ne 0) { throw "could not install cibuildwheel" }

# --- Build ------------------------------------------------------------------------------------
# cp312-* etc.; the other settings (skips, tests, test extras) are in cpp\pyproject.toml.
$env:CIBW_BUILD = ($Versions | ForEach-Object { "cp" + ($_ -replace "\.", "") + "-*" }) -join " "
$env:CIBW_ARCHS = $Arch
if ($NoTest) { $env:CIBW_TEST_SKIP = "*" } else { Remove-Item Env:CIBW_TEST_SKIP -ErrorAction SilentlyContinue }

Write-Host "Building $($env:CIBW_BUILD) for $Arch into $OutDir"
# cibuildwheel runs from the repository root: {project} in the test command is the root, so the
# tests of cpp\bindings\tests are found; "cpp" is the package folder with pyproject.toml.
Push-Location $root
try {
    & (Join-Path $tools "Scripts\cibuildwheel.exe") --platform windows --output-dir $OutDir cpp
    if ($LASTEXITCODE -ne 0) { throw "cibuildwheel failed (exit code $LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host "=== wheels in $OutDir ==="
Get-ChildItem (Join-Path $OutDir "*win*.whl") | ForEach-Object { Write-Host $_.FullName }
