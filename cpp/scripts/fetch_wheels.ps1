<#
.SYNOPSIS
    Downloads the nabia_python_bridge wheels published on GitHub into ..\dist (Windows), replacing the wheels there.

.DESCRIPTION
    The PowerShell counterpart of fetch_wheels.sh. Needs the GitHub CLI, logged in with access to the
    repository (gh auth login).

.PARAMETER Source
    auto (default: the newest successful workflow run with wheels, else the newest release), run or release.

.PARAMETER Ref
    A run id (Source run) or a tag (Source release). Default: the newest.

.EXAMPLE
    scripts\fetch_wheels.ps1
.EXAMPLE
    scripts\fetch_wheels.ps1 -Source release -Ref v0.1.0
#>
[CmdletBinding()]
param(
    [ValidateSet("auto", "run", "release")] [string] $Source = "auto",
    [string] $Ref = ""
)
$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $PSScriptRoot
$repo = if ($env:NABIA_REPO) { $env:NABIA_REPO } else { "frannuca/nabia" }
$wheels = Join-Path (Split-Path -Parent $project) "dist"
$tmp = "$wheels.tmp"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "the GitHub CLI (gh) is required: https://cli.github.com" }

function Get-LatestRun {
    $ids = gh run list --repo $repo --workflow wheels.yml --status success --limit 20 --json databaseId --jq '.[].databaseId'
    foreach ($id in $ids) {
        $names = gh api "repos/$repo/actions/runs/$id/artifacts" --jq '.artifacts[].name'
        if ($names | Where-Object { $_ -like "wheels-*" }) { return $id }
    }
    return $null
}

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory $tmp | Out-Null
if ($Source -eq "release") {
    if ($Ref) { gh release download $Ref --repo $repo --pattern "*.whl" -D $tmp }
    else { gh release download --repo $repo --pattern "*.whl" -D $tmp }
} else {
    $id = if ($Ref) { $Ref } else { Get-LatestRun }
    if ($id) {
        Write-Host "workflow run ${id}: artifacts wheels-*"
        gh run download $id --repo $repo --pattern "wheels-*" -D $tmp
        # one folder per artifact: keep only the wheel files, side by side
        Get-ChildItem $tmp -Recurse -Filter *.whl | Where-Object { $_.DirectoryName -ne (Resolve-Path $tmp).Path } |
            Move-Item -Destination $tmp
        Get-ChildItem $tmp -Directory | Remove-Item -Recurse -Force
    } elseif ($Source -eq "auto") {
        Write-Host "no successful workflow run with wheels; using the newest release"
        gh release download --repo $repo --pattern "*.whl" -D $tmp
    } else {
        throw "no successful workflow run with wheels"
    }
}
if ($LASTEXITCODE -ne 0) { throw "download failed" }

New-Item -ItemType Directory -Force $wheels | Out-Null
Remove-Item (Join-Path $wheels "nabia_python_bridge-*.whl") -ErrorAction SilentlyContinue
Get-ChildItem $tmp -Filter *.whl | Move-Item -Destination $wheels
Remove-Item -Recurse -Force $tmp
Get-ChildItem (Join-Path $wheels "nabia_python_bridge-*.whl") | ForEach-Object { $_.Name }
