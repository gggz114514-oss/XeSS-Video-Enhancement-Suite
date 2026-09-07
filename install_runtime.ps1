param([string]$AssetPath = "", [switch]$Force, [string]$Python = "")
$ErrorActionPreference = "Stop"
$NodeRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$Portable = [IO.Path]::GetFullPath((Join-Path $NodeRoot "..\..\.."))
$Candidates = @($Python, $env:XESS_PYTHON,
    (Join-Path $Portable "python_embeded\python.exe"),
    (Join-Path $Portable "python\python.exe"),
    (Join-Path $NodeRoot ".runtime\engine\python\python.exe"))
$Chosen = $Candidates | Where-Object { $_ -and [IO.File]::Exists($_) } | Select-Object -First 1
if (-not $Chosen) {
    $Command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Command) { $Chosen = $Command.Source }
}
if (-not $Chosen) { throw "Python not found. Run with -Python <ComfyUI python.exe>. No packages will be installed." }
$Arguments = @((Join-Path $NodeRoot "runtime_manager.py"), "ensure")
if ($AssetPath) { $Arguments += @("--asset", $AssetPath) }
if ($Force) { $Arguments += "--force" }
& $Chosen @Arguments
exit $LASTEXITCODE
