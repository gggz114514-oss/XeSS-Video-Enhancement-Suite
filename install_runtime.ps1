param(
    [string]$AssetPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ManifestPath = Join-Path $RepoRoot "runtime_manifest.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$ConfiguredRoot = [Environment]::GetEnvironmentVariable("COMFYUI_XESS_RUNTIME")
if ([string]::IsNullOrWhiteSpace($ConfiguredRoot)) {
    $RuntimeRoot = Join-Path $RepoRoot ".runtime"
} else {
    $RuntimeRoot = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($ConfiguredRoot))
    if ([System.IO.Path]::GetFileName($RuntimeRoot).Equals("engine", [System.StringComparison]::OrdinalIgnoreCase)) {
        $RuntimeRoot = [System.IO.Directory]::GetParent($RuntimeRoot).FullName
    }
}
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$Engine = Join-Path $RuntimeRoot "engine"
$StateName = ".runtime-state.json"

function Assert-InRuntimeRoot([string]$Path) {
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    $Prefix = $RuntimeRoot.TrimEnd('\') + '\'
    if (-not $Resolved.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe runtime path: $Resolved"
    }
}

function Test-Runtime([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    foreach ($Relative in $Manifest.required_files) {
        $File = Join-Path $Path ($Relative -replace '/', '\')
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { return $false }
    }
    $StatePath = Join-Path $Path $StateName
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        try {
            $State = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($State.runtime_version -eq $Manifest.runtime_version -and
                $State.asset_sha256 -eq $Manifest.sha256) { return $true }
        } catch {}
    }
    foreach ($Property in $Manifest.file_hashes.PSObject.Properties) {
        $File = Join-Path $Path ($Property.Name -replace '/', '\')
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { return $false }
        if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            ([string]$Property.Value).ToLowerInvariant()) { return $false }
    }
    return $true
}

function Sync-Pipeline([string]$Destination) {
    $Pipeline = Join-Path $RepoRoot "pipeline"
    foreach ($Source in Get-ChildItem -LiteralPath $Pipeline -Recurse -File) {
        if ($Source.Extension -in @(".pyc", ".pyo") -or $Source.FullName -match "__pycache__") { continue }
        $Relative = $Source.FullName.Substring($Pipeline.Length).TrimStart('\')
        $Target = Join-Path $Destination $Relative
        $Parent = Split-Path -Parent $Target
        if (-not (Test-Path -LiteralPath $Parent)) {
            New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        }
        Copy-Item -LiteralPath $Source.FullName -Destination $Target -Force
    }
}

function Write-State([string]$Destination) {
    $State = [ordered]@{
        runtime_version = $Manifest.runtime_version
        asset_name = $Manifest.asset_name
        asset_sha256 = $Manifest.sha256
        installed_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    }
    $State | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination $StateName) -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$LockPath = Join-Path $RuntimeRoot "install.lock"
$Lock = $null
$Downloaded = $false
$Archive = $null
$Staging = $null
$Succeeded = $false
$Deadline = [DateTime]::UtcNow.AddMinutes(20)
while ($null -eq $Lock) {
    try {
        $Lock = [System.IO.File]::Open($LockPath, [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    } catch [System.IO.IOException] {
        if ([DateTime]::UtcNow -ge $Deadline) { throw "Timed out waiting for $LockPath" }
        Start-Sleep -Seconds 1
    }
}

try {
    if (-not $Force -and (Test-Runtime $Engine)) {
        Sync-Pipeline $Engine
        Write-State $Engine
        Write-Host "[XeSS runtime] already current: $Engine"
        $Succeeded = $true
        exit 0
    }

    $Required = [int64]$Manifest.archive_size + [int64]$Manifest.installed_size + 512MB
    $Drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($RuntimeRoot))
    if ($Drive.AvailableFreeSpace -lt $Required) {
        throw ("Not enough free space. Need about {0:N2} GiB, free {1:N2} GiB." -f
            ($Required / 1GB), ($Drive.AvailableFreeSpace / 1GB))
    }

    $Downloaded = $false
    if ([string]::IsNullOrWhiteSpace($AssetPath)) {
        $DownloadDir = Join-Path $RuntimeRoot "downloads"
        New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
        $Archive = Join-Path $DownloadDir $Manifest.asset_name
        Write-Host "[XeSS runtime] downloading $($Manifest.download_url)"
        Invoke-WebRequest -Uri $Manifest.download_url -OutFile $Archive -UseBasicParsing
        $Downloaded = $true
    } else {
        $Archive = [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($AssetPath))
        if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
            throw "Runtime asset not found: $Archive"
        }
    }
    $ActualHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne ([string]$Manifest.sha256).ToLowerInvariant()) {
        throw "Runtime SHA256 mismatch: expected $($Manifest.sha256), got $ActualHash"
    }

    $Staging = Join-Path $RuntimeRoot ("installing-" + $PID)
    Assert-InRuntimeRoot $Staging
    if (Test-Path -LiteralPath $Staging) { Remove-Item -LiteralPath $Staging -Recurse -Force }
    New-Item -ItemType Directory -Path $Staging | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Staging
    $Candidate = Join-Path $Staging $Manifest.archive_root
    if (-not (Test-Runtime $Candidate)) { throw "Extracted runtime failed validation" }
    Sync-Pipeline $Candidate
    Write-State $Candidate

    $Backup = Join-Path $RuntimeRoot ("engine-backup-" + $PID)
    Assert-InRuntimeRoot $Backup
    if (Test-Path -LiteralPath $Backup) { Remove-Item -LiteralPath $Backup -Recurse -Force }
    if (Test-Path -LiteralPath $Engine) { Move-Item -LiteralPath $Engine -Destination $Backup }
    try {
        Move-Item -LiteralPath $Candidate -Destination $Engine
    } catch {
        if ((Test-Path -LiteralPath $Backup) -and -not (Test-Path -LiteralPath $Engine)) {
            Move-Item -LiteralPath $Backup -Destination $Engine
        }
        throw
    }
    if (Test-Path -LiteralPath $Backup) { Remove-Item -LiteralPath $Backup -Recurse -Force }
    if (Test-Path -LiteralPath $Staging) { Remove-Item -LiteralPath $Staging -Recurse -Force }
    if ($Downloaded -and (Test-Path -LiteralPath $Archive)) { Remove-Item -LiteralPath $Archive -Force }
    Write-Host "[XeSS runtime] ready: $Engine"
    $Succeeded = $true
    exit 0
} finally {
    if (-not $Succeeded) {
        if ($null -ne $Staging -and (Test-Path -LiteralPath $Staging)) {
            Assert-InRuntimeRoot $Staging
            Remove-Item -LiteralPath $Staging -Recurse -Force
        }
        if ($Downloaded -and $null -ne $Archive -and (Test-Path -LiteralPath $Archive)) {
            Assert-InRuntimeRoot $Archive
            Remove-Item -LiteralPath $Archive -Force
        }
    }
    if ($null -ne $Lock) { $Lock.Dispose() }
    if (Test-Path -LiteralPath $LockPath) { Remove-Item -LiteralPath $LockPath -Force }
}
