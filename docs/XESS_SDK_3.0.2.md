# Intel XeSS SDK 3.0.2 intake record

The SDK is fetched only from the Intel `intel/xess` GitHub Release.  The ZIP is
kept outside Git under the task workspace:

`<repo>/work/xess-sdk3-upgrade/XeSS_SDK_3.0.2.zip`

This is an intake record for a future SDK 3.0.2 source migration.  R4 Gate 0
does not compile native workers with this package or switch the existing r3
runtime; the current R4 build continues to use its existing SDK until a later
migration gate passes.

| Field | Value |
|---|---|
| Repository | `https://github.com/intel/xess` |
| Release/tag | `v3.0.2` |
| Release page | `https://github.com/intel/xess/releases/tag/v3.0.2` |
| Asset | `XeSS_SDK_3.0.2.zip` |
| Asset URL | `https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip` |
| Asset SHA-256 | `88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990` |
| Release commit | `8fe81bd` |
| Release date | `2026-07-24` |

The release notes say this revision updates XeLL and fixes the `minimumIntervalUs`
units, a non-Intel XeLL marker leak, and Streamline proxy unwrapping.  No SR or
FG header change exists between the 3.0.1 and 3.0.2 packages; the DLL changes are
recorded below.

## Exact SDK files for the future build

The following hashes identify the one-package inputs expected by that future
build.  No native build is performed from these files during this Gate 0
import.

| Component | File version | Bytes | SHA-256 |
|---|---:|---:|---|
| XeSS-SR `bin/libxess.dll` | `2.0.2.68` | 77,795,704 | `251659dd84a3e84de67c886a4186e01f3eca49b00641906fe38bb6b807e5d5b7` |
| XeSS-FG `bin/libxess_fg.dll` | `1.3.1.78` | 22,957,432 | `ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27` |
| XeLL `bin/libxell.dll` | `1.3.2.10` | 415,368 | `d2030dcd694fda8f2ec7e044b13e6db8f0b56d4ba9113a5efad334e3f3ded8c7` |
| `lib/libxess.lib` | — | 20,004 | `5e16bf3745358b54ecfb52f04bc4327536409f981a317819005075f57e0be813` |
| `lib/libxess_fg.lib` | — | 10,354 | `f41aa9ca6c31b2fc1a6dc96182b4e7d19e9642ba0e0b01df044c56687480f414` |
| `lib/libxell.lib` | — | 13,450 | `6c7f7578b12b5710cd09fb730ded190c13e98d712a7cf7ff71e17233ec4cd3a4` |

Header hashes used by the 3.0.2 compile are recorded in
`tools/xess_sdk_3_0_2.json`.  The SDK's `LICENSE.txt` is retained with the staged
package; the repository copy of the Intel SDK license remains unchanged.

## Build contract

For the future SDK 3.0.2 migration, `build.bat` must require `XESS_SDK_ROOT` to
point at this exact extracted package (a directory containing `inc`, `lib`, and
`bin`).  The standalone `tools/check_xess_sdk.py` checker validates the listed
headers, import libraries, and runtime DLL hashes.  In this Gate 0 import,
`build.bat` itself only checks that the required header exists and native source
migration is intentionally deferred.

Example (PowerShell):

```powershell
$env:XESS_SDK_ROOT = '<repo>/work/xess-sdk3-upgrade/sdk/XeSS_SDK_3.0.2'
.\build.bat
```

The future source and SDK package versions are intentionally separate: project
version `1.3.0`, SDK package `3.0.2`, SR DLL `2.0.2.68`, FG DLL `1.3.1.78`, and
XeLL DLL `1.3.2.10`.
