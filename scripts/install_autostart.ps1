# Register the pipeline to start at every logon and immediately resume work.
# Reliable across reboots (Task Scheduler, not the flaky Startup folder), runs
# hidden with no console window, and restarts itself if it ever dies.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# Uninstall:  scripts\uninstall_autostart.ps1
param(
    # optional: path to a ComfyUI portable dir (ComfyUI_windows_portable). When
    # given, a second logon task starts ComfyUI too, so image/mesh waves work
    # unattended after every reboot.
    [string]$ComfyDir = ""
)
$ErrorActionPreference = "Stop"

$TaskName = "AIGameDevPipeline"
$repo     = (Resolve-Path "$PSScriptRoot\..").Path
# venv may be named .venv or venv depending on how it was created
$pythonw  = @("$repo\.venv\Scripts\pythonw.exe", "$repo\venv\Scripts\pythonw.exe") |
            Where-Object { Test-Path $_ } | Select-Object -First 1
$runpy    = Join-Path $repo "run.py"

if (-not $pythonw) {
    # ASCII only in this file: PowerShell 5.1 reads BOM-less files as ANSI, and a
    # UTF-8 em dash misreads as a curly quote that breaks parsing
    throw "venv not found - create it first: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
}

# `run.py gui` serves the page AND auto-resumes every interrupted run on startup
$action = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$runpy`" gui" -WorkingDirectory $repo

# fire at logon; if the machine was off mid-run, this is when it picks back up
$trigger = New-ScheduledTaskTrigger -AtLogOn

# keep it alive like a service: no time limit, restart on failure
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "Installed '$TaskName': starts at logon, auto-resumes runs, restarts if it dies."

if ($ComfyDir) {
    $cpy = Join-Path $ComfyDir "python_embeded\python.exe"
    $cmain = Join-Path $ComfyDir "ComfyUI\main.py"
    if (-not (Test-Path $cpy) -or -not (Test-Path $cmain)) {
        throw "ComfyDir doesn't look like a ComfyUI portable dir: $ComfyDir"
    }
    $caction = New-ScheduledTaskAction -Execute $cpy `
        -Argument "-s `"$cmain`" --listen 127.0.0.1 --port 8188" `
        -WorkingDirectory $ComfyDir
    Register-ScheduledTask -TaskName "AIGameDevComfyUI" -Action $caction -Trigger $trigger `
        -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Installed 'AIGameDevComfyUI': ComfyUI serves 127.0.0.1:8188 at logon."
}

Write-Host "Start now without logging out:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "GUI will be at http://127.0.0.1:8500"
