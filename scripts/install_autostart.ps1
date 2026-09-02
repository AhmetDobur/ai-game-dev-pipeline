# Register the pipeline to start at every logon and immediately resume work.
# Reliable across reboots (Task Scheduler, not the flaky Startup folder), runs
# hidden with no console window, and restarts itself if it ever dies.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# Uninstall:  scripts\uninstall_autostart.ps1
$ErrorActionPreference = "Stop"

$TaskName = "AIGameDevPipeline"
$repo     = (Resolve-Path "$PSScriptRoot\..").Path
$pythonw  = Join-Path $repo ".venv\Scripts\pythonw.exe"   # windowless python
$runpy    = Join-Path $repo "run.py"

if (-not (Test-Path $pythonw)) {
    throw "venv not found at $pythonw — create it first: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
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
Write-Host "Start it now without logging out:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "GUI will be at http://127.0.0.1:8500"
