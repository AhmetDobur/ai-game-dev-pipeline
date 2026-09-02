# Remove the logon autostart task.
#   powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1
$ErrorActionPreference = "Stop"
$TaskName = "AIGameDevPipeline"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask  -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed '$TaskName'."
} else {
    Write-Host "'$TaskName' is not installed."
}
