for(;;) {
 try {
  # invoke the worker script
    ./gpu_logger.ps1 >> gpulog.txt
 }
 catch {
  # do something with $_, log it, more likely
    Write-Host "An error occurred: $($_.Exception.Message)"
 }

 # wait for 1 minute
    Start-Sleep 60
}

# .\gpu_log_executor.ps1