# Run nvidia-smi and capture the output
$raw = nvidia-smi | Out-String

# Split into lines
$lines = $raw -split "`r?`n"

# Grab the first line (timestamp)
$timestamp = $lines[0]

# Find the line with metrics (the one containing Temp, Power, Memory, Utilization)
$metricsLine = $lines | Where-Object { $_ -match 'C\s+P\d+\s+\d+W\s*/' }

if ($metricsLine) {
    $parts = $metricsLine -split '\s+'

    # Extract values
    $temp        = $parts[2]                          # e.g. "58C"
    $power       = $parts[4] + " / " + $parts[6]      # e.g. "21W / 78W"
    $memory      = $parts[8] + " / " + $parts[10]     # e.g. "7924MiB / 8188MiB"
    $utilization = $parts[12]                         # e.g. "99%"

    # Output nicely
    $output = @"
Timestamp   = $timestamp
    Temperature = $temp                     Memory      = $memory
    Power       = $power               Utilization = $utilization
=====================================
"@

    $output
}
else {
    Write-Error "Could not find metrics line in nvidia-smi output."
}
