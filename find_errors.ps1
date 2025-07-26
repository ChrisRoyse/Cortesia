cargo check 2>&1 | Out-File -FilePath build_output.txt
Get-Content build_output.txt | Select-String "error\[E[0-9]+\]" -Context 10,2