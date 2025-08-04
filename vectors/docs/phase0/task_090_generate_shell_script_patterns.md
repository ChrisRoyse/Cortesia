# Micro-Task 090: Generate Shell Script Patterns

## Objective
Generate shell script files with Windows batch and PowerShell patterns containing special characters for command-line interface testing.

## Context
Shell scripts contain unique special character patterns for command execution, piping, redirection, and variable substitution that differ from programming languages and require specialized vector search handling.

## Prerequisites
- Task 089 completed (Code with special characters generated)

## Time Estimate
8 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create shell script generation script `generate_shell_patterns.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate shell script patterns for vector search testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_shell_scripts():
       """Generate shell script files with special character patterns."""
       generator = TestFileGenerator()
       
       # Windows Batch Script
       batch_script = '''@echo off
   REM Windows batch script with special characters for testing
   REM Tests vector search handling of batch-specific syntax
   
   setlocal enabledelayedexpansion
   
   REM Variable definitions with special characters
   set "APP_NAME=VectorSearchApp"
   set "VERSION=1.0.0"
   set "BUILD_DIR=%~dp0build"
   set "LOG_FILE=%TEMP%\\%APP_NAME%_%DATE:/=-%_%TIME::=-%_build.log"
   
   REM Command line argument processing
   if "%1"=="" (
       echo Usage: %0 ^<command^> [options]
       echo Commands:
       echo   build    - Build the application
       echo   test     - Run tests
       echo   clean    - Clean build directory  
       echo   deploy   - Deploy to production
       goto :eof
   )
   
   REM Main command processing
   if /i "%1"=="build" goto :build
   if /i "%1"=="test" goto :test
   if /i "%1"=="clean" goto :clean
   if /i "%1"=="deploy" goto :deploy
   
   echo Error: Unknown command "%1"
   exit /b 1
   
   :build
   echo Building %APP_NAME% v%VERSION%...
   
   REM Create directories with error handling
   if not exist "%BUILD_DIR%" (
       mkdir "%BUILD_DIR%" 2>nul
       if errorlevel 1 (
           echo Error: Cannot create build directory
           exit /b 1
       )
   )
   
   REM File operations with special character handling
   echo Compiling source files...
   for %%f in (src\\*.rs) do (
       echo   Processing %%f...
       rustc "%%f" -o "%BUILD_DIR%\\%%~nf.exe" 2>>"%LOG_FILE%"
       if errorlevel 1 (
           echo Error compiling %%f
           type "%LOG_FILE%"
           exit /b 1
       )
   )
   
   REM Archive creation with compression
   echo Creating release archive...
   powershell -Command "Compress-Archive -Path '%BUILD_DIR%\\*' -DestinationPath '%BUILD_DIR%\\%APP_NAME%_v%VERSION%.zip' -Force"
   
   echo Build completed successfully!
   echo Output: %BUILD_DIR%\\%APP_NAME%_v%VERSION%.zip
   goto :eof
   
   :test
   echo Running tests for %APP_NAME%...
   
   REM Test execution with output capture
   set "TEST_RESULTS=%BUILD_DIR%\\test_results.xml"
   cargo test --manifest-path Cargo.toml -- --test-threads=1 --format=json > "%TEST_RESULTS%" 2>&1
   
   if errorlevel 1 (
       echo Tests failed! Check %TEST_RESULTS%
       type "%TEST_RESULTS%"
       exit /b 1
   )
   
   echo All tests passed!
   goto :eof
   
   :clean
   echo Cleaning build directory...
   
   if exist "%BUILD_DIR%" (
       rmdir /s /q "%BUILD_DIR%"
       echo Build directory cleaned
   ) else (
       echo Nothing to clean
   )
   goto :eof
   
   :deploy
   echo Deploying %APP_NAME% v%VERSION%...
   
   REM Environment validation
   if "%DEPLOY_TARGET%"=="" (
       set /p DEPLOY_TARGET="Enter deployment target (dev/staging/prod): "
   )
   
   REM Deployment commands with error handling
   echo Deploying to %DEPLOY_TARGET%...
   scp "%BUILD_DIR%\\%APP_NAME%_v%VERSION%.zip" user@%DEPLOY_TARGET%.example.com:/opt/releases/
   if errorlevel 1 (
       echo Error: Failed to upload release
       exit /b 1
   )
   
   ssh user@%DEPLOY_TARGET%.example.com "cd /opt/releases && unzip -o %APP_NAME%_v%VERSION%.zip && systemctl restart %APP_NAME%"
   if errorlevel 1 (
       echo Error: Failed to deploy and restart service
       exit /b 1
   )
   
   echo Deployment completed successfully!
   goto :eof'''
   
       # PowerShell Script
       powershell_script = '''#!/usr/bin/env pwsh
   # PowerShell script with special characters for testing
   # Tests vector search handling of PowerShell-specific syntax
   
   [CmdletBinding()]
   param(
       [Parameter(Mandatory=$true, Position=0)]
       [ValidateSet("Build", "Test", "Clean", "Deploy")]
       [string]$Command,
       
       [Parameter()]
       [string]$Configuration = "Release",
       
       [Parameter()]
       [string]$OutputPath = "./build",
       
       [Parameter()]
       [switch]$Verbose
   )
   
   # Script variables with special characters
   $script:AppName = "VectorSearchApp"
   $script:Version = "1.0.0"
   $script:BuildDir = Resolve-Path $OutputPath -ErrorAction SilentlyContinue
   $script:LogFile = Join-Path $env:TEMP "$AppName_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')_build.log"
   
   # Function definitions with advanced parameters
   function Write-LogMessage {
       [CmdletBinding()]
       param(
           [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
           [string]$Message,
           
           [Parameter()]
           [ValidateSet("Info", "Warning", "Error")]
           [string]$Level = "Info",
           
           [Parameter()]
           [switch]$WriteToConsole = $true
       )
       
       process {
           $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
           $logEntry = "[$timestamp] [$Level] $Message"
           
           # Write to log file
           Add-Content -Path $script:LogFile -Value $logEntry -Encoding UTF8
           
           # Write to console with colors
           if ($WriteToConsole) {
               switch ($Level) {
                   "Info" { Write-Host $logEntry -ForegroundColor Green }
                   "Warning" { Write-Host $logEntry -ForegroundColor Yellow }
                   "Error" { Write-Host $logEntry -ForegroundColor Red }
               }
           }
       }
   }
   
   function Invoke-BuildProcess {
       [CmdletBinding()]
       param(
           [Parameter()]
           [string]$ProjectPath = "./Cargo.toml"
       )
       
       Write-LogMessage "Starting build process for $script:AppName v$script:Version"
       
       try {
           # Create build directory
           if (-not (Test-Path $script:BuildDir)) {
               New-Item -ItemType Directory -Path $script:BuildDir -Force | Out-Null
               Write-LogMessage "Created build directory: $script:BuildDir"
           }
           
           # Build with Cargo
           $buildArgs = @(
               "build"
               "--release"
               "--manifest-path"
               $ProjectPath
               "--target-dir"
               $script:BuildDir
           )
           
           Write-LogMessage "Executing: cargo $($buildArgs -join ' ')"
           $result = Start-Process -FilePath "cargo" -ArgumentList $buildArgs -Wait -PassThru -NoNewWindow
           
           if ($result.ExitCode -ne 0) {
               throw "Build failed with exit code $($result.ExitCode)"
           }
           
           # Create release archive
           $archivePath = Join-Path $script:BuildDir "$script:AppName`_v$script:Version.zip"
           $sourceFiles = Get-ChildItem -Path "$script:BuildDir/release" -File -Recurse
           
           Compress-Archive -Path $sourceFiles.FullName -DestinationPath $archivePath -Force
           Write-LogMessage "Created release archive: $archivePath"
           
           return $archivePath
       }
       catch {
           Write-LogMessage "Build failed: $($_.Exception.Message)" -Level Error
           throw
       }
   }
   
   function Invoke-TestSuite {
       [CmdletBinding()]
       param(
           [Parameter()]
           [string]$TestFilter = "*",
           
           [Parameter()]
           [int]$TestThreads = 1
       )
       
       Write-LogMessage "Running test suite with filter: $TestFilter"
       
       try {
           $testArgs = @(
               "test"
               "--release"
               "--"
               "--test-threads=$TestThreads"
               "--format=json"
           )
           
           if ($TestFilter -ne "*") {
               $testArgs += $TestFilter
           }
           
           $testOutput = & cargo @testArgs 2>&1
           $testResults = $testOutput | Where-Object { $_ -match '^{.*}$' } | ConvertFrom-Json
           
           # Process test results
           $passed = ($testResults | Where-Object { $_.type -eq "test" -and $_.event -eq "ok" }).Count
           $failed = ($testResults | Where-Object { $_.type -eq "test" -and $_.event -eq "failed" }).Count
           
           Write-LogMessage "Test Results: $passed passed, $failed failed"
           
           if ($failed -gt 0) {
               $failedTests = $testResults | Where-Object { $_.type -eq "test" -and $_.event -eq "failed" }
               foreach ($test in $failedTests) {
                   Write-LogMessage "FAILED: $($test.name)" -Level Error
               }
               throw "Test suite failed with $failed failed tests"
           }
           
           return @{
               Passed = $passed
               Failed = $failed
               Total = $passed + $failed
           }
       }
       catch {
           Write-LogMessage "Test execution failed: $($_.Exception.Message)" -Level Error
           throw
       }
   }
   
   function Remove-BuildArtifacts {
       [CmdletBinding()]
       param(
           [Parameter()]
           [switch]$IncludeDependencies
       )
       
       Write-LogMessage "Cleaning build artifacts..."
       
       try {
           # Remove build directory
           if (Test-Path $script:BuildDir) {
               Remove-Item -Path $script:BuildDir -Recurse -Force
               Write-LogMessage "Removed build directory: $script:BuildDir"
           }
           
           # Clean Cargo cache if requested
           if ($IncludeDependencies) {
               & cargo clean
               Write-LogMessage "Cleaned Cargo cache"
           }
           
           # Remove log files older than 7 days
           $oldLogs = Get-ChildItem -Path $env:TEMP -Filter "$script:AppName*_build.log" |
                      Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
           
           foreach ($log in $oldLogs) {
               Remove-Item -Path $log.FullName -Force
               Write-LogMessage "Removed old log file: $($log.Name)"
           }
       }
       catch {
           Write-LogMessage "Clean operation failed: $($_.Exception.Message)" -Level Error
           throw
       }
   }
   
   # Main execution logic
   try {
       Write-LogMessage "Starting $Command operation..."
       
       switch ($Command) {
           "Build" {
               $archivePath = Invoke-BuildProcess
               Write-LogMessage "Build completed successfully: $archivePath"
           }
           
           "Test" {
               $results = Invoke-TestSuite
               Write-LogMessage "All $($results.Total) tests passed successfully"
           }
           
           "Clean" {
               Remove-BuildArtifacts -IncludeDependencies
               Write-LogMessage "Clean operation completed"
           }
           
           "Deploy" {
               $archivePath = Invoke-BuildProcess
               # Deployment logic would go here
               Write-LogMessage "Deployment completed (simulated)"
           }
       }
       
       Write-LogMessage "Operation completed successfully"
       exit 0
   }
   catch {
       Write-LogMessage "Operation failed: $($_.Exception.Message)" -Level Error
       exit 1
   }'''
   
       # Generate shell script files
       samples = [
           ("build_script.bat", "Windows batch script with special characters", batch_script),
           ("build_script.ps1", "PowerShell script with special characters", powershell_script)
       ]
       
       generated_files = []
       for filename, pattern_focus, content in samples:
           output_path = generator.generate_text_file(
               filename,
               "code_samples",
               pattern_focus,
               content,
               "code_samples"
           )
           generated_files.append(output_path)
           print(f"Generated: {output_path}")
       
       return generated_files
   
   def main():
       """Main generation function."""
       print("Generating shell script patterns...")
       
       # Ensure output directory exists
       os.makedirs("code_samples", exist_ok=True)
       
       try:
           files = generate_shell_scripts()
           print(f"\nSuccessfully generated {len(files)} shell script files:")
           for file_path in files:
               print(f"  - {file_path}")
           
           print("\nShell script pattern generation completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating shell scripts: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_shell_patterns.py`
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\generate_shell_patterns.py data\test_files\code_samples && git commit -m "task_090: Generate shell script patterns"`

## Expected Output
- Windows batch script with CMD-specific syntax
- PowerShell script with advanced PowerShell patterns
- Command-line special characters and operators
- Variable substitution and control flow patterns

## Success Criteria
- [ ] Batch script file generated with Windows patterns
- [ ] PowerShell script file generated with modern syntax
- [ ] Shell-specific special characters included
- [ ] Files follow template format and UTF-8 encoding

## Validation Commands
```cmd
cd data\test_files
python generate_shell_patterns.py
dir code_samples
```

## Next Task
task_091_generate_markup_patterns.md

## Notes
- Shell scripts test command-line interface pattern recognition
- Windows-specific patterns ensure platform compatibility
- Variable substitution tests dynamic content handling
- Files provide realistic automation script examples