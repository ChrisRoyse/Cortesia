# Micro-Task 027: Setup Development Scripts

## Objective
Create additional development scripts for common tasks beyond the basic Makefile commands.

## Context
Beyond basic build/test/format commands, developers need scripts for specific tasks like cleaning data directories, running specific test suites, and generating reports. This task creates these convenience scripts.

## Prerequisites
- Task 026 completed (Memory profiling configured)
- Makefile.bat already created
- Understanding of development workflow needs

## Time Estimate
9 minutes

## Instructions
1. Create `scripts` directory: `mkdir scripts`
2. Create `scripts/clean_all.bat`:
   ```batch
   @echo off
   echo Cleaning all build artifacts and data...
   cargo clean
   if exist "data\temp\*" del /q "data\temp\*"
   if exist "data\logs\*" del /q "data\logs\*"
   if exist "data\benchmarks\*" del /q "data\benchmarks\*"
   echo Clean complete.
   ```
3. Create `scripts/run_tests.bat`:
   ```batch
   @echo off
   echo Running comprehensive test suite...
   echo.
   echo === Unit Tests ===
   cargo test --lib --workspace
   echo.
   echo === Integration Tests ===
   cargo test --test "*" --workspace
   echo.
   echo === Doc Tests ===
   cargo test --doc --workspace
   echo.
   echo Test suite complete.
   ```
4. Create `scripts/generate_docs.bat`:
   ```batch
   @echo off
   echo Generating documentation...
   cargo doc --workspace --no-deps --open
   echo Documentation generated and opened.
   ```
5. Create `scripts/performance_check.bat`:
   ```batch
   @echo off
   echo Running performance benchmarks...
   cargo bench --workspace
   echo.
   echo Benchmark results saved to data/benchmarks/
   echo Opening benchmark report...
   if exist "data\benchmarks\report\index.html" (
       start data\benchmarks\report\index.html
   ) else (
       echo No benchmark report found.
   )
   ```
6. Create `scripts/README.md`:
   ```markdown
   # Development Scripts
   
   ## Available Scripts
   
   - `clean_all.bat` - Clean all build artifacts and temporary data
   - `run_tests.bat` - Run comprehensive test suite
   - `generate_docs.bat` - Generate and open documentation
   - `performance_check.bat` - Run benchmarks and open report
   
   ## Usage
   
   Run from project root directory:
   ```
   scripts\clean_all.bat
   scripts\run_tests.bat
   scripts\generate_docs.bat
   scripts\performance_check.bat
   ```
   ```
7. Test each script:
   - `scripts\clean_all.bat`
   - `scripts\run_tests.bat` (will show "no tests" - that's expected)
8. Commit scripts: `git add scripts/ && git commit -m "Add development convenience scripts"`

## Expected Output
- Development scripts created for common tasks
- Scripts tested and working
- Script documentation provided
- All scripts committed to repository

## Success Criteria
- [ ] All 4 development scripts created in scripts/ directory
- [ ] Scripts execute without syntax errors
- [ ] Script README.md documents usage
- [ ] Development scripts committed to Git

## Next Task
task_028_finalize_environment_setup.md