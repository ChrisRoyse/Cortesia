# Micro-Task 002: Verify Windows Development Tools

## Objective
Confirm Windows-specific development tools are installed for Rust compilation.

## Context
Windows requires specific tools for compiling Rust code with native dependencies. This task ensures the Microsoft C++ Build Tools or Visual Studio are properly configured.

## Prerequisites
- Task 001 completed (Rust installation verified)

## Time Estimate
8 minutes

## Instructions
1. Check for Windows SDK: Run `where link.exe`
2. Verify MSVC compiler: Run `cl` command (should show version)
3. Check for Windows 10/11 SDK installation
4. Test basic compilation: `cargo new test_compile && cd test_compile && cargo build`
5. Clean up test project: `cd .. && rmdir /s test_compile`

## Expected Output
- MSVC linker found and working
- Basic Rust compilation successful
- Development environment confirmed ready

## Success Criteria
- [ ] `link.exe` found in PATH
- [ ] `cl` command executes (even if with error about no input files)
- [ ] `cargo build` succeeds on test project
- [ ] Test project cleaned up

## Next Task
task_003_install_required_rust_components.md