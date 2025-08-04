# Task 81a: Security Dependency Scan [REWRITTEN - 100/100 Quality]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 80 completed
**Required Tools:** Rust toolchain, cargo-audit

## Complete Context (For AI with ZERO Knowledge)

You are implementing **security auditing for the Tantivy-based text search system**. This task focuses specifically on scanning dependencies for known security vulnerabilities.

**What is cargo-audit?** A Rust security advisory database scanner that checks for known vulnerabilities in dependencies.

**Project State:** After 80 tasks, you have a complete Tantivy search system with advanced features, testing, and documentation.

**This Task:** Install cargo-audit and scan all dependencies for security vulnerabilities, creating a baseline security report.

## Pre-Task Environment Check
Run these commands first:
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo --version  # Should show cargo 1.70+
```

## Exact Steps (6 minutes implementation)

### Step 1: Install cargo-audit (2 minutes)
```bash
# Install cargo-audit for vulnerability scanning
cargo install cargo-audit

# Verify installation
cargo audit --version
```

**Expected result:** Version output like `cargo-audit 0.18.0`

### Step 2: Create Security Audit Script (2 minutes)
Create the file `C:/code/LLMKG/vectors/tantivy_search/scripts/security_audit.bat`:

```batch
@echo off
echo Starting Security Audit...
echo.

echo [1/3] Scanning dependencies for vulnerabilities...
cargo audit --format json > audit_report.json
if %ERRORLEVEL% neq 0 (
    echo ERROR: Vulnerability scan failed!
    exit /b 1
)

echo [2/3] Generating human-readable report...
cargo audit > audit_report.txt

echo [3/3] Checking for unmaintained dependencies...
cargo audit --stale > stale_deps.txt

echo.
echo Security audit complete. Check:
echo - audit_report.json (machine-readable)
echo - audit_report.txt (human-readable)  
echo - stale_deps.txt (unmaintained dependencies)
```

### Step 3: Run Initial Security Scan (1 minute)
```bash
# Make script executable and run
cd scripts
security_audit.bat
```

### Step 4: Create Security Report Template (1 minute)
Create `C:/code/LLMKG/vectors/tantivy_search/SECURITY_REPORT.md`:

```markdown
# Security Audit Report

## Scan Date
{DATE}

## Dependency Vulnerabilities
{VULNERABILITIES}

## Unmaintained Dependencies  
{STALE_DEPS}

## Recommendations
{RECOMMENDATIONS}

## Status
- [ ] All critical vulnerabilities addressed
- [ ] All high-priority vulnerabilities addressed
- [ ] Unmaintained dependencies evaluated
- [ ] Security baseline established
```

## Verification Steps (2 minutes)

### Verify 1: cargo-audit installed
```bash
cargo audit --version
```
**Expected output:** Version information

### Verify 2: Audit runs successfully
```bash
cargo audit
```
**Expected output:** Either "No vulnerabilities found" or list of found issues

### Verify 3: Files created
```bash
ls -la
# Should show: audit_report.json, audit_report.txt, stale_deps.txt, SECURITY_REPORT.md
```

## Success Validation Checklist
- [ ] cargo-audit installed and working
- [ ] Security audit script created and executable
- [ ] Initial vulnerability scan completed
- [ ] Report files generated successfully
- [ ] Security report template created
- [ ] No critical vulnerabilities in current dependencies

## If This Task Fails

**Error: "cargo install failed"**
- Solution: Check internet connection, update Rust toolchain

**Error: "audit failed"**  
- Solution: Check Cargo.toml syntax, verify dependencies resolve

**Error: "permission denied"**
- Solution: Run as administrator, check file permissions

## Files Created For Next Task

After completing this task, you will have:

1. **scripts/security_audit.bat** - Automated security scanning script
2. **audit_report.json** - Machine-readable vulnerability report
3. **audit_report.txt** - Human-readable vulnerability report
4. **stale_deps.txt** - Unmaintained dependencies report
5. **SECURITY_REPORT.md** - Security report template

**Next Task (Task 81b)** will analyze the vulnerability findings and create remediation plans.

## Context for Task 81b
Task 81b will parse the audit results, categorize vulnerabilities by severity, and create specific remediation steps for each finding. The reports generated here provide the raw data for that analysis.