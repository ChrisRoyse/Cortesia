#!/usr/bin/env python3
"""
Rust Integration Verification Script

This script verifies that the Rust LLMKG system can be built and tested.
Since the full codebase has compilation issues, this focuses on testing
the specific tools that were fixed.
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and return success/failure with output"""
    print(f"\n[RUNNING] {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description}")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}...")
            return True, result.stdout
        else:
            print(f"[FAILED] {description}")
            print(f"Error: {result.stderr[:1000]}...")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description}")
        return False, "Command timed out"
    except Exception as e:
        print(f"[EXCEPTION] {description} - {e}")
        return False, str(e)

def check_rust_toolchain():
    """Verify Rust toolchain is available"""
    success, output = run_command(['cargo', '--version'], "Checking Rust toolchain")
    if success:
        print(f"Rust version: {output.strip()}")
    return success

def check_code_compilation():
    """Check if the Rust code compiles"""
    print("\n[COMPILATION CHECK]")
    
    # Try to check specific components that should work
    success, output = run_command(
        ['cargo', 'check', '--lib'], 
        "Checking library compilation",
        cwd=Path.cwd()
    )
    
    if not success:
        print("[FAILED] Library compilation failed")
        print("This is expected due to known issues in the codebase")
        return False
    
    return True

def run_specific_tests():
    """Try to run specific tests that might work"""
    print("\n[TESTING MODULES]")
    
    # Try to run tests for individual modules that might work
    test_modules = [
        "core::knowledge_engine",
        "core::triple",
        "mcp::llm_friendly_server::types"
    ]
    
    results = {}
    for module in test_modules:
        success, output = run_command(
            ['cargo', 'test', module, '--lib'], 
            f"Testing {module}",
            cwd=Path.cwd()
        )
        results[module] = success
    
    return results

def verify_handler_functions():
    """Verify the handler functions exist in the code"""
    print("\n[VERIFYING HANDLERS]")
    
    handlers_to_check = [
        "handle_generate_graph_query",
        "handle_get_stats", 
        "handle_validate_knowledge",
        "handle_neural_importance_scoring"
    ]
    
    results = {}
    for handler in handlers_to_check:
        # Check if function exists in the codebase
        success, output = run_command(
            ['grep', '-r', f'pub async fn {handler}', 'src/'], 
            f"Finding {handler} function",
            cwd=Path.cwd()
        )
        results[handler] = success
        if success:
            print(f"[FOUND] {handler}")
        else:
            print(f"[MISSING] {handler}")
    
    return results

def create_integration_test_report():
    """Create a comprehensive report of the integration test status"""
    print("\n[CREATING REPORT]")
    
    report = {
        "timestamp": subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
        "rust_toolchain": False,
        "compilation_status": False,
        "handler_functions": {},
        "test_results": {},
        "recommendations": []
    }
    
    # Check Rust toolchain
    report["rust_toolchain"] = check_rust_toolchain()
    
    # Check compilation
    report["compilation_status"] = check_code_compilation()
    
    # Verify handler functions exist
    report["handler_functions"] = verify_handler_functions()
    
    # Try to run tests if compilation works
    if report["compilation_status"]:
        report["test_results"] = run_specific_tests()
    else:
        print("[SKIPPED] Tests skipped due to compilation failures")
    
    # Generate recommendations
    if not report["compilation_status"]:
        report["recommendations"].append(
            "Fix compilation errors before running integration tests"
        )
    
    if all(report["handler_functions"].values()):
        report["recommendations"].append(
            "All handler functions found - ready for integration testing once compilation is fixed"
        )
    else:
        report["recommendations"].append(
            "Some handler functions missing - check implementation"
        )
    
    # Save report
    with open("integration_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[REPORT] Saved to: integration_test_report.json")
    return report

def main():
    """Main function to run all verification steps"""
    print("RUST INTEGRATION VERIFICATION")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    print(f"Working directory: {Path.cwd()}")
    
    # Create the report
    report = create_integration_test_report()
    
    # Print summary
    print("\nSUMMARY")
    print("=" * 30)
    print(f"Rust Toolchain: {'PASS' if report['rust_toolchain'] else 'FAIL'}")
    print(f"Compilation: {'PASS' if report['compilation_status'] else 'FAIL'}")
    
    handler_count = sum(1 for v in report['handler_functions'].values() if v)
    total_handlers = len(report['handler_functions'])
    print(f"Handler Functions: {handler_count}/{total_handlers} found")
    
    if report['test_results']:
        test_count = sum(1 for v in report['test_results'].values() if v)
        total_tests = len(report['test_results'])
        print(f"Test Modules: {test_count}/{total_tests} passed")
    
    print("\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Return appropriate exit code
    if report['compilation_status'] and all(report['handler_functions'].values()):
        print("\n[SUCCESS] Integration verification completed successfully!")
        return 0
    else:
        print("\n[WARNING] Integration verification found issues that need to be addressed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())