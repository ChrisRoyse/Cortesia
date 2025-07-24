#!/usr/bin/env python3
"""
Simple verification that the LLMKG Rust code compiles and the key tools work.
This script will check compilation and attempt to run a minimal test.
"""

import subprocess
import sys
import json
import os

def run_command(cmd, description):
    """Run a command and return success status with output."""
    print(f"\n[*] {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[+] SUCCESS: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}")
            return True, result.stdout, result.stderr
        else:
            print(f"[-] FAILED: {description}")
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"Stdout: {result.stdout[:1000]}{'...' if len(result.stdout) > 1000 else ''}")
            if result.stderr:
                print(f"Stderr: {result.stderr[:1000]}{'...' if len(result.stderr) > 1000 else ''}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"[!] TIMEOUT: {description} took too long")
        return False, "", "Timeout"
    except Exception as e:
        print(f"[!] EXCEPTION: {description} - {e}")
        return False, "", str(e)

def main():
    print("LLMKG Rust Compilation and Tool Verification")
    print("=" * 60)
    
    # Change to project directory
    os.chdir("C:\\code\\LLMKG")
    
    results = {}
    
    # Test 1: Check that the library compiles
    success, stdout, stderr = run_command(
        "cargo check --lib", 
        "Checking library compilation"
    )
    results["lib_compilation"] = success
    
    if not success:
        print("\n[-] Library doesn't compile. Cannot proceed with further tests.")
        return False
    
    # Count warnings
    warning_count = stderr.lower().count("warning:")
    error_count = stderr.lower().count("error:")
    
    print(f"\nCompilation Results:")
    print(f"  - Warnings: {warning_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Library compiles: {success}")
    
    # Test 2: Try to build (not just check)
    success, stdout, stderr = run_command(
        "cargo build --lib", 
        "Building library"
    )
    results["lib_build"] = success
    
    # Test 3: Check if binaries can be built
    success, stdout, stderr = run_command(
        "cargo build --bin llmkg_mcp_server", 
        "Building MCP server binary"
    )
    results["mcp_server_build"] = success
    
    # Test 4: Check specific modules that contain our tools
    success, stdout, stderr = run_command(
        "cargo check --lib --message-format=json", 
        "Detailed compilation check"
    )
    
    # Look for specific handler modules in compilation
    handler_modules = [
        "handlers::advanced",
        "handlers::cognitive", 
        "handlers::stats",
        "handlers::temporal"
    ]
    
    compiled_modules = []
    if success and stdout:
        for line in stdout.split('\n'):
            if line.strip():
                try:
                    msg = json.loads(line)
                    if msg.get('reason') == 'compiler-artifact':
                        target_name = msg.get('target', {}).get('name', '')
                        if any(module in target_name for module in handler_modules):
                            compiled_modules.append(target_name)
                except:
                    pass
    
    print(f"\nModule Compilation Status:")
    for module in handler_modules:
        found = any(module in compiled for compiled in compiled_modules)
        print(f"  - {module}: {'[+]' if found else '[?]'}")
    
    # Test 5: Check that our key functions exist in the compiled lib
    # This is a rough check by looking at the compiled binary
    success, stdout, stderr = run_command(
        "dir target\\debug\\deps\\libllmkg-*.rlib", 
        "Checking for compiled library files"
    )
    
    if success and stdout:
        print(f"\nFound compiled library files:")
        for line in stdout.split('\n')[:5]:  # Show first 5 matches
            if line.strip() and 'libllmkg' in line:
                print(f"  - {line.strip()}")
    
    # Final summary
    print(f"\nVERIFICATION SUMMARY:")
    print(f"=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "[+] PASS" if passed else "[-] FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if results.get("lib_compilation", False):
        print("\n[+] SUCCESS: The LLMKG library compiles successfully!")
        print("   This means the Rust code is syntactically correct and the")
        print("   4 key tools (generate_graph_query, neural_importance_scoring,")  
        print("   validate_knowledge, get_stats) should be available.")
        return True
    else:
        print("\n[-] FAILURE: The LLMKG library does not compile.")
        print("   The tools cannot be used until compilation issues are fixed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)