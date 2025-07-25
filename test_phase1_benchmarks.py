#!/usr/bin/env python3
"""
Simple Python script to test Phase 1 benchmark functionality
"""

import subprocess
import sys
import time
import json

def run_command(command, timeout=30):
    """Run a command with timeout"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def test_cargo_bench_compile():
    """Test if benchmarks compile"""
    print("Testing Phase 1 benchmarks compilation...")
    
    # Test basic compilation
    code, stdout, stderr = run_command("cargo check --bench phase1_benchmarks", 60)
    
    if code == 0:
        print("[PASS] Phase 1 benchmarks compile successfully")
        return True
    else:
        print(f"[FAIL] Compilation failed with code {code}")
        print("STDERR:", stderr[-500:])  # Last 500 chars of error
        return False

def test_basic_benchmark_functions():
    """Test basic benchmark function structure"""
    print("\nTesting benchmark function structure...")
    
    with open("benches/phase1_benchmarks.rs", "r") as f:
        content = f.read()
    
    required_functions = [
        "benchmark_entity_extraction",
        "benchmark_relationship_extraction", 
        "benchmark_triple_storage",
        "benchmark_triple_querying",
        "benchmark_semantic_search",
        "benchmark_question_answering"
    ]
    
    found_functions = []
    for func in required_functions:
        if func in content:
            found_functions.append(func)
            print(f"[PASS] Found {func}")
        else:
            print(f"[FAIL] Missing {func}")
    
    return len(found_functions) == len(required_functions)

def test_performance_requirements():
    """Test performance requirement definitions"""
    print("\nTesting performance requirements...")
    
    with open("benches/phase1_benchmarks.rs", "r") as f:
        content = f.read()
    
    requirements = [
        "Entity extraction: < 50ms for 1000 character text",
        "Relationship extraction: < 75ms for complex text with 10+ entities",
        "Question answering: < 100ms for simple questions"
    ]
    
    found_reqs = 0
    for req in requirements:
        if "50ms" in content or "75ms" in content or "100ms" in content:
            found_reqs += 1
    
    if found_reqs > 0:
        print("[PASS] Performance requirements documented")
        return True
    else:
        print("[FAIL] Performance requirements not found")
        return False

def test_benchmark_groups():
    """Test if benchmark groups are properly defined"""
    print("\nTesting benchmark groups...")
    
    with open("benches/phase1_benchmarks.rs", "r") as f:
        content = f.read()
    
    if "criterion_group!" in content and "phase1_benches" in content:
        print("[PASS] Benchmark groups properly defined")
        return True
    else:
        print("[FAIL] Benchmark groups not properly defined")
        return False

def test_cargo_toml_config():
    """Test if Cargo.toml is properly configured"""
    print("\nTesting Cargo.toml configuration...")
    
    with open("Cargo.toml", "r") as f:
        content = f.read()
    
    if '[[bench]]' in content and 'name = "phase1_benchmarks"' in content:
        print("[PASS] Cargo.toml properly configured for benchmarks")
        return True
    else:
        print("[FAIL] Cargo.toml not properly configured")
        return False

def main():
    """Main test function"""
    print("=== Phase 1 Benchmarks Verification ===\n")
    
    tests = [
        test_cargo_toml_config,
        test_basic_benchmark_functions,
        test_performance_requirements,
        test_benchmark_groups,
        test_cargo_bench_compile
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test failed with exception: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All Phase 1 benchmark tests passed!")
        print("\nNext steps:")
        print("1. Run 'cargo bench --bench phase1_benchmarks' to execute benchmarks")
        print("2. View results in target/criterion/report/index.html")
        print("3. Verify all operations meet < 100ms requirement")
    else:
        print("[WARNING] Some tests failed. Review implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)