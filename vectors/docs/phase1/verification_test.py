#!/usr/bin/env python3
"""
TDD Test for Phase 1 Task Verification Enhancement
This test validates that all tasks have comprehensive failure recovery procedures.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def test_task_has_comprehensive_verification(task_content: str, task_name: str) -> bool:
    """Test that each task has comprehensive failure recovery procedures"""
    
    required_sections = [
        "## Verification Steps",
        "### If Primary Verification Fails", 
        "## Troubleshooting Checklist",
        "## Recovery Procedures"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in task_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"FAIL {task_name}: Missing sections: {missing_sections}")
        return False
    
    # Check for specific error scenarios (minimum 3)
    error_patterns = [
        r"Error \d+:",
        r"Solution:",
        r"cargo.*failed",
        r"permission denied", 
        r"not found",
        r"compilation.*fail",
        r"network.*fail"
    ]
    
    error_count = 0
    for pattern in error_patterns:
        if re.search(pattern, task_content, re.IGNORECASE):
            error_count += 1
    
    if error_count < 3:
        print(f"FAIL {task_name}: Only {error_count} error scenarios, need 3+")
        return False
        
    # Check for platform-specific solutions
    platform_coverage = [
        "Windows",
        "Unix", 
        "icacls",
        "chmod"
    ]
    
    platform_count = sum(1 for platform in platform_coverage if platform in task_content)
    if platform_count < 2:
        print(f"FAIL {task_name}: Insufficient platform coverage ({platform_count}/2)")
        return False
    
    # Check for expected output examples
    if "**Expected output:**" not in task_content:
        print(f"FAIL {task_name}: Missing expected output examples")
        return False
        
    # Check for specific commands with exact syntax
    if not re.search(r"```bash\n.*cargo.*\n```", task_content, re.MULTILINE):
        print(f"FAIL {task_name}: Missing specific cargo commands")
        return False
    
    print(f"PASS {task_name}: All verification requirements met")
    return True

def test_all_phase1_tasks() -> Dict[str, bool]:
    """Test all Phase 1 tasks for comprehensive verification"""
    
    phase1_dir = Path("C:/code/LLMKG/vectors/docs/phase1")
    results = {}
    
    if not phase1_dir.exists():
        print(f"ERROR Phase 1 directory not found: {phase1_dir}")
        return results
    
    task_files = list(phase1_dir.glob("task_*.md"))
    print(f"Testing {len(task_files)} Phase 1 tasks...")
    
    for task_file in sorted(task_files):
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            task_name = task_file.stem
            results[task_name] = test_task_has_comprehensive_verification(content, task_name)
            
        except Exception as e:
            print(f"ERROR {task_file.name}: Error reading file - {e}")
            results[task_file.stem] = False
    
    return results

def main():
    """Run the comprehensive verification test"""
    print("RED Phase: Testing for comprehensive verification procedures...")
    print("This test SHOULD FAIL initially - we haven't implemented comprehensive verification yet.\n")
    
    results = test_all_phase1_tasks()
    
    total_tasks = len(results)
    passed_tasks = sum(1 for passed in results.values() if passed)
    failed_tasks = total_tasks - passed_tasks
    
    print(f"\nTest Results:")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Passed: {passed_tasks}")
    print(f"   Failed: {failed_tasks}")
    print(f"   Success rate: {(passed_tasks/total_tasks)*100:.1f}%")
    
    if failed_tasks > 0:
        print(f"\nTarget: Enhance {failed_tasks} tasks with comprehensive verification")
        print("   Required: 3+ error scenarios, platform coverage, expected outputs, exact commands")
        return False
    else:
        print(f"\nAll tasks have comprehensive verification procedures!")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)