#!/usr/bin/env python3
"""
Combined script that runs auto-fixes and then validates the results.
This provides a complete workflow for fixing embedding dimension issues.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main workflow."""
    print("🚀 LLMKG Embedding Dimension Fix & Validation Workflow")
    print("This script will:")
    print("1. Run initial validation to identify issues")
    print("2. Apply automatic fixes for common patterns") 
    print("3. Re-run validation to check remaining issues")
    print("4. Provide next steps")
    
    if not os.path.exists("Cargo.toml"):
        print("\n❌ ERROR: Not in project root. Please run from the directory containing Cargo.toml")
        return False
    
    # Step 1: Initial validation
    print("\n🔍 STEP 1: Running initial validation...")
    initial_success = run_command("python validate_embedding_fixes.py", "Initial Validation")
    
    # Step 2: Apply automatic fixes
    print("\n🔧 STEP 2: Applying automatic fixes...")
    fix_success = run_command("python auto_fix_dimensions.py", "Automatic Fixes")
    
    # Step 3: Final validation
    print("\n✅ STEP 3: Running final validation...")
    final_success = run_command("python validate_embedding_fixes.py", "Final Validation")
    
    # Step 4: Summary and next steps
    print("\n📊 WORKFLOW SUMMARY")
    print("="*60)
    
    if fix_success:
        print("Auto-fixes were applied to problematic files.")
        print("Check the validation report to see remaining issues.")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review the changes in modified files")
        print("2. Run 'cargo test' to verify tests pass")
        print("3. For remaining issues, manual fixes may be needed")
        print("4. Check EMBEDDING_DIMENSION_VALIDATION_REPORT.md for details")
        print("5. Check QUICK_FIX_SUMMARY.md for manual fix patterns")
        
    else:
        print("No auto-fixes were needed or applied.")
        
    # Check if we still have issues
    if os.path.exists("embedding_validation_report.json"):
        import json
        with open("embedding_validation_report.json", 'r') as f:
            report = json.load(f)
            
        issues_count = report.get("files_with_issues", 0)
        if issues_count == 0:
            print("\n🎉 SUCCESS: All embedding dimension issues have been resolved!")
        else:
            print(f"\n⚠️  WARNING: {issues_count} files still have dimension issues.")
            print("Manual intervention may be required for complex cases.")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)