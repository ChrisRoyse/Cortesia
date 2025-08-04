#!/usr/bin/env python3
"""Full analysis of task files to find all duplicates"""

import os
import re
from pathlib import Path
from collections import defaultdict

def main():
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    
    # Get all task files
    all_files = sorted([f.name for f in phase5_dir.glob("task_*.md")])
    
    print(f"Total task files: {len(all_files)}\n")
    
    # Look for files that start with same task number
    task_num_groups = defaultdict(list)
    
    for filename in all_files:
        # Extract just the number part
        if filename.startswith("task_"):
            # Get the numeric part after task_
            parts = filename[5:].split('_', 1)  # Split after "task_"
            num_part = parts[0].split('.')[0]  # Remove .md if directly attached
            try:
                num = int(num_part)
                task_num_groups[num].append(filename)
            except ValueError:
                print(f"Could not parse number from: {filename}")
    
    # Find all duplicates
    duplicates = {}
    for num, files in sorted(task_num_groups.items()):
        if len(files) > 1:
            duplicates[num] = files
    
    print(f"Found {len(duplicates)} task numbers with duplicates:\n")
    
    for num, files in sorted(duplicates.items()):
        print(f"Task {num:03d} ({len(files)} files):")
        for f in sorted(files):
            print(f"  - {f}")
        print()
    
    # Check for specific ranges
    print("\nDetailed check for 101-120 range:")
    for i in range(101, 121):
        if i in task_num_groups:
            files = task_num_groups[i]
            if len(files) > 1:
                print(f"  {i}: {len(files)} files - DUPLICATE")
            else:
                print(f"  {i}: 1 file - OK")
        else:
            print(f"  {i}: MISSING")
    
    # List all unique task numbers
    all_nums = sorted(task_num_groups.keys())
    print(f"\nTask number range: {min(all_nums)} to {max(all_nums)}")
    print(f"Unique task numbers: {len(all_nums)}")
    print(f"Total files: {len(all_files)}")
    print(f"Extra files due to duplicates: {len(all_files) - len(all_nums)}")

if __name__ == "__main__":
    main()