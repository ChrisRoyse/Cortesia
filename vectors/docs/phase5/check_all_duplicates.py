#!/usr/bin/env python3
"""Check all duplicate patterns in task files"""

import os
import re
from pathlib import Path
from collections import defaultdict

def main():
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    task_files = sorted([f.name for f in phase5_dir.glob("task_*.md")])
    
    # Parse task numbers more carefully
    task_patterns = defaultdict(list)
    
    for filename in task_files:
        # Match pattern like task_101_ or task_101.md
        match = re.match(r"task_(\d+)(_|\.)", filename)
        if match:
            num = match.group(1)
            task_patterns[num].append(filename)
    
    print(f"Total files: {len(task_files)}")
    print("\nTask number distribution:")
    
    duplicates = []
    for num in sorted(task_patterns.keys(), key=int):
        files = task_patterns[num]
        if len(files) > 1:
            duplicates.append((num, files))
            print(f"\nTask {num} has {len(files)} variants:")
            for f in files:
                print(f"  - {f}")
    
    print(f"\n\nSummary: Found {len(duplicates)} task numbers with multiple files")
    
    # Check for gaps in numbering
    all_nums = sorted([int(n) for n in task_patterns.keys()])
    gaps = []
    for i in range(1, max(all_nums)):
        if i not in all_nums:
            gaps.append(i)
    
    if gaps:
        print(f"\nGaps in numbering: {gaps}")
    
    return duplicates, gaps

if __name__ == "__main__":
    dups, gaps = main()