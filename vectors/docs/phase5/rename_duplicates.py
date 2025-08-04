#!/usr/bin/env python3
"""Fix duplicate task numbering in Phase 5 microtasks"""

import os
import re
from pathlib import Path

def main():
    # Get all task files
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    task_files = sorted([f for f in phase5_dir.glob("task_*.md")])
    
    # Group files by task number
    task_groups = {}
    for f in task_files:
        match = re.match(r"task_(\d+)", f.name)
        if match:
            num = int(match.group(1))
            if num not in task_groups:
                task_groups[num] = []
            task_groups[num].append(f)
    
    # Find duplicates
    duplicates = {k: v for k, v in task_groups.items() if len(v) > 1}
    
    print(f"Found {len(task_files)} total task files")
    print(f"Found {len(duplicates)} task numbers with duplicates:")
    
    for num, files in sorted(duplicates.items()):
        print(f"\nTask {num:03d} has {len(files)} files:")
        for f in files:
            print(f"  - {f.name}")
    
    # Create renaming plan
    renaming_plan = []
    next_num = 160  # Start renumbering duplicates from 160
    
    for num in sorted(duplicates.keys()):
        files = duplicates[num]
        # Keep the first file with original number
        for i, f in enumerate(files[1:], 1):  # Skip first file
            new_name = f"task_{next_num:03d}_{f.name.split('_', 2)[2] if '_' in f.name.split('task_')[1] else f.name.split('_', 1)[1]}"
            renaming_plan.append((f, phase5_dir / new_name))
            next_num += 1
    
    print(f"\n\nRenaming Plan ({len(renaming_plan)} files):")
    for old, new in renaming_plan:
        print(f"  {old.name} -> {new.name}")
    
    return renaming_plan, duplicates

if __name__ == "__main__":
    plan, dups = main()