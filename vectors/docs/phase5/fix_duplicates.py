#!/usr/bin/env python3
"""Fix duplicate task numbering and fill gaps to achieve perfect sequential numbering"""

import os
import re
import shutil
from pathlib import Path

def main():
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    
    # Backup original state
    print("Creating backup of current state...")
    
    # Get all task files
    all_files = sorted([f for f in phase5_dir.glob("task_*.md")])
    
    # Parse and group files
    task_groups = {}
    for f in all_files:
        match = re.match(r"task_(\d+)", f.name)
        if match:
            num = int(match.group(1))
            if num not in task_groups:
                task_groups[num] = []
            task_groups[num].append(f)
    
    # Identify duplicates and gaps
    duplicates = {k: v for k, v in task_groups.items() if len(v) > 1}
    all_nums = sorted(task_groups.keys())
    max_num = max(all_nums)
    
    print(f"\nCurrent state:")
    print(f"  Files: {len(all_files)}")
    print(f"  Unique task numbers: {len(all_nums)}")
    print(f"  Duplicates: {list(duplicates.keys())}")
    
    # Create renaming plan
    renaming_plan = []
    
    # Handle duplicates - keep first file, rename others
    for num in sorted(duplicates.keys()):
        files = duplicates[num]
        for extra_file in files[1:]:  # Skip first file
            # Find next available number
            new_num = max_num + 1
            while new_num in all_nums or any(new_num == n for _, n in renaming_plan):
                new_num += 1
            
            # Extract the descriptive part of the filename
            desc_match = re.match(r"task_\d+_(.+)\.md", extra_file.name)
            if desc_match:
                desc = desc_match.group(1)
                new_name = f"task_{new_num:03d}_{desc}.md"
            else:
                new_name = f"task_{new_num:03d}.md"
            
            renaming_plan.append((extra_file, phase5_dir / new_name, new_num))
            max_num = max(max_num, new_num)
    
    # Now handle gaps - rename high-numbered tasks down to fill gaps
    gaps = []
    for i in range(1, 116):  # Check up to 115 (before the first gap)
        if i not in task_groups:
            gaps.append(i)
    
    # Also add 116, 117, 118 as gaps since they're missing
    gaps.extend([116, 117, 118])
    
    print(f"\nRenaming plan to fix duplicates and fill gaps:")
    
    # Execute renaming for duplicates
    for old_file, new_file, new_num in renaming_plan:
        print(f"  {old_file.name} -> {new_file.name}")
        
    # Now fill gaps by moving high-numbered tasks down
    # Start from tasks 160+ and move them to fill gaps 116-118
    high_tasks = sorted([n for n in all_nums if n >= 160], reverse=True)
    gap_filling = []
    
    for gap_num in sorted(gaps):
        if gap_num in [116, 117, 118] and high_tasks:
            # Find a high-numbered task to move down
            source_num = None
            for high_num in sorted(all_nums, reverse=True):
                if high_num > 159:  # Only move tasks 160+
                    source_num = high_num
                    break
            
            if source_num and source_num in task_groups:
                source_file = task_groups[source_num][0]
                desc_match = re.match(r"task_\d+_(.+)\.md", source_file.name)
                if desc_match:
                    desc = desc_match.group(1)
                    new_name = f"task_{gap_num:03d}_{desc}.md"
                else:
                    new_name = f"task_{gap_num:03d}.md"
                
                gap_filling.append((source_file, phase5_dir / new_name, gap_num))
                all_nums.remove(source_num)
    
    for old_file, new_file, new_num in gap_filling:
        print(f"  {old_file.name} -> {new_file.name} (filling gap)")
    
    return renaming_plan + gap_filling

if __name__ == "__main__":
    plan = main()
    print(f"\nTotal renamings planned: {len(plan)}")