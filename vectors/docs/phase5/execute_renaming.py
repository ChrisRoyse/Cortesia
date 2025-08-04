#!/usr/bin/env python3
"""Execute the renaming to fix duplicate task numbers"""

import os
import shutil
from pathlib import Path

def main():
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    
    # Simple renaming plan for the 4 duplicate files
    # We'll rename them to use gaps 116, 117, 118 and 176
    renaming_plan = [
        ("task_110_integrate_health_score_in_metrics.md", "task_116_integrate_health_score_in_metrics.md"),
        ("task_119_add_performance_optimization_recommendations_alerts.md", "task_117_add_performance_optimization_recommendations_alerts.md"),
        ("task_119_add_start_method.md", "task_118_add_start_method.md"),
        ("task_120_add_stop_method.md", "task_176_add_stop_method.md"),
    ]
    
    print("Executing renaming plan:")
    print("=" * 60)
    
    for old_name, new_name in renaming_plan:
        old_path = phase5_dir / old_name
        new_path = phase5_dir / new_name
        
        if old_path.exists():
            print(f"Renaming: {old_name}")
            print(f"      To: {new_name}")
            
            # Read the file content
            content = old_path.read_text(encoding='utf-8')
            
            # Update the task number in the title
            old_num = old_name.split('_')[1]
            new_num = new_name.split('_')[1]
            content = content.replace(f"# Task {old_num}:", f"# Task {new_num}:")
            
            # Update Next Task references if needed
            if "task_110_" in old_name:
                content = content.replace("Task 111:", "Task 111:")  # No change needed
            elif "task_119_add_performance" in old_name:
                content = content.replace("Task 120:", "Task 118:")
                content = content.replace("task_120_", "task_118_")
            elif "task_119_add_start" in old_name:
                content = content.replace("Task 120:", "Task 119:")
                content = content.replace("task_120_", "task_119_")
            elif "task_120_add_stop" in old_name:
                content = content.replace("Task 121:", "Task 120:")
                content = content.replace("task_121_", "task_120_")
            
            # Write new file
            new_path.write_text(content, encoding='utf-8')
            
            # Delete old file
            old_path.unlink()
            
            print(f"  ✓ Complete\n")
        else:
            print(f"  ✗ File not found: {old_name}\n")
    
    print("\nVerifying results...")
    
    # Check for any remaining duplicates
    all_files = sorted([f.name for f in phase5_dir.glob("task_*.md")])
    task_nums = {}
    for f in all_files:
        if f.startswith("task_"):
            num = f.split('_')[1].split('.')[0]
            if num not in task_nums:
                task_nums[num] = []
            task_nums[num].append(f)
    
    duplicates = [k for k, v in task_nums.items() if len(v) > 1]
    
    print(f"Total files: {len(all_files)}")
    print(f"Unique task numbers: {len(task_nums)}")
    print(f"Remaining duplicates: {duplicates if duplicates else 'None'}")
    
    return len(duplicates) == 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ SUCCESS: All duplicates resolved!")
    else:
        print("\n✗ FAILED: Duplicates still exist")