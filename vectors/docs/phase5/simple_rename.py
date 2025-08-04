#!/usr/bin/env python3
"""Simple renaming to fix duplicates"""

from pathlib import Path

def main():
    phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
    
    # Renaming plan
    renames = [
        ("task_110_integrate_health_score_in_metrics.md", "task_116_integrate_health_score_in_metrics.md"),
        ("task_119_add_performance_optimization_recommendations_alerts.md", "task_117_add_performance_optimization_recommendations_alerts.md"),
        ("task_119_add_start_method.md", "task_118_add_start_method.md"),
        ("task_120_add_stop_method.md", "task_176_add_stop_method.md"),
    ]
    
    for old, new in renames:
        old_path = phase5_dir / old
        new_path = phase5_dir / new
        
        if old_path.exists():
            print(f"Renaming {old} -> {new}")
            
            # Read content
            content = old_path.read_text(encoding='utf-8')
            
            # Update task number in header
            old_num = old.split('_')[1]
            new_num = new.split('_')[1]
            content = content.replace(f"Task {old_num}:", f"Task {new_num}:")
            
            # Write new file
            new_path.write_text(content, encoding='utf-8')
            
            # Delete old
            old_path.unlink()
            print("  Done")
        else:
            print(f"  Not found: {old}")
    
    # Verify
    print("\nVerifying...")
    files = list(phase5_dir.glob("task_*.md"))
    nums = set()
    dups = []
    
    for f in files:
        num = f.name.split('_')[1]
        if num in nums:
            dups.append(num)
        nums.add(num)
    
    print(f"Total: {len(files)} files")
    print(f"Duplicates: {dups if dups else 'None'}")

if __name__ == "__main__":
    main()