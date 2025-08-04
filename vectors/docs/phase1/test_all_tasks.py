#!/usr/bin/env python3
"""Test all tasks quality"""

import os
from pathlib import Path
from accurate_test import test_single_task

# Get all task files except examples
task_files = sorted([f for f in os.listdir(".") if f.startswith("task_") and f.endswith(".md") 
                    and not "FIXED" in f and not "COMPLETE" in f and not "OLD" in f])

print(f"Testing {len(task_files)} tasks...")
print("="*60)

scores = []
failing_tasks = []

for task_file in task_files:
    try:
        score, breakdown = test_single_task(task_file)
        scores.append(score)
        
        if score < 100:
            failing_tasks.append((task_file, score, breakdown))
            
    except Exception as e:
        scores.append(0)
        failing_tasks.append((task_file, 0, {"error": str(e)}))

# Summary
total_score = sum(scores)
max_score = len(task_files) * 100
avg_score = total_score / len(task_files) if task_files else 0

print(f"\nSUMMARY:")
print(f"Total Score: {total_score}/{max_score}")
print(f"Average: {avg_score:.1f}/100")
print(f"Perfect Tasks: {len([s for s in scores if s == 100])}/{len(task_files)}")
print(f"Failing Tasks: {len(failing_tasks)}/{len(task_files)}")

print(f"\nFIRST 10 FAILING TASKS:")
for task, score, breakdown in failing_tasks[:10]:
    print(f"{task[:25]:<25}: {score:3.0f}/100")

# Check distribution
print(f"\nSCORE DISTRIBUTION:")
ranges = [(0,20,"0-20"), (20,40,"20-40"), (40,60,"40-60"), (60,80,"60-80"), (80,100,"80-100"), (100,100,"Perfect")]
for min_s, max_s, label in ranges:
    count = len([s for s in scores if min_s <= s <= max_s])
    print(f"{label}: {count} tasks")