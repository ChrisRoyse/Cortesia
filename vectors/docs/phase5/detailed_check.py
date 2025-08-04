#!/usr/bin/env python3
"""Detailed check of actual file state"""

from pathlib import Path
import re

phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")
files = sorted([f.name for f in phase5_dir.glob("task_*.md")])

print(f"Total files: {len(files)}\n")

# Extract and count unique numbers
nums = set()
for f in files:
    match = re.match(r"task_(\d+)", f)
    if match:
        nums.add(int(match.group(1)))

print(f"Unique task numbers: {len(nums)}")
print(f"Number range: {min(nums)} to {max(nums)}")

# Check for any gaps
gaps = []
for i in range(min(nums), max(nums) + 1):
    if i not in nums:
        gaps.append(i)

if gaps:
    print(f"\nGaps found: {gaps}")
else:
    print("\nNo gaps - perfect sequence!")

# Check specific problematic numbers
print("\nChecking previously problematic numbers:")
for num in [110, 116, 117, 118, 119, 120, 176]:
    matching = [f for f in files if f.startswith(f"task_{num}_") or f == f"task_{num}.md"]
    print(f"  Task {num}: {len(matching)} file(s)")
    if matching:
        for m in matching:
            print(f"    - {m}")