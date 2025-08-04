#!/usr/bin/env python3
"""Verify what looked like duplicates"""

from pathlib import Path

phase5_dir = Path(r"C:\code\LLMKG\vectors\docs\phase5")

# Check the files I thought were duplicates
patterns = [
    "task_101*.md",
    "task_102*.md", 
    "task_103*.md",
    "task_104*.md",
    "task_106*.md",
    "task_107*.md",
    "task_108*.md",
    "task_109*.md",
    "task_111*.md",
    "task_112*.md",
    "task_113*.md",
    "task_114*.md",
    "task_117*.md",
    "task_118*.md"
]

for pattern in patterns:
    files = list(phase5_dir.glob(pattern))
    if len(files) > 1:
        print(f"\n{pattern} has {len(files)} files:")
        for f in sorted(files):
            print(f"  - {f.name}")
    elif len(files) == 1:
        print(f"{pattern}: 1 file - {files[0].name}")
    else:
        print(f"{pattern}: NO FILES")