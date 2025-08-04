#!/usr/bin/env python3
"""Accurate validation of task quality"""

import os
from pathlib import Path

def test_single_task(filepath):
    content = Path(filepath).read_text(encoding='utf-8')
    
    scores = {}
    
    # Format test (20 points)
    format_items = [
        r"**Time:**",
        r"**Prerequisites:**", 
        r"**Input Files:**",
        "Complete Context (For AI with ZERO Knowledge)",
        "Exact Steps",
        "Verification Steps"
    ]
    
    format_score = sum(20/len(format_items) for item in format_items if item in content)
    scores['format'] = format_score
    
    # Timing test (20 points) 
    timing_score = 20 if "2 min read, 6 min implement, 2 min verify" in content else 0
    scores['timing'] = timing_score
    
    # Verification test (20 points)
    verification_items = [
        "If This Task Fails",
        "Error 1:",
        "Error 2:", 
        "Error 3:",
        "Solution:",
        "Troubleshooting Checklist",
        "Recovery Procedures"
    ]
    
    verification_score = sum(20/len(verification_items) for item in verification_items if item in content)
    scores['verification'] = verification_score
    
    # Context test (20 points)
    context_score = 0
    if "Complete Context (For AI with ZERO Knowledge)" in content:
        context_score += 5
    if "What is" in content:
        context_score += 5  
    if "Why" in content:
        context_score += 5
    if len(content) > 3000:
        context_score += 5
    scores['context'] = context_score
    
    # Code test (20 points)
    code_score = 0
    if content.count("```") >= 2:
        code_score += 10
    if "use " in content or "import " in content:
        code_score += 5
    if "C:/code/LLMKG" in content:
        code_score += 5
    scores['code'] = code_score
    
    total = sum(scores.values())
    return total, scores

# Test first 10 tasks
task_files = sorted([f for f in os.listdir(".") if f.startswith("task_") and f.endswith(".md") and not "FIXED" in f and not "COMPLETE" in f])[:10]

print("Accurate Task Quality Assessment:")
print("="*50)

total_score = 0
for task_file in task_files:
    score, breakdown = test_single_task(task_file)
    total_score += score
    print(f"{task_file[:20]:<20}: {score:3.0f}/100 (F:{breakdown['format']:.0f} T:{breakdown['timing']:.0f} V:{breakdown['verification']:.0f} C:{breakdown['context']:.0f} Co:{breakdown['code']:.0f})")

print(f"\nAverage: {total_score/len(task_files):.1f}/100")
print(f"Tasks scoring <100: {sum(1 for f in task_files if test_single_task(f)[0] < 100)}/{len(task_files)}")