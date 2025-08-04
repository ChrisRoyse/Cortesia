#!/usr/bin/env python3
"""
TDD Validation Suite for Phase 1 Tasks - Following CLAUDE.md Protocol
RED Phase: These tests MUST fail initially, proving the gap exists
"""

import os
import re
from pathlib import Path

class TaskValidator:
    """Validates tasks meet 100/100 quality standard"""
    
    def __init__(self, task_dir="C:/code/LLMKG/vectors/docs/phase1"):
        self.task_dir = Path(task_dir)
        self.failures = []
        
    def test_all_tasks_meet_standard(self):
        """Main test that MUST pass for 100/100"""
        task_files = sorted(self.task_dir.glob("task_*.md"))
        
        total_score = 0
        max_score = 0
        
        for task_file in task_files:
            if "FIXED_EXAMPLE" in task_file.name or "COMPLETE_100" in task_file.name:
                continue  # Skip examples
                
            task_content = task_file.read_text(encoding='utf-8')
            task_num = task_file.stem
            
            # Test each quality criterion
            scores = {
                'format': self._test_format(task_content, task_num),
                'verification': self._test_verification(task_content, task_num),
                'context': self._test_context(task_content, task_num),
                'code': self._test_code_completeness(task_content, task_num),
                'timing': self._test_timing(task_content, task_num),
            }
            
            task_score = sum(scores.values())
            total_score += task_score
            max_score += 100
            
            if task_score < 100:
                self.failures.append(f"{task_num}: {task_score}/100 - Missing: {[k for k,v in scores.items() if v < 20]}")
        
        # This test WILL FAIL initially (proving the gap)
        assert total_score == max_score, f"Score {total_score}/{max_score} - NOT 100/100\nFailures:\n" + "\n".join(self.failures[:10])
        
    def _test_format(self, content, task_num):
        """Test: Proper format structure (20 points)"""
        required = [
            r"\*\*Time:\*\* 10 minutes \(2 min read, 6 min implement, 2 min verify\)",
            r"\*\*Prerequisites:\*\*",
            r"\*\*Input Files:\*\*",
            r"## Complete Context \(For AI with ZERO Knowledge\)",
            r"## Exact Steps",
            r"## Verification Steps",
        ]
        
        score = 0
        for pattern in required:
            if re.search(pattern, content):
                score += 20/len(required)
        
        if score < 20:
            self.failures.append(f"{task_num}: Format incomplete ({score}/20)")
        
        return score
    
    def _test_verification(self, content, task_num):
        """Test: Comprehensive verification section (20 points)"""
        required = [
            "## If This Task Fails",
            "Error 1:",
            "Error 2:", 
            "Error 3:",
            "Solution:",
            "## Troubleshooting Checklist",
            "## Recovery Procedures",
        ]
        
        score = 0
        for item in required:
            if item in content:
                score += 20/len(required)
                
        if score < 20:
            self.failures.append(f"{task_num}: Verification incomplete ({score}/20)")
            
        return score
    
    def _test_context(self, content, task_num):
        """Test: Complete context for zero knowledge (20 points)"""
        indicators = [
            "What is",  # Technology explanations
            "Why",      # Purpose explanations
            len(content) > 3000,  # Sufficient detail
            "For AI with ZERO Knowledge" in content,
            content.count("**") > 10,  # Proper emphasis
        ]
        
        score = sum(20/len(indicators) for ind in indicators if ind)
        
        if score < 20:
            self.failures.append(f"{task_num}: Context insufficient ({score}/20)")
            
        return score
    
    def _test_code_completeness(self, content, task_num):
        """Test: Complete runnable code (20 points)"""
        code_blocks = content.count("```")
        has_imports = "use " in content or "import " in content or "require" in content
        has_exact_paths = "C:/code/LLMKG" in content or "C:\\code\\LLMKG" in content
        
        score = 0
        if code_blocks >= 2:
            score += 10
        if has_imports:
            score += 5
        if has_exact_paths:
            score += 5
            
        if score < 20:
            self.failures.append(f"{task_num}: Code incomplete ({score}/20)")
            
        return score
    
    def _test_timing(self, content, task_num):
        """Test: Validated 10-minute timing (20 points)"""
        has_time_structure = "2 min read, 6 min implement, 2 min verify" in content
        step_count = content.count("### Step")
        
        score = 0
        if has_time_structure:
            score += 10
        if 2 <= step_count <= 4:
            score += 10
            
        if score < 20:
            self.failures.append(f"{task_num}: Timing invalid ({score}/20)")
            
        return score

if __name__ == "__main__":
    print("=== RED PHASE: Running Failing Tests ===")
    print("These tests MUST fail to prove the gap exists\n")
    
    validator = TaskValidator()
    
    try:
        validator.test_all_tasks_meet_standard()
        print("ERROR: Tests passed but shouldn't have! Gap doesn't exist!")
    except AssertionError as e:
        print(f"✓ TEST FAILED AS EXPECTED (Good!):\n{e}")
        print("\nNow proceeding to GREEN phase to fix these failures...")