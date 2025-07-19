#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid

class SubagentOrchestrator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.tasks_dir = self.project_root / ".claude" / "tasks"
        self.pending_dir = self.tasks_dir / "pending"
        self.completed_dir = self.tasks_dir / "completed"
        
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)

    def parse_hook_input(self):
        try:
            return json.load(sys.stdin)
        except json.JSONDecodeError:
            return {}

    def create_analysis_tasks(self, target_file_path: str, context: dict):
        """Creates task files for all verification agents."""
        agents = [
            "factual_verifier", "logical_consistency_checker", "source_validator",
            "semantic_analyzer", "bias_detector", "confidence_scorer",
            "gap_analyzer", "meta_reviewer"
        ]
        
        for agent in agents:
            task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            task_payload = {
                "task_id": task_id,
                "agent_name": agent,
                "target_file_path": target_file_path,
                "context": context
            }
            task_file = self.pending_dir / f"{task_id}.json"
            with open(task_file, 'w') as f:
                json.dump(task_payload, f, indent=2)
        
        print(f"[ORCHESTRATOR] Created {len(agents)} analysis tasks.")

    def run(self):
        hook_input = self.parse_hook_input()
        
        if "autonomous_correction_task.md" in hook_input.get("tool_input", {}).get("file_path", ""):
            sys.exit(0)

        if hook_input.get("tool_name") in ["Write", "Edit", "MultiEdit"]:
            file_path = hook_input.get("tool_input", {}).get("file_path", "")
            if file_path and Path(file_path).exists():
                context = {
                    "file_path": file_path,
                    "tool_used": hook_input.get("tool_name"),
                    "session_id": hook_input.get("session_id", "")
                }
                self.create_analysis_tasks(file_path, context)
            else:
                sys.exit(0)
        
        # Note: Triangulation logic would need to be adapted to read from the completed directory.
        # This is a more advanced step once the task creation is confirmed to be working.
        print("[ORCHESTRATOR] Task creation complete. Sub-agent manager will process them.")
        sys.exit(0)

if __name__ == "__main__":
    orchestrator = SubagentOrchestrator()
    orchestrator.run()