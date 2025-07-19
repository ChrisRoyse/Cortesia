#!/usr/bin/env python3
"""
Feedback Processor - Implements the autonomous feedback loop
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import hashlib

class FeedbackProcessor:
    def __init__(self):
        self.project_root = Path.cwd()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.db_path = self.logs_dir / "feedback_history.db"
        self.init_database()
        
    def init_database(self):
        """Initialize feedback history database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT,
                file_path TEXT,
                content_hash TEXT,
                overall_confidence REAL,
                actions_taken INTEGER,
                success_rate REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                category TEXT,
                finding TEXT,
                confidence REAL,
                resolved BOOLEAN,
                resolution_session_id TEXT,
                FOREIGN KEY (session_id) REFERENCES verification_sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                finding_id INTEGER,
                action_type TEXT,
                action_description TEXT,
                success BOOLEAN,
                error_message TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES verification_sessions(session_id),
                FOREIGN KEY (finding_id) REFERENCES findings(finding_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process_verification_results(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification results and update feedback loop"""
        
        session_id = session_data.get("session_id", self.generate_session_id())
        triangulation = session_data.get("triangulation_results", {})
        hook_context = session_data.get("hook_context", {})
        
        # Calculate content hash
        content = session_data.get("content", "")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check for previous verifications of the same content
        similar_sessions = self.find_similar_sessions(content_hash)
        
        # Analyze patterns
        patterns = self.analyze_patterns(triangulation, similar_sessions)
        
        # Store session data
        self.store_session(
            session_id=session_id,
            file_path=hook_context.get("file_path", ""),
            content_hash=content_hash,
            triangulation=triangulation
        )
        
        # Generate adaptive corrections
        corrections = self.generate_adaptive_corrections(
            triangulation=triangulation,
            patterns=patterns,
            previous_sessions=similar_sessions
        )
        
        return {
            "session_id": session_id,
            "patterns_detected": patterns,
            "adaptive_corrections": corrections,
            "learning_insights": self.extract_learning_insights(patterns),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"session_{timestamp}"
    
    def find_similar_sessions(self, content_hash: str) -> List[Dict[str, Any]]:
        """Find previous verification sessions for similar content"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, timestamp, overall_confidence, success_rate
            FROM verification_sessions
            WHERE content_hash = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (content_hash,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "timestamp": row[1],
                "overall_confidence": row[2],
                "success_rate": row[3]
            })
        
        conn.close()
        return sessions
    
    def analyze_patterns(self, current_triangulation: Dict[str, Any], previous_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across verification sessions"""
        
        patterns = {
            "recurring_issues": [],
            "improvement_trends": [],
            "persistent_problems": [],
            "resolution_effectiveness": {}
        }
        
        if not previous_sessions:
            return patterns
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analyze recurring issues
        for category in ["what_broken", "works_but_shouldnt", "doesnt_but_pretends"]:
            current_findings = set()
            
            # Get current findings
            convergent = current_triangulation.get("convergent_findings", {}).get(category, [])
            for finding in convergent:
                current_findings.add(finding["finding"])
            
            # Check against previous sessions
            for session in previous_sessions:
                cursor.execute("""
                    SELECT finding, confidence, resolved
                    FROM findings
                    WHERE session_id = ? AND category = ?
                """, (session["session_id"], category))
                
                for row in cursor.fetchall():
                    finding, confidence, resolved = row
                    if finding in current_findings and not resolved:
                        patterns["recurring_issues"].append({
                            "finding": finding,
                            "category": category,
                            "occurrences": len([s for s in previous_sessions]),
                            "average_confidence": confidence
                        })
        
        # Analyze improvement trends
        if len(previous_sessions) >= 2:
            confidence_trend = [s["overall_confidence"] for s in previous_sessions]
            if len(confidence_trend) >= 2:
                trend_direction = "improving" if confidence_trend[0] > confidence_trend[-1] else "declining"
                patterns["improvement_trends"].append({
                    "metric": "overall_confidence",
                    "direction": trend_direction,
                    "change": confidence_trend[0] - confidence_trend[-1]
                })
        
        conn.close()
        return patterns
    
    def store_session(self, session_id: str, file_path: str, content_hash: str, triangulation: Dict[str, Any]):
        """Store verification session data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate metrics
        overall_confidence = triangulation.get("overall_confidence", 0)
        
        # Store session
        cursor.execute("""
            INSERT INTO verification_sessions 
            (session_id, timestamp, file_path, content_hash, overall_confidence, actions_taken, success_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            file_path,
            content_hash,
            overall_confidence,
            0,  # Will be updated after corrections
            0.0  # Will be updated after corrections
        ))
        
        # Store findings
        for category, findings_data in triangulation.get("convergent_findings", {}).items():
            for finding in findings_data:
                cursor.execute("""
                    INSERT INTO findings 
                    (session_id, category, finding, confidence, resolved, resolution_session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    category,
                    finding["finding"],
                    finding["confidence"],
                    False,
                    None
                ))
        
        conn.commit()
        conn.close()
    
    def generate_adaptive_corrections(self, triangulation: Dict[str, Any], patterns: Dict[str, Any], previous_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate corrections that adapt based on historical patterns"""
        
        corrections = []
        
        # Prioritize recurring issues
        recurring_issues = {issue["finding"]: issue for issue in patterns.get("recurring_issues", [])}
        
        # Process critical alerts with pattern awareness
        for alert in triangulation.get("critical_alerts", []):
            finding = alert["finding"]
            
            correction = {
                "finding": finding,
                "category": alert["category"],
                "priority": "critical",
                "confidence": alert["confidence"],
                "action": f"Fix: {finding}"
            }
            
            # Enhance correction if it's a recurring issue
            if finding in recurring_issues:
                issue_data = recurring_issues[finding]
                correction["enhanced_action"] = f"Permanent fix required - occurred {issue_data['occurrences']} times"
                correction["priority"] = "urgent"
                correction["recommended_approach"] = self.get_recommended_approach(issue_data)
            
            corrections.append(correction)
        
        # Add pattern-based corrections
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == "persistent_problems":
                for problem in pattern_data:
                    corrections.append({
                        "finding": problem["description"],
                        "category": "systematic",
                        "priority": "high",
                        "action": f"Address systematic issue: {problem['description']}",
                        "confidence": 85
                    })
        
        # Sort by priority
        priority_order = {"urgent": 0, "critical": 1, "high": 2, "medium": 3, "low": 4}
        corrections.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return corrections
    
    def get_recommended_approach(self, issue_data: Dict[str, Any]) -> str:
        """Get recommended approach based on issue history"""
        
        occurrences = issue_data.get("occurrences", 0)
        category = issue_data.get("category", "")
        
        if occurrences >= 3:
            if category == "what_broken":
                return "Refactor the entire component - multiple fix attempts have failed"
            elif category == "works_but_shouldnt":
                return "Redesign the architecture - the current approach is fundamentally flawed"
            elif category == "doesnt_but_pretends":
                return "Implement comprehensive error handling and logging"
        elif occurrences >= 2:
            return "Apply a more thorough fix - previous attempt was incomplete"
        else:
            return "Standard fix approach"
    
    def extract_learning_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Extract learning insights from patterns"""
        
        insights = []
        
        # Insights from recurring issues
        recurring = patterns.get("recurring_issues", [])
        if recurring:
            insights.append(f"Found {len(recurring)} recurring issues that need permanent solutions")
            
            # Group by category
            by_category = {}
            for issue in recurring:
                cat = issue["category"]
                if cat not in by_category:
                    by_category[cat] = 0
                by_category[cat] += 1
            
            for cat, count in by_category.items():
                insights.append(f"{count} recurring issues in category '{cat}'")
        
        # Insights from trends
        trends = patterns.get("improvement_trends", [])
        for trend in trends:
            if trend["direction"] == "improving":
                insights.append(f"Code quality improving - {trend['metric']} increased by {trend['change']:.1f}")
            else:
                insights.append(f"Code quality declining - {trend['metric']} decreased by {abs(trend['change']):.1f}")
        
        # Insights from resolution effectiveness
        effectiveness = patterns.get("resolution_effectiveness", {})
        if effectiveness:
            avg_success = sum(effectiveness.values()) / len(effectiveness)
            insights.append(f"Average correction success rate: {avg_success:.1f}%")
        
        return insights
    
    def update_correction_result(self, session_id: str, finding_id: int, success: bool, error_message: Optional[str] = None):
        """Update the result of a correction attempt"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO corrections 
            (session_id, finding_id, action_type, action_description, success, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            finding_id,
            "automated_fix",
            "Automated correction attempt",
            success,
            error_message,
            datetime.now().isoformat()
        ))
        
        if success:
            cursor.execute("""
                UPDATE findings 
                SET resolved = ?, resolution_session_id = ?
                WHERE finding_id = ?
            """, (True, session_id, finding_id))
        
        conn.commit()
        conn.close()
    
    def get_learning_summary(self, time_period_days: int = 7) -> Dict[str, Any]:
        """Get a summary of learning from the feedback loop"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=time_period_days)).isoformat()
        
        # Get session statistics
        cursor.execute("""
            SELECT COUNT(*), AVG(overall_confidence), AVG(success_rate)
            FROM verification_sessions
            WHERE timestamp > ?
        """, (cutoff_date,))
        
        session_count, avg_confidence, avg_success = cursor.fetchone()
        
        # Get finding statistics
        cursor.execute("""
            SELECT category, COUNT(*), SUM(resolved)
            FROM findings f
            JOIN verification_sessions v ON f.session_id = v.session_id
            WHERE v.timestamp > ?
            GROUP BY category
        """, (cutoff_date,))
        
        finding_stats = {}
        for row in cursor.fetchall():
            category, total, resolved = row
            finding_stats[category] = {
                "total": total,
                "resolved": resolved or 0,
                "resolution_rate": (resolved or 0) / total if total > 0 else 0
            }
        
        conn.close()
        
        return {
            "time_period_days": time_period_days,
            "total_sessions": session_count or 0,
            "average_confidence": avg_confidence or 0,
            "average_success_rate": avg_success or 0,
            "finding_statistics": finding_stats,
            "generated_at": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the feedback processor
    processor = FeedbackProcessor()
    
    # Simulate processing results
    test_session_data = {
        "session_id": "test_001",
        "content": "test code content",
        "hook_context": {"file_path": "test.py"},
        "triangulation_results": {
            "overall_confidence": 85,
            "convergent_findings": {
                "what_broken": [
                    {"finding": "Null pointer exception", "confidence": 90}
                ]
            },
            "critical_alerts": [
                {"finding": "Null pointer exception", "category": "what_broken", "confidence": 90}
            ]
        }
    }
    
    results = processor.process_verification_results(test_session_data)
    print("Processing Results:", json.dumps(results, indent=2))
    
    # Get learning summary
    summary = processor.get_learning_summary()
    print("\nLearning Summary:", json.dumps(summary, indent=2))