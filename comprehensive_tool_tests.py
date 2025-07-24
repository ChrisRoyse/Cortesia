#!/usr/bin/env python3
"""
Comprehensive LLMKG MCP Server Tool Testing
5 diverse test scenarios for each of the 10 available tools
"""

import json
import subprocess
import time
import sys
import os
from typing import Dict, Any, Optional, List

class ComprehensiveToolTester:
    def __init__(self, exe_path: str = 'target/debug/llmkg_mcp_server_test.exe'):
        self.exe_path = exe_path
        self.process = None
        self.test_results = []
        self.current_test_id = 0
        
    def start_server(self):
        """Start the MCP server process"""
        print("Starting LLMKG MCP Server for comprehensive tool testing...")
        
        env = os.environ.copy()
        env['RUST_LOG'] = 'warn'  # Reduce log noise
        
        self.process = subprocess.Popen(
            [self.exe_path, '--data-dir', './test_data'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        time.sleep(0.5)
        
        # Initialize the server
        self.send_request({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 0
        })
        
    def send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a request and get response"""
        if not self.process:
            return None
            
        request_str = json.dumps(request)
        
        try:
            self.process.stdin.write(request_str + '\n')
            self.process.stdin.flush()
        except Exception as e:
            print(f"Failed to send request: {e}")
            return None
        
        # Read response with timeout
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout per test
            try:
                line = self.process.stdout.readline()
                if line:
                    response = json.loads(line.strip())
                    return response
            except:
                time.sleep(0.01)
                continue
                
        return None
        
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a specific tool"""
        self.current_test_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": self.current_test_id
        }
        return self.send_request(request)
        
    def test_tool(self, tool_name: str, test_name: str, params: Dict[str, Any], expected_success: bool = True) -> bool:
        """Test a tool and record results"""
        print(f"    {test_name}: ", end="", flush=True)
        
        start_time = time.time()
        response = self.call_tool(tool_name, params)
        elapsed = time.time() - start_time
        
        if response is None:
            result = {
                'tool': tool_name,
                'test': test_name,
                'success': False,
                'error': 'No response (timeout)',
                'elapsed_ms': elapsed * 1000
            }
            self.test_results.append(result)
            print(f"FAIL (timeout)")
            return False
            
        # Check if response indicates success
        success = False
        error_msg = None
        
        if 'result' in response:
            success = True
        elif 'error' in response:
            error_msg = response['error'].get('message', 'Unknown error')
            success = not expected_success  # If we expected failure, error is success
        
        result = {
            'tool': tool_name,
            'test': test_name,
            'success': success,
            'error': error_msg,
            'elapsed_ms': elapsed * 1000,
            'response_size': len(json.dumps(response))
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        time_str = f"{elapsed*1000:.0f}ms"
        if error_msg:
            print(f"{status} ({time_str}) - {error_msg}")
        else:
            print(f"{status} ({time_str})")
        
        return success
    
    def run_all_tool_tests(self):
        """Run comprehensive tests for all 10 tools"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TOOL TESTING - 5 TESTS PER TOOL")
        print("="*80)
        
        # TOOL 1: store_fact
        print(f"\n[1/10] Testing store_fact - Basic triple storage")
        
        self.test_tool("store_fact", "Basic fact storage", {
            "subject": "Einstein",
            "predicate": "is", 
            "object": "scientist"
        })
        
        self.test_tool("store_fact", "Fact with confidence", {
            "subject": "Quantum_Computer",
            "predicate": "invented_by",
            "object": "Multiple_Scientists",
            "confidence": 0.85
        })
        
        self.test_tool("store_fact", "Unicode/International", {
            "subject": "北京",
            "predicate": "capital_of",
            "object": "中国"
        })
        
        self.test_tool("store_fact", "Complex entity names", {
            "subject": "Machine_Learning_Algorithm_XGBoost",
            "predicate": "developed_by",
            "object": "Tianqi_Chen_and_Carlos_Guestrin"
        })
        
        self.test_tool("store_fact", "Duplicate handling", {
            "subject": "Einstein",
            "predicate": "is",
            "object": "scientist"  # Same as first test
        })
        
        # TOOL 2: store_knowledge
        print(f"\n[2/10] Testing store_knowledge - Knowledge chunk storage")
        
        self.test_tool("store_knowledge", "Biography content", {
            "title": "Einstein Biography",
            "content": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity. He received the Nobel Prize in Physics in 1921.",
            "category": "biography"
        })
        
        self.test_tool("store_knowledge", "Technical documentation", {
            "title": "Python List Methods",
            "content": "Python lists have methods like append(), extend(), insert(), remove(), pop(), clear(), index(), count(), sort(), and reverse(). These methods modify the list in-place.",
            "category": "technical",
            "source": "Python Documentation"
        })
        
        self.test_tool("store_knowledge", "Historical events", {
            "title": "World War II Timeline",
            "content": "World War II began on September 1, 1939, when Germany invaded Poland. The war ended on September 2, 1945, with Japan's formal surrender.",
            "category": "historical"
        })
        
        self.test_tool("store_knowledge", "Scientific concepts", {
            "title": "Quantum Entanglement",
            "content": "Quantum entanglement is a physical phenomenon where particles become interconnected and the quantum state of each particle cannot be described independently.",
            "category": "science",
            "source": "Physics Textbook"
        })
        
        self.test_tool("store_knowledge", "Large content chunk", {
            "title": "Comprehensive AI Overview",
            "content": "Artificial Intelligence (AI) is a broad field of computer science. " * 50,  # Large content
            "category": "technology"
        })
        
        # TOOL 3: find_facts
        print(f"\n[3/10] Testing find_facts - Triple pattern matching")
        
        self.test_tool("find_facts", "Subject-only query", {
            "query": {"subject": "Einstein"},
            "limit": 10
        })
        
        self.test_tool("find_facts", "Predicate-only query", {
            "query": {"predicate": "is"},
            "limit": 5  
        })
        
        self.test_tool("find_facts", "Object-only query", {
            "query": {"object": "scientist"},
            "limit": 10
        })
        
        self.test_tool("find_facts", "Combined subject+predicate", {
            "query": {
                "subject": "Einstein",
                "predicate": "is"
            },
            "limit": 10
        })
        
        self.test_tool("find_facts", "Non-existent entity", {
            "query": {"subject": "NonexistentEntity_12345"}
        })
        
        # TOOL 4: ask_question
        print(f"\n[4/10] Testing ask_question - Natural language queries")
        
        self.test_tool("ask_question", "Simple who question", {
            "question": "Who is Einstein?",
            "max_results": 5
        })
        
        self.test_tool("ask_question", "What question with context", {
            "question": "What did Einstein discover?",
            "context": "focusing on physics theories", 
            "max_results": 3
        })
        
        self.test_tool("ask_question", "Where/when question", {
            "question": "When was Einstein born?",
            "max_results": 5
        })
        
        self.test_tool("ask_question", "Complex analytical question", {
            "question": "What are the main contributions of Einstein to modern physics?",
            "context": "looking for specific theories and discoveries",
            "max_results": 10
        })
        
        self.test_tool("ask_question", "Unanswerable question", {
            "question": "What is Einstein's favorite ice cream flavor?",
            "max_results": 5
        })
        
        # TOOL 5: explore_connections
        print(f"\n[5/10] Testing explore_connections - Entity relationship exploration")
        
        self.test_tool("explore_connections", "Single entity exploration", {
            "start_entity": "Einstein",
            "max_depth": 2
        })
        
        self.test_tool("explore_connections", "Deep path exploration", {
            "start_entity": "Einstein",
            "end_entity": "scientist",
            "max_depth": 3
        })
        
        self.test_tool("explore_connections", "Shallow exploration", {
            "start_entity": "Quantum_Computer",
            "max_depth": 1
        })
        
        self.test_tool("explore_connections", "Filtered by relationship", {
            "start_entity": "Einstein",
            "relationship_types": ["is", "developed"],
            "max_depth": 2
        })
        
        self.test_tool("explore_connections", "Non-existent connection", {
            "start_entity": "Einstein",
            "end_entity": "NonexistentEntity_99999",
            "max_depth": 4
        })
        
        # TOOL 6: get_suggestions
        print(f"\n[6/10] Testing get_suggestions - Intelligent suggestions")
        
        self.test_tool("get_suggestions", "Missing facts for entity", {
            "suggestion_type": "missing_facts",
            "focus_area": "Einstein",
            "limit": 5
        })
        
        self.test_tool("get_suggestions", "Interesting questions", {
            "suggestion_type": "interesting_questions",
            "limit": 3
        })
        
        self.test_tool("get_suggestions", "Potential connections", {
            "suggestion_type": "potential_connections",
            "focus_area": "Physics",
            "limit": 7
        })
        
        self.test_tool("get_suggestions", "Knowledge gaps", {
            "suggestion_type": "knowledge_gaps",
            "limit": 4
        })
        
        self.test_tool("get_suggestions", "Large suggestion set", {
            "suggestion_type": "missing_facts",
            "limit": 10
        })
        
        # TOOL 7: get_stats
        print(f"\n[7/10] Testing get_stats - Graph statistics")
        
        self.test_tool("get_stats", "Basic statistics", {
            "include_details": False
        })
        
        self.test_tool("get_stats", "Detailed statistics", {
            "include_details": True
        })
        
        self.test_tool("get_stats", "Stats after data operations", {
            "include_details": False
        })
        
        # Add some more data first
        self.call_tool("store_fact", {
            "subject": "Newton", 
            "predicate": "is", 
            "object": "physicist"
        })
        
        self.test_tool("get_stats", "Stats with more data", {
            "include_details": True
        })
        
        self.test_tool("get_stats", "Repeated stats call", {
            "include_details": False
        })
        
        # TOOL 8: generate_graph_query
        print(f"\n[8/10] Testing generate_graph_query - Query language conversion")
        
        self.test_tool("generate_graph_query", "Simple Cypher query", {
            "natural_query": "Find all scientists",
            "query_language": "cypher"
        })
        
        self.test_tool("generate_graph_query", "SPARQL generation", {
            "natural_query": "What did Einstein invent?",
            "query_language": "sparql"
        })
        
        self.test_tool("generate_graph_query", "Gremlin traversal", {
            "natural_query": "Show relationships between Einstein and physics",
            "query_language": "gremlin",
            "include_explanation": True
        })
        
        self.test_tool("generate_graph_query", "Complex query pattern", {
            "natural_query": "Find all physicists who worked on relativity theory",
            "query_language": "cypher",
            "include_explanation": False
        })
        
        self.test_tool("generate_graph_query", "Aggregation query", {
            "natural_query": "Count all scientists in the database",
            "query_language": "sparql"
        })
        
        # TOOL 9: hybrid_search
        print(f"\n[9/10] Testing hybrid_search - Advanced search")
        
        self.test_tool("hybrid_search", "Hybrid physics search", {
            "query": "quantum physics theories",
            "search_type": "hybrid"
        })
        
        self.test_tool("hybrid_search", "Semantic-only search", {
            "query": "revolutionary scientific discoveries",
            "search_type": "semantic",
            "limit": 5
        })
        
        self.test_tool("hybrid_search", "Structural search", {
            "query": "highly connected entities",
            "search_type": "structural",
            "limit": 3
        })
        
        self.test_tool("hybrid_search", "Keyword search", {
            "query": "Einstein relativity physics",
            "search_type": "keyword"
        })
        
        self.test_tool("hybrid_search", "Filtered search", {
            "query": "science physics",
            "search_type": "hybrid",
            "filters": {
                "entity_types": ["person", "theory"],
                "min_confidence": 0.8
            },
            "limit": 8
        })
        
        # TOOL 10: validate_knowledge
        print(f"\n[10/10] Testing validate_knowledge - Quality assurance")
        
        self.test_tool("validate_knowledge", "Full validation", {
            "validation_type": "all"
        })
        
        self.test_tool("validate_knowledge", "Consistency check", {
            "validation_type": "consistency",
            "entity": "Einstein"
        })
        
        self.test_tool("validate_knowledge", "Conflict detection", {
            "validation_type": "conflicts"
        })
        
        self.test_tool("validate_knowledge", "Quality assessment", {
            "validation_type": "quality",
            "entity": "Einstein"
        })
        
        self.test_tool("validate_knowledge", "Completeness check", {
            "validation_type": "completeness",
            "fix_issues": False
        })
        
    def generate_comprehensive_report(self):
        """Generate detailed test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Overall stats
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n[RESULTS] OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Performance metrics
        response_times = [r['elapsed_ms'] for r in self.test_results if r['success']]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"\n[PERFORMANCE] PERFORMANCE METRICS:")
            print(f"   Average Response Time: {avg_time:.0f}ms")
            print(f"   Fastest Response: {min_time:.0f}ms")
            print(f"   Slowest Response: {max_time:.0f}ms")
        
        # Tool-by-tool breakdown
        tools = {}
        for result in self.test_results:
            tool = result['tool']
            if tool not in tools:
                tools[tool] = {'total': 0, 'passed': 0, 'times': []}
            tools[tool]['total'] += 1
            if result['success']:
                tools[tool]['passed'] += 1
                tools[tool]['times'].append(result['elapsed_ms'])
        
        print(f"\n[TOOLS] TOOL-BY-TOOL RESULTS:")
        for i, (tool, stats) in enumerate(tools.items(), 1):
            success_rate = (stats['passed'] / stats['total']) * 100
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            status = "PASS" if success_rate == 100 else "FAIL" if success_rate == 0 else "WARN"
            print(f"   {status} {tool}: {stats['passed']}/{stats['total']} ({success_rate:.0f}%) - {avg_time:.0f}ms avg")
        
        # Failed tests details
        failed = [r for r in self.test_results if not r['success']]
        if failed:
            print(f"\n[FAILED] FAILED TESTS:")
            for result in failed:
                print(f"   • {result['tool']} - {result['test']}: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n[SUCCESS] ALL TESTS PASSED!")
        
        # Quality assessment
        tools_100_percent = sum(1 for stats in tools.values() if stats['passed'] == stats['total'])
        quality_score = (tools_100_percent / len(tools)) * 100
        
        print(f"\n[QUALITY] QUALITY ASSESSMENT:")
        print(f"   Tools with 100% pass rate: {tools_100_percent}/{len(tools)}")
        print(f"   Overall quality score: {quality_score:.0f}%")
        
        if quality_score >= 90:
            verdict = "EXCELLENT - Production ready"
        elif quality_score >= 75:
            verdict = "GOOD - Minor issues to address"
        elif quality_score >= 50:
            verdict = "FAIR - Significant issues need fixing"
        else:
            verdict = "POOR - Major fixes required"
            
        print(f"   Verdict: {verdict}")
        
        return quality_score >= 75
        
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None


def main():
    # Check if binary exists
    exe_path = 'target/debug/llmkg_mcp_server_test.exe'
    if not os.path.exists(exe_path):
        print(f"ERROR: MCP server binary not found at {exe_path}")
        print("Please build the project first: cargo build --bin llmkg_mcp_server_test")
        sys.exit(1)
        
    # Create and run comprehensive test
    print("Starting comprehensive LLMKG MCP Server testing...")
    print("Testing 10 tools with 5 scenarios each (50 total tests)")
    
    tester = ComprehensiveToolTester(exe_path)
    
    try:
        tester.start_server()
        tester.run_all_tool_tests()
        success = tester.generate_comprehensive_report()
        
        if success:
            print(f"\n[COMPLETE] TEST SUITE COMPLETED SUCCESSFULLY!")
            return True
        else:
            print(f"\n[WARNING] TEST SUITE COMPLETED WITH ISSUES!")
            return False
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n[ERROR] Test suite failed with error: {e}")
        return False
    finally:
        tester.stop_server()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)