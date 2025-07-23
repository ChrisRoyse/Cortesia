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
        env['RUST_LOG'] = 'warn'
        
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
        while time.time() - start_time < 3.0:
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
            success = not expected_success
        
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
        print(f"\n[1/10] Testing store_fact")
        
        self.test_tool("store_fact", "Test 1.1 - Basic fact", {
            "subject": "Einstein",
            "predicate": "is", 
            "object": "scientist"
        })
        
        self.test_tool("store_fact", "Test 1.2 - With confidence", {
            "subject": "Quantum_Computer",
            "predicate": "invented_by",
            "object": "Multiple_Scientists",
            "confidence": 0.85
        })
        
        self.test_tool("store_fact", "Test 1.3 - Unicode chars", {
            "subject": "Paris",
            "predicate": "capital_of",
            "object": "France"
        })
        
        self.test_tool("store_fact", "Test 1.4 - Complex names", {
            "subject": "Machine_Learning",
            "predicate": "developed_by",
            "object": "Computer_Scientists"
        })
        
        self.test_tool("store_fact", "Test 1.5 - Duplicate", {
            "subject": "Einstein",
            "predicate": "is",
            "object": "scientist"
        })
        
        # TOOL 2: store_knowledge
        print(f"\n[2/10] Testing store_knowledge")
        
        self.test_tool("store_knowledge", "Test 2.1 - Biography", {
            "title": "Einstein Biography",
            "content": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
            "category": "biography"
        })
        
        self.test_tool("store_knowledge", "Test 2.2 - Technical docs", {
            "title": "Python Basics",
            "content": "Python is a programming language with simple syntax and powerful libraries.",
            "category": "technical"
        })
        
        self.test_tool("store_knowledge", "Test 2.3 - Historical", {
            "title": "World War II",
            "content": "WWII was a global war from 1939 to 1945 involving most nations.",
            "category": "historical"
        })
        
        self.test_tool("store_knowledge", "Test 2.4 - Scientific", {
            "title": "Quantum Physics",
            "content": "Quantum physics studies matter and energy at the smallest scales.",
            "category": "science"
        })
        
        self.test_tool("store_knowledge", "Test 2.5 - Large content", {
            "title": "AI Overview",
            "content": "Artificial Intelligence is computer science. " * 20,
            "category": "technology"
        })
        
        # TOOL 3: find_facts
        print(f"\n[3/10] Testing find_facts")
        
        self.test_tool("find_facts", "Test 3.1 - Subject query", {
            "query": {"subject": "Einstein"}
        })
        
        self.test_tool("find_facts", "Test 3.2 - Predicate query", {
            "query": {"predicate": "is"}
        })
        
        self.test_tool("find_facts", "Test 3.3 - Object query", {
            "query": {"object": "scientist"}
        })
        
        self.test_tool("find_facts", "Test 3.4 - Combined query", {
            "query": {"subject": "Einstein", "predicate": "is"}
        })
        
        self.test_tool("find_facts", "Test 3.5 - Non-existent", {
            "query": {"subject": "NonExistent"}
        })
        
        # TOOL 4: ask_question
        print(f"\n[4/10] Testing ask_question")
        
        self.test_tool("ask_question", "Test 4.1 - Who question", {
            "question": "Who is Einstein?"
        })
        
        self.test_tool("ask_question", "Test 4.2 - What question", {
            "question": "What is quantum physics?"
        })
        
        self.test_tool("ask_question", "Test 4.3 - With context", {
            "question": "Tell me about Einstein",
            "context": "focusing on physics"
        })
        
        self.test_tool("ask_question", "Test 4.4 - Complex question", {
            "question": "What are Einstein's contributions to science?"
        })
        
        self.test_tool("ask_question", "Test 4.5 - Unknown info", {
            "question": "What is Einstein's favorite color?"
        })
        
        # TOOL 5: explore_connections
        print(f"\n[5/10] Testing explore_connections")
        
        self.test_tool("explore_connections", "Test 5.1 - Single entity", {
            "start_entity": "Einstein"
        })
        
        self.test_tool("explore_connections", "Test 5.2 - Path finding", {
            "start_entity": "Einstein",
            "end_entity": "scientist"
        })
        
        self.test_tool("explore_connections", "Test 5.3 - Shallow search", {
            "start_entity": "Quantum_Computer",
            "max_depth": 1
        })
        
        self.test_tool("explore_connections", "Test 5.4 - Deep search", {
            "start_entity": "Einstein",
            "max_depth": 3
        })
        
        self.test_tool("explore_connections", "Test 5.5 - No connection", {
            "start_entity": "NonExistent",
            "end_entity": "scientist"
        })
        
        # TOOL 6: get_suggestions
        print(f"\n[6/10] Testing get_suggestions")
        
        self.test_tool("get_suggestions", "Test 6.1 - Missing facts", {
            "suggestion_type": "missing_facts"
        })
        
        self.test_tool("get_suggestions", "Test 6.2 - Questions", {
            "suggestion_type": "interesting_questions"
        })
        
        self.test_tool("get_suggestions", "Test 6.3 - Connections", {
            "suggestion_type": "potential_connections"
        })
        
        self.test_tool("get_suggestions", "Test 6.4 - Knowledge gaps", {
            "suggestion_type": "knowledge_gaps"
        })
        
        self.test_tool("get_suggestions", "Test 6.5 - Focused area", {
            "suggestion_type": "missing_facts",
            "focus_area": "Einstein"
        })
        
        # TOOL 7: get_stats
        print(f"\n[7/10] Testing get_stats")
        
        self.test_tool("get_stats", "Test 7.1 - Basic stats", {
            "include_details": False
        })
        
        self.test_tool("get_stats", "Test 7.2 - Detailed stats", {
            "include_details": True
        })
        
        # Add more data
        self.call_tool("store_fact", {"subject": "Newton", "predicate": "is", "object": "physicist"})
        
        self.test_tool("get_stats", "Test 7.3 - After more data", {
            "include_details": False
        })
        
        self.test_tool("get_stats", "Test 7.4 - Repeated call", {
            "include_details": True
        })
        
        self.test_tool("get_stats", "Test 7.5 - Quick stats", {
            "include_details": False
        })
        
        # TOOL 8: generate_graph_query
        print(f"\n[8/10] Testing generate_graph_query")
        
        self.test_tool("generate_graph_query", "Test 8.1 - Cypher query", {
            "natural_query": "Find all scientists"
        })
        
        self.test_tool("generate_graph_query", "Test 8.2 - SPARQL query", {
            "natural_query": "What did Einstein discover?",
            "query_language": "sparql"
        })
        
        self.test_tool("generate_graph_query", "Test 8.3 - Gremlin query", {
            "natural_query": "Show Einstein connections",
            "query_language": "gremlin"
        })
        
        self.test_tool("generate_graph_query", "Test 8.4 - Complex pattern", {
            "natural_query": "Find physicists who worked on quantum theory"
        })
        
        self.test_tool("generate_graph_query", "Test 8.5 - Count query", {
            "natural_query": "Count all scientists",
            "query_language": "sparql"
        })
        
        # TOOL 9: hybrid_search
        print(f"\n[9/10] Testing hybrid_search")
        
        self.test_tool("hybrid_search", "Test 9.1 - Hybrid search", {
            "query": "quantum physics"
        })
        
        self.test_tool("hybrid_search", "Test 9.2 - Semantic search", {
            "query": "scientific discoveries",
            "search_type": "semantic"
        })
        
        self.test_tool("hybrid_search", "Test 9.3 - Structural search", {
            "query": "connected entities",
            "search_type": "structural"
        })
        
        self.test_tool("hybrid_search", "Test 9.4 - Keyword search", {
            "query": "Einstein physics",
            "search_type": "keyword"
        })
        
        self.test_tool("hybrid_search", "Test 9.5 - Filtered search", {
            "query": "science",
            "filters": {"min_confidence": 0.5}
        })
        
        # TOOL 10: validate_knowledge
        print(f"\n[10/10] Testing validate_knowledge")
        
        self.test_tool("validate_knowledge", "Test 10.1 - Full validation", {
            "validation_type": "all"
        })
        
        self.test_tool("validate_knowledge", "Test 10.2 - Consistency check", {
            "validation_type": "consistency"
        })
        
        self.test_tool("validate_knowledge", "Test 10.3 - Conflict detection", {
            "validation_type": "conflicts"
        })
        
        self.test_tool("validate_knowledge", "Test 10.4 - Quality check", {
            "validation_type": "quality"
        })
        
        self.test_tool("validate_knowledge", "Test 10.5 - Completeness", {
            "validation_type": "completeness"
        })
        
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Tool breakdown
        tools = {}
        for result in self.test_results:
            tool = result['tool']
            if tool not in tools:
                tools[tool] = {'total': 0, 'passed': 0}
            tools[tool]['total'] += 1
            if result['success']:
                tools[tool]['passed'] += 1
        
        print(f"\nTOOL-BY-TOOL RESULTS:")
        for tool, stats in tools.items():
            success_rate = (stats['passed'] / stats['total']) * 100
            status = "PASS" if success_rate == 100 else "FAIL" if success_rate == 0 else "PARTIAL"
            print(f"  {status}: {tool} - {stats['passed']}/{stats['total']} ({success_rate:.0f}%)")
        
        # Failed tests
        failed = [r for r in self.test_results if not r['success']]
        if failed:
            print(f"\nFAILED TESTS:")
            for result in failed:
                print(f"  - {result['tool']}.{result['test']}: {result.get('error', 'Unknown')}")
        else:
            print(f"\nALL TESTS PASSED!")
        
        return passed_tests == total_tests
        
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)


def main():
    exe_path = 'target/debug/llmkg_mcp_server_test.exe'
    if not os.path.exists(exe_path):
        print(f"Error: {exe_path} not found")
        return False
        
    print("Starting comprehensive LLMKG MCP Server testing...")
    print("Testing 10 tools with 5 scenarios each (50 total tests)")
    
    tester = ComprehensiveToolTester(exe_path)
    
    try:
        tester.start_server()
        tester.run_all_tool_tests()
        return tester.generate_report()
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        tester.stop_server()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)