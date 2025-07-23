#!/usr/bin/env python3
"""
LLMKG MCP Server Test Harness
Tests MCP server functionality and identifies hanging issues
"""

import json
import subprocess
import time
import sys
import os
from typing import Dict, Any, Optional, List
import threading
import queue

class MCPTestHarness:
    def __init__(self, exe_path: str = 'target/debug/llmkg_mcp_server.exe'):
        self.exe_path = exe_path
        self.process = None
        self.test_results = []
        self.response_queue = queue.Queue()
        self.reader_thread = None
        
    def start_server(self, verbose: bool = True):
        """Start the MCP server process"""
        print(f"Starting MCP server from: {self.exe_path}")
        
        # Set environment for debug logging
        env = os.environ.copy()
        env['RUST_LOG'] = 'debug' if verbose else 'info'
        
        cmd = [self.exe_path, '--data-dir', './test_data']
        if verbose:
            cmd.append('-v')
            
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_responses)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        # Give server time to initialize
        time.sleep(1)
        print("MCP server started")
        
    def _read_responses(self):
        """Read responses from the server in a separate thread"""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            response = json.loads(line)
                            self.response_queue.put(response)
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse response: {line}")
                            print(f"Error: {e}")
            except Exception as e:
                print(f"Reader thread error: {e}")
                break
                
    def send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response"""
        if not self.process:
            return None
            
        request_str = json.dumps(request)
        print(f"\n-> Sending: {request_str}")
        
        try:
            self.process.stdin.write(request_str + '\n')
            self.process.stdin.flush()
        except Exception as e:
            print(f"Failed to send request: {e}")
            return None
        
        # Wait for response with timeout
        timeout = 5.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                print(f"<- Received: {json.dumps(response, indent=2)}")
                return response
            except queue.Empty:
                continue
                
        print(f"WARNING: Timeout waiting for response after {timeout}s")
        
        # Check stderr for any errors
        try:
            stderr_output = self.process.stderr.read()
            if stderr_output:
                print(f"Server stderr: {stderr_output}")
        except:
            pass
            
        return None
        
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a specific tool"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": 1
        }
        return self.send_request(request)
        
    def initialize(self) -> bool:
        """Initialize the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 0
        }
        response = self.send_request(request)
        return response is not None and "result" in response
        
    def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Get list of available tools"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        response = self.send_request(request)
        if response and "result" in response:
            return response["result"].get("tools", [])
        return None
        
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            print("\nStopping MCP server...")
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
            
    def run_basic_tests(self):
        """Run basic connectivity tests"""
        print("\n" + "="*60)
        print("LLMKG MCP Server Basic Tests")
        print("="*60)
        
        # Test 1: Initialize
        print("\nTest 1: Initialize server")
        if self.initialize():
            print("PASS: Server initialized successfully")
            self.test_results.append(("Initialize", True, "Success"))
        else:
            print("FAIL: Failed to initialize server")
            self.test_results.append(("Initialize", False, "No response"))
            return
            
        # Test 2: List tools
        print("\nTest 2: List available tools")
        tools = self.list_tools()
        if tools:
            print(f"PASS: Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description'][:50]}...")
            self.test_results.append(("List tools", True, f"{len(tools)} tools"))
        else:
            print("FAIL: Failed to list tools")
            self.test_results.append(("List tools", False, "No response"))
            return
            
        # Test 3: Simple tool call (get_stats)
        print("\nTest 3: Call get_stats tool")
        response = self.call_tool("get_stats", {"include_details": False})
        if response:
            print("PASS: get_stats responded")
            self.test_results.append(("get_stats", True, "Success"))
        else:
            print("FAIL: get_stats failed to respond")
            self.test_results.append(("get_stats", False, "Timeout"))
            
        # Test 4: Store a fact
        print("\nTest 4: Store a simple fact")
        response = self.call_tool("store_fact", {
            "subject": "TestEntity",
            "predicate": "is",
            "object": "test_object"
        })
        if response:
            print("PASS: store_fact responded")
            self.test_results.append(("store_fact", True, "Success"))
        else:
            print("FAIL: store_fact failed to respond")
            self.test_results.append(("store_fact", False, "Timeout"))
            
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "PASS" if success else "FAIL"
            print(f"{status} {test_name}: {message}")
            
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed < total:
            print("\nWARNING: Some tests failed. Check server logs for details.")


def main():
    # Check if binary exists
    exe_path = 'target/debug/llmkg_mcp_server.exe'
    if not os.path.exists(exe_path):
        print(f"Error: MCP server binary not found at {exe_path}")
        print("Please build the project first: cargo build --bin llmkg_mcp_server")
        sys.exit(1)
        
    # Create test harness
    harness = MCPTestHarness(exe_path)
    
    try:
        # Start server
        harness.start_server(verbose=True)
        
        # Run tests
        harness.run_basic_tests()
        
        # Print summary
        harness.print_summary()
        
    finally:
        # Always stop server
        harness.stop_server()


if __name__ == "__main__":
    main()