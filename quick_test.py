#!/usr/bin/env python3
"""Quick test of core MCP functionality"""

import json
import subprocess
import time
import sys
import os

def quick_test():
    exe_path = 'target/debug/llmkg_mcp_server_test.exe'
    
    if not os.path.exists(exe_path):
        print(f"Error: {exe_path} not found")
        return False
        
    print("Running quick MCP functionality test...")
    
    env = os.environ.copy()
    env['RUST_LOG'] = 'warn'
    
    process = subprocess.Popen(
        [exe_path, '--data-dir', './test_data'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    def send_request(request):
        request_str = json.dumps(request)
        process.stdin.write(request_str + '\n')
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        return None
    
    try:
        # Test 1: Initialize
        print("1. Initialize...")
        response = send_request({
            "jsonrpc": "2.0",
            "method": "initialize", 
            "params": {},
            "id": 0
        })
        if not response or 'result' not in response:
            print("   FAIL: Initialize failed")
            return False
        print("   PASS")
        
        # Test 2: Store fact
        print("2. Store fact...")
        response = send_request({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "store_fact",
                "arguments": {
                    "subject": "Einstein",
                    "predicate": "is", 
                    "object": "scientist"
                }
            },
            "id": 1
        })
        if not response or 'result' not in response:
            print("   FAIL: Store fact failed")
            return False
        print("   PASS")
        
        # Test 3: Get stats
        print("3. Get stats...")
        response = send_request({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_stats",
                "arguments": {"include_details": False}
            },
            "id": 2
        })
        if not response or 'result' not in response:
            print("   FAIL: Get stats failed")
            return False
        print("   PASS")
        
        # Test 4: Find facts
        print("4. Find facts...")
        response = send_request({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "find_facts",
                "arguments": {
                    "query": {"subject": "Einstein"}
                }
            },
            "id": 3
        })
        if not response or 'result' not in response:
            print("   FAIL: Find facts failed")
            return False
        print("   PASS")
        
        # Test 5: Ask question
        print("5. Ask question...")
        response = send_request({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "ask_question",
                "arguments": {
                    "question": "Who is Einstein?"
                }
            },
            "id": 4
        })
        if not response or 'result' not in response:
            print("   FAIL: Ask question failed")
            return False
        print("   PASS")
        
        print("\nAll core tests PASSED!")
        return True
        
    finally:
        process.terminate()
        process.wait(timeout=5)

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)