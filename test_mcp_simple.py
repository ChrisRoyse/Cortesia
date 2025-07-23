#!/usr/bin/env python3
"""
Simple MCP Server Test
Direct test without threading complexity
"""

import json
import subprocess
import time
import sys
import os

def test_mcp_server():
    exe_path = 'target/debug/llmkg_mcp_server.exe'
    
    # Check if binary exists
    if not os.path.exists(exe_path):
        print(f"Error: MCP server binary not found at {exe_path}")
        sys.exit(1)
    
    print("Starting MCP server...")
    
    # Set environment
    env = os.environ.copy()
    env['RUST_LOG'] = 'debug'
    
    # Start process
    process = subprocess.Popen(
        [exe_path, '--data-dir', './test_data', '-v'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    
    time.sleep(0.5)  # Give server time to start
    
    # Test 1: Initialize
    print("\nTest 1: Initialize")
    request = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {},
        "id": 0
    })
    
    print(f"Sending: {request}")
    process.stdin.write(request + '\n')
    process.stdin.flush()
    
    # Read response
    response_line = process.stdout.readline()
    if response_line:
        response = json.loads(response_line.strip())
        print(f"Response: {json.dumps(response, indent=2)}")
        
        if "result" in response:
            print("PASS: Initialize successful")
        else:
            print("FAIL: Initialize failed")
    else:
        print("FAIL: No response")
        
    # Test 2: List tools
    print("\nTest 2: List tools")
    request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 1
    })
    
    print(f"Sending: {request}")
    process.stdin.write(request + '\n')
    process.stdin.flush()
    
    response_line = process.stdout.readline()
    if response_line:
        response = json.loads(response_line.strip())
        print(f"Response: {json.dumps(response, indent=2)}")
        
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"PASS: Found {len(tools)} tools")
            for tool in tools[:5]:  # Show first 5 tools
                print(f"  - {tool['name']}")
        else:
            print("FAIL: List tools failed")
    else:
        print("FAIL: No response")
        
    # Test 3: Call get_stats
    print("\nTest 3: Call get_stats")
    request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_stats",
            "arguments": {"include_details": False}
        },
        "id": 2
    })
    
    print(f"Sending: {request}")
    process.stdin.write(request + '\n')
    process.stdin.flush()
    
    # Wait for response with timeout
    start_time = time.time()
    response_line = None
    
    while time.time() - start_time < 5.0:  # 5 second timeout
        try:
            response_line = process.stdout.readline()
            if response_line:
                break
        except:
            pass
        time.sleep(0.1)
    
    if response_line:
        response = json.loads(response_line.strip())
        print(f"Response: {json.dumps(response, indent=2)}")
        
        if "result" in response:
            print("PASS: get_stats successful")
        else:
            print("FAIL: get_stats failed")
    else:
        print("FAIL: Timeout waiting for get_stats response")
        
        # Check stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"Server stderr:\n{stderr_output}")
    
    # Cleanup
    print("\nStopping server...")
    process.terminate()
    process.wait(timeout=5)
    print("Server stopped")

if __name__ == "__main__":
    test_mcp_server()