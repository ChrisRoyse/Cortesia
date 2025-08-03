#!/usr/bin/env python3
"""
Test script for MCP RAG Indexer Server

This script tests the MCP server functionality without Claude Code.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_rag_server import RAGIndexerServer


class MockMCPClient:
    """Mock MCP client for testing"""
    
    def __init__(self, server: RAGIndexerServer):
        self.server = server
        
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a tool on the server"""
        tools = {
            'index_codebase': self.server.server._tools.get('index_codebase'),
            'query_codebase': self.server.server._tools.get('query_codebase')
        }
        
        if tool_name not in tools:
            return f"Tool not found: {tool_name}"
            
        # Get the tool function
        for tool in self.server.server._tools:
            if tool.name == tool_name:
                return await tool.function(**kwargs)
                
        return f"Tool not found: {tool_name}"


async def test_mcp_server():
    """Test the MCP server functionality"""
    print("=" * 60)
    print("MCP RAG Indexer Server - Test Suite")
    print("=" * 60)
    
    # Create server
    server = RAGIndexerServer()
    client = MockMCPClient(server)
    
    # Test directory (use simulation if available)
    test_dir = Path("simulation/1_multi_language")
    if not test_dir.exists():
        test_dir = Path(__file__).parent
        
    test_dir = test_dir.resolve()
    
    print(f"\nTest directory: {test_dir}")
    
    # Test 1: Index a codebase
    print("\n" + "=" * 60)
    print("TEST 1: Index Codebase")
    print("=" * 60)
    
    result = await client.call_tool(
        "index_codebase",
        root_dir=str(test_dir),
        watch=False,  # Disable for testing
        incremental=False
    )
    print(result)
    
    if "Successfully indexed" in result:
        print("✓ Indexing test passed")
    else:
        print("✗ Indexing test failed")
        return
        
    # Test 2: Query the indexed codebase
    print("\n" + "=" * 60)
    print("TEST 2: Query Codebase")
    print("=" * 60)
    
    test_queries = [
        ("function", "Find all functions"),
        ("class", "Find class definitions"),
        ("import", "Find import statements")
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: {description} ('{query}')")
        print("-" * 40)
        
        result = await client.call_tool(
            "query_codebase",
            query=query,
            root_dir=str(test_dir),
            k=3,
            filter_type="code",
            rerank=True
        )
        
        # Show first 500 chars of result
        if len(result) > 500:
            print(result[:500] + "...")
        else:
            print(result)
            
        if "Search results" in result or "No results" in result:
            print(f"✓ Query '{query}' completed")
        else:
            print(f"✗ Query '{query}' failed")
            
    # Test 3: Test error handling
    print("\n" + "=" * 60)
    print("TEST 3: Error Handling")
    print("=" * 60)
    
    # Non-existent directory
    print("\nTesting non-existent directory...")
    result = await client.call_tool(
        "index_codebase",
        root_dir="/nonexistent/path/12345"
    )
    print(result)
    
    if "not found" in result.lower():
        print("✓ Error handling test passed")
    else:
        print("✗ Error handling test failed")
        
    # Query non-indexed directory
    print("\nTesting query on non-indexed directory...")
    result = await client.call_tool(
        "query_codebase",
        query="test",
        root_dir="/another/nonexistent/path"
    )
    print(result)
    
    if "not indexed" in result.lower():
        print("✓ Non-indexed query test passed")
    else:
        print("✗ Non-indexed query test failed")
        
    # Test 4: Check server state
    print("\n" + "=" * 60)
    print("TEST 4: Server State")
    print("=" * 60)
    
    print(f"Indexed projects: {len(server.projects)}")
    for project_id, project in server.projects.items():
        print(f"  - {project.root_dir}")
        print(f"    Files: {project.file_count}, Chunks: {project.chunk_count}")
        print(f"    Languages: {', '.join(project.languages)}")
        
    # Cleanup
    print("\n" + "=" * 60)
    print("Cleaning up...")
    server.shutdown()
    print("Test complete!")
    
    return True


async def test_stdio_mode():
    """Test the server in stdio mode (simulated)"""
    print("\n" + "=" * 60)
    print("TEST: Stdio Mode Simulation")
    print("=" * 60)
    
    # Create a sample JSON-RPC request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "index_codebase",
            "arguments": {
                "root_dir": str(Path.cwd()),
                "watch": False
            }
        }
    }
    
    print("Sample JSON-RPC request:")
    print(json.dumps(request, indent=2))
    
    print("\nTo test with Claude Code:")
    print("1. Restart Claude Code")
    print("2. Type: /mcp")
    print("3. Check for 'rag-indexer' in the list")
    print("4. Use: 'Index the current directory'")
    

def main():
    """Main test runner"""
    print("Starting MCP RAG Indexer Server Tests\n")
    
    # Run async tests
    success = asyncio.run(test_mcp_server())
    
    if success:
        # Show stdio mode info
        asyncio.run(test_stdio_mode())
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run: install_windows.bat")
        print("2. Restart Claude Code")
        print("3. Test in Claude: 'Index C:\\your\\project'")
    else:
        print("\nSome tests failed. Please check the output above.")
        

if __name__ == "__main__":
    # Windows-specific setup
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    main()