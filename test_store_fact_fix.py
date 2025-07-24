#!/usr/bin/env python3

import asyncio
import json
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_store_fact():
    """Test the store_fact tool to see if the hanging issue is fixed"""
    
    server_params = StdioServerParameters(
        command="cargo",
        args=["run", "--bin", "llmkg_mcp_server"]
    )
    
    print("Starting MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Connected to MCP server")
            
            # Test store_fact with timeout
            print("Testing store_fact (should complete quickly)...")
            
            start_time = time.time()
            
            try:
                # Set a timeout of 10 seconds - should complete much faster
                result = await asyncio.wait_for(
                    session.call_tool("store_fact", {
                        "subject": "Claude",
                        "predicate": "is",
                        "object": "AI assistant",
                        "confidence": 0.95
                    }),
                    timeout=10.0
                )
                
                elapsed = time.time() - start_time
                print(f"SUCCESS: store_fact completed in {elapsed:.3f}s")
                print(f"Result: {json.dumps(result.content, indent=2)}")
                
                if elapsed > 1.0:
                    print(f"WARNING: Took {elapsed:.3f}s - should be much faster")
                else:
                    print("EXCELLENT: Completed in under 1 second!")
                    
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"FAILED: store_fact timed out after {elapsed:.3f}s")
                return False
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ERROR: store_fact failed after {elapsed:.3f}s with: {e}")
                return False
                
    return True

if __name__ == "__main__":
    asyncio.run(test_store_fact())