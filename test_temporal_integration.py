#!/usr/bin/env python3
"""Test temporal tracking integration with storage operations."""

import json
import asyncio
import websockets
from datetime import datetime
import time

async def test_temporal_tracking():
    """Test that storage operations are tracked in temporal index."""
    uri = "ws://localhost:33994"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to LLMKG MCP Server")
        
        # Test 1: Create a new fact
        print("\n1. Testing CREATE operation tracking...")
        create_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "store_fact",
                "arguments": {
                    "subject": "Einstein",
                    "predicate": "occupation",
                    "object": "physicist",
                    "confidence": 1.0
                }
            },
            "id": 1
        }
        
        await websocket.send(json.dumps(create_request))
        response = await websocket.recv()
        result = json.loads(response)
        print(f"Create response: {json.dumps(result, indent=2)}")
        
        # Wait a moment
        await asyncio.sleep(0.5)
        
        # Test 2: Update the same fact (same subject-predicate, different object)
        print("\n2. Testing UPDATE operation tracking...")
        update_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "store_fact",
                "arguments": {
                    "subject": "Einstein",
                    "predicate": "occupation",
                    "object": "theoretical physicist",
                    "confidence": 1.0
                }
            },
            "id": 2
        }
        
        await websocket.send(json.dumps(update_request))
        response = await websocket.recv()
        result = json.loads(response)
        print(f"Update response: {json.dumps(result, indent=2)}")
        
        # Wait a moment
        await asyncio.sleep(0.5)
        
        # Test 3: Query temporal history
        print("\n3. Querying temporal history for Einstein...")
        temporal_query = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "time_travel_query",
                "arguments": {
                    "entity": "Einstein",
                    "query_type": "evolution_tracking"
                }
            },
            "id": 3
        }
        
        await websocket.send(json.dumps(temporal_query))
        response = await websocket.recv()
        result = json.loads(response)
        
        if "result" in result:
            content = result["result"].get("content", [])
            if content and isinstance(content[0], dict):
                data = content[0].get("data", {})
                print(f"\nTemporal history found:")
                print(f"Total changes: {data.get('total_changes', 0)}")
                print(f"Query type: {data.get('query_type', 'N/A')}")
                
                if "results" in data:
                    for item in data["results"]:
                        print(f"\nTimestamp: {item.get('timestamp', 'N/A')}")
                        print(f"Entity: {item.get('entity', 'N/A')}")
                        for change in item.get("changes", []):
                            triple = change.get("triple", {})
                            print(f"  - Operation: {change.get('operation', 'N/A')}")
                            print(f"    Triple: {triple.get('subject', 'N/A')} {triple.get('predicate', 'N/A')} {triple.get('object', 'N/A')}")
                            if change.get("previous_value"):
                                print(f"    Previous value: {change['previous_value']}")
                
                print(f"\nInsights:")
                for insight in data.get("insights", []):
                    print(f"  - {insight}")
        
        # Test 4: Store knowledge and verify tracking
        print("\n\n4. Testing knowledge storage tracking...")
        knowledge_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "store_knowledge",
                "arguments": {
                    "content": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
                    "title": "Einstein Biography",
                    "category": "biography"
                }
            },
            "id": 4
        }
        
        await websocket.send(json.dumps(knowledge_request))
        response = await websocket.recv()
        result = json.loads(response)
        print(f"Knowledge storage response: {json.dumps(result, indent=2)}")
        
        # Test 5: Point-in-time query
        print("\n5. Testing point-in-time query...")
        current_time = datetime.utcnow().isoformat() + "Z"
        pit_query = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "time_travel_query",
                "arguments": {
                    "entity": "Einstein",
                    "query_type": "point_in_time",
                    "timestamp": current_time
                }
            },
            "id": 5
        }
        
        await websocket.send(json.dumps(pit_query))
        response = await websocket.recv()
        result = json.loads(response)
        
        if "result" in result:
            content = result["result"].get("content", [])
            if content and isinstance(content[0], dict):
                data = content[0].get("data", {})
                print(f"\nPoint-in-time state at {current_time}:")
                print(f"Total historical changes: {data.get('total_changes', 0)}")
                
                if "results" in data and data["results"]:
                    for item in data["results"]:
                        print(f"\nEntity: {item.get('entity', 'N/A')}")
                        for change in item.get("changes", []):
                            triple = change.get("triple", {})
                            print(f"  - {triple.get('predicate', 'N/A')}: {triple.get('object', 'N/A')}")

if __name__ == "__main__":
    print("Testing Temporal Tracking Integration")
    print("=====================================")
    asyncio.run(test_temporal_tracking())