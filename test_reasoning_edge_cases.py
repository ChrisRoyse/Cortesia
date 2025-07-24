"""Test edge cases in reasoning engine"""
import json
import asyncio
import aiohttp

async def test_reasoning_edge_cases():
    """Test the edge cases we implemented"""
    
    # MCP server URL - adjust if needed
    url = "http://localhost:3000/mcp/v1/tools/call"
    
    test_cases = [
        {
            "name": "Empty knowledge base test",
            "description": "Test reasoning with no stored knowledge",
            "tool": "cognitive_reasoning_chains",
            "params": {
                "premise": "Einstein",
                "reasoning_type": "deductive"
            }
        },
        {
            "name": "Cycle detection test",
            "description": "Test cycle detection in reasoning chains",
            "setup": [
                {"tool": "store_fact", "params": {"subject": "A", "predicate": "leads_to", "object": "B"}},
                {"tool": "store_fact", "params": {"subject": "B", "predicate": "leads_to", "object": "C"}},
                {"tool": "store_fact", "params": {"subject": "C", "predicate": "leads_to", "object": "A"}}
            ],
            "tool": "cognitive_reasoning_chains",
            "params": {
                "premise": "A",
                "reasoning_type": "deductive"
            }
        },
        {
            "name": "Contradiction detection test",
            "description": "Test contradiction detection",
            "setup": [
                {"tool": "store_fact", "params": {"subject": "Einstein", "predicate": "is", "object": "physicist"}},
                {"tool": "store_fact", "params": {"subject": "Einstein", "predicate": "is_not", "object": "physicist"}}
            ],
            "tool": "cognitive_reasoning_chains",
            "params": {
                "premise": "Einstein",
                "reasoning_type": "deductive"
            }
        },
        {
            "name": "Mutual exclusion test",
            "description": "Test mutual exclusion detection",
            "setup": [
                {"tool": "store_fact", "params": {"subject": "Schrodinger's cat", "predicate": "is", "object": "alive"}},
                {"tool": "store_fact", "params": {"subject": "Schrodinger's cat", "predicate": "is", "object": "dead"}}
            ],
            "tool": "cognitive_reasoning_chains",
            "params": {
                "premise": "Schrodinger's cat",
                "reasoning_type": "deductive"
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        results = []
        
        for test in test_cases:
            print(f"\n{'='*60}")
            print(f"Running: {test['name']}")
            print(f"Description: {test['description']}")
            print(f"{'='*60}")
            
            # Run setup if needed
            if "setup" in test:
                print("\nSetting up test data...")
                for setup_step in test["setup"]:
                    async with session.post(url, json={
                        "tool": setup_step["tool"],
                        "arguments": setup_step["params"]
                    }) as resp:
                        result = await resp.json()
                        print(f"  - {setup_step['tool']}: {setup_step['params']}")
            
            # Run the actual test
            print(f"\nTesting {test['tool']} with params: {test['params']}")
            async with session.post(url, json={
                "tool": test["tool"],
                "arguments": test["params"]
            }) as resp:
                result = await resp.json()
                
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                else:
                    content = result.get("content", [{}])[0].get("text", "")
                    data = json.loads(content) if content else {}
                    
                    print(f"\nResult:")
                    print(f"Primary Conclusion: {data.get('primary_conclusion', 'N/A')}")
                    print(f"Logical Validity: {data.get('logical_validity', 0)}")
                    print(f"Number of chains: {len(data.get('chains', []))}")
                    
                    # Check for cycles and contradictions
                    if data.get('chains'):
                        for i, chain in enumerate(data['chains']):
                            if chain.get('cycles_detected'):
                                print(f"  Chain {i+1}: CYCLE DETECTED!")
                            if chain.get('contradictions'):
                                print(f"  Chain {i+1}: Contradictions: {chain['contradictions']}")
                    
                    if data.get('counterarguments'):
                        print(f"Counterarguments: {data['counterarguments']}")
                
                results.append({
                    "test": test['name'],
                    "passed": "error" not in result,
                    "result": result
                })
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        passed = sum(1 for r in results if r['passed'])
        print(f"Passed: {passed}/{len(results)}")
        for r in results:
            status = "✓" if r['passed'] else "✗"
            print(f"  {status} {r['test']}")

if __name__ == "__main__":
    asyncio.run(test_reasoning_edge_cases())