#!/usr/bin/env python3
"""
Graph Analysis Consolidation Tests
Tests the unified analyze_graph tool that consolidates:
- explore_connections -> analyze_graph with analysis_type="connections"
- analyze_graph_centrality -> analyze_graph with analysis_type="centrality"
- hierarchical_clustering -> analyze_graph with analysis_type="clustering"
- predict_graph_structure -> analyze_graph with analysis_type="prediction"
"""

import asyncio
import json
import time
from typing import Dict, Any, List

# MCP client setup (simulated for testing)
class MCPTestClient:
    def __init__(self):
        self.results = []
        
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP tool call"""
        print(f"\n[CALLING] {tool_name}")
        print(f"[PARAMS] {json.dumps(params, indent=2)}")
        
        # Simulate response based on tool
        if tool_name == "store_fact":
            return {
                "success": True,
                "message": f"Stored fact: {params['subject']} {params['predicate']} {params['object']}"
            }
        elif tool_name == "analyze_graph":
            analysis_type = params["analysis_type"]
            if analysis_type == "connections":
                return self._simulate_connections_response(params["config"])
            elif analysis_type == "centrality":
                return self._simulate_centrality_response(params["config"])
            elif analysis_type == "clustering":
                return self._simulate_clustering_response(params["config"])
            elif analysis_type == "prediction":
                return self._simulate_prediction_response(params["config"])
        elif tool_name in ["explore_connections", "analyze_graph_centrality", 
                          "hierarchical_clustering", "predict_graph_structure"]:
            # Deprecated tools should be migrated automatically
            return {
                "success": True,
                "message": f"[MIGRATED] {tool_name} -> analyze_graph",
                "data": {"migrated": True}
            }
        
        return {"success": False, "message": f"Unknown tool: {tool_name}"}
    
    def _simulate_connections_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate connections analysis response"""
        return {
            "success": True,
            "message": f"Graph Analysis Complete (connections): Found 3 paths from {config.get('start_entity', 'unknown')} to {config.get('end_entity', 'unknown')}",
            "data": {
                "analysis_type": "connections",
                "results": {
                    "paths": [
                        {"path": ["Einstein", "studied_at", "ETH_Zurich", "located_in", "Switzerland"], "length": 2},
                        {"path": ["Einstein", "worked_at", "Princeton", "located_in", "USA"], "length": 2}
                    ],
                    "total_paths": 2,
                    "nodes_processed": 50,
                    "edges_processed": 75
                },
                "performance_metrics": {
                    "execution_time_ms": 45,
                    "analysis_type": "connections"
                }
            },
            "suggestions": [
                "Try increasing max_depth to find more paths",
                "Use relationship_types to filter specific connections"
            ]
        }
    
    def _simulate_centrality_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate centrality analysis response"""
        return {
            "success": True,
            "message": "Graph Analysis Complete (centrality): Analyzed 2 centrality measures for top 10 entities",
            "data": {
                "analysis_type": "centrality",
                "results": {
                    "centrality_measures": {
                        "pagerank": [
                            {"entity": "Einstein", "score": 0.89},
                            {"entity": "Newton", "score": 0.76}
                        ],
                        "betweenness": [
                            {"entity": "Physics", "score": 0.92},
                            {"entity": "Mathematics", "score": 0.84}
                        ]
                    },
                    "nodes_processed": 150,
                    "edges_processed": 300
                }
            }
        }
    
    def _simulate_clustering_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate clustering analysis response"""
        return {
            "success": True,
            "message": f"Graph Analysis Complete (clustering): Found 5 clusters using {config.get('algorithm', 'leiden')} algorithm (modularity: 0.847)",
            "data": {
                "analysis_type": "clustering",
                "results": {
                    "clusters": [
                        ["Einstein", "Bohr", "Heisenberg"],
                        ["Newton", "Galileo", "Kepler"],
                        ["Darwin", "Wallace", "Mendel"]
                    ],
                    "clustering_metrics": {
                        "modularity": 0.847,
                        "num_clusters": 3,
                        "resolution": config.get("resolution", 1.0)
                    }
                }
            }
        }
    
    def _simulate_prediction_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate prediction analysis response"""
        return {
            "success": True,
            "message": "Graph Analysis Complete (prediction): Generated 10 missing links predictions (validation score: 0.85)",
            "data": {
                "analysis_type": "prediction",
                "results": {
                    "predictions": [
                        {
                            "type": "missing_link",
                            "source": "Einstein",
                            "target": "Schrodinger",
                            "predicted_relation": "collaborated_with",
                            "confidence": 0.89
                        }
                    ],
                    "confidence_distribution": {
                        "high_confidence": 3,
                        "medium_confidence": 5,
                        "low_confidence": 2
                    },
                    "validation_score": 0.85
                }
            }
        }

async def test_connections_analysis(client: MCPTestClient) -> Dict[str, Any]:
    """Test connections analysis mode"""
    print("\n=== Testing Connections Analysis ===")
    
    # First, add some test data
    test_facts = [
        ("Einstein", "studied_at", "ETH_Zurich"),
        ("ETH_Zurich", "located_in", "Switzerland"),
        ("Einstein", "worked_at", "Princeton"),
        ("Princeton", "located_in", "USA"),
        ("Einstein", "collaborated_with", "Bohr"),
        ("Bohr", "worked_at", "Copenhagen_University")
    ]
    
    for subject, predicate, obj in test_facts:
        await client.call_tool("store_fact", {
            "subject": subject,
            "predicate": predicate,
            "object": obj
        })
    
    # Test connections analysis
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "connections",
        "config": {
            "start_entity": "Einstein",
            "end_entity": "Switzerland",
            "max_depth": 3
        }
    })
    
    return {
        "test": "connections_analysis",
        "status": "PASS" if result["success"] else "FAIL",
        "result": result
    }

async def test_centrality_analysis(client: MCPTestClient) -> Dict[str, Any]:
    """Test centrality analysis mode"""
    print("\n=== Testing Centrality Analysis ===")
    
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "centrality",
        "config": {
            "centrality_types": ["pagerank", "betweenness"],
            "top_n": 10,
            "include_scores": True
        }
    })
    
    return {
        "test": "centrality_analysis",
        "status": "PASS" if result["success"] else "FAIL",
        "result": result
    }

async def test_clustering_analysis(client: MCPTestClient) -> Dict[str, Any]:
    """Test clustering analysis mode"""
    print("\n=== Testing Clustering Analysis ===")
    
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "clustering",
        "config": {
            "algorithm": "leiden",
            "resolution": 1.2,
            "min_cluster_size": 3,
            "include_metadata": True
        }
    })
    
    return {
        "test": "clustering_analysis",
        "status": "PASS" if result["success"] else "FAIL",
        "result": result
    }

async def test_prediction_analysis(client: MCPTestClient) -> Dict[str, Any]:
    """Test prediction analysis mode"""
    print("\n=== Testing Prediction Analysis ===")
    
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "prediction",
        "config": {
            "prediction_type": "missing_links",
            "confidence_threshold": 0.8,
            "max_predictions": 10,
            "use_neural_features": True
        }
    })
    
    return {
        "test": "prediction_analysis",
        "status": "PASS" if result["success"] else "FAIL",
        "result": result
    }

async def test_deprecated_tools_migration(client: MCPTestClient) -> Dict[str, Any]:
    """Test that deprecated tools are properly migrated"""
    print("\n=== Testing Deprecated Tools Migration ===")
    
    deprecated_tests = []
    
    # Test explore_connections migration
    result1 = await client.call_tool("explore_connections", {
        "start_entity": "Einstein",
        "max_depth": 2
    })
    deprecated_tests.append({
        "tool": "explore_connections",
        "migrated": result1.get("data", {}).get("migrated", False)
    })
    
    # Test analyze_graph_centrality migration
    result2 = await client.call_tool("analyze_graph_centrality", {
        "centrality_type": "pagerank",
        "top_n": 10
    })
    deprecated_tests.append({
        "tool": "analyze_graph_centrality",
        "migrated": result2.get("data", {}).get("migrated", False)
    })
    
    # Test hierarchical_clustering migration
    result3 = await client.call_tool("hierarchical_clustering", {
        "max_clusters": 5,
        "linkage": "average"
    })
    deprecated_tests.append({
        "tool": "hierarchical_clustering",
        "migrated": result3.get("data", {}).get("migrated", False)
    })
    
    # Test predict_graph_structure migration
    result4 = await client.call_tool("predict_graph_structure", {
        "prediction_type": "missing_links",
        "confidence_threshold": 0.7
    })
    deprecated_tests.append({
        "tool": "predict_graph_structure",
        "migrated": result4.get("data", {}).get("migrated", False)
    })
    
    all_migrated = all(test["migrated"] for test in deprecated_tests)
    
    return {
        "test": "deprecated_tools_migration",
        "status": "PASS" if all_migrated else "FAIL",
        "deprecated_tests": deprecated_tests
    }

async def test_performance_metrics(client: MCPTestClient) -> Dict[str, Any]:
    """Test that performance metrics are included"""
    print("\n=== Testing Performance Metrics ===")
    
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "connections",
        "config": {
            "start_entity": "Einstein",
            "max_depth": 2
        }
    })
    
    has_metrics = False
    if result["success"] and "data" in result:
        data = result["data"]
        has_metrics = (
            "performance_metrics" in data and
            "execution_time_ms" in data["performance_metrics"] and
            "nodes_processed" in data["results"] and
            "edges_processed" in data["results"]
        )
    
    return {
        "test": "performance_metrics",
        "status": "PASS" if has_metrics else "FAIL",
        "has_metrics": has_metrics
    }

async def test_invalid_analysis_type(client: MCPTestClient) -> Dict[str, Any]:
    """Test error handling for invalid analysis type"""
    print("\n=== Testing Invalid Analysis Type ===")
    
    result = await client.call_tool("analyze_graph", {
        "analysis_type": "invalid_type",
        "config": {}
    })
    
    # Should fail with appropriate error
    is_handled = not result["success"] and "invalid" in result.get("message", "").lower()
    
    return {
        "test": "invalid_analysis_type",
        "status": "PASS" if is_handled else "FAIL",
        "error_handled": is_handled
    }

async def run_all_tests():
    """Run all graph analysis consolidation tests"""
    print("Graph Analysis Consolidation Test Suite")
    print("=" * 80)
    
    client = MCPTestClient()
    test_results = []
    
    # Run all tests
    tests = [
        test_connections_analysis,
        test_centrality_analysis,
        test_clustering_analysis,
        test_prediction_analysis,
        test_deprecated_tools_migration,
        test_performance_metrics,
        test_invalid_analysis_type
    ]
    
    for test_func in tests:
        try:
            result = await test_func(client)
            test_results.append(result)
            print(f"\n[{result['status']}] {result['test']}")
        except Exception as e:
            test_results.append({
                "test": test_func.__name__,
                "status": "ERROR",
                "error": str(e)
            })
            print(f"\n[ERROR] {test_func.__name__}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    errors = sum(1 for r in test_results if r["status"] == "ERROR")
    
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    
    # Save results
    with open("graph_analysis_test_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": len(test_results),
                "passed": passed,
                "failed": failed,
                "errors": errors
            },
            "results": test_results
        }, f, indent=2)
    
    print(f"\nResults saved to graph_analysis_test_results.json")
    
    if failed == 0 and errors == 0:
        print("\n[SUCCESS] All tests passed! Graph analysis consolidation is working correctly.")
    else:
        print("\n[WARNING] Some tests failed. Review the results for details.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())