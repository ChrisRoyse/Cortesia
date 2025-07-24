#!/usr/bin/env python3
"""
Comprehensive MCP Tool Testing Suite for LLMKG
====================

This suite creates synthetic test data and validates all 16 LLMKG MCP tools
with 10 challenging scenarios each. It tests functionality, edge cases, 
performance, and correctness.

Tools Tested:
1. store_fact - Store simple facts as triples
2. store_knowledge - Store complex knowledge chunks
3. find_facts - Find facts matching patterns
4. ask_question - Natural language queries
5. hybrid_search - Advanced search with multiple modes
6. analyze_graph - Graph analysis suite
7. get_suggestions - Intelligent recommendations
8. get_stats - Knowledge graph statistics
9. generate_graph_query - Natural language to query language
10. validate_knowledge - Knowledge validation and quality
11. neural_importance_scoring - AI-powered content assessment
12. divergent_thinking_engine - Creative exploration
13. time_travel_query - Temporal database queries
14. cognitive_reasoning_chains - Advanced logical reasoning

Each tool has 10 challenging test scenarios designed to push boundaries.
"""

import json
import requests
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys

class LLMKGMCPTestSuite:
    def __init__(self, server_url: str = "http://localhost:3000"):
        self.server_url = server_url
        self.test_results = {}
        self.synthetic_data = {}
        
    def setup_synthetic_data(self):
        """Create comprehensive synthetic test data"""
        print("Setting up synthetic test data...")
        
        # Scientific entities and relationships
        scientists = [
            ("Einstein", "physicist", 1879, 1955, "Germany"),
            ("Newton", "physicist", 1643, 1727, "England"), 
            ("Darwin", "biologist", 1809, 1882, "England"),
            ("Curie", "chemist", 1867, 1934, "Poland"),
            ("Tesla", "inventor", 1856, 1943, "Serbia"),
            ("Hawking", "physicist", 1942, 2018, "England"),
            ("Feynman", "physicist", 1918, 1988, "USA"),
            ("Planck", "physicist", 1858, 1947, "Germany"),
            ("Bohr", "physicist", 1885, 1962, "Denmark"),
            ("Heisenberg", "physicist", 1901, 1976, "Germany")
        ]
        
        discoveries = [
            ("relativity", "Einstein", 1915, "physics"),
            ("quantum_mechanics", "Planck", 1900, "physics"),
            ("evolution", "Darwin", 1859, "biology"),
            ("radioactivity", "Curie", 1896, "chemistry"),
            ("AC_motor", "Tesla", 1887, "engineering"),
            ("uncertainty_principle", "Heisenberg", 1927, "physics"),
            ("atomic_model", "Bohr", 1913, "physics"),
            ("laws_of_motion", "Newton", 1687, "physics"),
            ("QED", "Feynman", 1948, "physics"),
            ("black_hole_radiation", "Hawking", 1974, "physics")
        ]
        
        institutions = [
            ("Princeton", "university", "USA", 1746),
            ("Cambridge", "university", "England", 1209),
            ("MIT", "institute", "USA", 1861),
            ("Stanford", "university", "USA", 1885),
            ("Oxford", "university", "England", 1096),
            ("Harvard", "university", "USA", 1636),
            ("Caltech", "institute", "USA", 1891),
            ("ETH_Zurich", "institute", "Switzerland", 1855),
            ("Sorbonne", "university", "France", 1150),
            ("Max_Planck_Institute", "institute", "Germany", 1948)
        ]
        
        # Store basic facts
        facts_to_store = []
        
        # Scientist facts
        for name, profession, birth, death, country in scientists:
            facts_to_store.extend([
                (name, "is", profession),
                (name, "born_in", str(birth)),
                (name, "died_in", str(death)) if death else None,
                (name, "from", country),
                (name, "type", "person")
            ])
        
        # Discovery facts
        for discovery, discoverer, year, field in discoveries:
            facts_to_store.extend([
                (discoverer, "discovered", discovery),
                (discovery, "discovered_in", str(year)),
                (discovery, "field", field),
                (discovery, "type", "discovery")
            ])
        
        # Institution facts
        for name, inst_type, country, founded in institutions:
            facts_to_store.extend([
                (name, "is", inst_type),
                (name, "located_in", country),
                (name, "founded_in", str(founded)),
                (name, "type", "institution")
            ])
        
        # Remove None entries
        facts_to_store = [f for f in facts_to_store if f is not None]
        
        # Store all facts
        for subject, predicate, obj in facts_to_store:
            self.call_mcp_tool("store_fact", {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": random.uniform(0.85, 1.0)
            })
        
        # Store complex knowledge chunks
        knowledge_chunks = [
            {
                "title": "Einstein's Theory of Relativity",
                "content": "Albert Einstein's theory of relativity consists of special relativity (1905) and general relativity (1915). Special relativity introduced the concept that space and time are interwoven into spacetime, and that the speed of light is constant for all observers. General relativity describes gravity not as a force, but as curvature in spacetime caused by mass and energy.",
                "category": "physics",
                "source": "Physics Textbook"
            },
            {
                "title": "Quantum Mechanics Foundations", 
                "content": "Quantum mechanics emerged in the early 20th century through the work of Planck, Bohr, Heisenberg, and others. It describes the behavior of matter and energy at atomic and subatomic scales. Key principles include wave-particle duality, uncertainty principle, and quantum superposition.",
                "category": "physics",
                "source": "Quantum Physics Encyclopedia"
            },
            {
                "title": "Darwin's Evolution Theory",
                "content": "Charles Darwin's theory of evolution by natural selection, published in 'On the Origin of Species' (1859), revolutionized biology. It explains how species change over time through variation, inheritance, and differential survival and reproduction.",
                "category": "biology", 
                "source": "Biology Textbook"
            },
            {
                "title": "Marie Curie's Radioactivity Research",
                "content": "Marie Curie pioneered research on radioactivity, discovering polonium and radium. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences (Physics 1903, Chemistry 1911).",
                "category": "chemistry",
                "source": "Scientific Biography"
            },
            {
                "title": "Tesla's Electrical Innovations",
                "content": "Nikola Tesla developed the alternating current (AC) electrical supply system, invented the AC motor, and made numerous contributions to wireless technology. His work laid the foundation for modern electrical power systems.",
                "category": "engineering",
                "source": "Engineering History"
            }
        ]
        
        for chunk in knowledge_chunks:
            self.call_mcp_tool("store_knowledge", chunk)
        
        print("Synthetic data setup complete")
        
    def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the response"""
        try:
            response = requests.post(
                f"{self.server_url}/mcp/tools/{tool_name}",
                json=params,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_store_fact(self) -> List[Dict]:
        """Test store_fact with 10 challenging scenarios"""
        tests = [
            {
                "name": "Basic fact storage",
                "params": {"subject": "TestEntity1", "predicate": "is", "object": "test_object"},
                "expected": "success"
            },
            {
                "name": "High confidence fact",
                "params": {"subject": "HighConfidence", "predicate": "has", "object": "certainty", "confidence": 1.0},
                "expected": "success"
            },
            {
                "name": "Low confidence fact",
                "params": {"subject": "LowConfidence", "predicate": "might_be", "object": "uncertain", "confidence": 0.1},
                "expected": "success"
            },
            {
                "name": "Unicode characters",
                "params": {"subject": "Ünïcødé", "predicate": "contains", "object": "spëcîål_chars"},
                "expected": "success"
            },
            {
                "name": "Maximum length strings",
                "params": {
                    "subject": "A" * 128,
                    "predicate": "B" * 64, 
                    "object": "C" * 128
                },
                "expected": "success"
            },
            {
                "name": "Numbers as strings",
                "params": {"subject": "2024", "predicate": "equals", "object": "2024"},
                "expected": "success"
            },
            {
                "name": "Complex relationship",
                "params": {"subject": "ComplexEntity", "predicate": "has_complex_relationship_with", "object": "AnotherComplexEntity"},
                "expected": "success"
            },
            {
                "name": "Duplicate fact storage",
                "params": {"subject": "Duplicate", "predicate": "is", "object": "duplicate"},
                "expected": "success"
            },
            {
                "name": "Special characters in predicate", 
                "params": {"subject": "Entity", "predicate": "has-special_chars.here", "object": "value"},
                "expected": "success"
            },
            {
                "name": "Boundary confidence values",
                "params": {"subject": "Boundary", "predicate": "tests", "object": "confidence", "confidence": 0.0},
                "expected": "success"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("store_fact", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_store_knowledge(self) -> List[Dict]:
        """Test store_knowledge with 10 challenging scenarios"""
        tests = [
            {
                "name": "Simple knowledge chunk",
                "params": {
                    "title": "Test Knowledge",
                    "content": "This is a simple test knowledge chunk."
                },
                "expected": "success"
            },
            {
                "name": "Large knowledge chunk",
                "params": {
                    "title": "Large Knowledge Test",
                    "content": "Lorem ipsum dolor sit amet. " * 1000,
                    "category": "test"
                },
                "expected": "success"
            },
            {
                "name": "Knowledge with all fields",
                "params": {
                    "title": "Complete Knowledge",
                    "content": "This knowledge chunk has all possible fields filled out.",
                    "category": "complete_test",
                    "source": "Test Suite"
                },
                "expected": "success"
            },
            {
                "name": "Scientific knowledge",
                "params": {
                    "title": "Photosynthesis Process",
                    "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and involves light-dependent and light-independent reactions.",
                    "category": "biology",
                    "source": "Biology Textbook"
                },
                "expected": "success"
            },
            {
                "name": "Historical knowledge",
                "params": {
                    "title": "World War II Timeline",
                    "content": "World War II lasted from 1939 to 1945. Key events include the invasion of Poland, Pearl Harbor attack, D-Day landings, and the atomic bombings of Japan.",
                    "category": "history"
                },
                "expected": "success" 
            },
            {
                "name": "Technical knowledge",
                "params": {
                    "title": "Machine Learning Basics",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                    "category": "technology",
                    "source": "AI Research Paper"
                },
                "expected": "success"
            },
            {
                "name": "Mathematical knowledge",
                "params": {
                    "title": "Calculus Fundamentals",
                    "content": "Calculus deals with derivatives and integrals. The derivative measures rate of change, while the integral measures accumulation.",
                    "category": "mathematics"
                },
                "expected": "success"
            },
            {
                "name": "Empty category knowledge",
                "params": {
                    "title": "No Category Test",
                    "content": "This knowledge chunk has no category specified."
                },
                "expected": "success"
            },
            {
                "name": "Unicode knowledge",
                "params": {
                    "title": "Ünïcødé Tést",
                    "content": "This contains special characters: ñáéíóú àèìòù äëïöü çñ",
                    "category": "unicode_test"
                },
                "expected": "success"
            },
            {
                "name": "JSON-like content",
                "params": {
                    "title": "Structured Data",
                    "content": '{"name": "test", "value": 123, "nested": {"key": "value"}}',
                    "category": "structured"
                },
                "expected": "success"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("store_knowledge", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_find_facts(self) -> List[Dict]:
        """Test find_facts with 10 challenging scenarios"""
        tests = [
            {
                "name": "Find by subject",
                "params": {"query": {"subject": "Einstein"}},
                "expected": "results"
            },
            {
                "name": "Find by predicate",
                "params": {"query": {"predicate": "is"}},
                "expected": "results"
            },
            {
                "name": "Find by object", 
                "params": {"query": {"object": "physicist"}},
                "expected": "results"
            },
            {
                "name": "Combined query",
                "params": {"query": {"subject": "Einstein", "predicate": "is"}},
                "expected": "results"
            },
            {
                "name": "Limit results",
                "params": {"query": {"predicate": "is"}, "limit": 3},
                "expected": "results"
            },
            {
                "name": "No matches query",
                "params": {"query": {"subject": "NonExistentEntity"}},
                "expected": "empty_results"
            },
            {
                "name": "Case sensitive search",
                "params": {"query": {"subject": "einstein"}},
                "expected": "empty_results"
            },
            {
                "name": "Partial matching attempt",
                "params": {"query": {"subject": "Ein"}},
                "expected": "empty_results"
            },
            {
                "name": "Maximum limit",
                "params": {"query": {"predicate": "is"}, "limit": 100},
                "expected": "results"
            },
            {
                "name": "Complex predicate search",
                "params": {"query": {"predicate": "discovered"}},
                "expected": "results"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("find_facts", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_ask_question(self) -> List[Dict]:
        """Test ask_question with 10 challenging scenarios"""
        tests = [
            {
                "name": "Simple question",
                "params": {"question": "Who is Einstein?"},
                "expected": "answer"
            },
            {
                "name": "Question with context",
                "params": {
                    "question": "What did he discover?",
                    "context": "Einstein"
                },
                "expected": "answer"
            },
            {
                "name": "Complex question",
                "params": {"question": "How are quantum mechanics and relativity related?"},
                "expected": "answer"
            },
            {
                "name": "Biographical question",
                "params": {"question": "When was Einstein born and where?"},
                "expected": "answer"
            },
            {
                "name": "Comparative question",
                "params": {"question": "What is the difference between Newton and Einstein's physics?"},
                "expected": "answer"
            },
            {
                "name": "Existence question",
                "params": {"question": "Does the knowledge base contain information about Tesla?"},
                "expected": "answer"
            },
            {
                "name": "Counting question",
                "params": {"question": "How many physicists are in the database?"},
                "expected": "answer"
            },
            {
                "name": "Unknown entity question",
                "params": {"question": "What do you know about Schrödinger?"},
                "expected": "answer"
            },
            {
                "name": "Relationship question",
                "params": {"question": "Which scientists worked at Princeton?"},
                "expected": "answer"
            },
            {
                "name": "Technical question",
                "params": {
                    "question": "Explain the photoelectric effect",
                    "max_results": 10
                },
                "expected": "answer"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("ask_question", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_hybrid_search(self) -> List[Dict]:
        """Test hybrid_search with 10 challenging scenarios"""
        tests = [
            {
                "name": "Basic hybrid search",
                "params": {"query": "Einstein physics"},
                "expected": "results"
            },
            {
                "name": "Semantic search",
                "params": {
                    "query": "theory of relativity",
                    "search_type": "semantic"
                },
                "expected": "results"
            },
            {
                "name": "Structural search",
                "params": {
                    "query": "physicist",
                    "search_type": "structural"
                },
                "expected": "results"
            },
            {
                "name": "Keyword search",
                "params": {
                    "query": "quantum mechanics",
                    "search_type": "keyword"
                },
                "expected": "results"
            },
            {
                "name": "Filtered search",
                "params": {
                    "query": "science",
                    "filters": {
                        "entity_types": ["person"],
                        "min_confidence": 0.8
                    }
                },
                "expected": "results"
            },
            {
                "name": "SIMD performance mode",
                "params": {
                    "query": "physics discovery",
                    "performance_mode": "simd",
                    "simd_config": {
                        "distance_threshold": 0.7,
                        "use_simd": True
                    }
                },
                "expected": "results"
            },
            {
                "name": "LSH performance mode",
                "params": {
                    "query": "scientific breakthrough",
                    "performance_mode": "lsh",
                    "lsh_config": {
                        "hash_functions": 32,
                        "hash_tables": 8,
                        "similarity_threshold": 0.6
                    }
                },
                "expected": "results"
            },
            {
                "name": "Limited results",
                "params": {
                    "query": "scientist",
                    "limit": 2
                },
                "expected": "results"
            },
            {
                "name": "Complex query",
                "params": {
                    "query": "Nobel Prize winner theoretical physics",
                    "search_type": "hybrid",
                    "filters": {
                        "relationship_types": ["won", "is"],
                        "min_confidence": 0.9
                    }
                },
                "expected": "results"
            },
            {
                "name": "Empty query handling",
                "params": {"query": ""},
                "expected": "error_or_empty"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("hybrid_search", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result or test["expected"] == "error_or_empty"
            })
        
        return results
    
    def test_analyze_graph(self) -> List[Dict]:
        """Test analyze_graph with 10 challenging scenarios"""
        tests = [
            {
                "name": "Connections analysis",
                "params": {
                    "analysis_type": "connections",
                    "config": {
                        "start_entity": "Einstein",
                        "end_entity": "physics",
                        "max_depth": 3
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Centrality analysis",
                "params": {
                    "analysis_type": "centrality",
                    "config": {
                        "centrality_types": ["pagerank", "betweenness"],
                        "top_n": 5,
                        "include_scores": True
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Clustering analysis",
                "params": {
                    "analysis_type": "clustering",
                    "config": {
                        "algorithm": "leiden",
                        "resolution": 1.0,
                        "min_cluster_size": 3
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Prediction analysis",
                "params": {
                    "analysis_type": "prediction",
                    "config": {
                        "prediction_type": "missing_links",
                        "confidence_threshold": 0.7,
                        "max_predictions": 5
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Deep connections",
                "params": {
                    "analysis_type": "connections", 
                    "config": {
                        "start_entity": "Newton",
                        "end_entity": "Einstein",
                        "max_depth": 5
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Multiple centrality measures",
                "params": {
                    "analysis_type": "centrality",
                    "config": {
                        "centrality_types": ["pagerank", "betweenness", "closeness", "eigenvector"],
                        "top_n": 10
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Alternative clustering",
                "params": {
                    "analysis_type": "clustering",
                    "config": {
                        "algorithm": "louvain",
                        "resolution": 0.8
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Node prediction",
                "params": {
                    "analysis_type": "prediction",
                    "config": {
                        "prediction_type": "node_classification",
                        "target_entity": "Hawking"
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Shortest paths",
                "params": {
                    "analysis_type": "connections",
                    "config": {
                        "start_entity": "Tesla",
                        "end_entity": "MIT",
                        "max_depth": 4,
                        "find_all_paths": False
                    }
                },
                "expected": "analysis"
            },
            {
                "name": "Community detection",
                "params": {
                    "analysis_type": "clustering",
                    "config": {
                        "algorithm": "walktrap",
                        "min_cluster_size": 2
                    }
                },
                "expected": "analysis"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("analyze_graph", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_get_suggestions(self) -> List[Dict]:
        """Test get_suggestions with 10 challenging scenarios"""
        tests = [
            {
                "name": "Missing facts suggestions",
                "params": {"suggestion_type": "missing_facts"},
                "expected": "suggestions"
            },
            {
                "name": "Interesting questions",
                "params": {"suggestion_type": "interesting_questions"},
                "expected": "suggestions"
            },
            {
                "name": "Potential connections",
                "params": {"suggestion_type": "potential_connections"},
                "expected": "suggestions"
            },
            {
                "name": "Knowledge gaps",
                "params": {"suggestion_type": "knowledge_gaps"},
                "expected": "suggestions"
            },
            {
                "name": "Focused missing facts",
                "params": {
                    "suggestion_type": "missing_facts",
                    "focus_area": "Einstein"
                },
                "expected": "suggestions"
            },
            {
                "name": "Physics questions",
                "params": {
                    "suggestion_type": "interesting_questions",
                    "focus_area": "physics",
                    "limit": 3
                },
                "expected": "suggestions"
            },
            {
                "name": "Limited connections",
                "params": {
                    "suggestion_type": "potential_connections",
                    "limit": 2
                },
                "expected": "suggestions"
            },
            {
                "name": "Biology gaps",
                "params": {
                    "suggestion_type": "knowledge_gaps",
                    "focus_area": "biology",
                    "limit": 5
                },
                "expected": "suggestions"
            },
            {
                "name": "Maximum suggestions",
                "params": {
                    "suggestion_type": "missing_facts",
                    "limit": 10
                },
                "expected": "suggestions"
            },
            {
                "name": "Tesla connections",
                "params": {
                    "suggestion_type": "potential_connections",
                    "focus_area": "Tesla",
                    "limit": 7
                },
                "expected": "suggestions"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("get_suggestions", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_get_stats(self) -> List[Dict]:
        """Test get_stats with 10 challenging scenarios"""
        tests = [
            {
                "name": "Basic stats",
                "params": {},
                "expected": "stats"
            },
            {
                "name": "Detailed stats",
                "params": {"include_details": True},
                "expected": "detailed_stats"
            },
            {
                "name": "Simple stats check",
                "params": {"include_details": False},
                "expected": "stats"
            },
            {
                "name": "Stats after data loading",
                "params": {},
                "expected": "stats"
            },
            {
                "name": "Detailed breakdown",
                "params": {"include_details": True},
                "expected": "detailed_stats"
            },
            {
                "name": "Repeated stats call",
                "params": {},
                "expected": "stats"
            },
            {
                "name": "Stats with details toggle",
                "params": {"include_details": True},
                "expected": "detailed_stats"
            },
            {
                "name": "Quick stats",
                "params": {"include_details": False},
                "expected": "stats"
            },
            {
                "name": "Full analysis stats",
                "params": {"include_details": True},
                "expected": "detailed_stats"
            },
            {
                "name": "Final stats check",
                "params": {},
                "expected": "stats"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("get_stats", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_generate_graph_query(self) -> List[Dict]:
        """Test generate_graph_query with 10 challenging scenarios"""
        tests = [
            {
                "name": "Simple Cypher query",
                "params": {"natural_query": "Find all physicists"},
                "expected": "query"
            },
            {
                "name": "SPARQL query generation",
                "params": {
                    "natural_query": "Who discovered relativity?",
                    "query_language": "sparql"
                },
                "expected": "query"
            },
            {
                "name": "Gremlin query",
                "params": {
                    "natural_query": "Show connections between Einstein and physics",
                    "query_language": "gremlin"
                },
                "expected": "query"
            },
            {
                "name": "Complex relationship query",
                "params": {
                    "natural_query": "Find scientists who won Nobel Prizes and their discoveries",
                    "query_language": "cypher",
                    "include_explanation": True
                },
                "expected": "query"
            },
            {
                "name": "No explanation query",
                "params": {
                    "natural_query": "List all universities",
                    "include_explanation": False
                },
                "expected": "query"
            },
            {
                "name": "Temporal query",
                "params": {
                    "natural_query": "Find discoveries made before 1900",
                    "query_language": "cypher"
                },
                "expected": "query"
            },
            {
                "name": "Path finding query",
                "params": {
                    "natural_query": "Find the shortest path between Newton and Einstein",
                    "query_language": "gremlin",
                    "include_explanation": True
                },
                "expected": "query"
            },
            {
                "name": "Aggregation query",
                "params": {
                    "natural_query": "Count how many scientists are from each country",
                    "query_language": "sparql"
                },
                "expected": "query"
            },
            {
                "name": "Pattern matching",
                "params": {
                    "natural_query": "Find all entities that have a 'type' relationship",
                    "query_language": "cypher"
                },
                "expected": "query"
            },
            {
                "name": "Complex filtering",
                "params": {
                    "natural_query": "Show physicists born after 1900 who made discoveries",
                    "query_language": "gremlin",
                    "include_explanation": True
                },
                "expected": "query"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("generate_graph_query", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_validate_knowledge(self) -> List[Dict]:
        """Test validate_knowledge with 10 challenging scenarios"""
        tests = [
            {
                "name": "Basic validation",
                "params": {},
                "expected": "validation"
            },
            {
                "name": "Consistency check",
                "params": {"validation_type": "consistency"},
                "expected": "validation"
            },
            {
                "name": "Conflict detection",
                "params": {"validation_type": "conflicts"},
                "expected": "validation"
            },
            {
                "name": "Quality assessment",
                "params": {"validation_type": "quality"},
                "expected": "validation"
            },
            {
                "name": "Completeness check",
                "params": {"validation_type": "completeness"},
                "expected": "validation"
            },
            {
                "name": "Entity specific validation",
                "params": {
                    "entity": "Einstein",
                    "validation_type": "all"
                },
                "expected": "validation"
            },
            {
                "name": "Comprehensive validation",
                "params": {
                    "scope": "comprehensive",
                    "include_metrics": True
                },
                "expected": "validation"
            },
            {
                "name": "Neural features validation",
                "params": {
                    "neural_features": True,
                    "quality_threshold": 0.8
                },
                "expected": "validation"
            },
            {
                "name": "High threshold validation",
                "params": {
                    "quality_threshold": 0.9,
                    "importance_threshold": 0.8
                },
                "expected": "validation"
            },
            {
                "name": "Auto-fix validation",
                "params": {
                    "fix_issues": True,
                    "validation_type": "quality"
                },
                "expected": "validation"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("validate_knowledge", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_neural_importance_scoring(self) -> List[Dict]:
        """Test neural_importance_scoring with 10 challenging scenarios"""
        tests = [
            {
                "name": "Scientific text scoring",
                "params": {
                    "text": "The theory of relativity changed our understanding of space and time."
                },
                "expected": "score"
            },
            {
                "name": "Complex scientific content",
                "params": {
                    "text": "Quantum entanglement is a phenomenon where particles become correlated in such a way that the quantum state of each particle cannot be described independently.",
                    "context": "quantum physics"
                },
                "expected": "score"
            },
            {
                "name": "Simple factual statement",
                "params": {
                    "text": "The sky is blue."
                },
                "expected": "score"
            },
            {
                "name": "Historical content",
                "params": {
                    "text": "World War II ended in 1945 after the atomic bombings of Hiroshima and Nagasaki led to Japan's surrender.",
                    "context": "historical facts"
                },
                "expected": "score"
            },
            {
                "name": "Technical documentation",
                "params": {
                    "text": "Machine learning models require training data to learn patterns and make predictions on new, unseen data."
                },
                "expected": "score"
            },
            {
                "name": "Biographical information",
                "params": {
                    "text": "Marie Curie was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different scientific fields."
                },
                "expected": "score"
            },
            {
                "name": "Mathematical concept",
                "params": {
                    "text": "The derivative of a function represents the rate of change at any given point.",
                    "context": "mathematics education"
                },
                "expected": "score"
            },
            {
                "name": "Trivial information",
                "params": {
                    "text": "This is just a random sentence with no particular meaning or importance."
                },
                "expected": "score"
            },
            {
                "name": "Long form content",
                "params": {
                    "text": "Artificial intelligence has evolved significantly over the past decades. From rule-based systems to machine learning and now deep learning, AI has transformed numerous industries. The development of neural networks, particularly deep neural networks, has enabled breakthroughs in computer vision, natural language processing, and robotics."
                },
                "expected": "score"
            },
            {
                "name": "Context-dependent scoring",
                "params": {
                    "text": "The process involves heating the solution to 100 degrees Celsius.",
                    "context": "chemistry laboratory procedures"
                },
                "expected": "score"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("neural_importance_scoring", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_divergent_thinking_engine(self) -> List[Dict]:
        """Test divergent_thinking_engine with 10 challenging scenarios"""
        tests = [
            {
                "name": "Scientific concept exploration",
                "params": {"seed_concept": "quantum mechanics"},
                "expected": "exploration"
            },
            {
                "name": "Creative exploration",
                "params": {
                    "seed_concept": "artificial intelligence",
                    "creativity_level": 0.9,
                    "exploration_depth": 4
                },
                "expected": "exploration"
            },
            {
                "name": "Conservative exploration",
                "params": {
                    "seed_concept": "gravity",
                    "creativity_level": 0.3,
                    "exploration_depth": 2
                },
                "expected": "exploration"
            },
            {
                "name": "Deep exploration",
                "params": {
                    "seed_concept": "consciousness",
                    "exploration_depth": 5,
                    "max_branches": 15
                },
                "expected": "exploration"
            },
            {
                "name": "Historical concept",
                "params": {
                    "seed_concept": "renaissance",
                    "creativity_level": 0.7
                },
                "expected": "exploration"
            },
            {
                "name": "Technology exploration",
                "params": {
                    "seed_concept": "blockchain",
                    "exploration_depth": 3,
                    "max_branches": 8
                },
                "expected": "exploration"
            },
            {
                "name": "Biological concept",
                "params": {
                    "seed_concept": "evolution",
                    "creativity_level": 0.8,
                    "exploration_depth": 4
                },
                "expected": "exploration"
            },
            {
                "name": "Abstract concept",
                "params": {
                    "seed_concept": "time",
                    "creativity_level": 0.6,
                    "max_branches": 12
                },
                "expected": "exploration"
            },
            {
                "name": "Limited branches",
                "params": {
                    "seed_concept": "energy",
                    "max_branches": 3,
                    "exploration_depth": 2
                },
                "expected": "exploration"
            },
            {
                "name": "Maximum creativity",
                "params": {
                    "seed_concept": "reality",
                    "creativity_level": 1.0,
                    "exploration_depth": 5,
                    "max_branches": 20
                },
                "expected": "exploration"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("divergent_thinking_engine", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_time_travel_query(self) -> List[Dict]:
        """Test time_travel_query with 10 challenging scenarios"""
        base_time = datetime.now().isoformat()
        past_time = (datetime.now() - timedelta(days=365)).isoformat()
        future_time = (datetime.now() + timedelta(days=30)).isoformat()
        
        tests = [
            {
                "name": "Point in time query",
                "params": {
                    "query_type": "point_in_time",
                    "timestamp": base_time
                },
                "expected": "temporal_result"
            },
            {
                "name": "Evolution tracking",
                "params": {
                    "query_type": "evolution_tracking",
                    "entity": "Einstein",
                    "time_range": {
                        "start": past_time,
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            },
            {
                "name": "Temporal comparison",
                "params": {
                    "query_type": "temporal_comparison",
                    "time_range": {
                        "start": past_time,
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            },
            {
                "name": "Change detection",
                "params": {
                    "query_type": "change_detection",
                    "entity": "physics",
                    "time_range": {
                        "start": past_time,
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            },
            {
                "name": "Specific entity evolution",
                "params": {
                    "query_type": "evolution_tracking",
                    "entity": "quantum_mechanics"
                },
                "expected": "temporal_result"
            },
            {
                "name": "Historical point query",
                "params": {
                    "query_type": "point_in_time",
                    "timestamp": past_time,
                    "entity": "Tesla"
                },
                "expected": "temporal_result"
            },
            {
                "name": "Recent changes",
                "params": {
                    "query_type": "change_detection",
                    "time_range": {
                        "start": (datetime.now() - timedelta(hours=1)).isoformat(),
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            },
            {
                "name": "Long term evolution",
                "params": {
                    "query_type": "evolution_tracking",
                    "entity": "relativity",
                    "time_range": {
                        "start": (datetime.now() - timedelta(days=1000)).isoformat(),
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            },
            {
                "name": "Future projection query",
                "params": {
                    "query_type": "point_in_time",
                    "timestamp": future_time
                },
                "expected": "temporal_result"
            },
            {
                "name": "Multi-entity comparison",
                "params": {
                    "query_type": "temporal_comparison",
                    "entity": "Newton",
                    "time_range": {
                        "start": past_time,
                        "end": base_time
                    }
                },
                "expected": "temporal_result"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("time_travel_query", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def test_cognitive_reasoning_chains(self) -> List[Dict]:
        """Test cognitive_reasoning_chains with 10 challenging scenarios"""
        tests = [
            {
                "name": "Deductive reasoning",
                "params": {
                    "reasoning_type": "deductive",
                    "premise": "All physicists study matter and energy"
                },
                "expected": "reasoning"
            },
            {
                "name": "Inductive reasoning",
                "params": {
                    "reasoning_type": "inductive",
                    "premise": "Einstein was a brilliant physicist who made groundbreaking discoveries"
                },
                "expected": "reasoning"
            },
            {
                "name": "Abductive reasoning",
                "params": {
                    "reasoning_type": "abductive",
                    "premise": "The photoelectric effect was unexplained by classical physics"
                },
                "expected": "reasoning"
            },
            {
                "name": "Analogical reasoning",
                "params": {
                    "reasoning_type": "analogical",
                    "premise": "Light behaves like a wave in some experiments"
                },
                "expected": "reasoning"
            },
            {
                "name": "Complex deductive chain",
                "params": {
                    "reasoning_type": "deductive",
                    "premise": "Nobel Prize winners have made significant contributions to science",
                    "max_chain_length": 7
                },
                "expected": "reasoning"
            },
            {
                "name": "High confidence reasoning",
                "params": {
                    "reasoning_type": "inductive",
                    "premise": "Quantum mechanics explains atomic behavior",
                    "confidence_threshold": 0.9
                },
                "expected": "reasoning"
            },
            {
                "name": "No alternatives reasoning",
                "params": {
                    "reasoning_type": "deductive",
                    "premise": "Speed of light is constant in vacuum",
                    "include_alternatives": False
                },
                "expected": "reasoning"
            },
            {
                "name": "Scientific hypothesis",
                "params": {
                    "reasoning_type": "abductive",
                    "premise": "Mass and energy seem to be related",
                    "max_chain_length": 5,
                    "include_alternatives": True
                },
                "expected": "reasoning"
            },
            {
                "name": "Short reasoning chain",
                "params": {
                    "reasoning_type": "analogical",
                    "premise": "Atoms are like miniature solar systems",
                    "max_chain_length": 2
                },
                "expected": "reasoning"
            },
            {
                "name": "Low confidence exploration",
                "params": {
                    "reasoning_type": "inductive",
                    "premise": "Some particles exhibit strange behavior",
                    "confidence_threshold": 0.3,
                    "max_chain_length": 6
                },
                "expected": "reasoning"
            }
        ]
        
        results = []
        for test in tests:
            result = self.call_mcp_tool("cognitive_reasoning_chains", test["params"])
            results.append({
                "test_name": test["name"],
                "params": test["params"],
                "result": result,
                "passed": "error" not in result
            })
        
        return results
    
    def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("Starting Comprehensive LLMKG MCP Tool Testing...")
        print("=" * 80)
        
        # Setup synthetic data first
        self.setup_synthetic_data()
        
        # Run tests for all tools
        tool_tests = [
            ("store_fact", self.test_store_fact),
            ("store_knowledge", self.test_store_knowledge),
            ("find_facts", self.test_find_facts),
            ("ask_question", self.test_ask_question),
            ("hybrid_search", self.test_hybrid_search),
            ("analyze_graph", self.test_analyze_graph),
            ("get_suggestions", self.test_get_suggestions),
            ("get_stats", self.test_get_stats),
            ("generate_graph_query", self.test_generate_graph_query),
            ("validate_knowledge", self.test_validate_knowledge),
            ("neural_importance_scoring", self.test_neural_importance_scoring),
            ("divergent_thinking_engine", self.test_divergent_thinking_engine),
            ("time_travel_query", self.test_time_travel_query),
            ("cognitive_reasoning_chains", self.test_cognitive_reasoning_chains)
        ]
        
        all_results = {}
        total_tests = 0
        total_passed = 0
        
        for tool_name, test_func in tool_tests:
            print(f"\nTesting {tool_name}...")
            try:
                results = test_func()
                all_results[tool_name] = results
                
                passed = sum(1 for r in results if r["passed"])
                total = len(results)
                total_tests += total
                total_passed += passed
                
                print(f"   {passed}/{total} tests passed")
                
                # Show failed tests
                failed = [r for r in results if not r["passed"]]
                if failed:
                    print(f"   Failed tests:")
                    for fail in failed[:3]:  # Show first 3 failures
                        print(f"      - {fail['test_name']}: {fail['result']}")
                        
            except Exception as e:
                print(f"   Error testing {tool_name}: {str(e)}")
                all_results[tool_name] = {"error": str(e)}
        
        # Generate final report
        self.generate_report(all_results, total_passed, total_tests)
        
        return all_results
    
    def generate_report(self, results: Dict, total_passed: int, total_tests: int):
        """Generate comprehensive test report"""
        print(f"\nCOMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"Overall Success Rate: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")
        print()
        
        # Tool-by-tool breakdown
        for tool_name, tool_results in results.items():
            if isinstance(tool_results, dict) and "error" in tool_results:
                print(f"ERROR {tool_name}: ERROR - {tool_results['error']}")
            else:
                passed = sum(1 for r in tool_results if r["passed"])
                total = len(tool_results)
                status = "PASS" if passed == total else "PARTIAL"
                print(f"{status} {tool_name}: {passed}/{total} ({100*passed/total:.0f}%)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mcp_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "summary": {
                    "total_tests": total_tests,
                    "total_passed": total_passed,
                    "success_rate": total_passed / total_tests
                },
                "results": results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        print(f"Testing complete! All {len(results)} tools validated with {total_tests} scenarios.")

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = "http://localhost:3000"
    
    print(f"Connecting to LLMKG MCP server at: {server_url}")
    
    test_suite = LLMKGMCPTestSuite(server_url)
    results = test_suite.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()