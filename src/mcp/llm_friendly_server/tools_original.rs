//! Tool definitions for the LLM-friendly MCP server

use crate::mcp::llm_friendly_server::types::{LLMMCPTool, LLMExample};
use serde_json::json;

/// Get all available tools with LLM-friendly descriptions and examples
pub fn get_tools() -> Vec<LLMMCPTool> {
    vec![
        LLMMCPTool {
            name: "store_fact".to_string(),
            description: "Store a simple fact as a Subject-Predicate-Object triple. This is the most basic way to add knowledge. Use short, clear predicates (1-3 words max).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The entity or thing the fact is about (e.g., 'Einstein', 'Python', 'New York')",
                        "maxLength": 128
                    },
                    "predicate": {
                        "type": "string", 
                        "description": "The relationship or property (e.g., 'is', 'invented', 'located_in', 'has'). Keep it short!",
                        "maxLength": 64
                    },
                    "object": {
                        "type": "string",
                        "description": "What the subject is related to (e.g., 'scientist', 'relativity', 'USA')",
                        "maxLength": 128
                    },
                    "confidence": {
                        "type": "number",
                        "description": "How confident you are in this fact (0.0 to 1.0, default: 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 1.0
                    }
                },
                "required": ["subject", "predicate", "object"]
            }),
            examples: vec![
                LLMExample {
                    description: "Store that Einstein was a scientist".to_string(),
                    input: json!({
                        "subject": "Einstein",
                        "predicate": "is",
                        "object": "scientist"
                    }),
                    expected_output: "Stored fact: Einstein is scientist".to_string(),
                },
                LLMExample {
                    description: "Store that Python was created by Guido van Rossum".to_string(),
                    input: json!({
                        "subject": "Python",
                        "predicate": "created_by",
                        "object": "Guido van Rossum",
                        "confidence": 1.0
                    }),
                    expected_output: "Stored fact: Python created_by Guido van Rossum (confidence: 1.0)".to_string(),
                }
            ],
            tips: vec![
                "Use consistent naming (e.g., always 'New_York' not sometimes 'new york')".to_string(),
                "Keep predicates simple and reusable (prefer 'located_in' over 'is_located_in_the_country_of')".to_string(),
                "When unsure about confidence, use 0.8 instead of 1.0".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "store_knowledge".to_string(),
            description: "Store more complex knowledge as a chunk of text. Use this for descriptions, explanations, or any text longer than a simple fact. The system will automatically extract entities and relationships.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The knowledge content to store (can be a paragraph, list, or structured text)",
                        "maxLength": 50000
                    },
                    "title": {
                        "type": "string",
                        "description": "A short title or summary of this knowledge",
                        "maxLength": 200
                    },
                    "category": {
                        "type": "string",
                        "description": "Category or type of knowledge (e.g., 'biography', 'technical', 'historical')",
                        "maxLength": 50
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this knowledge came from (optional)",
                        "maxLength": 200
                    }
                },
                "required": ["content", "title"]
            }),
            examples: vec![
                LLMExample {
                    description: "Store a biography".to_string(),
                    input: json!({
                        "title": "Albert Einstein Biography",
                        "content": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity. He received the Nobel Prize in Physics in 1921.",
                        "category": "biography",
                        "source": "Wikipedia"
                    }),
                    expected_output: "Stored knowledge chunk 'Albert Einstein Biography' with 3 extracted entities and 5 relationships".to_string(),
                }
            ],
            tips: vec![
                "Break very long texts into logical chunks (e.g., one per topic)".to_string(),
                "Include dates, names, and specific facts for better extraction".to_string(),
                "Use the category field to help with later retrieval".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "find_facts".to_string(),
            description: "Find facts (triples) about a subject or matching a pattern. Returns Subject-Predicate-Object triples. At least one of subject, predicate, or object must be provided.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Query parameters - must provide at least one of subject, predicate, or object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Find facts about this subject"
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Find facts with this relationship type"
                            },
                            "object": {
                                "type": "string",
                                "description": "Find facts pointing to this object"
                            }
                        },
                        "minProperties": 1,
                        "additionalProperties": false
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of facts to return (default: 10, max: 100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            examples: vec![
                LLMExample {
                    description: "Find all facts about Einstein".to_string(),
                    input: json!({
                        "query": {
                            "subject": "Einstein"
                        },
                        "limit": 5
                    }),
                    expected_output: "Found 5 facts:\n1. Einstein is scientist\n2. Einstein invented relativity\n3. Einstein born_in Germany\n4. Einstein won Nobel_Prize\n5. Einstein died_in 1955".to_string(),
                },
                LLMExample {
                    description: "Find who invented what".to_string(),
                    input: json!({
                        "query": {
                            "predicate": "invented"
                        },
                        "limit": 3
                    }),
                    expected_output: "Found 3 facts:\n1. Einstein invented relativity\n2. Edison invented light_bulb\n3. Tesla invented AC_motor".to_string(),
                }
            ],
            tips: vec![
                "Leave fields empty to make them wildcards".to_string(),
                "Use partial matching by including part of a name".to_string(),
                "Combine multiple fields for more specific queries".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "ask_question".to_string(),
            description: "Ask a natural language question about the stored knowledge. The system will search for relevant facts and knowledge chunks to answer your question.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question in natural language",
                        "maxLength": 500
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context to help answer the question (optional)",
                        "maxLength": 500
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of relevant pieces to return (default: 5)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["question"]
            }),
            examples: vec![
                LLMExample {
                    description: "Ask about Einstein's contributions".to_string(),
                    input: json!({
                        "question": "What did Einstein invent or discover?",
                        "max_results": 3
                    }),
                    expected_output: "Based on the knowledge graph:\n1. Einstein invented the theory of relativity\n2. Einstein discovered the photoelectric effect\n3. Einstein developed E=mc²\n\nRelevant knowledge chunks: [Biography of Einstein, Einstein's Major Works]".to_string(),
                }
            ],
            tips: vec![
                "Be specific in your questions for better results".to_string(),
                "Use the context field to disambiguate (e.g., 'context: \"in physics\"')".to_string(),
                "Questions about relationships work well (e.g., 'How are X and Y related?')".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "explore_connections".to_string(),
            description: "Explore connections between entities. Find paths, common relationships, or network patterns. Great for discovering non-obvious relationships.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start_entity": {
                        "type": "string",
                        "description": "Starting entity to explore from",
                        "maxLength": 128
                    },
                    "end_entity": {
                        "type": "string",
                        "description": "Target entity to find connections to (optional)",
                        "maxLength": 128
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "How many relationship hops to explore (default: 2, max: 4)",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 2
                    },
                    "relationship_types": {
                        "type": "array",
                        "description": "Specific relationship types to follow (optional)",
                        "items": {"type": "string"}
                    }
                },
                "required": ["start_entity"]
            }),
            examples: vec![
                LLMExample {
                    description: "Find connections between Einstein and Nobel Prize".to_string(),
                    input: json!({
                        "start_entity": "Einstein",
                        "end_entity": "Nobel_Prize",
                        "max_depth": 3
                    }),
                    expected_output: "Found 2 paths:\n1. Einstein -> won -> Nobel_Prize (length: 1)\n2. Einstein -> contemporary_of -> Bohr -> won -> Nobel_Prize (length: 2)".to_string(),
                }
            ],
            tips: vec![
                "Use max_depth=1 to find direct connections only".to_string(),
                "Omit end_entity to explore all connections from start_entity".to_string(),
                "Use relationship_types to focus on specific kinds of connections".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "get_stats".to_string(),
            description: "Get statistics about the knowledge graph including size, coverage, and usage patterns. Useful for understanding the current state of your knowledge base.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed breakdown by category/type (default: false)",
                        "default": false
                    }
                },
                "additionalProperties": false
            }),
            examples: vec![
                LLMExample {
                    description: "Get basic statistics".to_string(),
                    input: json!({}),
                    expected_output: "Knowledge Graph Statistics:\n- Total facts (triples): 1,247\n- Total entities: 523\n- Total relationships: 42 types\n- Knowledge chunks: 89\n- Average facts per entity: 2.4\n- Memory efficiency: 94.3%".to_string(),
                }
            ],
            tips: vec![
                "Check stats regularly to monitor growth".to_string(),
                "Use include_details=true for category breakdowns".to_string(),
                "Memory efficiency shows how well data is compressed".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "hybrid_search".to_string(),
            description: "Advanced search with multiple performance modes. Combines semantic similarity, graph structure, and text matching with optional hardware acceleration.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language or keywords)",
                        "maxLength": 500
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search to perform",
                        "enum": ["semantic", "structural", "keyword", "hybrid"],
                        "default": "hybrid"
                    },
                    "performance_mode": {
                        "type": "string",
                        "description": "Performance optimization mode",
                        "enum": ["standard", "simd", "lsh"],
                        "default": "standard"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters",
                        "properties": {
                            "entity_types": {"type": "array", "items": {"type": "string"}},
                            "relationship_types": {"type": "array", "items": {"type": "string"}},
                            "min_confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 10)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    },
                    "simd_config": {
                        "type": "object",
                        "description": "SIMD-specific configuration (when performance_mode='simd')",
                        "properties": {
                            "distance_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                            "use_simd": {"type": "boolean", "default": true}
                        }
                    },
                    "lsh_config": {
                        "type": "object",
                        "description": "LSH-specific configuration (when performance_mode='lsh')",
                        "properties": {
                            "hash_functions": {"type": "integer", "minimum": 8, "maximum": 128, "default": 64},
                            "hash_tables": {"type": "integer", "minimum": 2, "maximum": 32, "default": 8},
                            "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7}
                        }
                    }
                },
                "required": ["query"]
            }),
            examples: vec![
                LLMExample {
                    description: "Standard hybrid search".to_string(),
                    input: json!({
                        "query": "quantum physics discoveries",
                        "search_type": "hybrid",
                        "filters": {
                            "min_confidence": 0.8
                        }
                    }),
                    expected_output: "Found 8 results (hybrid search, standard mode):\n\n1. **Quantum Mechanics** (score: 0.92)\n   - Semantic: High relevance to 'quantum physics'\n   - Structural: Central node with 15 connections\n   - Contains: 'revolutionary physics discovery 1920s'".to_string(),
                },
                LLMExample {
                    description: "SIMD-accelerated search".to_string(),
                    input: json!({
                        "query": "Einstein relativity",
                        "search_type": "semantic",
                        "performance_mode": "simd",
                        "limit": 5
                    }),
                    expected_output: "Found 5 results (semantic search, simd mode):\n⚡ Search Time: 0.34ms\n🚀 Throughput: 15.2 million vectors/sec\n\n1. Einstein -> invented -> special_relativity\n2. Einstein -> developed -> general_relativity".to_string(),
                },
                LLMExample {
                    description: "LSH approximate search".to_string(),
                    input: json!({
                        "query": "machine learning algorithms",
                        "performance_mode": "lsh",
                        "lsh_config": {
                            "hash_functions": 32,
                            "hash_tables": 16
                        }
                    }),
                    expected_output: "Found 12 results (hybrid search, lsh mode):\n⚡ Speedup: 8.5x vs standard\n🎯 Recall: 0.89\n\n1. Decision_Trees -> type_of -> machine_learning".to_string(),
                }
            ],
            tips: vec![
                "Use 'standard' mode for highest accuracy (default)".to_string(),
                "Use 'simd' mode for 10x faster searches on large datasets".to_string(),
                "Use 'lsh' mode for ultra-fast approximate searches".to_string(),
                "Combine search_type with performance_mode for optimal results".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "validate_knowledge".to_string(),
            description: "Validate stored knowledge for consistency, conflicts, and quality. Helps maintain a high-quality knowledge graph.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "validation_type": {
                        "type": "string",
                        "description": "What to validate",
                        "enum": ["consistency", "conflicts", "quality", "completeness", "all"],
                        "default": "all"
                    },
                    "entity": {
                        "type": "string",
                        "description": "Specific entity to validate (optional, validates everything if omitted)",
                        "maxLength": 128
                    },
                    "fix_issues": {
                        "type": "boolean",
                        "description": "Attempt to automatically fix found issues (default: false)",
                        "default": false
                    }
                },
                "additionalProperties": false
            }),
            examples: vec![
                LLMExample {
                    description: "Validate all knowledge".to_string(),
                    input: json!({
                        "validation_type": "all"
                    }),
                    expected_output: "Validation Results:\n\n**Consistency**: ✓ Passed\n- All entity references valid\n- No orphaned relationships\n\n**Conflicts**: ⚠ 2 issues found\n1. Einstein birth date: '1879' vs '1878' (different sources)\n2. Python creation date: '1989' vs '1991'\n\n**Quality**: ✓ Good (score: 8.7/10)\n- Average confidence: 0.87\n- Most facts have sources\n\n**Completeness**: ⚠ Could improve\n- 15 entities missing descriptions\n- 23 relationships lack confidence scores".to_string(),
                }
            ],
            tips: vec![
                "Run validation periodically to maintain quality".to_string(),
                "Use fix_issues=true carefully, review changes".to_string(),
                "Focus validation on specific entities when debugging".to_string(),
            ],
        },

        // ========= TIER 1 ADVANCED COGNITIVE TOOLS =========
        
        LLMMCPTool {
            name: "divergent_thinking_engine".to_string(),
            description: "Creative exploration and ideation engine that generates novel connections, alternative perspectives, and innovative insights from seed concepts.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "seed_concept": {
                        "type": "string",
                        "description": "Starting concept for creative exploration",
                        "maxLength": 200
                    },
                    "exploration_depth": {
                        "type": "integer",
                        "description": "How many conceptual layers to explore (default: 3, max: 5)",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3
                    },
                    "creativity_level": {
                        "type": "number",
                        "description": "Creativity vs relevance balance (0.0 = conservative, 1.0 = highly creative, default: 0.7)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7
                    },
                    "max_branches": {
                        "type": "integer",
                        "description": "Maximum exploration branches (default: 10)",
                        "minimum": 3,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": ["seed_concept"]
            }),
            examples: vec![
                LLMExample {
                    description: "Explore creative connections from artificial intelligence".to_string(),
                    input: json!({
                        "seed_concept": "artificial intelligence",
                        "exploration_depth": 3,
                        "creativity_level": 0.8
                    }),
                    expected_output: "Divergent Thinking Exploration:\n🧠 Seed Concept: artificial intelligence\n🌟 Generated 8 creative paths\n🔗 Found 12 novel connections\n💡 6 cross-domain ideas\n📊 Creativity Score: 0.83/1.0\n🎯 Novelty Score: 0.76/1.0".to_string(),
                }
            ],
            tips: vec![
                "Use higher creativity_level (0.8-0.9) for more novel ideas".to_string(),
                "Increase exploration_depth for deeper insights".to_string(),
                "Store interesting paths as new knowledge".to_string(),
            ],
        },

        LLMMCPTool {
            name: "time_travel_query".to_string(),
            description: "Query knowledge at any point in time using temporal database capabilities. Track evolution, detect changes, and analyze trends over time.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "description": "Type of temporal query",
                        "enum": ["point_in_time", "evolution_tracking", "temporal_comparison", "change_detection"],
                        "default": "point_in_time"
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "ISO timestamp for point-in-time queries",
                        "format": "date-time"
                    },
                    "entity": {
                        "type": "string",
                        "description": "Entity to track over time",
                        "maxLength": 200
                    },
                    "time_range": {
                        "type": "object",
                        "description": "Time range for comparison/evolution queries",
                        "properties": {
                            "start": {"type": "string", "format": "date-time"},
                            "end": {"type": "string", "format": "date-time"}
                        }
                    }
                },
                "required": []
            }),
            examples: vec![
                LLMExample {
                    description: "Track how Einstein's reputation evolved".to_string(),
                    input: json!({
                        "query_type": "evolution_tracking",
                        "entity": "Einstein",
                        "time_range": {
                            "start": "1900-01-01T00:00:00Z",
                            "end": "1955-12-31T23:59:59Z"
                        }
                    }),
                    expected_output: "Time Travel Query Results:\n⏰ Query Type: evolution_tracking\n📊 Data Points: 47\n📈 Changes Detected: 12\n🕰️ Time Span: 55 years\n🔍 Key Insights: Nobel Prize 1921, relativity acceptance grew gradually, peak recognition 1919 eclipse".to_string(),
                }
            ],
            tips: vec![
                "Use 'evolution_tracking' to see how entities change over time".to_string(),
                "Compare different time periods with 'temporal_comparison'".to_string(),
                "Detect anomalies with 'change_detection' queries".to_string(),
            ],
        },

        LLMMCPTool {
            name: "simd_ultra_fast_search".to_string(),
            description: "Hardware-accelerated similarity search using SIMD instructions for ultra-high performance. 10x faster than standard search with configurable precision.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Text to search for (will be converted to vector)",
                        "maxLength": 1000
                    },
                    "query_vector": {
                        "type": "array",
                        "description": "Pre-computed query vector (384 dimensions)",
                        "items": {"type": "number"},
                        "minItems": 384,
                        "maxItems": 384
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    },
                    "distance_threshold": {
                        "type": "number",
                        "description": "Similarity threshold (default: 0.8)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.8
                    },
                    "use_simd": {
                        "type": "boolean",
                        "description": "Enable SIMD acceleration (default: true)",
                        "default": true
                    }
                },
                "anyOf": [
                    {"required": ["query_text"]},
                    {"required": ["query_vector"]}
                ]
            }),
            examples: vec![
                LLMExample {
                    description: "Ultra-fast search for quantum physics content".to_string(),
                    input: json!({
                        "query_text": "quantum mechanics wave function",
                        "top_k": 5,
                        "use_simd": true
                    }),
                    expected_output: "SIMD Ultra-Fast Search Results:\n⚡ Search Time: 0.34ms (341 μs)\n🚀 Throughput: 15.2 million vectors/sec\n🎯 Found 5 matches from 125,000 candidates\n🔧 SIMD Acceleration: Enabled\n📊 Top Match Similarity: 0.954".to_string(),
                }
            ],
            tips: vec![
                "Enable SIMD for 10x faster search performance".to_string(),
                "Adjust distance_threshold for precision/recall balance".to_string(),
                "Use larger top_k for comprehensive results".to_string(),
            ],
        },

        LLMMCPTool {
            name: "analyze_graph_centrality".to_string(),
            description: "Advanced graph analysis using centrality measures (PageRank, betweenness, closeness) to identify the most important and influential entities.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "centrality_types": {
                        "type": "array",
                        "description": "Types of centrality to calculate",
                        "items": {
                            "type": "string",
                            "enum": ["pagerank", "betweenness", "closeness", "eigenvector", "degree"]
                        },
                        "default": ["pagerank", "betweenness", "closeness"]
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top entities to return per measure (default: 20)",
                        "minimum": 5,
                        "maximum": 100,
                        "default": 20
                    },
                    "include_scores": {
                        "type": "boolean",
                        "description": "Include numerical centrality scores (default: true)",
                        "default": true
                    },
                    "entity_filter": {
                        "type": "string",
                        "description": "Filter to specific entity pattern (optional)",
                        "maxLength": 100
                    }
                },
                "required": []
            }),
            examples: vec![
                LLMExample {
                    description: "Find most influential scientists using PageRank".to_string(),
                    input: json!({
                        "centrality_types": ["pagerank", "betweenness"],
                        "top_n": 10,
                        "entity_filter": "scientist"
                    }),
                    expected_output: "Graph Centrality Analysis:\n📊 Analyzed 2 centrality measures\n🎯 Top 10 entities per measure\n📈 Total entities: 15,847\n🏆 Most influential entity: Einstein\n💡 Key insights: 5".to_string(),
                }
            ],
            tips: vec![
                "Use PageRank to find globally important entities".to_string(),
                "Use Betweenness to identify bridge entities".to_string(),
                "Combine multiple centrality measures for comprehensive analysis".to_string(),
            ],
        },

        // ========= TIER 2 ADVANCED TOOLS =========

        LLMMCPTool {
            name: "hierarchical_clustering".to_string(),
            description: "Advanced community detection and clustering using Leiden algorithm, Louvain optimization, and hierarchical methods to discover hidden structure.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Clustering algorithm to use",
                        "enum": ["leiden", "louvain", "hierarchical"],
                        "default": "leiden"
                    },
                    "resolution": {
                        "type": "number",
                        "description": "Resolution parameter for community detection (default: 1.0)",
                        "minimum": 0.1,
                        "maximum": 5.0,
                        "default": 1.0
                    },
                    "min_cluster_size": {
                        "type": "integer",
                        "description": "Minimum cluster size (default: 3)",
                        "minimum": 2,
                        "maximum": 100,
                        "default": 3
                    },
                    "max_clusters": {
                        "type": "integer",
                        "description": "Maximum number of clusters (default: 50)",
                        "minimum": 2,
                        "maximum": 200,
                        "default": 50
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include detailed cluster metadata (default: true)",
                        "default": true
                    }
                },
                "required": []
            }),
            examples: vec![
                LLMExample {
                    description: "Find communities of related scientists".to_string(),
                    input: json!({
                        "algorithm": "leiden",
                        "resolution": 1.2,
                        "min_cluster_size": 5
                    }),
                    expected_output: "Hierarchical Clustering Results:\n🧮 Algorithm: leiden (Community Detection)\n🎯 Found 12 clusters from 486 entities\n📊 Modularity: 0.847\n⏱️ Execution Time: 127ms\n🏆 Quality Score: 0.823".to_string(),
                }
            ],
            tips: vec![
                "Use Leiden algorithm for large-scale community detection".to_string(),
                "Adjust resolution parameter to control cluster granularity".to_string(),
                "Combine with centrality analysis for comprehensive insights".to_string(),
            ],
        },

        LLMMCPTool {
            name: "predict_graph_structure".to_string(),
            description: "Heuristic prediction of missing links, future connections, and knowledge gaps using graph-based algorithms.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prediction_type": {
                        "type": "string",
                        "description": "Type of prediction to make",
                        "enum": ["missing_links", "future_connections", "community_evolution", "knowledge_gaps"],
                        "default": "missing_links"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for predictions (default: 0.7)",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.7
                    },
                    "max_predictions": {
                        "type": "integer",
                        "description": "Maximum number of predictions (default: 20)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20
                    },
                    "entity_filter": {
                        "type": "string",
                        "description": "Focus predictions on specific entity pattern",
                        "maxLength": 100
                    }
                },
                "required": []
            }),
            examples: vec![
                LLMExample {
                    description: "Predict missing connections between scientists".to_string(),
                    input: json!({
                        "prediction_type": "missing_links",
                        "entity_filter": "scientist",
                        "confidence_threshold": 0.8
                    }),
                    expected_output: "Heuristic Structure Prediction:\n🧠 Prediction Type: missing_links\n🎯 Generated 15 predictions\n📊 Avg Confidence: 0.847\n⚡ Processing Time: 234ms\n✅ Validation Score: 0.912".to_string(),
                }
            ],
            tips: vec![
                "Use 'missing_links' for discovering hidden connections".to_string(),
                "Enable advanced features for more accurate predictions".to_string(),
                "Validate predictions before incorporating into knowledge base".to_string(),
            ],
        },

        LLMMCPTool {
            name: "cognitive_reasoning_chains".to_string(),
            description: "Advanced logical reasoning engine supporting deductive, inductive, abductive, and analogical reasoning with chain generation and validation.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "reasoning_type": {
                        "type": "string",
                        "description": "Type of reasoning to perform",
                        "enum": ["deductive", "inductive", "abductive", "analogical"],
                        "default": "deductive"
                    },
                    "premise": {
                        "type": "string",
                        "description": "Starting premise for reasoning",
                        "maxLength": 500
                    },
                    "max_chain_length": {
                        "type": "integer",
                        "description": "Maximum reasoning chain length (default: 5)",
                        "minimum": 2,
                        "maximum": 10,
                        "default": 5
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for reasoning steps (default: 0.6)",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.6
                    },
                    "include_alternatives": {
                        "type": "boolean",
                        "description": "Generate alternative reasoning paths (default: true)",
                        "default": true
                    }
                },
                "required": ["premise"]
            }),
            examples: vec![
                LLMExample {
                    description: "Deduce conclusions about Einstein's work".to_string(),
                    input: json!({
                        "reasoning_type": "deductive",
                        "premise": "Einstein developed special relativity",
                        "max_chain_length": 4
                    }),
                    expected_output: "Cognitive Reasoning Analysis:\n🧠 Reasoning Type: deductive\n📝 Generated 3 reasoning chains\n🎯 Primary Conclusion: Einstein's work revolutionized physics\n📊 Avg Confidence: 0.876\n⏱️ Processing Time: 156ms".to_string(),
                }
            ],
            tips: vec![
                "Use deductive reasoning for logical conclusions".to_string(),
                "Try inductive reasoning for pattern discovery".to_string(),
                "Enable alternatives for comprehensive analysis".to_string(),
            ],
        },

        LLMMCPTool {
            name: "approximate_similarity_search".to_string(),
            description: "Locality Sensitive Hashing (LSH) for ultra-fast approximate similarity search with configurable precision/recall tradeoffs.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Text to search for",
                        "maxLength": 1000
                    },
                    "query_vector": {
                        "type": "array",
                        "description": "Pre-computed query vector",
                        "items": {"type": "number"}
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Similarity threshold (default: 0.7)",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.7
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20
                    },
                    "hash_functions": {
                        "type": "integer",
                        "description": "Number of LSH hash functions (default: 64)",
                        "minimum": 8,
                        "maximum": 128,
                        "default": 64
                    },
                    "hash_tables": {
                        "type": "integer",
                        "description": "Number of hash tables (default: 8)",
                        "minimum": 2,
                        "maximum": 32,
                        "default": 8
                    }
                },
                "anyOf": [
                    {"required": ["query_text"]},
                    {"required": ["query_vector"]}
                ]
            }),
            examples: vec![
                LLMExample {
                    description: "Fast approximate search for physics concepts".to_string(),
                    input: json!({
                        "query_text": "quantum entanglement",
                        "hash_functions": 32,
                        "hash_tables": 16
                    }),
                    expected_output: "LSH Approximate Similarity Search:\n⚡ Found 12 matches in 3ms\n🎯 Similarity Threshold: 0.70\n📊 Speedup: 47x vs brute force\n🎪 Recall Estimate: 0.891\n🏹 Precision Estimate: 0.923".to_string(),
                }
            ],
            tips: vec![
                "Increase hash_functions for better precision".to_string(),
                "Increase hash_tables for better recall".to_string(),
                "Lower threshold for more diverse results".to_string(),
            ],
        },

        LLMMCPTool {
            name: "knowledge_quality_metrics".to_string(),
            description: "Comprehensive quality assessment of knowledge including entity completeness, relationship consistency, content quality, and improvement recommendations.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "assessment_scope": {
                        "type": "string",
                        "description": "Scope of quality assessment",
                        "enum": ["comprehensive", "entities", "relationships", "content"],
                        "default": "comprehensive"
                    },
                    "include_entity_analysis": {
                        "type": "boolean",
                        "description": "Include detailed entity analysis (default: true)",
                        "default": true
                    },
                    "include_relationship_analysis": {
                        "type": "boolean",
                        "description": "Include relationship analysis (default: true)",
                        "default": true
                    },
                    "include_content_analysis": {
                        "type": "boolean",
                        "description": "Include content quality analysis (default: true)",
                        "default": true
                    },
                    "quality_threshold": {
                        "type": "number",
                        "description": "Quality threshold for issue detection (default: 0.6)",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.6
                    }
                },
                "required": []
            }),
            examples: vec![
                LLMExample {
                    description: "Comprehensive quality assessment".to_string(),
                    input: json!({
                        "assessment_scope": "comprehensive",
                        "quality_threshold": 0.7
                    }),
                    expected_output: "Knowledge Quality Assessment:\n📊 Overall Quality Score: 0.847/1.0\n🎯 Assessment Scope: comprehensive\n⚠️ Quality Issues Found: 7\n📈 Improvement Opportunities: 12\n⏱️ Analysis Time: 89ms".to_string(),
                }
            ],
            tips: vec![
                "Focus on high-impact quality improvements first".to_string(),
                "Use entity analysis to identify knowledge gaps".to_string(),
                "Monitor quality trends over time".to_string(),
            ],
        },
    ]
}

/// Get tool by name
pub fn get_tool_by_name(name: &str) -> Option<LLMMCPTool> {
    get_tools().into_iter().find(|tool| tool.name == name)
}