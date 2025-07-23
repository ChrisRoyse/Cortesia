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
            name: "get_suggestions".to_string(),
            description: "Get intelligent suggestions for what knowledge to add next, what questions to ask, or what connections to explore. Helps you build a comprehensive knowledge graph.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "suggestion_type": {
                        "type": "string",
                        "description": "Type of suggestions wanted",
                        "enum": ["missing_facts", "interesting_questions", "potential_connections", "knowledge_gaps"]
                    },
                    "focus_area": {
                        "type": "string",
                        "description": "Area to focus suggestions on (optional)",
                        "maxLength": 100
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of suggestions to return (default: 5)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["suggestion_type"]
            }),
            examples: vec![
                LLMExample {
                    description: "Get suggestions for missing facts about Einstein".to_string(),
                    input: json!({
                        "suggestion_type": "missing_facts",
                        "focus_area": "Einstein",
                        "limit": 3
                    }),
                    expected_output: "Suggested facts to add:\n1. Einstein's birth date (we have death date but not birth)\n2. Einstein's education (no university information found)\n3. Einstein's nationality (birthplace is Germany but citizenship unclear)".to_string(),
                }
            ],
            tips: vec![
                "Use 'knowledge_gaps' to find areas that need more information".to_string(),
                "Use 'potential_connections' to discover missing links between entities".to_string(),
                "Focus suggestions on specific topics with focus_area".to_string(),
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
            name: "generate_graph_query".to_string(),
            description: "Convert natural language to graph query languages (Cypher, SPARQL, Gremlin). Helps you learn graph query languages or integrate with other systems.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "natural_query": {
                        "type": "string",
                        "description": "Your query in natural language",
                        "maxLength": 500
                    },
                    "query_language": {
                        "type": "string",
                        "description": "Target query language",
                        "enum": ["cypher", "sparql", "gremlin"],
                        "default": "cypher"
                    },
                    "include_explanation": {
                        "type": "boolean",
                        "description": "Include explanation of the query (default: true)",
                        "default": true
                    }
                },
                "required": ["natural_query"]
            }),
            examples: vec![
                LLMExample {
                    description: "Generate Cypher query".to_string(),
                    input: json!({
                        "natural_query": "Find all scientists who won a Nobel Prize",
                        "query_language": "cypher"
                    }),
                    expected_output: "Cypher Query:\n```\nMATCH (s:Entity)-[:is]->(t:Entity {name: 'scientist'})\nMATCH (s)-[:won]->(n:Entity {name: 'Nobel_Prize'})\nRETURN s.name\n```\n\nExplanation: This query finds entities that have an 'is' relationship to 'scientist' AND a 'won' relationship to 'Nobel_Prize'".to_string(),
                }
            ],
            tips: vec![
                "Use this to learn graph query languages".to_string(),
                "Generated queries can be used in Neo4j, Blazegraph, etc.".to_string(),
                "Start with simple queries and build complexity".to_string(),
            ],
        },
        
        LLMMCPTool {
            name: "hybrid_search".to_string(),
            description: "Advanced search combining semantic similarity, graph structure, and text matching. Provides the most comprehensive search results.".to_string(),
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
                    }
                },
                "required": ["query"]
            }),
            examples: vec![
                LLMExample {
                    description: "Hybrid search for quantum physics".to_string(),
                    input: json!({
                        "query": "quantum physics discoveries",
                        "search_type": "hybrid",
                        "filters": {
                            "min_confidence": 0.8
                        }
                    }),
                    expected_output: "Found 8 results (hybrid search):\n\n1. **Quantum Mechanics** (score: 0.92)\n   - Semantic: High relevance to 'quantum physics'\n   - Structural: Central node with 15 connections\n   - Contains: 'revolutionary physics discovery 1920s'\n\n2. **Einstein Photoelectric Effect** (score: 0.87)\n   - Links quantum mechanics to Einstein\n   - Knowledge chunk with detailed explanation".to_string(),
                }
            ],
            tips: vec![
                "Hybrid search gives best results but is slower".to_string(),
                "Use filters to narrow down large result sets".to_string(),
                "Semantic search is best for conceptual queries".to_string(),
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
    ]
}

/// Get tool by name
pub fn get_tool_by_name(name: &str) -> Option<LLMMCPTool> {
    get_tools().into_iter().find(|tool| tool.name == name)
}