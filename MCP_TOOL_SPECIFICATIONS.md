# MCP Tool Specifications - Ultimate RAG System

## Executive Summary

This document specifies Model Context Protocol (MCP) tools for the Ultimate RAG System - a high-accuracy codebase search system achieving 95-97% accuracy through multi-method search ensemble, specialized embeddings, and temporal analysis. The MCP server provides LLMs with advanced search capabilities via standardized JSON-RPC 2.0 interfaces.

## System Overview

The Ultimate RAG System implements a four-layer intelligence stack:

1. **Multi-Method Search Foundation**: Exact match (ripgrep), token search (Tantivy), fuzzy search, AST search (tree-sitter), statistical (BM25/TF-IDF)
2. **Multi-Embedding Semantic Search**: Specialized embeddings per content type (code, docs, comments, identifiers, SQL, config, errors) 
3. **Temporal Analysis**: Git history integration for regression detection and change correlation
4. **Intelligent Synthesis**: Weighted voting and confidence scoring across all methods

### Tiered Execution Strategy

- **Tier 1**: Fast local search (<50ms, 85-90% accuracy, <$0.001/query)
- **Tier 2**: Balanced hybrid search (<500ms, 92-95% accuracy, $0.01/query)  
- **Tier 3**: Deep analysis (<2s, 95-97% accuracy, $0.05/query)

## MCP Tool Definitions

### Tool 1: index_codebase

**Purpose**: Index a codebase using all search methods with progress reporting

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to codebase root directory"
    },
    "include_patterns": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Glob patterns for files to include (e.g., ['*.rs', '*.py', '*.js'])",
      "default": ["*"]
    },
    "exclude_patterns": {
      "type": "array", 
      "items": {"type": "string"},
      "description": "Glob patterns for files to exclude (e.g., ['target/*', 'node_modules/*'])",
      "default": [".git/*", "target/*", "node_modules/*"]
    },
    "enable_git_tracking": {
      "type": "boolean",
      "description": "Enable git history analysis for temporal features",
      "default": true
    },
    "embedding_config": {
      "type": "object",
      "properties": {
        "enable_specialized_embeddings": {"type": "boolean", "default": true},
        "local_model_only": {"type": "boolean", "default": false},
        "cache_embeddings": {"type": "boolean", "default": true}
      }
    }
  },
  "required": ["path"]
}
```

**Response**:
```json
{
  "type": "object",
  "properties": {
    "index_id": {"type": "string", "description": "Unique identifier for this index"},
    "status": {"type": "string", "enum": ["indexing", "completed", "failed"]},
    "progress": {
      "type": "object", 
      "properties": {
        "files_processed": {"type": "integer"},
        "total_files": {"type": "integer"},
        "current_phase": {"type": "string"},
        "estimated_completion": {"type": "string", "format": "date-time"}
      }
    },
    "index_stats": {
      "type": "object",
      "properties": {
        "total_documents": {"type": "integer"},
        "ripgrep_indexed": {"type": "integer"},
        "tantivy_documents": {"type": "integer"},
        "ast_parsed_files": {"type": "integer"},
        "embeddings_generated": {"type": "integer"},
        "git_commits_analyzed": {"type": "integer"}
      }
    },
    "performance_metrics": {
      "type": "object",
      "properties": {
        "indexing_time_ms": {"type": "integer"},
        "memory_usage_mb": {"type": "integer"},
        "disk_usage_mb": {"type": "integer"}
      }
    }
  }
}
```

### Tool 2: search

**Purpose**: Multi-method search with tiered execution and confidence scoring

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query text"
    },
    "tier": {
      "type": "integer",
      "enum": [1, 2, 3],
      "description": "Search tier (1=fast, 2=balanced, 3=deep analysis)",
      "default": 2
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "description": "Maximum number of results to return",
      "default": 10
    },
    "file_patterns": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Limit search to files matching these patterns"
    },
    "content_types": {
      "type": "array", 
      "items": {"type": "string", "enum": ["code", "docs", "comments", "config", "sql", "errors"]},
      "description": "Limit search to specific content types"
    },
    "include_explanations": {
      "type": "boolean",
      "description": "Include reasoning for why matches were found",
      "default": false
    },
    "similarity_threshold": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Minimum similarity score for semantic matches",
      "default": 0.7
    }
  },
  "required": ["query"]
}
```

**Response**:
```json
{
  "type": "object",
  "properties": {
    "matches": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "line_number": {"type": "integer"},
          "content": {"type": "string"},
          "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
          "match_type": {"type": "string", "enum": ["exact", "fuzzy", "semantic", "structural", "statistical"]},
          "content_type": {"type": "string"},
          "context_before": {"type": "array", "items": {"type": "string"}},
          "context_after": {"type": "array", "items": {"type": "string"}},
          "highlighting": {
            "type": "array",
            "items": {
              "type": "object", 
              "properties": {
                "start": {"type": "integer"},
                "end": {"type": "integer"},
                "type": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "search_metadata": {
      "type": "object",
      "properties": {
        "tier_used": {"type": "integer"},
        "total_matches": {"type": "integer"},
        "search_time_ms": {"type": "integer"},
        "methods_used": {"type": "array", "items": {"type": "string"}},
        "cache_hit": {"type": "boolean"},
        "cost_estimate": {"type": "number"}
      }
    },
    "method_breakdown": {
      "type": "object",
      "properties": {
        "ripgrep_matches": {"type": "integer"},
        "tantivy_matches": {"type": "integer"},
        "fuzzy_matches": {"type": "integer"},
        "ast_matches": {"type": "integer"},
        "semantic_matches": {"type": "integer"}
      }
    }
  }
}
```

### Tool 3: semantic_search

**Purpose**: Specialized semantic search using content-aware embeddings

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string", 
      "description": "Semantic search query"
    },
    "similarity_threshold": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Minimum cosine similarity threshold",
      "default": 0.7
    },
    "content_type": {
      "type": "string",
      "enum": ["auto", "code", "documentation", "comments", "identifiers", "sql", "config", "errors"],
      "description": "Force specific embedding model or auto-detect",
      "default": "auto"
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 50,
      "description": "Maximum results to return",
      "default": 10
    },
    "include_embeddings": {
      "type": "boolean",
      "description": "Include embedding vectors in response",
      "default": false
    },
    "fusion_method": {
      "type": "string",
      "enum": ["single", "multi_embed", "rrf"],
      "description": "Result fusion strategy",
      "default": "multi_embed"
    }
  },
  "required": ["query"]
}
```

**Response**:
```json
{
  "type": "object",
  "properties": {
    "matches": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "content": {"type": "string"},
          "similarity_score": {"type": "number"},
          "content_type_detected": {"type": "string"},
          "embedding_model_used": {"type": "string"},
          "chunk_index": {"type": "integer"},
          "embedding_vector": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Only included if include_embeddings=true"
          }
        }
      }
    },
    "embedding_stats": {
      "type": "object",
      "properties": {
        "models_used": {"type": "array", "items": {"type": "string"}},
        "total_comparisons": {"type": "integer"},
        "search_time_ms": {"type": "integer"},
        "cache_hits": {"type": "integer"},
        "api_calls_made": {"type": "integer"}
      }
    }
  }
}
```

### Tool 4: track_repository

**Purpose**: Setup git repository tracking with automatic reindexing

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "repo_path": {
      "type": "string",
      "description": "Path to git repository root"
    },
    "auto_update": {
      "type": "boolean", 
      "description": "Enable automatic reindexing on git changes",
      "default": true
    },
    "hook_types": {
      "type": "array",
      "items": {"type": "string", "enum": ["post-commit", "post-merge", "post-checkout"]},
      "description": "Git hooks to install for automatic updates",
      "default": ["post-commit", "post-merge"]
    },
    "incremental_indexing": {
      "type": "boolean",
      "description": "Use incremental indexing for changed files only",
      "default": true
    },
    "temporal_analysis": {
      "type": "object",
      "properties": {
        "enable_blame_tracking": {"type": "boolean", "default": true},
        "enable_change_correlation": {"type": "boolean", "default": true},
        "max_history_depth": {"type": "integer", "default": 1000}
      }
    }
  },
  "required": ["repo_path"]
}
```

**Response**:
```json
{
  "type": "object", 
  "properties": {
    "tracking_id": {"type": "string"},
    "status": {"type": "string", "enum": ["active", "inactive", "error"]},
    "hooks_installed": {"type": "array", "items": {"type": "string"}},
    "initial_analysis": {
      "type": "object",
      "properties": {
        "total_commits": {"type": "integer"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "file_change_frequency": {"type": "object"},
        "hot_spots": {"type": "array", "items": {"type": "string"}}
      }
    },
    "monitoring_config": {
      "type": "object",
      "properties": {
        "watch_interval_ms": {"type": "integer"},
        "max_queue_size": {"type": "integer"},
        "retry_failed_updates": {"type": "boolean"}
      }
    }
  }
}
```

### Tool 5: get_context

**Purpose**: Retrieve surrounding code context with AST awareness

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Path to the file"
    },
    "line_number": {
      "type": "integer",
      "minimum": 1,
      "description": "Line number to get context for"
    },
    "context_size": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "description": "Number of lines before/after to include",
      "default": 5
    },
    "ast_aware": {
      "type": "boolean",
      "description": "Use AST to determine semantic boundaries",
      "default": true
    },
    "include_definitions": {
      "type": "boolean",
      "description": "Include related function/class definitions",
      "default": false
    },
    "include_imports": {
      "type": "boolean", 
      "description": "Include relevant import statements",
      "default": false
    }
  },
  "required": ["file_path", "line_number"]
}
```

**Response**:
```json
{
  "type": "object",
  "properties": {
    "context": {
      "type": "object",
      "properties": {
        "before": {"type": "array", "items": {"type": "string"}},
        "target_line": {"type": "string"},
        "after": {"type": "array", "items": {"type": "string"}},
        "line_numbers": {
          "type": "object",
          "properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "target": {"type": "integer"}
          }
        }
      }
    },
    "ast_info": {
      "type": "object",
      "properties": {
        "language": {"type": "string"},
        "current_function": {"type": "string"},
        "current_class": {"type": "string"},
        "current_scope": {"type": "string"},
        "node_type": {"type": "string"}
      }
    },
    "related_definitions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string", "enum": ["function", "class", "variable", "import"]},
          "file_path": {"type": "string"},
          "line_number": {"type": "integer"},
          "signature": {"type": "string"}
        }
      }
    },
    "semantic_boundaries": {
      "type": "object",
      "properties": {
        "function_start": {"type": "integer"},
        "function_end": {"type": "integer"},
        "class_start": {"type": "integer"},
        "class_end": {"type": "integer"}
      }
    }
  }
}
```

### Tool 6: explain_matches

**Purpose**: Provide detailed explanation of why matches were found

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "search_results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "line_number": {"type": "integer"},
          "content": {"type": "string"},
          "match_type": {"type": "string"},
          "confidence_score": {"type": "number"}
        }
      },
      "description": "Search results to explain"
    },
    "original_query": {
      "type": "string",
      "description": "Original search query for context"
    },
    "explanation_level": {
      "type": "string",
      "enum": ["basic", "detailed", "technical"],
      "description": "Level of detail in explanations",
      "default": "detailed"
    },
    "include_alternatives": {
      "type": "boolean",
      "description": "Suggest alternative search strategies",
      "default": true
    }
  },
  "required": ["search_results", "original_query"]
}
```

**Response**:
```json
{
  "type": "object",
  "properties": {
    "explanations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "result_index": {"type": "integer"},
          "match_reasoning": {"type": "string"},
          "contributing_factors": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "factor": {"type": "string"},
                "weight": {"type": "number"},
                "description": {"type": "string"}
              }
            }
          },
          "method_contributions": {
            "type": "object",
            "properties": {
              "exact_match": {"type": "number"},
              "semantic_similarity": {"type": "number"},  
              "structural_match": {"type": "number"},
              "temporal_relevance": {"type": "number"}
            }
          }
        }
      }
    },
    "query_analysis": {
      "type": "object",
      "properties": {
        "query_intent": {"type": "string"},
        "detected_patterns": {"type": "array", "items": {"type": "string"}},
        "search_strategy_used": {"type": "string"},
        "effectiveness_score": {"type": "number"}
      }
    },
    "suggestions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {"type": "string", "enum": ["refinement", "alternative", "expansion"]},
          "suggestion": {"type": "string"},
          "expected_improvement": {"type": "string"}
        }
      }
    }
  }
}
```

## JSON-RPC 2.0 Message Examples

### index_codebase Request/Response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "index_codebase",
    "arguments": {
      "path": "/home/user/my-project",
      "include_patterns": ["*.rs", "*.toml", "*.md"],
      "exclude_patterns": ["target/*", ".git/*"],
      "enable_git_tracking": true,
      "embedding_config": {
        "enable_specialized_embeddings": true,
        "local_model_only": false,
        "cache_embeddings": true
      }
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0", 
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Indexing started for codebase at /home/user/my-project"
      }
    ],
    "isError": false,
    "_meta": {
      "index_id": "idx_789abc123def",
      "status": "indexing",
      "progress": {
        "files_processed": 45,
        "total_files": 342,
        "current_phase": "AST parsing",
        "estimated_completion": "2024-01-20T15:30:00Z"
      },
      "index_stats": {
        "total_documents": 342,
        "ripgrep_indexed": 342,
        "tantivy_documents": 295,
        "ast_parsed_files": 45,
        "embeddings_generated": 127,
        "git_commits_analyzed": 89
      },
      "performance_metrics": {
        "indexing_time_ms": 12500,
        "memory_usage_mb": 245,
        "disk_usage_mb": 89
      }
    }
  }
}
```

### search Request/Response  

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call", 
  "params": {
    "name": "search",
    "arguments": {
      "query": "async function database connection pool",
      "tier": 2,
      "limit": 5,
      "include_explanations": true,
      "similarity_threshold": 0.8
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text", 
        "text": "Found 5 high-confidence matches using Tier 2 search (92-95% accuracy)"
      }
    ],
    "isError": false,
    "_meta": {
      "matches": [
        {
          "file_path": "src/database/pool.rs",
          "line_number": 23,
          "content": "pub async fn create_connection_pool(config: &PoolConfig) -> Result<Pool<Postgres>, PoolError> {",
          "confidence_score": 0.94,
          "match_type": "semantic",
          "content_type": "code",
          "context_before": ["use sqlx::{Pool, Postgres};", "use std::time::Duration;"],
          "context_after": ["    let pool = Pool::builder()", "        .max_connections(config.max_size)"],
          "highlighting": [
            {"start": 10, "end": 15, "type": "keyword"},
            {"start": 16, "end": 45, "type": "function_name"}
          ]
        }
      ],
      "search_metadata": {
        "tier_used": 2,
        "total_matches": 5,
        "search_time_ms": 342,
        "methods_used": ["tantivy", "semantic", "ast"],
        "cache_hit": false,
        "cost_estimate": 0.008
      },
      "method_breakdown": {
        "ripgrep_matches": 0,
        "tantivy_matches": 3,
        "fuzzy_matches": 1,
        "ast_matches": 2,
        "semantic_matches": 4
      }
    }
  }
}
```

### semantic_search Request/Response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "semantic_search", 
    "arguments": {
      "query": "error handling and logging utilities",
      "similarity_threshold": 0.75,
      "content_type": "auto",
      "limit": 8,
      "fusion_method": "multi_embed"
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 8 semantically similar matches using multi-embedding fusion"
      }
    ],
    "isError": false,
    "_meta": {
      "matches": [
        {
          "file_path": "src/utils/error.rs",
          "content": "/// Comprehensive error handling utilities with structured logging support",
          "similarity_score": 0.89,
          "content_type_detected": "code",
          "embedding_model_used": "VoyageCode2",
          "chunk_index": 0
        },
        {
          "file_path": "docs/logging.md",
          "content": "# Logging Configuration\n\nThis document describes error logging patterns and utilities...",
          "similarity_score": 0.82,
          "content_type_detected": "documentation", 
          "embedding_model_used": "E5Mistral7B",
          "chunk_index": 2
        }
      ],
      "embedding_stats": {
        "models_used": ["VoyageCode2", "E5Mistral7B", "CodeBERT"],
        "total_comparisons": 2847,
        "search_time_ms": 156,
        "cache_hits": 12,
        "api_calls_made": 3
      }
    }
  }
}
```

### track_repository Request/Response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "track_repository",
    "arguments": {
      "repo_path": "/home/user/my-project",
      "auto_update": true,
      "hook_types": ["post-commit", "post-merge"],
      "incremental_indexing": true,
      "temporal_analysis": {
        "enable_blame_tracking": true,
        "enable_change_correlation": true,
        "max_history_depth": 500
      }
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Repository tracking enabled with automatic reindexing on commits and merges"
      }
    ],
    "isError": false,
    "_meta": {
      "tracking_id": "track_abc123def456",
      "status": "active",
      "hooks_installed": ["post-commit", "post-merge"],
      "initial_analysis": {
        "total_commits": 1247,
        "authors": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        "file_change_frequency": {
          "src/main.rs": 45,
          "src/lib.rs": 23,
          "Cargo.toml": 12
        },
        "hot_spots": ["src/database/", "src/api/", "tests/integration/"]
      },
      "monitoring_config": {
        "watch_interval_ms": 1000,
        "max_queue_size": 100,
        "retry_failed_updates": true
      }
    }
  }
}
```

### get_context Request/Response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "get_context",
    "arguments": {
      "file_path": "src/database/pool.rs",
      "line_number": 23,
      "context_size": 8,
      "ast_aware": true,
      "include_definitions": true,
      "include_imports": true
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Retrieved AST-aware context for src/database/pool.rs:23"
      }
    ],
    "isError": false,
    "_meta": {
      "context": {
        "before": [
          "use sqlx::{Pool, Postgres};", 
          "use std::time::Duration;",
          "",
          "/// Configuration for database connection pool",
          "pub struct PoolConfig {",
          "    pub max_size: u32,",
          "    pub timeout: Duration,",
          "}"
        ],
        "target_line": "pub async fn create_connection_pool(config: &PoolConfig) -> Result<Pool<Postgres>, PoolError> {",
        "after": [
          "    let pool = Pool::builder()",
          "        .max_connections(config.max_size)",
          "        .connect_timeout(config.timeout)",
          "        .build(&database_url)",
          "        .await?;",
          "",
          "    Ok(pool)",
          "}"
        ],
        "line_numbers": {
          "start": 15,
          "end": 31,
          "target": 23
        }
      },
      "ast_info": {
        "language": "rust",
        "current_function": "create_connection_pool",
        "current_class": null,
        "current_scope": "module::pool",
        "node_type": "function_item"
      },
      "related_definitions": [
        {
          "name": "PoolConfig",
          "type": "class",
          "file_path": "src/database/pool.rs", 
          "line_number": 19,
          "signature": "pub struct PoolConfig"
        },
        {
          "name": "PoolError",
          "type": "class",
          "file_path": "src/database/error.rs",
          "line_number": 12,
          "signature": "pub enum PoolError"
        }
      ],
      "semantic_boundaries": {
        "function_start": 23,
        "function_end": 31,
        "class_start": null,
        "class_end": null
      }
    }
  }
}
```

### explain_matches Request/Response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "explain_matches",
    "arguments": {
      "search_results": [
        {
          "file_path": "src/database/pool.rs",
          "line_number": 23,
          "content": "pub async fn create_connection_pool(config: &PoolConfig) -> Result<Pool<Postgres>, PoolError> {",
          "match_type": "semantic",
          "confidence_score": 0.94
        }
      ],
      "original_query": "async function database connection pool",
      "explanation_level": "detailed",
      "include_alternatives": true
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Detailed explanation of why matches were found for query: 'async function database connection pool'"
      }
    ],
    "isError": false,
    "_meta": {
      "explanations": [
        {
          "result_index": 0,
          "match_reasoning": "High semantic similarity match found through VoyageCode2 embedding model. The function signature contains key terms 'async', 'connection_pool' and returns a database Pool type, directly matching the query intent.",
          "contributing_factors": [
            {
              "factor": "async_keyword_match",
              "weight": 0.25,
              "description": "Exact match on 'async' keyword in function signature"
            },
            {
              "factor": "semantic_similarity",
              "weight": 0.45,
              "description": "High embedding similarity (0.94) between query and function purpose"
            },
            {
              "factor": "return_type_relevance", 
              "weight": 0.20,
              "description": "Return type Pool<Postgres> strongly indicates database connection pooling"
            },
            {
              "factor": "function_name_match",
              "weight": 0.10,
              "description": "Function name 'create_connection_pool' directly matches query intent"
            }
          ],
          "method_contributions": {
            "exact_match": 0.25,
            "semantic_similarity": 0.94,
            "structural_match": 0.85,
            "temporal_relevance": 0.12
          }
        }
      ],
      "query_analysis": {
        "query_intent": "find_async_database_functions",
        "detected_patterns": ["async_function", "database_related", "connection_management"],
        "search_strategy_used": "multi_method_tier2",
        "effectiveness_score": 0.91
      },
      "suggestions": [
        {
          "type": "refinement",
          "suggestion": "Add 'PostgreSQL' or 'sqlx' to query for more specific database matches",
          "expected_improvement": "Higher precision for specific database technology"
        },
        {
          "type": "expansion", 
          "suggestion": "Try 'connection pool management' or 'database pool configuration' for related functionality",
          "expected_improvement": "Find related configuration and management code"
        },
        {
          "type": "alternative",
          "suggestion": "Use structural search: 'async fn *_pool(' to find all async pool functions",
          "expected_improvement": "More comprehensive coverage of async pool functions"
        }
      ]
    }
  }
}
```

## Error Handling

All tools follow standard MCP error response format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": {
      "error_type": "ValidationError",
      "details": "Parameter 'path' is required for index_codebase",
      "recovery_suggestions": [
        "Provide an absolute path to the codebase directory",
        "Ensure the path exists and is readable"
      ]
    }
  }
}
```

**Error Codes**:
- `-32600`: Invalid Request (malformed JSON-RPC)
- `-32601`: Method Not Found (unknown tool)
- `-32602`: Invalid Params (parameter validation failed)  
- `-32603`: Internal Error (system error during execution)
- `-32000` to `-32099`: Custom errors (indexing failed, search timeout, etc.)

## Implementation Notes

### Performance Considerations
- All search operations include timeout handling (30s default)
- Large result sets are paginated automatically
- Embedding generation is cached to avoid redundant API calls
- Indexing operations report progress via streaming responses

### Security
- File path validation prevents directory traversal attacks
- Query sanitization prevents injection attacks  
- Rate limiting on expensive operations (embedding generation)
- Authentication required for repository modification operations

### Scalability
- Concurrent request handling with proper resource management
- Memory-aware processing for large codebases
- Incremental indexing for efficiency
- Configurable resource limits per operation

This MCP server specification provides LLMs with comprehensive, high-accuracy codebase search capabilities through a standardized protocol interface, achieving the Ultimate RAG System's 95-97% accuracy targets while maintaining sub-second response times for most operations.