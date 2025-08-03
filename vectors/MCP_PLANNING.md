# MCP Server Implementation Plan for Universal RAG Indexing System

## Executive Summary

This document outlines the implementation plan for converting the Universal RAG Indexing System into a Model Context Protocol (MCP) server. The MCP server will expose two primary tools:
1. **Index Tool**: Initialize and maintain embeddings for any codebase with Git tracking
2. **Query Tool**: Search and retrieve relevant code/documentation from indexed content

## What is MCP (Model Context Protocol)?

MCP is an open standard protocol that enables LLM applications to connect with external data sources and tools. Key features:
- **Built on JSON-RPC 2.0**: Standardized message format
- **Stateful sessions**: Maintains context between client and server
- **Three primitives**: Tools (actions), Resources (data), and Prompts (templates)
- **Transport flexibility**: Supports stdio, SSE, and HTTP transports

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code (Client)                  │
├─────────────────────────────────────────────────────────┤
│                    MCP Protocol Layer                    │
│                     (JSON-RPC 2.0)                      │
├─────────────────────────────────────────────────────────┤
│              RAG MCP Server (mcp-rag-indexer)           │
├──────────────────────┬───────────────────────────────────┤
│     Index Tool       │        Query Tool                 │
├──────────────────────┴───────────────────────────────────┤
│                  Core RAG System                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • UniversalIndexer    • GitChangeTracker        │  │
│  │ • UniversalQuerier    • CacheManager            │  │
│  │ • UniversalCodeParser • ChromaDB Integration    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Implementation Components

### 1. MCP Server Structure

```python
mcp-rag-indexer/
├── server.py              # Main MCP server implementation
├── tools/
│   ├── index_tool.py     # Indexing tool implementation
│   └── query_tool.py     # Query tool implementation
├── core/                  # Existing RAG system (imported)
│   ├── indexer_universal.py
│   ├── query_universal.py
│   ├── git_tracker.py
│   └── cache_manager.py
├── watchers/
│   └── git_watcher.py    # Efficient Git monitoring
├── config/
│   └── settings.py       # Configuration management
├── requirements.txt      # Dependencies
├── README.md            # Documentation
└── install.py           # Installation script
```

### 2. Tool Specifications

#### Index Tool
```json
{
  "name": "index_codebase",
  "description": "Index a codebase for RAG retrieval with Git tracking",
  "inputSchema": {
    "type": "object",
    "properties": {
      "root_dir": {
        "type": "string",
        "description": "Absolute path to codebase root directory"
      },
      "watch": {
        "type": "boolean",
        "description": "Enable Git change monitoring",
        "default": true
      },
      "incremental": {
        "type": "boolean",
        "description": "Only index changed files",
        "default": true
      },
      "languages": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Languages to index (empty = all)",
        "default": []
      }
    },
    "required": ["root_dir"]
  }
}
```

#### Query Tool
```json
{
  "name": "query_codebase",
  "description": "Query indexed codebase with semantic search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "root_dir": {
        "type": "string",
        "description": "Codebase to search (must be indexed)"
      },
      "k": {
        "type": "integer",
        "description": "Number of results",
        "default": 5
      },
      "filter_type": {
        "type": "string",
        "enum": ["code", "docs", "config", "all"],
        "default": "all"
      },
      "rerank": {
        "type": "boolean",
        "description": "Enable multi-factor reranking",
        "default": true
      }
    },
    "required": ["query", "root_dir"]
  }
}
```

### 3. Resource Management

#### Database Management
- **Per-project databases**: Each indexed codebase gets its own ChromaDB instance
- **Location**: `~/.mcp-rag-indexer/databases/{project_hash}/`
- **Metadata tracking**: Store indexing timestamps, file counts, settings

#### Memory Management
- **Lazy loading**: Only load embeddings when queried
- **Memory limits**: Monitor and enforce < 2GB usage per session
- **Cleanup**: Automatic cleanup of inactive databases

#### Git Monitoring Strategy
- **Polling interval**: 30 seconds (configurable)
- **Event debouncing**: Batch changes within 5-second window
- **Resource usage**: < 0.1% CPU when idle
- **Smart detection**: Only check when files are modified

### 4. Caching Strategy

```python
class MCPCacheManager:
    """Enhanced cache for MCP server"""
    
    def __init__(self):
        self.query_cache = {}      # In-memory query cache
        self.index_cache = {}       # Track indexed projects
        self.ttl = 3600            # 1 hour default TTL
        
    def get_cached_query(self, project_id: str, query: str, params: dict):
        """Get cached query result with project isolation"""
        
    def cache_index_status(self, project_id: str, status: dict):
        """Cache indexing status for quick checks"""
```

### 5. Error Handling

```python
class MCPError(Exception):
    """Base MCP error with JSON-RPC error codes"""
    
    ERROR_CODES = {
        "PARSE_ERROR": -32700,
        "INVALID_REQUEST": -32600,
        "METHOD_NOT_FOUND": -32601,
        "INVALID_PARAMS": -32602,
        "INTERNAL_ERROR": -32603,
        # Custom errors
        "NOT_INDEXED": -32000,
        "INDEX_IN_PROGRESS": -32001,
        "QUERY_FAILED": -32002,
        "GIT_ERROR": -32003,
    }
```

## Implementation Details

### Phase 1: Core MCP Server (Day 1-2)
1. Set up MCP server skeleton using official Python SDK
2. Implement JSON-RPC message handling
3. Create stdio transport for Claude Code integration
4. Basic error handling and logging

### Phase 2: Tool Implementation (Day 2-3)
1. Implement index_codebase tool
   - Validate input parameters
   - Create/update project database
   - Handle incremental indexing
   - Return indexing statistics
   
2. Implement query_codebase tool
   - Validate project is indexed
   - Execute semantic search
   - Apply reranking
   - Format results for LLM consumption

### Phase 3: Git Monitoring (Day 3-4)
1. Implement efficient file watcher
   - Use watchdog library for cross-platform support
   - Debounce rapid changes
   - Queue incremental updates
   
2. Background worker thread
   - Non-blocking operation
   - Graceful shutdown
   - Error recovery

### Phase 4: Resource Optimization (Day 4-5)
1. Memory management
   - Implement LRU cache for embeddings
   - Monitor memory usage
   - Automatic cleanup
   
2. Performance optimization
   - Batch processing
   - Async operations where possible
   - Connection pooling for ChromaDB

### Phase 5: Testing & Documentation (Day 5-6)
1. Unit tests for all components
2. Integration tests with mock MCP client
3. Performance benchmarks
4. Installation documentation
5. Usage examples

## Windows-Specific Considerations

Based on the Neo4j MCP example:

1. **Executable Generation**
   - Create `.exe` wrapper using PyInstaller or similar
   - Place in Scripts directory for easy access
   
2. **Path Handling**
   - Use raw strings or Path objects
   - Handle spaces in paths
   - Proper JSON escaping in config

3. **Configuration Format**
   ```json
   {
     "mcpServers": {
       "rag-indexer": {
         "type": "stdio",
         "command": "C:\\\\Users\\\\{user}\\\\AppData\\\\Roaming\\\\Python\\\\Python313\\\\Scripts\\\\mcp-rag-indexer.exe",
         "args": ["--log-level", "info"]
       }
     }
   }
   ```

## Performance Targets

- **Indexing**: 50+ chunks/second
- **Query latency**: < 100ms (cached), < 500ms (uncached)
- **Memory usage**: < 2GB per session
- **Git monitoring overhead**: < 0.1% CPU when idle
- **Startup time**: < 2 seconds

## Security Considerations

1. **Path validation**: Prevent directory traversal attacks
2. **Resource limits**: Cap memory and CPU usage
3. **Input sanitization**: Validate all tool inputs
4. **Access control**: Only index allowed directories
5. **Secure storage**: Encrypt sensitive metadata

## Installation Process

### For End Users
```bash
# Install via pip
pip install mcp-rag-indexer

# Configure Claude Code
mcp-rag-indexer configure

# Test connection
mcp-rag-indexer test
```

### Configuration in Claude Code
```json
{
  "mcpServers": {
    "rag-indexer": {
      "type": "stdio",
      "command": "mcp-rag-indexer",
      "args": ["serve"]
    }
  }
}
```

## Usage Examples

### Indexing a Project
```
User: Index my project at C:/projects/myapp
Claude: I'll index your project for semantic search.
[Uses index_codebase tool]
Result: Indexed 1,247 files with 5,892 chunks in 23.4 seconds
```

### Querying Code
```
User: Find the authentication implementation
Claude: I'll search for authentication code in your project.
[Uses query_codebase tool]
Result: Found 5 relevant sections:
1. auth_service.py:23 - AuthenticationService class
2. middleware.py:45 - authenticate_request function
...
```

## Success Metrics

1. **Functionality**: All existing RAG features work through MCP
2. **Performance**: Meets or exceeds standalone performance
3. **Reliability**: 99.9% uptime with automatic recovery
4. **Usability**: Simple configuration and usage in Claude Code
5. **Resource efficiency**: Minimal overhead when idle

## Timeline

- **Week 1**: Core implementation and testing
- **Week 2**: Optimization and documentation
- **Week 3**: Beta testing and refinement
- **Week 4**: Release and support

## Next Steps

1. Create the MCP server implementation
2. Port existing RAG system as importable modules
3. Implement efficient Git monitoring
4. Create installation and configuration scripts
5. Test with Claude Code on Windows
6. Document and publish

This plan provides a comprehensive roadmap for converting the Universal RAG Indexing System into a production-ready MCP server that seamlessly integrates with Claude Code while maintaining all existing functionality and performance characteristics.