# MicroPhase 9: Documentation and Examples

**Duration**: 4-5 hours  
**Priority**: High - Essential for adoption and maintenance  
**Prerequisites**: All previous MicroPhases (1-8)

## Overview

Break down documentation into atomic micro-tasks that each AI can complete in 15-20 minutes. Each task produces a single, complete documentation file.

## AI-Actionable Micro-Tasks

### Micro-Task 9.1.1: Create API Overview Documentation
**Estimated Time**: 18 minutes  
**File**: `docs/api/README.md`
**Expected Deliverable**: Complete API reference overview

**Task Prompt for AI**:
Create a comprehensive API overview document covering authentication, tools overview, and quick start guide.

```markdown
# CortexKG MCP Server API Reference

The CortexKG MCP (Model Context Protocol) server provides neuromorphic memory capabilities through a standardized tool interface. This documentation covers all available tools, schemas, and usage patterns.

## Quick Start

```bash
# Start the server
./cortex_kg_mcp_server --config config/production.toml

# Health check
curl http://localhost:8080/health

# MCP tool execution (requires authentication)
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "store_memory",
    "params": {
      "content": "The capital of France is Paris",
      "tags": ["geography", "france"]
    },
    "id": "req_001"
  }'
```

## Authentication

All MCP tools require authentication via JWT tokens. Obtain tokens through the OAuth 2.1 flow:

1. **Authorization URL**: Get authorization URL
2. **Token Exchange**: Exchange authorization code for access token
3. **Tool Access**: Use access token in `Authorization: Bearer` header

### Permission Levels

- **Read**: `cortex_kg:read` - Access to `retrieve_memory`, `get_memory_stats`, `analyze_memory_graph`
- **Write**: `cortex_kg:write` - All read permissions plus `store_memory`, `update_memory`, `delete_memory`
- **Admin**: `cortex_kg:admin` - All permissions plus `configure_learning`

## Available Tools Overview

### Core Memory Operations

| Tool | Purpose | Required Permission |
|------|---------|-------------------|
| `store_memory` | Store new information | `cortex_kg:write` |
| `retrieve_memory` | Search and retrieve | `cortex_kg:read` |
| `update_memory` | Modify existing | `cortex_kg:write` |
| `delete_memory` | Remove memories | `cortex_kg:write` |

### Analysis & Management

| Tool | Purpose | Required Permission |
|------|---------|-------------------|
| `analyze_memory_graph` | Graph analysis | `cortex_kg:read` |
| `get_memory_stats` | System metrics | `cortex_kg:read` |
| `configure_learning` | Tune parameters | `cortex_kg:admin` |

### Retrieval Strategies

- **SemanticSimilarity**: Concept-based matching
- **StructuralMatching**: Graph topology analysis
- **TemporalProximity**: Time-based relevance
- **HybridApproach**: Multi-column consensus (default)
- **SpreadingActivation**: Neural propagation

## Rate Limits

| Tier | Requests/min | Burst Limit | Memory Ops/min |
|------|-------------|-------------|----------------|
| Standard | 60 | 10 | 30 |
| Premium | 120 | 20 | 100 |
| Enterprise | 300 | 50 | 500 |

## Error Codes

| Code | Error | Description |
|------|-------|-------------|
| 400 | ValidationError | Invalid input parameters |
| 401 | AuthenticationError | Missing or invalid token |
| 403 | AuthorizationError | Insufficient permissions |
| 404 | NotFoundError | Memory ID not found |
| 429 | RateLimitError | Rate limit exceeded |
| 500 | InternalError | Server processing error |

## Performance Characteristics

- **Response Time**: <100ms average, <200ms P95
- **Throughput**: >1000 operations/minute
- **Memory Usage**: <2GB for 1M memories
- **Availability**: 99.9% uptime target
- **Accuracy**: >95% allocation success rate
```

**Verification**: 
- Complete API overview with all tools listed
- Authentication clearly explained
- Rate limits and error codes documented

---

### Micro-Task 9.1.2: Create Store Memory Tool Documentation
**Estimated Time**: 16 minutes  
**File**: `docs/api/tools/store_memory.md`
**Expected Deliverable**: Complete store_memory tool specification

**Task Prompt for AI**:
Create detailed documentation for the store_memory tool with complete schemas, examples, and neuromorphic output explanation.

```markdown
# store_memory Tool

Store new information in the neuromorphic memory system with biologically-inspired allocation.

## Overview

The `store_memory` tool processes input through a 4-column cortical architecture to determine optimal memory allocation. Each memory storage operation involves neural pathway formation and synaptic weight updates.

## Input Schema

```json
{
  "content": "Information to store (required)",
  "context": "Optional contextual information",
  "source": "Optional source attribution", 
  "confidence": 0.95,
  "tags": ["tag1", "tag2"],
  "importance": 0.8
}
```

### Parameters

- **content** (string, required): The primary information to store
- **context** (string, optional): Additional contextual information
- **source** (string, optional): Attribution or source reference
- **confidence** (float, 0.0-1.0): Confidence in the information accuracy
- **tags** (array, optional): Classification tags for organization
- **importance** (float, 0.0-1.0): Relative importance for retention priority

## Output Schema

```json
{
  "memory_id": "mem_123abc",
  "allocation_path": ["semantic_column", "high_confidence_path"],
  "cortical_decision": {
    "primary_column": "Semantic",
    "decision_confidence": 0.92,
    "alternative_paths": [],
    "inhibition_strength": 0.8
  },
  "synaptic_changes": {
    "weights_updated": 45,
    "new_connections": 3,
    "strengthened_pathways": ["concept_similarity", "factual_knowledge"],
    "stdp_applications": 4
  },
  "storage_metadata": {
    "storage_location": "graph_node_456",
    "allocation_algorithm": "neuromorphic_consensus",
    "compression_ratio": 0.85,
    "predicted_retrieval_time_ms": 15.2
  }
}
```

### Response Fields

- **memory_id**: Unique identifier for the stored memory
- **allocation_path**: Neural pathway used for storage
- **cortical_decision**: Results from cortical column processing
- **synaptic_changes**: Neural network modifications made
- **storage_metadata**: Technical storage information

## Examples

### Basic Memory Storage

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "store_memory",
    "params": {
      "content": "Python is a programming language",
      "tags": ["programming", "python"],
      "confidence": 0.95
    },
    "id": "req_001"
  }'
```

### Complex Memory with Context

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "store_memory",
    "params": {
      "content": "The Eiffel Tower was completed in 1889",
      "context": "Built for the 1889 World Exposition in Paris",
      "source": "Historical records",
      "tags": ["architecture", "paris", "history"],
      "confidence": 0.98,
      "importance": 0.7
    },
    "id": "req_002"
  }'
```

## Neuromorphic Processing

Memory storage involves:

1. **TTFS Encoding**: Convert input to Time-To-First-Spike patterns
2. **Cortical Analysis**: 4 columns process the pattern simultaneously
3. **Lateral Inhibition**: Columns compete for allocation decision
4. **STDP Learning**: Synaptic weights updated based on success
5. **Graph Storage**: Memory allocated to optimal graph location

## Error Handling

- **400 ValidationError**: Invalid input parameters
- **401 AuthenticationError**: Missing or invalid token
- **403 AuthorizationError**: Insufficient write permissions
- **429 RateLimitError**: Rate limit exceeded
- **500 InternalError**: Storage system error
```

**Verification**: 
- Complete tool specification with schemas
- Clear examples and neuromorphic explanation
- Proper error handling documentation

---

### Micro-Task 9.1.3: Create Retrieve Memory Tool Documentation
**Estimated Time**: 19 minutes  
**File**: `docs/api/tools/retrieve_memory.md`
**Expected Deliverable**: Complete retrieve_memory tool specification

**Task Prompt for AI**:
Create detailed documentation for the retrieve_memory tool with all search strategies and neuromorphic response analysis.

```markdown
# retrieve_memory Tool

Search and retrieve memories using neuromorphic activation patterns and multiple search strategies.

## Overview

The `retrieve_memory` tool uses cortical column activation to find relevant memories through semantic similarity, structural matching, temporal proximity, or hybrid approaches.

## Input Schema

```json
{
  "query": "Search query (required)",
  "limit": 10,
  "threshold": 0.3,
  "include_reasoning": true,
  "tag_filter": ["geography"],
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-12-31T23:59:59Z"
  },
  "strategy": "HybridApproach"
}
```

### Parameters

- **query** (string, required): Natural language search query
- **limit** (integer, optional): Maximum number of results (default: 10)
- **threshold** (float, optional): Minimum similarity score (default: 0.3)
- **include_reasoning** (boolean, optional): Include neuromorphic analysis
- **tag_filter** (array, optional): Filter by specific tags
- **time_range** (object, optional): Filter by time period
- **strategy** (string, optional): Search algorithm to use

## Search Strategies

### SemanticSimilarity
- **Purpose**: Concept-based matching using embeddings
- **Best for**: Knowledge queries, concept exploration
- **Processing**: Semantic column activation patterns

### StructuralMatching
- **Purpose**: Graph topology and connection analysis
- **Best for**: Relationship discovery, pattern matching
- **Processing**: Structural column pathway analysis

### TemporalProximity
- **Purpose**: Time-based relationships and sequences
- **Best for**: Event sequences, time-sensitive information
- **Processing**: Temporal column activation timing

### HybridApproach (Default)
- **Purpose**: Multi-column consensus with learned weights
- **Best for**: General-purpose retrieval
- **Processing**: All columns contribute to final ranking

### SpreadingActivation
- **Purpose**: Neural activation propagation through graph
- **Best for**: Discovery of indirect relationships
- **Processing**: Activation spreads through neural pathways

## Output Schema

```json
{
  "memories": [
    {
      "memory_id": "mem_123abc",
      "content": "The capital of France is Paris",
      "similarity_score": 0.95,
      "context": "Geography information",
      "source": "Educational content",
      "tags": ["geography", "france"],
      "created_at": "2024-01-15T10:30:00Z",
      "retrieval_path": ["semantic_similarity", "high_match"],
      "confidence": 0.9
    }
  ],
  "retrieval_metadata": {
    "strategy_used": "HybridApproach",
    "search_depth": 3,
    "nodes_traversed": 156,
    "cortical_columns_activated": ["Semantic", "Structural"],
    "spreading_activation_hops": 0
  },
  "neural_activations": [
    {
      "column_type": "Semantic", 
      "activation_pattern": [0.9, 0.7, 0.5],
      "peak_activation": 0.9,
      "activation_duration_ms": 8.5
    }
  ],
  "similarity_scores": [0.95, 0.87, 0.76],
  "total_found": 3,
  "query_understanding": {
    "semantic_concepts": ["geography", "capital", "france"],
    "temporal_indicators": [],
    "structural_hints": ["location", "administrative"],
    "confidence_score": 0.88
  }
}
```

## Examples

### Basic Search

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "retrieve_memory",
    "params": {
      "query": "capital of France",
      "limit": 5
    },
    "id": "search_001"
  }'
```

### Advanced Search with Strategy

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "retrieve_memory",
    "params": {
      "query": "programming languages",
      "strategy": "SemanticSimilarity",
      "threshold": 0.7,
      "tag_filter": ["programming"],
      "include_reasoning": true
    },
    "id": "search_002"
  }'
```

## Performance Characteristics

- **Semantic Search**: 45-85ms average response time
- **Structural Search**: 30-60ms average response time
- **Temporal Search**: 25-50ms average response time
- **Hybrid Search**: 55-95ms average response time
- **Spreading Activation**: 100-200ms for complex graphs
```

**Verification**: 
- All search strategies documented
- Complete examples with different approaches
- Performance characteristics included

---

### Micro-Task 9.1.4: Create Tool Documentation for Update/Delete
**Estimated Time**: 15 minutes  
**File**: `docs/api/tools/memory_management.md`
**Expected Deliverable**: Complete update_memory and delete_memory specifications

**Task Prompt for AI**:
Create documentation for memory management tools: update_memory and delete_memory with synaptic plasticity details.

```markdown
# Memory Management Tools

Modify and remove existing memories with neuromorphic synaptic plasticity.

## update_memory Tool

### Overview

Update existing memories while maintaining neural pathway integrity through biologically-inspired synaptic plasticity.

### Input Schema

```json
{
  "memory_id": "mem_123abc",
  "new_content": "Updated information",
  "new_context": "Updated context",
  "new_confidence": 0.98,
  "add_tags": ["updated", "verified"],
  "remove_tags": ["temporary"],
  "new_importance": 0.9,
  "plasticity_mode": "Adaptive"
}
```

### Plasticity Modes

- **Conservative**: Minimal neural pathway changes, preserves existing connections
- **Adaptive**: Moderate synaptic updates with balanced reorganization (default)
- **Aggressive**: Significant neural pathway reorganization for major updates

### Example

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "update_memory",
    "params": {
      "memory_id": "mem_123abc",
      "new_content": "Python is a powerful programming language",
      "add_tags": ["powerful", "versatile"],
      "new_confidence": 0.98,
      "plasticity_mode": "Adaptive"
    },
    "id": "update_001"
  }'
```

## delete_memory Tool

### Overview

Remove memories from the system with configurable neural pathway cleanup to maintain graph integrity.

### Input Schema

```json
{
  "memory_id": "mem_123abc",
  "force_delete": false,
  "cleanup_mode": "Standard"
}
```

### Cleanup Modes

- **Minimal**: Mark as deleted, preserve all neural connections
- **Standard**: Remove direct connections, maintain structural integrity (default)
- **Aggressive**: Full pathway cleanup and graph reorganization

### Parameters

- **memory_id** (string, required): Unique identifier of memory to delete
- **force_delete** (boolean, optional): Override safety checks (default: false)
- **cleanup_mode** (string, optional): Neural pathway cleanup strategy

### Example

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "delete_memory",
    "params": {
      "memory_id": "mem_123abc",
      "cleanup_mode": "Standard",
      "force_delete": false
    },
    "id": "delete_001"
  }'
```

## Neural Safety Mechanisms

Both operations include safety mechanisms:

- **Consistency Checks**: Verify graph integrity before modifications
- **Rollback Capability**: Undo operations if they cause instability
- **Pathway Preservation**: Maintain critical neural pathways
- **Reference Counting**: Track memory usage before deletion

## Response Format

Both tools return operation metadata:

```json
{
  "success": true,
  "memory_id": "mem_123abc",
  "operation": "update",
  "synaptic_changes": {
    "pathways_modified": 12,
    "connections_added": 3,
    "connections_removed": 1
  },
  "graph_impact": {
    "nodes_affected": 8,
    "integrity_score": 0.94
  }
}
```
```

**Verification**: 
- Complete specifications for both management tools
- Safety mechanisms clearly explained
- Examples demonstrate proper usage

---

### Micro-Task 9.1.5: Create Analysis Tools Documentation
**Estimated Time**: 17 minutes  
**File**: `docs/api/tools/analysis_tools.md`
**Expected Deliverable**: Documentation for analyze_memory_graph and get_memory_stats

**Task Prompt for AI**:
Create comprehensive documentation for graph analysis and system statistics tools.

```markdown
# Analysis Tools

Advanced analysis capabilities for memory graph exploration and system performance monitoring.

## analyze_memory_graph Tool

### Overview

Perform sophisticated analysis of neural pathways, memory connections, and graph topology using neuromorphic principles.

### Input Schema

```json
{
  "analysis_type": "Connectivity",
  "memory_ids": ["mem_123", "mem_456"],
  "include_pathways": true,
  "max_depth": 5
}
```

### Analysis Types

#### Connectivity
- **Purpose**: Analyze connection patterns and network density
- **Output**: Node degrees, clustering coefficients, path lengths
- **Use case**: Understanding memory interconnectedness

#### Centrality
- **Purpose**: Identify important nodes and information hubs
- **Output**: Betweenness, closeness, and eigenvector centrality
- **Use case**: Finding key memories and knowledge bridges

#### Clustering
- **Purpose**: Discover memory groupings and communities
- **Output**: Cluster assignments, modularity scores
- **Use case**: Understanding knowledge organization

#### PathwayEfficiency
- **Purpose**: Analyze neural pathway optimization
- **Output**: Efficiency metrics, bottleneck identification
- **Use case**: Performance optimization insights

#### SynapticStrength
- **Purpose**: Evaluate connection strength distribution
- **Output**: Weight distributions, strength correlations
- **Use case**: Understanding learning patterns

#### MemoryIntegrity
- **Purpose**: Verify data consistency and graph health
- **Output**: Integrity scores, consistency checks
- **Use case**: System health monitoring

### Example

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "analyze_memory_graph",
    "params": {
      "analysis_type": "Connectivity",
      "include_pathways": true,
      "max_depth": 3
    },
    "id": "analysis_001"
  }'
```

## get_memory_stats Tool

### Overview

Retrieve comprehensive system performance metrics, health indicators, and operational statistics.

### Input Schema

```json
{
  "detailed_breakdown": true,
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "include_performance": true
}
```

### Parameters

- **detailed_breakdown** (boolean): Include per-component statistics
- **time_range** (object): Filter metrics by time period
- **include_performance** (boolean): Include performance metrics

### Response Schema

```json
{
  "system_stats": {
    "total_memories": 150000,
    "total_connections": 450000,
    "storage_size_mb": 2048,
    "average_retrieval_time_ms": 45.2
  },
  "cortical_stats": {
    "semantic_activations": 85000,
    "structural_activations": 45000,
    "temporal_activations": 32000,
    "exception_activations": 8000
  },
  "performance_metrics": {
    "requests_per_minute": 1200,
    "error_rate_percent": 0.3,
    "cache_hit_rate": 0.87,
    "memory_usage_mb": 1536
  },
  "health_indicators": {
    "overall_health_score": 0.92,
    "neural_integrity": 0.95,
    "graph_consistency": 0.98,
    "allocation_efficiency": 0.89
  }
}
```

### Example

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "get_memory_stats",
    "params": {
      "detailed_breakdown": true,
      "include_performance": true
    },
    "id": "stats_001"
  }'
```

## configure_learning Tool

### Overview

Configure neuromorphic learning parameters for system optimization (admin only).

### Input Schema

```json
{
  "learning_rate": 0.015,
  "decay_rate": 0.001,
  "activation_threshold": 0.75,
  "learning_mode": "STDP",
  "plasticity_window_ms": 50.0
}
```

### Learning Modes

- **Hebbian**: Basic Hebbian learning ("fire together, wire together")
- **AntiHebbian**: Inverse strengthening for competition
- **STDP**: Spike-timing dependent plasticity (default)
- **Homeostatic**: Balanced plasticity with stability
- **MetaLearning**: Adaptive learning parameter optimization

### Example

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "configure_learning",
    "params": {
      "learning_rate": 0.02,
      "learning_mode": "STDP",
      "activation_threshold": 0.7
    },
    "id": "config_001"
  }'
```
```

**Verification**: 
- All analysis types clearly documented
- System statistics comprehensively covered
- Learning configuration options explained

---

### Micro-Task 9.2.1: Create Basic Usage Examples
**Estimated Time**: 18 minutes  
**File**: `docs/examples/basic_usage.md`
**Expected Deliverable**: Simple integration examples and patterns

**Task Prompt for AI**:
Create practical usage examples covering basic memory operations, authentication, and error handling.

```markdown
# Basic Usage Examples

## Getting Started

### 1. Server Setup

```bash
# Clone and build
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server
cargo build --release

# Configure
cp config/development.toml config/local.toml
# Edit config/local.toml with your settings

# Generate JWT secret
openssl rand -hex 32 > config/jwt.key

# Start server
./target/release/cortex_kg_mcp_server --config config/local.toml
```

### 2. Authentication

For development, create a test token:

```python
import requests
import json

# For development, create a test token
# In production, use OAuth 2.1 flow
auth_response = requests.post('http://localhost:8080/auth/token', json={
    'username': 'test_user',
    'permissions': ['cortex_kg:read', 'cortex_kg:write']
})

token = auth_response.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}
```

## Basic Memory Operations

### Storing Information

```python
# Store a simple fact
response = requests.post('http://localhost:8080/mcp', 
    headers=headers,
    json={
        'method': 'store_memory',
        'params': {
            'content': 'The Pacific Ocean is the largest ocean on Earth',
            'tags': ['geography', 'ocean', 'facts'],
            'confidence': 0.95
        },
        'id': 'req_001'
    }
)

result = response.json()
memory_id = result['result']['memory_id']
print(f"Stored memory: {memory_id}")
```

### Retrieving Information

```python
# Search for related information
response = requests.post('http://localhost:8080/mcp',
    headers=headers, 
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'largest ocean',
            'limit': 5,
            'threshold': 0.3
        },
        'id': 'req_002'
    }
)

results = response.json()
for memory in results['result']['memories']:
    print(f"Found: {memory['content']} (score: {memory['similarity_score']:.2f})")
```

### Updating Memories

```python
# Update existing memory
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'update_memory', 
        'params': {
            'memory_id': memory_id,
            'new_content': 'The Pacific Ocean is the largest and deepest ocean on Earth',
            'add_tags': ['depth', 'size'],
            'new_confidence': 0.98
        },
        'id': 'req_003'
    }
)
```

## Building a Knowledge Base

```python
# Store interconnected facts
facts = [
    {
        'content': 'Paris is the capital of France',
        'tags': ['geography', 'france', 'capital', 'europe']
    },
    {
        'content': 'France is located in Western Europe', 
        'tags': ['geography', 'france', 'europe', 'location']
    },
    {
        'content': 'The Eiffel Tower is located in Paris',
        'tags': ['landmarks', 'paris', 'france', 'architecture']
    },
    {
        'content': 'French is the official language of France',
        'tags': ['language', 'france', 'french', 'official']
    }
]

memory_ids = []
for i, fact in enumerate(facts):
    response = requests.post('http://localhost:8080/mcp',
        headers=headers,
        json={
            'method': 'store_memory',
            'params': fact,
            'id': f'fact_{i}'
        }
    )
    memory_ids.append(response.json()['result']['memory_id'])

print(f"Stored {len(memory_ids)} interconnected facts")
```

## Error Handling

```python
def safe_mcp_request(method, params, request_id):
    try:
        response = requests.post('http://localhost:8080/mcp',
            headers=headers,
            json={
                'method': method,
                'params': params,
                'id': request_id
            },
            timeout=30
        )
        
        if response.status_code == 429:
            print("Rate limit exceeded, waiting...")
            time.sleep(60)
            return safe_mcp_request(method, params, request_id)
        
        response.raise_for_status()
        
        result = response.json()
        if 'error' in result:
            print(f"MCP Error: {result['error']['message']}")
            return None
            
        return result['result']
        
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Usage
result = safe_mcp_request('store_memory', {
    'content': 'Test content',
    'tags': ['test']
}, 'safe_req_001')

if result:
    print(f"Successfully stored: {result['memory_id']}")
```

## Quick Health Check

```bash
# Simple health check
curl http://localhost:8080/health

# Example response:
# {
#   "status": "healthy",
#   "health_score": 0.92,
#   "response_time_ms": 45.2,
#   "throughput_ops_per_min": 1250.0,
#   "error_rate_percent": 0.5,
#   "memory_usage_mb": 512.8,
#   "uptime_seconds": 86400
# }
```
```

**Verification**: 
- Complete setup instructions
- Basic operations with clear examples
- Error handling patterns
- Knowledge base building example

---

### Micro-Task 9.2.2: Create Search Strategy Examples
**Estimated Time**: 16 minutes  
**File**: `docs/examples/search_strategies.md`
**Expected Deliverable**: Examples of different retrieval strategies and use cases

**Task Prompt for AI**:
Create comprehensive examples demonstrating all search strategies with practical use cases and performance comparisons.

```markdown
# Search Strategy Examples

Demonstrating different retrieval approaches and their optimal use cases.

## Strategy Overview

The CortexKG MCP server supports five different search strategies, each optimized for specific types of queries and use cases.

## SemanticSimilarity Strategy

### Best For
- Concept exploration and knowledge discovery
- Finding related topics and ideas
- Educational content retrieval

### Example: Research Query

```python
# Find content related to machine learning
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'artificial intelligence and neural networks',
            'strategy': 'SemanticSimilarity',
            'limit': 10,
            'threshold': 0.6
        },
        'id': 'semantic_search'
    }
)

results = response.json()['result']
print(f"Found {len(results['memories'])} semantically related memories")
for memory in results['memories']:
    print(f"- {memory['content'][:60]}... (score: {memory['similarity_score']:.2f})")
```

### Performance Characteristics
- **Response Time**: 45-85ms
- **Accuracy**: High for conceptual queries
- **Best Threshold**: 0.5-0.8

## StructuralMatching Strategy

### Best For
- Relationship discovery
- Pattern matching in data
- Finding structural similarities

### Example: Relationship Exploration

```python
# Find memories with similar connection patterns
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'capital cities in Europe',
            'strategy': 'StructuralMatching',
            'limit': 15,
            'include_reasoning': True
        },
        'id': 'structural_search'
    }
)

results = response.json()['result']
print("Structural connections found:")
for memory in results['memories']:
    path = " -> ".join(memory['retrieval_path'])
    print(f"- {memory['content']} (path: {path})")
```

### Performance Characteristics
- **Response Time**: 30-60ms
- **Accuracy**: High for relationship queries
- **Best Threshold**: 0.3-0.6

## TemporalProximity Strategy

### Best For
- Event sequences and timelines
- Recent information retrieval
- Time-sensitive queries

### Example: Timeline Query

```python
# Find events related to a specific time period
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'technological developments',
            'strategy': 'TemporalProximity',
            'time_range': {
                'start': '2020-01-01T00:00:00Z',
                'end': '2024-01-01T00:00:00Z'
            },
            'limit': 20
        },
        'id': 'temporal_search'
    }
)

results = response.json()['result']
print("Recent technological developments:")
for memory in results['memories']:
    created = memory['created_at'][:10]  # Extract date
    print(f"- {created}: {memory['content'][:50]}...")
```

### Performance Characteristics
- **Response Time**: 25-50ms
- **Accuracy**: High for time-based queries
- **Best Threshold**: 0.2-0.5

## HybridApproach Strategy (Default)

### Best For
- General-purpose searches
- Complex queries with multiple aspects
- Balanced accuracy and performance

### Example: Complex Query

```python
# Multi-faceted search combining concepts, structure, and time
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'renewable energy policies in European countries',
            'strategy': 'HybridApproach',
            'limit': 12,
            'tag_filter': ['policy', 'energy', 'europe'],
            'include_reasoning': True
        },
        'id': 'hybrid_search'
    }
)

results = response.json()['result']
print(f"Hybrid search results (strategy: {results['retrieval_metadata']['strategy_used']}):")
print(f"Columns activated: {results['retrieval_metadata']['cortical_columns_activated']}")

for memory in results['memories']:
    score = memory['similarity_score']
    print(f"- {memory['content'][:70]}... (score: {score:.2f})")
```

### Performance Characteristics
- **Response Time**: 55-95ms
- **Accuracy**: Balanced across all query types
- **Best Threshold**: 0.3-0.7

## SpreadingActivation Strategy

### Best For
- Discovery of indirect relationships
- Exploring knowledge networks
- Finding unexpected connections

### Example: Network Exploration

```python
# Discover indirect relationships through activation spreading
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'climate change',
            'strategy': 'SpreadingActivation',
            'limit': 25,
            'threshold': 0.1,  # Lower threshold for broader exploration
            'include_reasoning': True
        },
        'id': 'spreading_search'
    }
)

results = response.json()['result']
activation_hops = results['retrieval_metadata']['spreading_activation_hops']
print(f"Spreading activation explored {activation_hops} hops")

for memory in results['memories']:
    path_length = len(memory['retrieval_path'])
    print(f"- {memory['content'][:60]}... (hops: {path_length})")
```

### Performance Characteristics
- **Response Time**: 100-200ms (depends on graph size)
- **Accuracy**: High for discovering connections
- **Best Threshold**: 0.1-0.4

## Strategy Comparison

```python
# Compare all strategies for the same query
query = "machine learning applications"
strategies = ['SemanticSimilarity', 'StructuralMatching', 'TemporalProximity', 'HybridApproach']

for strategy in strategies:
    start_time = time.time()
    
    response = requests.post('http://localhost:8080/mcp',
        headers=headers,
        json={
            'method': 'retrieve_memory',
            'params': {
                'query': query,
                'strategy': strategy,
                'limit': 5
            },
            'id': f'compare_{strategy.lower()}'
        }
    )
    
    end_time = time.time()
    results = response.json()['result']
    
    print(f"\n{strategy}:")
    print(f"  Response time: {(end_time - start_time)*1000:.1f}ms")
    print(f"  Results found: {len(results['memories'])}")
    print(f"  Top result: {results['memories'][0]['content'][:50]}...")
    print(f"  Score: {results['memories'][0]['similarity_score']:.2f}")
```

## Best Practices

### Strategy Selection Guidelines

1. **For factual queries**: Use SemanticSimilarity
2. **For relationship exploration**: Use StructuralMatching
3. **For recent information**: Use TemporalProximity
4. **For general searches**: Use HybridApproach (default)
5. **For discovery**: Use SpreadingActivation

### Performance Optimization

```python
# Use appropriate thresholds for each strategy
optimal_thresholds = {
    'SemanticSimilarity': 0.6,
    'StructuralMatching': 0.4,
    'TemporalProximity': 0.3,
    'HybridApproach': 0.5,
    'SpreadingActivation': 0.2
}

def search_with_optimal_threshold(query, strategy):
    return requests.post('http://localhost:8080/mcp',
        headers=headers,
        json={
            'method': 'retrieve_memory',
            'params': {
                'query': query,
                'strategy': strategy,
                'threshold': optimal_thresholds[strategy],
                'limit': 10
            },
            'id': f'optimized_{strategy.lower()}'
        }
    )
```
```

**Verification**: 
- All five search strategies demonstrated
- Performance characteristics documented
- Best practices and optimization tips included

---

### Micro-Task 9.2.3: Create Integration Patterns Documentation
**Estimated Time**: 20 minutes  
**File**: `docs/examples/integration_patterns.md`
**Expected Deliverable**: Real-world integration examples (Flask, async, batch operations)

**Task Prompt for AI**:
Create practical integration patterns for web applications, async processing, and production use cases.

```markdown
# Integration Patterns

Real-world examples for integrating CortexKG MCP server into applications.

## Flask Web Application Integration

```python
from flask import Flask, request, jsonify
import requests
import threading
import time
from functools import lru_cache

app = Flask(__name__)

class MemoryService:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
        self.request_id_counter = 0
        self.lock = threading.Lock()
    
    def _get_request_id(self):
        with self.lock:
            self.request_id_counter += 1
            return f'web_req_{self.request_id_counter}'
    
    def store(self, content, tags=None, confidence=0.8):
        response = requests.post(f'{self.base_url}/mcp', 
            headers=self.headers,
            json={
                'method': 'store_memory',
                'params': {
                    'content': content,
                    'tags': tags or [],
                    'confidence': confidence
                },
                'id': self._get_request_id()
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'error' not in result:
                return result['result']
        return None
    
    @lru_cache(maxsize=1000)
    def search(self, query, limit=10, strategy='HybridApproach'):
        cache_key = f"{query}_{limit}_{strategy}"
        
        response = requests.post(f'{self.base_url}/mcp',
            headers=self.headers,
            json={
                'method': 'retrieve_memory',
                'params': {
                    'query': query,
                    'limit': limit,
                    'strategy': strategy
                },
                'id': self._get_request_id()
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'error' not in result:
                return result['result']
        return None

# Initialize service
memory_service = MemoryService('http://localhost:8080', 'your_token_here')

@app.route('/api/memories', methods=['POST'])
def create_memory():
    data = request.json
    
    if not data or 'content' not in data:
        return jsonify({'error': 'Content is required'}), 400
    
    result = memory_service.store(
        data['content'], 
        data.get('tags'),
        data.get('confidence', 0.8)
    )
    
    if result:
        return jsonify({
            'memory_id': result['memory_id'],
            'allocation_path': result['allocation_path']
        })
    else:
        return jsonify({'error': 'Failed to store memory'}), 500

@app.route('/api/search')
def search_memories():
    query = request.args.get('q')
    limit = int(request.args.get('limit', 10))
    strategy = request.args.get('strategy', 'HybridApproach')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    result = memory_service.search(query, limit, strategy)
    
    if result:
        return jsonify({
            'memories': [{
                'content': mem['content'],
                'score': mem['similarity_score'],
                'tags': mem['tags']
            } for mem in result['memories']],
            'total': result['total_found'],
            'strategy_used': result['retrieval_metadata']['strategy_used']
        })
    else:
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/health')
def health_check():
    try:
        response = requests.get(f'{memory_service.base_url}/health', timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'status': 'unhealthy'}), 503
    except requests.RequestException:
        return jsonify({'status': 'service_unavailable'}), 503

if __name__ == '__main__':
    app.run(debug=True)
```

## Async Processing with Batch Operations

```python
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class MemoryBatch:
    content: str
    tags: List[str]
    confidence: float = 0.8
    importance: float = 0.5

class AsyncMemoryClient:
    def __init__(self, base_url: str, token: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, params: dict, request_id: str) -> Optional[dict]:
        async with self.semaphore:
            payload = {
                'method': method,
                'params': params,
                'id': request_id
            }
            
            try:
                async with self.session.post(f'{self.base_url}/mcp', json=payload) as response:
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(1)
                        return await self._request(method, params, request_id)
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'error' in data:
                        print(f"MCP Error: {data['error']['message']}")
                        return None
                    
                    return data.get('result')
                    
            except asyncio.TimeoutError:
                print(f"Request timeout for {request_id}")
                return None
            except aiohttp.ClientError as e:
                print(f"Request failed: {e}")
                return None
    
    async def store_batch(self, memories: List[MemoryBatch]) -> List[str]:
        """Store multiple memories concurrently"""
        tasks = []
        
        for i, memory in enumerate(memories):
            task = self._request(
                'store_memory',
                {
                    'content': memory.content,
                    'tags': memory.tags,
                    'confidence': memory.confidence,
                    'importance': memory.importance
                },
                f'batch_{i}'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        memory_ids = []
        for result in results:
            if isinstance(result, dict) and 'memory_id' in result:
                memory_ids.append(result['memory_id'])
        
        return memory_ids
    
    async def parallel_search(self, queries: List[str], strategy: str = 'HybridApproach') -> Dict[str, List[dict]]:
        """Execute multiple searches in parallel"""
        tasks = []
        
        for i, query in enumerate(queries):
            task = self._request(
                'retrieve_memory',
                {
                    'query': query,
                    'strategy': strategy,
                    'limit': 10
                },
                f'search_{i}'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        search_results = {}
        for i, (query, result) in enumerate(zip(queries, results)):
            if isinstance(result, dict) and 'memories' in result:
                search_results[query] = result['memories']
            else:
                search_results[query] = []
        
        return search_results

# Usage example
async def main():
    async with AsyncMemoryClient('http://localhost:8080', 'your_token') as client:
        
        # Batch store operation
        memories = [
            MemoryBatch(
                content=f"Machine learning fact #{i}",
                tags=['ml', 'ai', f'fact_{i}'],
                confidence=0.9
            )
            for i in range(100)
        ]
        
        print("Storing batch of memories...")
        memory_ids = await client.store_batch(memories)
        print(f"Stored {len(memory_ids)} memories")
        
        # Parallel search operations
        queries = [
            "machine learning algorithms",
            "neural network architectures",
            "data preprocessing techniques",
            "model evaluation metrics"
        ]
        
        print("Executing parallel searches...")
        search_results = await client.parallel_search(queries)
        
        for query, memories in search_results.items():
            print(f"\n{query}: {len(memories)} results")
            for memory in memories[:3]:  # Show top 3
                print(f"  - {memory['content'][:50]}... (score: {memory['similarity_score']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

## Background Processing with Queue

```python
import queue
import threading
import time
import requests
from typing import NamedTuple, Optional
import logging

class MemoryTask(NamedTuple):
    operation: str
    params: dict
    callback: Optional[callable] = None

class MemoryProcessor:
    def __init__(self, base_url: str, token: str, worker_count: int = 5):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
        self.task_queue = queue.Queue()
        self.workers = []
        self.worker_count = worker_count
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start background workers"""
        self.running = True
        for i in range(self.worker_count):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.worker_count} memory processing workers")
    
    def stop(self):
        """Stop background workers"""
        self.running = False
        
        # Add sentinel values to wake up workers
        for _ in range(self.worker_count):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        self.logger.info("All workers stopped")
    
    def _worker(self, worker_id: int):
        """Background worker thread"""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Sentinel value
                    break
                
                self._process_task(task, worker_id)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _process_task(self, task: MemoryTask, worker_id: int):
        """Process a single memory task"""
        try:
            response = requests.post(
                f'{self.base_url}/mcp',
                headers=self.headers,
                json={
                    'method': task.operation,
                    'params': task.params,
                    'id': f'worker_{worker_id}_{int(time.time())}'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result and task.callback:
                    task.callback(result['result'])
            else:
                self.logger.error(f"Request failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Task processing error: {e}")
    
    def queue_store(self, content: str, tags: list = None, callback: callable = None):
        """Queue a memory storage task"""
        task = MemoryTask(
            operation='store_memory',
            params={
                'content': content,
                'tags': tags or [],
                'confidence': 0.8
            },
            callback=callback
        )
        self.task_queue.put(task)
    
    def queue_search(self, query: str, callback: callable = None):
        """Queue a memory search task"""
        task = MemoryTask(
            operation='retrieve_memory',
            params={
                'query': query,
                'limit': 10
            },
            callback=callback
        )
        self.task_queue.put(task)

# Usage example
def on_memory_stored(result):
    print(f"Memory stored: {result['memory_id']}")

def on_search_complete(result):
    print(f"Search found {len(result['memories'])} results")

# Initialize and start processor
processor = MemoryProcessor('http://localhost:8080', 'your_token')
processor.start()

# Queue some tasks
for i in range(10):
    processor.queue_store(
        f"Background task #{i}",
        ['background', 'queue', f'task_{i}'],
        on_memory_stored
    )

processor.queue_search("background tasks", on_search_complete)

# Let it process
time.sleep(5)

# Clean shutdown
processor.stop()
```

## Production Monitoring Integration

```python
import requests
import time
import json
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

class CortexKGMonitor:
    def __init__(self, cortex_url: str, pushgateway_url: str):
        self.cortex_url = cortex_url
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.health_score = Gauge(
            'cortex_kg_health_score',
            'Overall system health score',
            registry=self.registry
        )
        
        self.response_time = Histogram(
            'cortex_kg_response_time_seconds',
            'Request response times',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.request_count = Counter(
            'cortex_kg_requests_total',
            'Total requests processed',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.memory_count = Gauge(
            'cortex_kg_memories_total',
            'Total memories stored',
            registry=self.registry
        )
    
    def collect_metrics(self):
        """Collect and push metrics to Prometheus"""
        try:
            # Health check
            start_time = time.time()
            health_response = requests.get(f"{self.cortex_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                self.health_score.set(health_data.get('health_score', 0))
                self.response_time.observe(response_time)
                self.request_count.labels(method='health', status='success').inc()
            else:
                self.request_count.labels(method='health', status='error').inc()
            
            # Memory statistics
            stats_response = requests.post(f"{self.cortex_url}/mcp", json={
                'method': 'get_memory_stats',
                'params': {'include_performance': True},
                'id': f'monitor_{int(time.time())}'
            }, timeout=30)
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json().get('result', {})
                system_stats = stats_data.get('system_stats', {})
                self.memory_count.set(system_stats.get('total_memories', 0))
                self.request_count.labels(method='get_memory_stats', status='success').inc()
            
            # Push to Prometheus
            push_to_gateway(
                self.pushgateway_url,
                job='cortex-kg-monitor',
                registry=self.registry
            )
            
        except Exception as e:
            print(f"Metrics collection failed: {e}")
            self.request_count.labels(method='monitoring', status='error').inc()

# Usage
monitor = CortexKGMonitor(
    'http://localhost:8080',
    'http://prometheus-pushgateway:9091'
)

# Collect metrics every minute
while True:
    monitor.collect_metrics()
    time.sleep(60)
```
```

**Verification**: 
- Complete Flask integration with caching
- Async batch processing patterns
- Background queue processing
- Production monitoring integration

---

### Micro-Task 9.3.1: Create Deployment Guide
**Estimated Time**: 19 minutes  
**File**: `docs/operations/deployment_guide.md`
**Expected Deliverable**: Complete production deployment instructions

**Task Prompt for AI**:
Create comprehensive deployment guide covering production setup, Docker, Kubernetes, and monitoring.

```markdown
# CortexKG MCP Server Deployment Guide

## Prerequisites

### System Requirements

**Minimum (Development)**
- CPU: 2 cores, 2.4GHz
- RAM: 4GB
- Storage: 10GB
- OS: Linux, macOS, Windows

**Recommended (Production)**
- CPU: 4+ cores, 3.0GHz with AVX2 support
- RAM: 8GB+ (16GB for large datasets)
- Storage: 50GB SSD
- OS: Linux (Ubuntu 22.04 LTS recommended)

### Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev curl

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel curl

# macOS
brew install openssl curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

## Production Deployment

### 1. Build Release Binary

```bash
# Clone repository
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server

# Build optimized release
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Verify binary
./target/release/cortex_kg_mcp_server --version
```

### 2. System Configuration

#### Create System User
```bash
sudo useradd -r -s /bin/false cortex-kg
sudo mkdir -p /var/lib/cortex-kg /var/log/cortex-kg /etc/cortex-kg
sudo chown cortex-kg:cortex-kg /var/lib/cortex-kg /var/log/cortex-kg
```

#### Generate Security Keys
```bash
# JWT signing key
sudo openssl rand -hex 32 > /etc/cortex-kg/jwt.key
sudo chown cortex-kg:cortex-kg /etc/cortex-kg/jwt.key
sudo chmod 600 /etc/cortex-kg/jwt.key

# TLS certificates (if using HTTPS)
sudo openssl req -x509 -newkey rsa:4096 -keyout /etc/cortex-kg/server.key \
  -out /etc/cortex-kg/server.crt -days 365 -nodes
```

#### Install Binary and Configuration
```bash
sudo cp target/release/cortex_kg_mcp_server /usr/local/bin/
sudo cp config/production.toml /etc/cortex-kg/
sudo chown cortex-kg:cortex-kg /etc/cortex-kg/production.toml
```

### 3. Systemd Service

Create `/etc/systemd/system/cortex-kg-mcp.service`:

```ini
[Unit]
Description=CortexKG MCP Server
After=network.target
Wants=network.target

[Service]
Type=exec
User=cortex-kg
Group=cortex-kg
ExecStart=/usr/local/bin/cortex_kg_mcp_server --config /etc/cortex-kg/production.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cortex-kg-mcp

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/cortex-kg /var/log/cortex-kg

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

#### Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cortex-kg-mcp
sudo systemctl start cortex-kg-mcp

# Check status
sudo systemctl status cortex-kg-mcp
```

## Docker Deployment

### 1. Production Dockerfile

```dockerfile
# Multi-stage build for optimal size
FROM rust:1.75-slim as builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates libssl3 curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 cortex && \
    mkdir -p /var/lib/cortex-kg /var/log/cortex-kg && \
    chown -R cortex:cortex /var/lib/cortex-kg /var/log/cortex-kg

WORKDIR /app
COPY --from=builder /app/target/release/cortex_kg_mcp_server /usr/local/bin/
COPY --chown=cortex:cortex config/ /app/config/

USER cortex

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 9090
CMD ["cortex_kg_mcp_server", "--config", "/app/config/production.toml"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  cortex-kg-mcp:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: cortex-kg-mcp
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./config/production.toml:/app/config/production.toml:ro
      - ./secrets:/etc/cortex-kg:ro
      - cortex-data:/var/lib/cortex-kg
      - cortex-logs:/var/log/cortex-kg
    environment:
      - RUST_LOG=info
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

volumes:
  cortex-data:
  cortex-logs:
```

## Kubernetes Deployment

### 1. ConfigMap and Secrets

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cortex-kg

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cortex-kg-config
  namespace: cortex-kg
data:
  production.toml: |
    [server]
    host = "0.0.0.0"
    port = 8080
    max_connections = 1000
    
    [neuromorphic]
    ttfs_precision_ms = 0.1
    cortical_columns = 4
    
    [performance]
    enable_simd = true
    cache_size_mb = 1024
    
    [security]
    enable_oauth = true
    jwt_secret_path = "/etc/secrets/jwt.key"

---
apiVersion: v1
kind: Secret
metadata:
  name: cortex-kg-secrets
  namespace: cortex-kg
type: Opaque
data:
  jwt.key: <base64-encoded-jwt-secret>
```

### 2. Deployment and Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-kg-mcp
  namespace: cortex-kg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cortex-kg-mcp
  template:
    metadata:
      labels:
        app: cortex-kg-mcp
    spec:
      containers:
      - name: cortex-kg-mcp
        image: ghcr.io/your-org/cortex-kg-mcp:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
        - name: data
          mountPath: /var/lib/cortex-kg
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: cortex-kg-config
      - name: secrets
        secret:
          secretName: cortex-kg-secrets
      - name: data
        persistentVolumeClaim:
          claimName: cortex-kg-data

---
apiVersion: v1
kind: Service
metadata:
  name: cortex-kg-mcp-service
  namespace: cortex-kg
spec:
  selector:
    app: cortex-kg-mcp
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

## Security Configuration

### 1. Firewall Setup

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp comment 'CortexKG MCP'
sudo ufw allow 9090/tcp comment 'Metrics'
sudo ufw enable
```

### 2. TLS Configuration

```bash
# Generate strong TLS certificate
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt \
  -days 365 -nodes -subj "/CN=api.cortex-kg.example.com"

# Set proper permissions
chmod 600 server.key
chmod 644 server.crt
```

## Monitoring Setup

### 1. Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Response:
# {
#   "status": "healthy",
#   "health_score": 0.92,
#   "response_time_ms": 45.2,
#   "throughput_ops_per_min": 1250.0,
#   "error_rate_percent": 0.5,
#   "memory_usage_mb": 512.8,
#   "uptime_seconds": 86400
# }
```

### 2. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cortex-kg-mcp'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 10s
    metrics_path: /metrics
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats cortex-kg-mcp
   
   # Reduce cache size in config
   cache_size_mb = 256
   ```

2. **Slow Response Times**
   ```bash
   # Enable SIMD acceleration
   enable_simd = true
   
   # Increase connection pool
   connection_pool_size = 200
   ```

3. **Authentication Failures**
   ```bash
   # Check JWT secret
   ls -la /etc/cortex-kg/jwt.key
   
   # Verify OAuth configuration
   curl -v https://auth.provider.com/.well-known/openid_configuration
   ```

### Log Analysis

```bash
# Check service logs
journalctl -u cortex-kg-mcp -f

# Docker logs
docker logs -f cortex-kg-mcp

# Search for errors
grep -i error /var/log/cortex-kg/server.log
```
```

**Verification**: 
- Complete deployment instructions for multiple platforms
- Security configuration included
- Monitoring and troubleshooting covered

---

### Micro-Task 9.3.2: Create Operations Manual
**Estimated Time**: 16 minutes  
**File**: `docs/operations/operations_manual.md`
**Expected Deliverable**: Day-to-day operations guide

**Task Prompt for AI**:
Create a comprehensive operations manual for day-to-day management, monitoring, and maintenance procedures.

```markdown
# Operations Manual

Daily operations guide for managing the CortexKG MCP server in production.

## Daily Operations

### 1. Health Monitoring

#### Morning Health Check
```bash
#!/bin/bash
# daily_health_check.sh

echo "🔍 CortexKG Daily Health Check - $(date)"
echo "==========================================="

# Service status
if systemctl is-active --quiet cortex-kg-mcp; then
    echo "✅ Service Status: Running"
else
    echo "❌ Service Status: Not Running"
    exit 1
fi

# Health endpoint
HEALTH_DATA=$(curl -s http://localhost:8080/health)
STATUS=$(echo "$HEALTH_DATA" | jq -r '.status')
HEALTH_SCORE=$(echo "$HEALTH_DATA" | jq -r '.health_score')
RESPONSE_TIME=$(echo "$HEALTH_DATA" | jq -r '.response_time_ms')
ERROR_RATE=$(echo "$HEALTH_DATA" | jq -r '.error_rate_percent')

echo "📊 Health Status: $STATUS"
echo "📈 Health Score: $HEALTH_SCORE"
echo "⏱️  Response Time: ${RESPONSE_TIME}ms"
echo "🚨 Error Rate: ${ERROR_RATE}%"

# Performance thresholds
if (( $(echo "$HEALTH_SCORE < 0.8" | bc -l) )); then
    echo "⚠️  LOW HEALTH SCORE: $HEALTH_SCORE"
fi

if (( $(echo "$RESPONSE_TIME > 100" | bc -l) )); then
    echo "⚠️  HIGH RESPONSE TIME: ${RESPONSE_TIME}ms"
fi

if (( $(echo "$ERROR_RATE > 1" | bc -l) )); then
    echo "⚠️  HIGH ERROR RATE: ${ERROR_RATE}%"
fi

# Memory statistics
MEMORY_STATS=$(curl -s -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "get_memory_stats", "params": {}, "id": "daily_check"}')

TOTAL_MEMORIES=$(echo "$MEMORY_STATS" | jq -r '.result.system_stats.total_memories')
STORAGE_SIZE=$(echo "$MEMORY_STATS" | jq -r '.result.system_stats.storage_size_mb')

echo "🧠 Total Memories: $TOTAL_MEMORIES"
echo "💾 Storage Size: ${STORAGE_SIZE}MB"

echo "✅ Daily health check complete"
```

#### Continuous Monitoring
```bash
# monitor_dashboard.sh - Real-time monitoring
#!/bin/bash

while true; do
    clear
    echo "🔍 CortexKG Real-time Monitor - $(date)"
    echo "======================================"
    
    # Container stats (if using Docker)
    if command -v docker &> /dev/null && docker ps | grep -q cortex-kg-mcp; then
        echo "🐳 Container Stats:"
        docker stats cortex-kg-mcp --no-stream --format "   CPU: {{.CPUPerc}} | Memory: {{.MemUsage}} | Network: {{.NetIO}}"
        echo ""
    fi
    
    # Service metrics
    HEALTH_DATA=$(curl -s http://localhost:8080/health 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo "📊 Service Metrics:"
        echo "   Status: $(echo "$HEALTH_DATA" | jq -r '.status')"
        echo "   Health Score: $(echo "$HEALTH_DATA" | jq -r '.health_score')"
        echo "   Response Time: $(echo "$HEALTH_DATA" | jq -r '.response_time_ms')ms"
        echo "   Throughput: $(echo "$HEALTH_DATA" | jq -r '.throughput_ops_per_min') ops/min"
        echo "   Memory Usage: $(echo "$HEALTH_DATA" | jq -r '.memory_usage_mb')MB"
    else
        echo "❌ Service not responding"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit. Refreshing in 10 seconds..."
    sleep 10
done
```

### 2. Log Management

#### Daily Log Analysis
```bash
# analyze_logs.sh
#!/bin/bash

LOG_DATE=$(date +%Y-%m-%d)
echo "📋 Log Analysis for $LOG_DATE"
echo "=============================="

# Error count
ERROR_COUNT=$(journalctl -u cortex-kg-mcp --since "$LOG_DATE" | grep -i error | wc -l)
echo "🚨 Errors today: $ERROR_COUNT"

# Warning count
WARN_COUNT=$(journalctl -u cortex-kg-mcp --since "$LOG_DATE" | grep -i warn | wc -l)
echo "⚠️  Warnings today: $WARN_COUNT"

# Request count
REQ_COUNT=$(journalctl -u cortex-kg-mcp --since "$LOG_DATE" | grep -i "request" | wc -l)
echo "📤 Requests today: $REQ_COUNT"

# Top errors
echo ""
echo "🔝 Top 5 Error Messages:"
journalctl -u cortex-kg-mcp --since "$LOG_DATE" | grep -i error | sort | uniq -c | sort -nr | head -5

# Performance issues
echo ""
echo "⏱️  Performance Issues:"
journalctl -u cortex-kg-mcp --since "$LOG_DATE" | grep -i "slow\|timeout\|high" | tail -10
```

#### Log Rotation Setup
```bash
# /etc/logrotate.d/cortex-kg-mcp
/var/log/cortex-kg/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    postrotate
        systemctl reload cortex-kg-mcp
    endscript
}
```

## Weekly Maintenance

### 1. Performance Analysis

```bash
# weekly_performance_report.sh
#!/bin/bash

WEEK_START=$(date -d '7 days ago' +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "📊 Weekly Performance Report: $WEEK_START to $TODAY"
echo "================================================"

# Average response times
echo "⏱️  Response Time Analysis:"
journalctl -u cortex-kg-mcp --since "$WEEK_START" | \
    grep "response_time" | \
    awk '{print $NF}' | \
    awk '{sum+=$1; count++} END {printf "   Average: %.2fms\n", sum/count}'

# Memory usage trends
echo ""
echo "💾 Memory Usage Trends:"
for i in {0..6}; do
    CHECK_DATE=$(date -d "$i days ago" +%Y-%m-%d)
    MEMORY_USAGE=$(journalctl -u cortex-kg-mcp --since "$CHECK_DATE 00:00:00" --until "$CHECK_DATE 23:59:59" | \
                   grep "memory_usage" | tail -1 | awk '{print $NF}')
    echo "   $CHECK_DATE: ${MEMORY_USAGE:-N/A}"
done

# Error rate trends
echo ""
echo "🚨 Error Rate Trends:"
for i in {0..6}; do
    CHECK_DATE=$(date -d "$i days ago" +%Y-%m-%d)
    ERROR_COUNT=$(journalctl -u cortex-kg-mcp --since "$CHECK_DATE 00:00:00" --until "$CHECK_DATE 23:59:59" | \
                  grep -i error | wc -l)
    echo "   $CHECK_DATE: $ERROR_COUNT errors"
done
```

### 2. System Optimization

```bash
# weekly_optimization.sh
#!/bin/bash

echo "🔧 Weekly System Optimization"
echo "============================"

# Clear temporary files
echo "🧹 Cleaning temporary files..."
sudo find /tmp -name "cortex-kg-*" -mtime +7 -delete
sudo find /var/tmp -name "cortex-kg-*" -mtime +7 -delete

# Analyze disk usage
echo "💾 Disk Usage Analysis:"
echo "   Data directory: $(du -sh /var/lib/cortex-kg | cut -f1)"
echo "   Log directory: $(du -sh /var/log/cortex-kg | cut -f1)"

# Check for memory leaks
echo "🔍 Memory Leak Check:"
ps aux | grep cortex_kg_mcp_server | grep -v grep | awk '{print "   RSS: " $6/1024 " MB, VSZ: " $5/1024 " MB"}'

# Database optimization (if applicable)
echo "🗄️  Database Optimization:"
SIZE_BEFORE=$(du -sh /var/lib/cortex-kg/knowledge.db 2>/dev/null | cut -f1 || echo "N/A")
echo "   Knowledge graph size: $SIZE_BEFORE"

# Restart service for memory cleanup
echo "🔄 Service restart for memory cleanup..."
sudo systemctl restart cortex-kg-mcp
sleep 10

if systemctl is-active --quiet cortex-kg-mcp; then
    echo "✅ Service restarted successfully"
else
    echo "❌ Service restart failed"
    sudo systemctl status cortex-kg-mcp
fi
```

## Monthly Procedures

### 1. Backup and Recovery

```bash
# monthly_backup.sh
#!/bin/bash

BACKUP_DATE=$(date +%Y-%m-%d)
BACKUP_DIR="/backup/cortex-kg/$BACKUP_DATE"

echo "💾 Monthly Backup - $BACKUP_DATE"
echo "=============================="

# Create backup directory
sudo mkdir -p "$BACKUP_DIR"

# Backup configuration
echo "📁 Backing up configuration..."
sudo cp -r /etc/cortex-kg "$BACKUP_DIR/config"

# Backup data
echo "🗄️  Backing up data..."
sudo tar -czf "$BACKUP_DIR/data.tar.gz" -C /var/lib/cortex-kg .

# Backup logs (last 30 days)
echo "📋 Backing up recent logs..."
sudo journalctl -u cortex-kg-mcp --since '30 days ago' > "$BACKUP_DIR/service.log"
sudo tar -czf "$BACKUP_DIR/logs.tar.gz" -C /var/log/cortex-kg .

# Create backup manifest
echo "📝 Creating backup manifest..."
sudo tee "$BACKUP_DIR/manifest.txt" << EOF
CortexKG MCP Server Backup
Date: $BACKUP_DATE
Version: $(cortex_kg_mcp_server --version 2>/dev/null || echo "Unknown")
System: $(uname -a)
Files:
$(ls -la "$BACKUP_DIR")
EOF

# Verify backup
echo "✅ Backup verification:"
echo "   Config size: $(du -sh "$BACKUP_DIR/config" | cut -f1)"
echo "   Data size: $(du -sh "$BACKUP_DIR/data.tar.gz" | cut -f1)"
echo "   Logs size: $(du -sh "$BACKUP_DIR/logs.tar.gz" | cut -f1)"
echo "   Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"

# Cleanup old backups (keep 3 months)
echo "🧹 Cleaning old backups..."
sudo find /backup/cortex-kg -type d -mtime +90 -exec rm -rf {} +

echo "✅ Monthly backup complete"
```

### 2. Security Audit

```bash
# security_audit.sh
#!/bin/bash

echo "🔒 Monthly Security Audit"
echo "======================="

# Check file permissions
echo "📁 File Permission Check:"
echo "   /etc/cortex-kg/jwt.key: $(ls -la /etc/cortex-kg/jwt.key | awk '{print $1, $3, $4}')"
echo "   /var/lib/cortex-kg: $(ls -ld /var/lib/cortex-kg | awk '{print $1, $3, $4}')"

# Check for failed login attempts
echo ""
echo "🚨 Failed Access Attempts:"
FAILED_AUTHS=$(journalctl -u cortex-kg-mcp --since '30 days ago' | grep -i "authentication.*failed" | wc -l)
echo "   Failed authentications: $FAILED_AUTHS"

# Check SSL certificate expiry
echo ""
echo "🔐 SSL Certificate Status:"
if [[ -f /etc/cortex-kg/server.crt ]]; then
    EXPIRY=$(openssl x509 -in /etc/cortex-kg/server.crt -noout -enddate | cut -d= -f2)
    echo "   Certificate expires: $EXPIRY"
    
    # Check if expiring in 30 days
    EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s)
    THIRTY_DAYS=$(date -d '+30 days' +%s)
    
    if [[ $EXPIRY_EPOCH -lt $THIRTY_DAYS ]]; then
        echo "   ⚠️  WARNING: Certificate expires within 30 days!"
    fi
else
    echo "   No SSL certificate found"
fi

# Check for security updates
echo ""
echo "🔄 Security Updates:"
sudo apt list --upgradable 2>/dev/null | grep -i security | wc -l | xargs echo "   Available security updates:"

echo "✅ Security audit complete"
```

## Emergency Procedures

### 1. Service Recovery

```bash
# emergency_recovery.sh
#!/bin/bash

echo "🚨 Emergency Recovery Procedure"
echo "=============================="

# Stop service
echo "🛑 Stopping service..."
sudo systemctl stop cortex-kg-mcp

# Check for core dumps
echo "🔍 Checking for core dumps..."
CORE_DUMPS=$(find /var/crash -name "core.*cortex*" 2>/dev/null | wc -l)
if [[ $CORE_DUMPS -gt 0 ]]; then
    echo "   Found $CORE_DUMPS core dumps in /var/crash"
fi

# Backup current state
echo "💾 Backing up current state..."
EMERGENCY_BACKUP="/tmp/emergency-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EMERGENCY_BACKUP"
cp -r /var/lib/cortex-kg "$EMERGENCY_BACKUP/data"
cp -r /etc/cortex-kg "$EMERGENCY_BACKUP/config"
journalctl -u cortex-kg-mcp --since '1 hour ago' > "$EMERGENCY_BACKUP/recent.log"

# Clear problematic files
echo "🧹 Clearing temporary files..."
rm -f /tmp/cortex-kg-*
rm -f /var/tmp/cortex-kg-*

# Reset permissions
echo "🔧 Resetting permissions..."
sudo chown -R cortex-kg:cortex-kg /var/lib/cortex-kg
sudo chown -R cortex-kg:cortex-kg /var/log/cortex-kg
sudo chmod 600 /etc/cortex-kg/jwt.key

# Start service
echo "▶️  Starting service..."
sudo systemctl start cortex-kg-mcp

# Wait and check
sleep 10
if systemctl is-active --quiet cortex-kg-mcp; then
    echo "✅ Service recovery successful"
    echo "   Health check: $(curl -s http://localhost:8080/health | jq -r '.status')"
else
    echo "❌ Service recovery failed"
    echo "   Status: $(sudo systemctl status cortex-kg-mcp --no-pager -l)"
    echo "   Emergency backup saved to: $EMERGENCY_BACKUP"
fi
```

### 2. Performance Emergency

```bash
# performance_emergency.sh
#!/bin/bash

echo "⚡ Performance Emergency Response"
echo "==============================="

# Get current metrics
HEALTH=$(curl -s http://localhost:8080/health)
HEALTH_SCORE=$(echo "$HEALTH" | jq -r '.health_score')
RESPONSE_TIME=$(echo "$HEALTH" | jq -r '.response_time_ms')
MEMORY_USAGE=$(echo "$HEALTH" | jq -r '.memory_usage_mb')

echo "📊 Current Performance:"
echo "   Health Score: $HEALTH_SCORE"
echo "   Response Time: ${RESPONSE_TIME}ms"
echo "   Memory Usage: ${MEMORY_USAGE}MB"

# Performance triage
if (( $(echo "$HEALTH_SCORE < 0.5" | bc -l) )); then
    echo "🚨 CRITICAL: Health score below 0.5"
    
    # Emergency cache clear
    echo "🧹 Clearing caches..."
    curl -s -X POST http://localhost:8080/mcp \
         -H "Content-Type: application/json" \
         -d '{"method": "clear_caches", "params": {}, "id": "emergency"}'
    
    # Reduce batch sizes
    echo "⚙️  Reducing batch sizes..."
    curl -s -X POST http://localhost:8080/mcp \
         -H "Content-Type: application/json" \
         -d '{"method": "configure_learning", "params": {"batch_size": 16}, "id": "emergency"}'
fi

if (( $(echo "$RESPONSE_TIME > 200" | bc -l) )); then
    echo "⚠️  HIGH RESPONSE TIME: ${RESPONSE_TIME}ms"
    
    # Enable SIMD if not already enabled
    echo "🚀 Enabling SIMD acceleration..."
    curl -s -X POST http://localhost:8080/mcp \
         -H "Content-Type: application/json" \
         -d '{"method": "enable_simd", "params": {}, "id": "emergency"}'
fi

echo "✅ Emergency response complete"
echo "   Re-run health check in 5 minutes"
```

## Automation Setup

### Cron Jobs

```bash
# /etc/cron.d/cortex-kg-maintenance

# Daily health check at 9 AM
0 9 * * * cortex-kg /opt/cortex-kg/scripts/daily_health_check.sh >> /var/log/cortex-kg/daily-check.log 2>&1

# Weekly performance report on Sundays at 6 AM
0 6 * * 0 cortex-kg /opt/cortex-kg/scripts/weekly_performance_report.sh >> /var/log/cortex-kg/weekly-report.log 2>&1

# Monthly backup on the 1st at 2 AM
0 2 1 * * root /opt/cortex-kg/scripts/monthly_backup.sh >> /var/log/cortex-kg/backup.log 2>&1

# Log analysis every 6 hours
0 */6 * * * cortex-kg /opt/cortex-kg/scripts/analyze_logs.sh >> /var/log/cortex-kg/log-analysis.log 2>&1
```

### Monitoring Alerts

```bash
# /opt/cortex-kg/scripts/alert_check.sh
#!/bin/bash

# Check critical metrics and send alerts
HEALTH=$(curl -s http://localhost:8080/health)
HEALTH_SCORE=$(echo "$HEALTH" | jq -r '.health_score')
ERROR_RATE=$(echo "$HEALTH" | jq -r '.error_rate_percent')

# Alert thresholds
CRITICAL_HEALTH=0.5
CRITICAL_ERROR_RATE=5.0

if (( $(echo "$HEALTH_SCORE < $CRITICAL_HEALTH" | bc -l) )); then
    echo "CRITICAL: CortexKG health score $HEALTH_SCORE below threshold $CRITICAL_HEALTH" | \
         mail -s "CortexKG Critical Alert" admin@example.com
fi

if (( $(echo "$ERROR_RATE > $CRITICAL_ERROR_RATE" | bc -l) )); then
    echo "CRITICAL: CortexKG error rate $ERROR_RATE% above threshold $CRITICAL_ERROR_RATE%" | \
         mail -s "CortexKG Error Rate Alert" admin@example.com
fi
```
```

**Verification**: 
- Complete daily, weekly, and monthly procedures
- Emergency response procedures documented
- Automation and alerting setup included

---

## Validation Checklist

- [ ] API overview documentation complete
- [ ] Individual tool documentation created
- [ ] Basic usage examples functional
- [ ] Search strategy examples comprehensive
- [ ] Integration patterns cover real-world use cases
- [ ] Deployment guide covers all platforms
- [ ] Operations manual provides practical procedures
- [ ] All documentation follows consistent format
- [ ] Examples are tested and working
- [ ] Documentation enables user adoption

## Next Phase Dependencies

This phase enables:
- **User Adoption**: Complete documentation for developers
- **Production Operations**: Operational procedures for DevOps teams
- **Community Growth**: Open source contribution guidelines
- **Support Systems**: Self-service documentation and troubleshooting

## Overview


## Cortical Column Architecture

### Four-Column Processing

The system uses four specialized cortical columns that process information in parallel:

1. **Semantic Column**: Analyzes conceptual meaning and relationships
2. **Structural Column**: Optimizes graph topology and connections
3. **Temporal Column**: Processes time-based patterns and sequences
4. **Exception Column**: Detects contradictions and anomalies

### Processing Flow

```
Input → TTFS Encoding → Parallel Column Processing → Lateral Inhibition → Consensus → Action
```

1. **TTFS Encoding**: Converts input to Time-To-First-Spike patterns
2. **Parallel Processing**: All four columns analyze the pattern simultaneously
3. **Lateral Inhibition**: Columns compete for allocation decisions
4. **Consensus Formation**: Winning column determines memory allocation
5. **STDP Learning**: Synaptic weights updated based on success

## Neural Network Integration

### Adaptive Network Selection

Each cortical column can utilize multiple neural network architectures:

- **Semantic**: MLP, TiDE, DeepAR, TSMixer
- **Structural**: StemGNN, iTransformer, PatchTST, TFT  
- **Temporal**: LSTM, TCN, NBEATS, GRU
- **Exception**: CascadeCorrelation, SparseConnected, DLinear

The system intelligently selects optimal networks (1-4 types from 29 available) based on:
- Input pattern characteristics
- Historical performance
- Current system load
- Memory constraints

### Performance Optimization

- **SIMD Acceleration**: Vectorized operations for similarity computation
- **Connection Pooling**: Efficient resource management
- **Multi-Level Caching**: L1/L2/L3 hierarchy for fast retrieval
- **Parallel Processing**: Concurrent column activation

## Memory Allocation Process

### 1. Pattern Analysis
```
Input: "The capital of France is Paris"
↓
TTFS Pattern: [(0.1, 1.0), (0.2, 2.0), (0.3, 3.0), ...]
```

### 2. Cortical Processing
```
Semantic Column:    Activation = 0.95 (geography, capital, location)
Structural Column:  Activation = 0.67 (factual, hierarchical)
Temporal Column:    Activation = 0.45 (static fact, low temporal)
Exception Column:   Activation = 0.12 (no contradictions detected)
```

### 3. Lateral Inhibition
```
Winner: Semantic Column (0.95 * 1.2 = 1.14)
Suppressed: Others (activation * 0.3)
Consensus Strength: 2.8
```

### 4. Memory Storage
```
Allocation Path: ["semantic_column", "geography_cluster", "europe_facts"]
Storage Location: graph_node_789
Neural Pathway: ["concept_similarity", "factual_knowledge", "location_hierarchy"]
```

## Learning Mechanisms

### STDP (Spike-Timing Dependent Plasticity)

Synaptic weights are updated based on temporal relationships:

```
if (pre_spike_time < post_spike_time):
    weight += learning_rate * reward_signal
else:
    weight -= learning_rate * punishment_signal
```

### Homeostatic Plasticity

Maintains network stability by:
- Scaling synaptic weights to prevent runaway excitation
- Balancing activation levels across columns
- Pruning weak connections during maintenance

### Meta-Learning

The system learns to:
- Select optimal neural networks for specific patterns
- Adjust learning rates based on performance
- Optimize allocation strategies over time

## Retrieval Strategies

### Semantic Similarity
- Computes conceptual distance using embeddings
- Considers synonyms and related concepts
- Best for: Knowledge queries, concept exploration

### Structural Matching
- Analyzes graph topology and connection patterns
- Finds structurally similar memory clusters
- Best for: Relationship discovery, pattern matching

### Temporal Proximity
- Considers time-based relationships and sequences
- Weighted by recency and temporal context
- Best for: Event sequences, time-sensitive information

### Hybrid Approach
- Combines all column outputs with learned weights
- Adapts strategy based on query characteristics
- Best for: General-purpose retrieval (default)

### Spreading Activation
- Simulates neural activation propagation
- Follows connection strengths through graph
- Best for: Discovery of indirect relationships

## Performance Characteristics

### Response Times by Operation
- **Simple Storage**: 15-30ms
- **Complex Retrieval**: 45-85ms
- **Graph Analysis**: 100-200ms
- **Learning Updates**: 5-15ms

### Throughput Scaling
- **Sequential**: ~1,200 ops/minute
- **Parallel (4 cores)**: ~4,500 ops/minute
- **SIMD Optimized**: ~7,000 ops/minute

### Memory Efficiency
- **Base System**: ~50MB
- **Per 100K Memories**: ~200MB
- **Cache Overhead**: ~100MB
- **Neural Networks**: ~500MB

## Tuning Parameters

### Learning Rate (0.001-0.1)
- **Low (0.001-0.01)**: Stable, slow adaptation
- **Medium (0.01-0.05)**: Balanced learning (default: 0.01)
- **High (0.05-0.1)**: Fast adaptation, potential instability

### Activation Threshold (0.1-0.9)
- **Low (0.1-0.3)**: Sensitive, many activations
- **Medium (0.3-0.7)**: Balanced (default: 0.6)
- **High (0.7-0.9)**: Selective, few activations

### Plasticity Window (1-100ms)
- **Narrow (1-10ms)**: Precise timing requirements
- **Medium (10-50ms)**: Standard STDP (default: 20ms)
- **Wide (50-100ms)**: Loose timing constraints

## Best Practices

### For Optimal Performance
1. Use batch operations when possible
2. Enable SIMD acceleration for large datasets
3. Configure appropriate cache sizes
4. Monitor column activation balance

### For Memory Quality
1. Provide rich context and tags
2. Set appropriate confidence scores
3. Use consistent terminology
4. Regular graph analysis and optimization

### For Learning Effectiveness
1. Allow sufficient training data
2. Monitor learning convergence
3. Adjust parameters based on domain
4. Use meta-learning for automatic tuning
```

**Success Criteria**:
- Complete API documentation with examples
- Neuromorphic concepts explained clearly
- All tools documented with schemas and outputs
- Performance characteristics documented
- Best practices provided for optimization

### Task 9.2: Usage Examples and Tutorials
**Estimated Time**: 25 minutes  
**File**: `docs/examples/basic_usage.md`

```markdown
# Basic Usage Examples

## Getting Started

### 1. Server Setup

```bash
# Clone the repository
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server

# Build the server
cargo build --release

# Generate JWT secret
openssl rand -hex 32 > config/jwt.key

# Start the server
./target/release/cortex_kg_mcp_server --config config/development.toml
```

### 2. Authentication

First, obtain an access token (simplified for development):

```python
import requests
import json

# For development, create a test token
# In production, use OAuth 2.1 flow
auth_response = requests.post('http://localhost:8080/auth/token', json={
    'username': 'test_user',
    'permissions': ['cortex_kg:read', 'cortex_kg:write']
})

token = auth_response.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}
```

## Basic Memory Operations

### Storing Information

```python
# Store a simple fact
response = requests.post('http://localhost:8080/mcp', 
    headers=headers,
    json={
        'method': 'store_memory',
        'params': {
            'content': 'The Pacific Ocean is the largest ocean on Earth',
            'tags': ['geography', 'ocean', 'facts'],
            'confidence': 0.95
        },
        'id': 'req_001'
    }
)

result = response.json()
memory_id = result['result']['memory_id']
print(f"Stored memory: {memory_id}")
```

### Retrieving Information

```python
# Search for related information
response = requests.post('http://localhost:8080/mcp',
    headers=headers, 
    json={
        'method': 'retrieve_memory',
        'params': {
            'query': 'largest ocean',
            'limit': 5,
            'threshold': 0.3
        },
        'id': 'req_002'
    }
)

results = response.json()
for memory in results['result']['memories']:
    print(f"Found: {memory['content']} (score: {memory['similarity_score']:.2f})")
```

### Updating Memories

```python
# Update existing memory
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'update_memory', 
        'params': {
            'memory_id': memory_id,
            'new_content': 'The Pacific Ocean is the largest and deepest ocean on Earth',
            'add_tags': ['depth', 'size'],
            'new_confidence': 0.98
        },
        'id': 'req_003'
    }
)
```

## Advanced Examples

### Building a Knowledge Base

```python
# Store interconnected facts
facts = [
    {
        'content': 'Paris is the capital of France',
        'tags': ['geography', 'france', 'capital', 'europe']
    },
    {
        'content': 'France is located in Western Europe', 
        'tags': ['geography', 'france', 'europe', 'location']
    },
    {
        'content': 'The Eiffel Tower is located in Paris',
        'tags': ['landmarks', 'paris', 'france', 'architecture']
    },
    {
        'content': 'French is the official language of France',
        'tags': ['language', 'france', 'french', 'official']
    }
]

memory_ids = []
for i, fact in enumerate(facts):
    response = requests.post('http://localhost:8080/mcp',
        headers=headers,
        json={
            'method': 'store_memory',
            'params': fact,
            'id': f'fact_{i}'
        }
    )
    memory_ids.append(response.json()['result']['memory_id'])

print(f"Stored {len(memory_ids)} interconnected facts")
```

### Semantic Search Examples

```python
# Different search strategies
queries = [
    ('What is the capital of France?', 'SemanticSimilarity'),
    ('Find connections to Paris', 'StructuralMatching'), 
    ('Recent information about France', 'TemporalProximity'),
    ('Everything about French culture', 'SpreadingActivation')
]

for query, strategy in queries:
    response = requests.post('http://localhost:8080/mcp',
        headers=headers,
        json={
            'method': 'retrieve_memory',
            'params': {
                'query': query,
                'strategy': strategy,
                'limit': 3,
                'include_reasoning': True
            },
            'id': f'search_{strategy.lower()}'
        }
    )
    
    result = response.json()['result']
    print(f"\n{strategy} search for '{query}':")
    print(f"Strategy used: {result['retrieval_metadata']['strategy_used']}")
    
    for memory in result['memories']:
        print(f"  - {memory['content'][:50]}... (score: {memory['similarity_score']:.2f})")
```

### Graph Analysis

```python
# Analyze memory connections
response = requests.post('http://localhost:8080/mcp',
    headers=headers,
    json={
        'method': 'analyze_memory_graph',
        'params': {
            'analysis_type': 'Connectivity',
            'memory_ids': memory_ids,
            'include_pathways': True,
            'max_depth': 3
        },
        'id': 'analysis_001'
    }
)

analysis = response.json()['result']
print(f"\nGraph Analysis Results:")
print(f"Total nodes: {analysis['graph_metrics']['total_nodes']}")
print(f"Total connections: {analysis['graph_metrics']['total_connections']}")
print(f"Average connectivity: {analysis['graph_metrics']['average_connectivity']:.2f}")

print("\nInsights:")
for insight in analysis['neural_insights']:
    print(f"  - {insight['description']} (confidence: {insight['confidence']:.2f})")
```

## Error Handling

```python
def safe_mcp_request(method, params, request_id):
    try:
        response = requests.post('http://localhost:8080/mcp',
            headers=headers,
            json={
                'method': method,
                'params': params,
                'id': request_id
            },
            timeout=30
        )
        
        if response.status_code == 429:
            print("Rate limit exceeded, waiting...")
            time.sleep(60)
            return safe_mcp_request(method, params, request_id)
        
        response.raise_for_status()
        
        result = response.json()
        if 'error' in result:
            print(f"MCP Error: {result['error']['message']}")
            return None
            
        return result['result']
        
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Usage
result = safe_mcp_request('store_memory', {
    'content': 'Test content',
    'tags': ['test']
}, 'safe_req_001')

if result:
    print(f"Successfully stored: {result['memory_id']}")
```

## Performance Optimization

### Batch Operations

```python
# Store multiple memories efficiently
batch_data = [
    {'content': f'Fact number {i}', 'tags': ['batch', f'item_{i}']}
    for i in range(100)
]

# Use concurrent requests for better throughput
import concurrent.futures
import threading

def store_memory(data, index):
    return safe_mcp_request('store_memory', data, f'batch_{index}')

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(store_memory, data, i) 
        for i, data in enumerate(batch_data)
    ]
    
    results = []
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            results.append(result['memory_id'])

print(f"Stored {len(results)} memories in batch")
```

### Caching Results

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_retrieve(query, strategy='HybridApproach', limit=10):
    # Create cache key
    cache_key = hashlib.md5(
        f"{query}_{strategy}_{limit}".encode()
    ).hexdigest()
    
    result = safe_mcp_request('retrieve_memory', {
        'query': query,
        'strategy': strategy, 
        'limit': limit
    }, f'cached_{cache_key}')
    
    return result

# Multiple calls to same query will use cache
for _ in range(5):
    result = cached_retrieve('capital of France')
    print(f"Found {len(result['memories'])} memories")
```

## Integration Patterns

### Flask Web Application

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

class MemoryService:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def store(self, content, tags=None, confidence=0.8):
        return safe_mcp_request('store_memory', {
            'content': content,
            'tags': tags or [],
            'confidence': confidence
        }, f'web_{hash(content)}')
    
    def search(self, query, limit=10):
        return safe_mcp_request('retrieve_memory', {
            'query': query,
            'limit': limit
        }, f'search_{hash(query)}')

memory_service = MemoryService('http://localhost:8080', token)

@app.route('/api/memories', methods=['POST'])
def create_memory():
    data = request.json
    result = memory_service.store(
        data['content'], 
        data.get('tags'),
        data.get('confidence', 0.8)
    )
    
    if result:
        return jsonify({'memory_id': result['memory_id']})
    else:
        return jsonify({'error': 'Failed to store memory'}), 500

@app.route('/api/search')
def search_memories():
    query = request.args.get('q')
    limit = int(request.args.get('limit', 10))
    
    result = memory_service.search(query, limit)
    
    if result:
        return jsonify({
            'memories': result['memories'],
            'total': result['total_found']
        })
    else:
        return jsonify({'error': 'Search failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Jupyter Notebook Integration

```python
# Memory-enhanced Jupyter notebook
class NotebookMemory:
    def __init__(self, token):
        self.token = token
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def remember(self, content, tags=None):
        """Store content with automatic cell context"""
        import inspect
        
        # Get caller context
        frame = inspect.currentframe().f_back
        cell_id = getattr(frame, 'f_globals', {}).get('_', 'unknown')
        
        tags = (tags or []) + ['jupyter', f'cell_{cell_id}']
        
        result = safe_mcp_request('store_memory', {
            'content': content,
            'tags': tags,
            'context': f'Jupyter notebook cell {cell_id}'
        }, f'nb_{cell_id}')
        
        if result:
            print(f"💾 Remembered: {content[:50]}...")
            return result['memory_id']
    
    def recall(self, query, limit=5):
        """Search memories with rich display"""
        result = safe_mcp_request('retrieve_memory', {
            'query': query,
            'limit': limit,
            'tag_filter': ['jupyter']  # Focus on notebook content
        }, f'recall_{hash(query)}')
        
        if result:
            from IPython.display import display, Markdown
            
            print(f"🧠 Found {len(result['memories'])} memories:")
            for memory in result['memories']:
                score = memory['similarity_score']
                content = memory['content']
                
                display(Markdown(f"""
**Score: {score:.2f}**  
{content}  
*Tags: {', '.join(memory['tags'])}*
---
"""))
        
        return result

# Usage in notebook
nb_memory = NotebookMemory(token)

# Store important findings
nb_memory.remember("Machine learning accuracy improved to 94% with new feature engineering")

# Later, recall related information
nb_memory.recall("machine learning accuracy")
```
```

**File**: `docs/examples/advanced_integration.py`

```python
"""
Advanced CortexKG MCP Server Integration Examples

This module demonstrates sophisticated integration patterns including:
- Async/await patterns for high performance
- Custom neuromorphic analysis
- Real-time learning adaptation
- Multi-tenant memory isolation
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

class MemoryTier(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important" 
    STANDARD = "standard"
    TEMPORARY = "temporary"

@dataclass
class MemoryContext:
    user_id: str
    session_id: str
    domain: str
    tier: MemoryTier = MemoryTier.STANDARD

class AsyncMemoryClient:
    """High-performance async client for CortexKG MCP server"""
    
    def __init__(self, base_url: str, token: str, max_connections: int = 100):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
        self.connector = aiohttp.TCPConnector(limit=max_connections)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, params: dict, request_id: str) -> Optional[dict]:
        """Execute MCP request with error handling and retries"""
        payload = {
            'method': method,
            'params': params,
            'id': request_id
        }
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with self.session.post(f'{self.base_url}/mcp', json=payload) as response:
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'error' in data:
                        print(f"MCP Error: {data['error']['message']}")
                        return None
                    
                    return data.get('result')
                    
            except asyncio.TimeoutError:
                print(f"Request timeout on attempt {attempt + 1}")
                if attempt == 2:  # Last attempt
                    return None
                await asyncio.sleep(1)
            
            except aiohttp.ClientError as e:
                print(f"Request failed: {e}")
                return None
        
        return None
    
    async def store_batch(self, memories: List[Dict], context: MemoryContext) -> List[str]:
        """Store multiple memories concurrently"""
        tasks = []
        
        for i, memory in enumerate(memories):
            # Add context-specific tags
            memory['tags'] = memory.get('tags', []) + [
                f'user:{context.user_id}',
                f'domain:{context.domain}',
                f'tier:{context.tier.value}'
            ]
            
            # Set importance based on tier
            tier_importance = {
                MemoryTier.CRITICAL: 0.95,
                MemoryTier.IMPORTANT: 0.8,
                MemoryTier.STANDARD: 0.6,
                MemoryTier.TEMPORARY: 0.3
            }
            memory['importance'] = tier_importance[context.tier]
            
            task = self._request('store_memory', memory, f'batch_{context.session_id}_{i}')
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        memory_ids = []
        for result in results:
            if isinstance(result, dict) and 'memory_id' in result:
                memory_ids.append(result['memory_id'])
        
        return memory_ids
    
    async def semantic_stream_search(
        self, 
        query: str, 
        context: MemoryContext,
        min_similarity: float = 0.3
    ) -> AsyncGenerator[Dict, None]:
        """Stream search results as they become available"""
        
        # Start with high similarity, gradually lower threshold
        for threshold in [0.9, 0.7, 0.5, min_similarity]:
            result = await self._request('retrieve_memory', {
                'query': query,
                'threshold': threshold,
                'tag_filter': [f'user:{context.user_id}', f'domain:{context.domain}'],
                'limit': 20,
                'strategy': 'SemanticSimilarity'
            }, f'stream_{context.session_id}_{threshold}')
            
            if result and result['memories']:
                for memory in result['memories']:
                    if memory['similarity_score'] >= threshold:
                        yield memory
            
            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.1)

class NeuromorphicAnalyzer:
    """Advanced neuromorphic pattern analysis"""
    
    def __init__(self, client: AsyncMemoryClient):
        self.client = client
    
    async def analyze_learning_patterns(self, context: MemoryContext) -> Dict:
        """Analyze how the system is learning for a specific context"""
        
        # Get system stats
        stats_result = await self.client._request('get_memory_stats', {
            'detailed_breakdown': True,
            'include_performance': True
        }, f'stats_{context.session_id}')
        
        if not stats_result:
            return {}
        
        # Analyze graph connectivity for user's memories
        graph_result = await self.client._request('analyze_memory_graph', {
            'analysis_type': 'Connectivity',
            'include_pathways': True,
            'max_depth': 5
        }, f'graph_{context.session_id}')
        
        # Compute learning metrics
        learning_metrics = {
            'column_efficiency': self._analyze_column_efficiency(stats_result),
            'connectivity_health': self._analyze_connectivity(graph_result),
            'learning_velocity': self._compute_learning_velocity(stats_result),
            'memory_consolidation': self._analyze_consolidation(stats_result)
        }
        
        return learning_metrics
    
    def _analyze_column_efficiency(self, stats: Dict) -> Dict:
        """Analyze how efficiently each cortical column is performing"""
        column_stats = stats.get('column_statistics', [])
        
        efficiency = {}
        for column in column_stats:
            success_rate = (column['successful_allocations'] / 
                          max(column['activations_count'], 1))
            avg_time = column['processing_time_ms_avg']
            
            # Efficiency score: high success rate, low processing time
            efficiency_score = success_rate * (100 / max(avg_time, 1))
            
            efficiency[column['column_type']] = {
                'success_rate': success_rate,
                'avg_processing_time': avg_time,
                'efficiency_score': efficiency_score,
                'recommendation': self._generate_column_recommendation(column)
            }
        
        return efficiency
    
    def _analyze_connectivity(self, graph: Optional[Dict]) -> Dict:
        """Analyze graph connectivity patterns"""
        if not graph:
            return {'status': 'analysis_failed'}
        
        metrics = graph.get('graph_metrics', {})
        
        # Compute connectivity health
        node_count = metrics.get('total_nodes', 0)
        connection_count = metrics.get('total_connections', 0)
        
        if node_count == 0:
            return {'status': 'no_data'}
        
        connectivity_ratio = connection_count / node_count
        clustering = metrics.get('clustering_coefficient', 0)
        efficiency = metrics.get('path_efficiency', 0)
        
        health_score = (connectivity_ratio * 0.4 + clustering * 0.3 + efficiency * 0.3)
        
        return {
            'connectivity_ratio': connectivity_ratio,
            'clustering_coefficient': clustering,
            'path_efficiency': efficiency,
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'needs_attention'
        }
    
    def _compute_learning_velocity(self, stats: Dict) -> float:
        """Compute how quickly the system is learning and adapting"""
        perf_metrics = stats.get('performance_metrics', {})
        
        response_time = perf_metrics.get('avg_response_time_ms', 100)
        throughput = perf_metrics.get('throughput_ops_per_minute', 100)
        error_rate = perf_metrics.get('error_rate_percent', 5)
        
        # Learning velocity: high throughput, low response time, low errors
        velocity = (throughput / response_time) * (1 - error_rate / 100)
        
        return min(velocity, 100)  # Cap at 100
    
    def _analyze_consolidation(self, stats: Dict) -> Dict:
        """Analyze memory consolidation patterns"""
        health = stats.get('health_indicators', {})
        
        return {
            'overall_health': health.get('overall_health_score', 0),
            'neural_integrity': health.get('neural_integrity', 0),
            'synaptic_health': health.get('synaptic_health', 0),
            'allocation_efficiency': health.get('allocation_efficiency', 0),
            'consolidation_status': 'good' if health.get('overall_health_score', 0) > 0.8 else 'monitoring'
        }
    
    def _generate_column_recommendation(self, column: Dict) -> str:
        """Generate optimization recommendations for cortical columns"""
        success_rate = column['successful_allocations'] / max(column['activations_count'], 1)
        avg_time = column['processing_time_ms_avg']
        
        if success_rate < 0.7:
            return "Consider increasing activation threshold"
        elif avg_time > 50:
            return "Enable SIMD acceleration or reduce batch size"
        elif success_rate > 0.95 and avg_time < 10:
            return "Performance optimal"
        else:
            return "Monitor performance trends"

class AdaptiveLearningManager:
    """Manages adaptive learning parameters based on performance"""
    
    def __init__(self, client: AsyncMemoryClient, analyzer: NeuromorphicAnalyzer):
        self.client = client
        self.analyzer = analyzer
        self.performance_history = []
    
    async def optimize_learning_parameters(self, context: MemoryContext) -> Dict:
        """Automatically optimize learning parameters based on performance"""
        
        # Analyze current performance
        learning_patterns = await self.analyzer.analyze_learning_patterns(context)
        
        # Get current configuration
        current_config = await self._get_current_config()
        
        # Compute optimal parameters
        optimal_params = self._compute_optimal_parameters(
            learning_patterns, 
            current_config,
            self.performance_history
        )
        
        # Apply optimizations if significant improvement expected
        if self._should_apply_optimization(optimal_params, current_config):
            result = await self.client._request('configure_learning', optimal_params, 
                                             f'optimize_{context.session_id}')
            
            if result:
                self.performance_history.append({
                    'timestamp': time.time(),
                    'config': optimal_params,
                    'performance': learning_patterns
                })
                
                return {
                    'optimization_applied': True,
                    'new_config': optimal_params,
                    'expected_improvement': self._estimate_improvement(optimal_params)
                }
        
        return {
            'optimization_applied': False,
            'current_performance': learning_patterns,
            'recommendation': 'Continue monitoring'
        }
    
    async def _get_current_config(self) -> Dict:
        """Get current learning configuration"""
        # This would query the current system configuration
        # For demo purposes, return default values
        return {
            'learning_rate': 0.01,
            'decay_rate': 0.001,
            'activation_threshold': 0.6,
            'learning_mode': 'STDP'
        }
    
    def _compute_optimal_parameters(self, patterns: Dict, current: Dict, history: List) -> Dict:
        """Compute optimal learning parameters based on analysis"""
        
        # Base optimization on column efficiency
        column_efficiency = patterns.get('column_efficiency', {})
        avg_efficiency = sum(col.get('efficiency_score', 0) 
                           for col in column_efficiency.values()) / max(len(column_efficiency), 1)
        
        learning_velocity = patterns.get('learning_velocity', 50)
        
        # Adjust learning rate based on performance
        if avg_efficiency < 0.5:
            learning_rate = min(current['learning_rate'] * 1.2, 0.05)  # Increase learning
        elif avg_efficiency > 0.9:
            learning_rate = max(current['learning_rate'] * 0.9, 0.005)  # Decrease learning
        else:
            learning_rate = current['learning_rate']  # Keep current
        
        # Adjust activation threshold based on connectivity
        connectivity = patterns.get('connectivity_health', {})
        health_score = connectivity.get('health_score', 0.5)
        
        if health_score < 0.6:
            activation_threshold = max(current['activation_threshold'] - 0.1, 0.3)
        elif health_score > 0.9:
            activation_threshold = min(current['activation_threshold'] + 0.1, 0.8)
        else:
            activation_threshold = current['activation_threshold']
        
        return {
            'learning_rate': learning_rate,
            'decay_rate': current['decay_rate'],  # Keep stable
            'activation_threshold': activation_threshold,
            'learning_mode': current['learning_mode']  # Keep current mode
        }
    
    def _should_apply_optimization(self, optimal: Dict, current: Dict) -> bool:
        """Determine if optimization should be applied"""
        
        # Check if changes are significant enough
        learning_rate_change = abs(optimal['learning_rate'] - current['learning_rate'])
        threshold_change = abs(optimal['activation_threshold'] - current['activation_threshold'])
        
        return learning_rate_change > 0.002 or threshold_change > 0.05
    
    def _estimate_improvement(self, params: Dict) -> str:
        """Estimate expected performance improvement"""
        # Simple heuristic for demo
        if params['learning_rate'] > 0.02:
            return "Faster adaptation expected"
        elif params['activation_threshold'] < 0.5:
            return "Increased sensitivity expected"
        else:
            return "Marginal improvement expected"

# Example usage
async def main():
    """Demonstration of advanced integration patterns"""
    
    token = "your_jwt_token_here"
    
    async with AsyncMemoryClient('http://localhost:8080', token) as client:
        
        # Set up analysis and learning components
        analyzer = NeuromorphicAnalyzer(client)
        learning_manager = AdaptiveLearningManager(client, analyzer)
        
        # Create memory context for a user session
        context = MemoryContext(
            user_id='user_123',
            session_id='session_456', 
            domain='research',
            tier=MemoryTier.IMPORTANT
        )
        
        # Store batch of related memories
        research_data = [
            {
                'content': 'Neural networks can learn complex patterns through backpropagation',
                'tags': ['machine-learning', 'neural-networks', 'algorithms']
            },
            {
                'content': 'LSTM networks are effective for sequence modeling tasks',
                'tags': ['lstm', 'sequence-modeling', 'deep-learning']
            },
            {
                'content': 'Attention mechanisms have revolutionized natural language processing',
                'tags': ['attention', 'nlp', 'transformers']
            }
        ]
        
        print("Storing research data...")
        memory_ids = await client.store_batch(research_data, context)
        print(f"Stored {len(memory_ids)} memories")
        
        # Perform streaming search
        print("\nStreaming search results:")
        async for memory in client.semantic_stream_search(
            "deep learning neural networks", 
            context,
            min_similarity=0.3
        ):
            print(f"  Found: {memory['content'][:50]}... (score: {memory['similarity_score']:.2f})")
        
        # Analyze learning patterns
        print("\nAnalyzing learning patterns...")
        patterns = await analyzer.analyze_learning_patterns(context)
        
        if patterns:
            print("Column Efficiency:")
            for column, metrics in patterns.get('column_efficiency', {}).items():
                print(f"  {column}: {metrics['efficiency_score']:.2f} - {metrics['recommendation']}")
            
            connectivity = patterns.get('connectivity_health', {})
            print(f"Connectivity Health: {connectivity.get('health_score', 0):.2f}")
            print(f"Learning Velocity: {patterns.get('learning_velocity', 0):.2f}")
        
        # Optimize learning parameters
        print("\nOptimizing learning parameters...")
        optimization = await learning_manager.optimize_learning_parameters(context)
        
        if optimization['optimization_applied']:
            print(f"Applied optimization: {optimization['expected_improvement']}")
            print(f"New config: {optimization['new_config']}")
        else:
            print("No optimization needed at this time")

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria**:
- Comprehensive usage examples from basic to advanced
- Real-world integration patterns (Flask, Jupyter, async)
- Performance optimization techniques demonstrated
- Error handling and retry logic included
- Advanced neuromorphic analysis examples

### Task 9.3: Operational Guides
**Estimated Time**: 20 minutes  
**File**: `docs/operations/deployment_guide.md`

```markdown
# CortexKG MCP Server Deployment Guide

## Prerequisites

### System Requirements

**Minimum (Development)**
- CPU: 2 cores, 2.4GHz
- RAM: 4GB
- Storage: 10GB
- OS: Linux, macOS, Windows

**Recommended (Production)**
- CPU: 4+ cores, 3.0GHz with AVX2 support
- RAM: 8GB+ (16GB for large datasets)
- Storage: 50GB SSD
- OS: Linux (Ubuntu 22.04 LTS recommended)

### Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev curl

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel curl

# macOS
brew install openssl curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

## Production Deployment

### 1. Build Release Binary

```bash
# Clone repository
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server

# Build optimized release
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Verify binary
./target/release/cortex_kg_mcp_server --version
```

### 2. System Configuration

#### Create System User
```bash
sudo useradd -r -s /bin/false cortex-kg
sudo mkdir -p /var/lib/cortex-kg /var/log/cortex-kg /etc/cortex-kg
sudo chown cortex-kg:cortex-kg /var/lib/cortex-kg /var/log/cortex-kg
```

#### Generate Security Keys
```bash
# JWT signing key
sudo openssl rand -hex 32 > /etc/cortex-kg/jwt.key
sudo chown cortex-kg:cortex-kg /etc/cortex-kg/jwt.key
sudo chmod 600 /etc/cortex-kg/jwt.key

# TLS certificates (if using HTTPS)
sudo openssl req -x509 -newkey rsa:4096 -keyout /etc/cortex-kg/server.key \
  -out /etc/cortex-kg/server.crt -days 365 -nodes
```

#### Install Binary and Configuration
```bash
sudo cp target/release/cortex_kg_mcp_server /usr/local/bin/
sudo cp config/production.toml /etc/cortex-kg/
sudo chown cortex-kg:cortex-kg /etc/cortex-kg/production.toml
```

### 3. Systemd Service

Create `/etc/systemd/system/cortex-kg-mcp.service`:

```ini
[Unit]
Description=CortexKG MCP Server
After=network.target
Wants=network.target

[Service]
Type=exec
User=cortex-kg
Group=cortex-kg
ExecStart=/usr/local/bin/cortex_kg_mcp_server --config /etc/cortex-kg/production.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cortex-kg-mcp

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/cortex-kg /var/log/cortex-kg

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

#### Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cortex-kg-mcp
sudo systemctl start cortex-kg-mcp

# Check status
sudo systemctl status cortex-kg-mcp
```

## Docker Deployment

### 1. Production Docker Setup

```dockerfile
# Dockerfile.production
FROM rust:1.75-slim as builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates libssl3 curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 cortex && \
    mkdir -p /var/lib/cortex-kg /var/log/cortex-kg && \
    chown -R cortex:cortex /var/lib/cortex-kg /var/log/cortex-kg

WORKDIR /app
COPY --from=builder /app/target/release/cortex_kg_mcp_server /usr/local/bin/
COPY --chown=cortex:cortex config/ /app/config/

USER cortex

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 9090
CMD ["cortex_kg_mcp_server", "--config", "/app/config/production.toml"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  cortex-kg-mcp:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: cortex-kg-mcp
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./config/production.toml:/app/config/production.toml:ro
      - ./secrets:/etc/cortex-kg:ro
      - cortex-data:/var/lib/cortex-kg
      - cortex-logs:/var/log/cortex-kg
    environment:
      - RUST_LOG=info
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

volumes:
  cortex-data:
  cortex-logs:
```

### 3. Container Management

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f cortex-kg-mcp

# Scale for high availability
docker-compose up -d --scale cortex-kg-mcp=3

# Update deployment
docker-compose pull
docker-compose up -d --no-deps cortex-kg-mcp
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cortex-kg

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cortex-kg-config
  namespace: cortex-kg
data:
  production.toml: |
    [server]
    host = "0.0.0.0"
    port = 8080
    max_connections = 1000
    
    [neuromorphic]
    ttfs_precision_ms = 0.1
    cortical_columns = 4
    
    [performance]
    enable_simd = true
    cache_size_mb = 1024
    
    [security]
    enable_oauth = true
    jwt_secret_path = "/etc/secrets/jwt.key"
```

### 2. Deployment and Service

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-kg-mcp
  namespace: cortex-kg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cortex-kg-mcp
  template:
    metadata:
      labels:
        app: cortex-kg-mcp
    spec:
      containers:
      - name: cortex-kg-mcp
        image: ghcr.io/your-org/cortex-kg-mcp:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
        - name: data
          mountPath: /var/lib/cortex-kg
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: cortex-kg-config
      - name: secrets
        secret:
          secretName: cortex-kg-secrets
      - name: data
        persistentVolumeClaim:
          claimName: cortex-kg-data

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cortex-kg-mcp-service
  namespace: cortex-kg
spec:
  selector:
    app: cortex-kg-mcp
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### 3. Ingress and TLS

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cortex-kg-mcp-ingress
  namespace: cortex-kg
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.cortex-kg.example.com
    secretName: cortex-kg-tls
  rules:
  - host: api.cortex-kg.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cortex-kg-mcp-service
            port:
              number: 8080
```

## Load Balancing and High Availability

### 1. Nginx Load Balancer

```nginx
# /etc/nginx/sites-available/cortex-kg-mcp
upstream cortex_kg_backend {
    least_conn;
    server 10.0.1.10:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.cortex-kg.example.com;

    ssl_certificate /etc/ssl/certs/cortex-kg.crt;
    ssl_certificate_key /etc/ssl/private/cortex-kg.key;

    location / {
        proxy_pass http://cortex_kg_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        access_log off;
        proxy_pass http://cortex_kg_backend;
        proxy_connect_timeout 2s;
        proxy_send_timeout 2s;
        proxy_read_timeout 2s;
    }

    location /metrics {
        access_log off;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://cortex_kg_backend;
    }
}
```

### 2. HAProxy Configuration

```
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend cortex_kg_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/cortex-kg.pem
    redirect scheme https if !{ ssl_fc }
    default_backend cortex_kg_servers

backend cortex_kg_servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server cortex1 10.0.1.10:8080 check inter 10s
    server cortex2 10.0.1.11:8080 check inter 10s
    server cortex3 10.0.1.12:8080 check inter 10s

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
```

## Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp comment 'CortexKG MCP'
sudo ufw allow 9090/tcp comment 'Metrics'
sudo ufw enable

# iptables
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -j DROP
```

### 2. TLS Configuration

```bash
# Generate strong TLS certificate
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt \
  -days 365 -nodes -subj "/CN=api.cortex-kg.example.com"

# Set proper permissions
chmod 600 server.key
chmod 644 server.crt
```

### 3. OAuth 2.1 Setup

```bash
# Environment variables
export OAUTH_CLIENT_ID="your_client_id"
export OAUTH_CLIENT_SECRET="your_client_secret"
export OAUTH_AUTH_URL="https://auth.provider.com/oauth2/authorize"
export OAUTH_TOKEN_URL="https://auth.provider.com/oauth2/token"
export OAUTH_REDIRECT_URL="https://api.cortex-kg.example.com/auth/callback"
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cortex-kg-mcp'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 10s
    metrics_path: /metrics
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "CortexKG MCP Server",
    "panels": [
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "cortex_kg_response_time_ms"
          }
        ]
      },
      {
        "title": "Throughput",
        "targets": [
          {
            "expr": "rate(cortex_kg_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats cortex-kg-mcp
   
   # Reduce cache size in config
   cache_size_mb = 256
   ```

2. **Slow Response Times**
   ```bash
   # Enable SIMD acceleration
   enable_simd = true
   
   # Increase connection pool
   connection_pool_size = 200
   ```

3. **Authentication Failures**
   ```bash
   # Check JWT secret
   ls -la /etc/cortex-kg/jwt.key
   
   # Verify OAuth configuration
   curl -v https://auth.provider.com/.well-known/openid_configuration
   ```

### Log Analysis

```bash
# Check service logs
journalctl -u cortex-kg-mcp -f

# Docker logs
docker logs -f cortex-kg-mcp

# Search for errors
grep -i error /var/log/cortex-kg/server.log

# Performance analysis
grep "response_time" /var/log/cortex-kg/server.log | tail -100
```
```

**File**: `docs/operations/monitoring_guide.md`

```markdown
# Monitoring and Observability Guide

## Metrics Overview

The CortexKG MCP server exposes comprehensive metrics for monitoring performance, health, and operational status.

### Core Metrics

#### Performance Metrics
- `cortex_kg_response_time_ms` - Request processing time
- `cortex_kg_throughput_ops_per_minute` - Operations throughput
- `cortex_kg_requests_total` - Total request count by method
- `cortex_kg_errors_total` - Error count by type

#### Neuromorphic Metrics
- `cortex_kg_cortical_activation_time_ms` - Column processing time
- `cortex_kg_consensus_formation_time_ms` - Lateral inhibition time
- `cortex_kg_ttfs_encoding_time_ms` - Pattern encoding time
- `cortex_kg_stdp_updates_total` - Learning updates count

#### Resource Metrics
- `cortex_kg_memory_usage_bytes` - Memory consumption
- `cortex_kg_connection_pool_active` - Active connections
- `cortex_kg_cache_hit_rate` - Cache performance
- `cortex_kg_simd_operations_total` - SIMD acceleration usage

## Prometheus Integration

### 1. Metrics Endpoint

The server exposes metrics at `http://localhost:9090/metrics`:

```
# HELP cortex_kg_response_time_ms Average response time in milliseconds
# TYPE cortex_kg_response_time_ms gauge
cortex_kg_response_time_ms{method="store_memory"} 45.2

# HELP cortex_kg_throughput_ops_per_minute Operations per minute
# TYPE cortex_kg_throughput_ops_per_minute gauge
cortex_kg_throughput_ops_per_minute 1250.0

# HELP cortex_kg_cortical_activation_time_ms Cortical column activation time
# TYPE cortex_kg_cortical_activation_time_ms gauge
cortex_kg_cortical_activation_time_ms{column="semantic"} 12.5
cortex_kg_cortical_activation_time_ms{column="structural"} 15.8
```

### 2. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "cortex_kg_rules.yml"

scrape_configs:
  - job_name: 'cortex-kg-mcp'
    static_configs:
      - targets: ['cortex-kg-mcp:9090']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'cortex-kg-health'
    static_configs:
      - targets: ['cortex-kg-mcp:8080']
    metrics_path: /health
    scrape_interval: 30s
```

### 3. Alerting Rules

```yaml
# cortex_kg_rules.yml
groups:
  - name: cortex_kg_alerts
    rules:
      - alert: HighResponseTime
        expr: cortex_kg_response_time_ms > 200
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CortexKG response time is high"
          description: "Response time {{ $value }}ms exceeds threshold"

      - alert: LowThroughput
        expr: cortex_kg_throughput_ops_per_minute < 500
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "CortexKG throughput is low"
          description: "Throughput {{ $value }} ops/min below threshold"

      - alert: HighErrorRate
        expr: rate(cortex_kg_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "CortexKG error rate is high"
          description: "Error rate {{ $value }} exceeds 10%"

      - alert: ServiceDown
        expr: up{job="cortex-kg-mcp"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CortexKG service is down"
          description: "CortexKG MCP server is not responding"

      - alert: LowCacheHitRate
        expr: cortex_kg_cache_hit_rate < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate is low"
          description: "Cache hit rate {{ $value }} below 70%"
```

## Grafana Dashboards

### 1. Main Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "CortexKG MCP Server Overview",
    "tags": ["cortex-kg", "mcp"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "cortex_kg_response_time_ms",
            "legendFormat": "{{method}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms",
            "min": 0,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 100},
                {"color": "red", "value": 200}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "cortex_kg_throughput_ops_per_minute",
            "legendFormat": "Ops/Min"
          }
        ],
        "yAxes": [
          {
            "label": "Operations per Minute",
            "min": 0
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cortex_kg_errors_total[5m]) * 100",
            "legendFormat": "{{error_type}}"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate %",
            "min": 0,
            "max": 10
          }
        ]
      },
      {
        "id": 4,
        "title": "Cortical Column Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "cortex_kg_cortical_activation_time_ms",
            "legendFormat": "{{column}}"
          }
        ],
        "yAxes": [
          {
            "label": "Activation Time (ms)",
            "min": 0
          }
        ]
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cortex_kg_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory MB"
          }
        ],
        "yAxes": [
          {
            "label": "Memory Usage (MB)",
            "min": 0
          }
        ]
      },
      {
        "id": 6,
        "title": "Cache Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "cortex_kg_cache_hit_rate * 100",
            "legendFormat": "Hit Rate %"
          }
        ],
        "yAxes": [
          {
            "label": "Cache Hit Rate %",
            "min": 0,
            "max": 100
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

### 2. Neuromorphic Dashboard

```json
{
  "dashboard": {
    "title": "CortexKG Neuromorphic Analysis",
    "panels": [
      {
        "title": "Column Activation Balance",
        "type": "piechart",
        "targets": [
          {
            "expr": "cortex_kg_column_activations_total",
            "legendFormat": "{{column}}"
          }
        ]
      },
      {
        "title": "Learning Rate Trends",
        "type": "graph",
        "targets": [
          {
            "expr": "cortex_kg_learning_rate",
            "legendFormat": "Learning Rate"
          }
        ]
      },
      {
        "title": "Synaptic Changes",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cortex_kg_stdp_updates_total[5m])",
            "legendFormat": "STDP Updates/sec"
          }
        ]
      }
    ]
  }
}
```

## Health Checks

### 1. HTTP Health Endpoint

```bash
# Basic health check
curl http://localhost:8080/health

# Response
{
  "status": "healthy",
  "health_score": 0.92,
  "response_time_ms": 45.2,
  "throughput_ops_per_min": 1250.0,
  "error_rate_percent": 0.5,
  "memory_usage_mb": 512.8,
  "uptime_seconds": 86400
}
```

### 2. Deep Health Check Script

```bash
#!/bin/bash
# health_check.sh - Comprehensive health verification

ENDPOINT="http://localhost:8080"
THRESHOLD_RESPONSE_TIME=100
THRESHOLD_ERROR_RATE=5
THRESHOLD_MEMORY_MB=2048

echo "🔍 CortexKG Health Check"
echo "======================="

# Check if service is responding
if ! curl -sf "$ENDPOINT/health" > /dev/null; then
    echo "❌ Service not responding"
    exit 1
fi

# Get health data
HEALTH_DATA=$(curl -s "$ENDPOINT/health")
STATUS=$(echo "$HEALTH_DATA" | jq -r '.status')
RESPONSE_TIME=$(echo "$HEALTH_DATA" | jq -r '.response_time_ms')
ERROR_RATE=$(echo "$HEALTH_DATA" | jq -r '.error_rate_percent')
MEMORY_MB=$(echo "$HEALTH_DATA" | jq -r '.memory_usage_mb')
HEALTH_SCORE=$(echo "$HEALTH_DATA" | jq -r '.health_score')

echo "Status: $STATUS"
echo "Health Score: $HEALTH_SCORE"
echo "Response Time: ${RESPONSE_TIME}ms"
echo "Error Rate: ${ERROR_RATE}%"
echo "Memory Usage: ${MEMORY_MB}MB"

# Check thresholds
WARNINGS=0

if (( $(echo "$RESPONSE_TIME > $THRESHOLD_RESPONSE_TIME" | bc -l) )); then
    echo "⚠️  High response time: ${RESPONSE_TIME}ms"
    WARNINGS=$((WARNINGS + 1))
fi

if (( $(echo "$ERROR_RATE > $THRESHOLD_ERROR_RATE" | bc -l) )); then
    echo "⚠️  High error rate: ${ERROR_RATE}%"
    WARNINGS=$((WARNINGS + 1))
fi

if (( $(echo "$MEMORY_MB > $THRESHOLD_MEMORY_MB" | bc -l) )); then
    echo "⚠️  High memory usage: ${MEMORY_MB}MB"
    WARNINGS=$((WARNINGS + 1))
fi

if (( $(echo "$HEALTH_SCORE < 0.8" | bc -l) )); then
    echo "⚠️  Low health score: $HEALTH_SCORE"
    WARNINGS=$((WARNINGS + 1))
fi

# Final result
if [ "$STATUS" = "healthy" ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All health checks passed"
    exit 0
elif [ $WARNINGS -gt 0 ]; then
    echo "⚠️  Health check passed with $WARNINGS warnings"
    exit 1
else
    echo "❌ Health check failed"
    exit 2
fi
```

## Log Management

### 1. Structured Logging

```toml
# In production.toml
[logging]
level = "info"
format = "json"
output = "file"
file_path = "/var/log/cortex-kg/server.log"
max_size_mb = 100
max_files = 10
```

### 2. Log Analysis with ELK Stack

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/cortex-kg/*.log
  json.keys_under_root: true
  json.add_error_key: true
  fields:
    service: cortex-kg-mcp
    environment: production

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "cortex-kg-%{+yyyy.MM.dd}"

setup.template.settings:
  index.number_of_shards: 1
  index.number_of_replicas: 0
```

### 3. Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [service] == "cortex-kg-mcp" {
    if [level] == "ERROR" {
      mutate {
        add_tag => ["error"]
      }
    }
    
    if [response_time_ms] {
      if [response_time_ms] > 100 {
        mutate {
          add_tag => ["slow_response"]
        }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "cortex-kg-%{+YYYY.MM.dd}"
  }
}
```

## Performance Monitoring

### 1. Custom Metrics Collection

```python
# metrics_collector.py
import requests
import time
import json
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class CortexKGMetricsCollector:
    def __init__(self, endpoint, pushgateway_url):
        self.endpoint = endpoint
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()
        
        # Define custom metrics
        self.health_score = Gauge(
            'cortex_kg_health_score', 
            'Overall health score',
            registry=self.registry
        )
        
        self.column_efficiency = Gauge(
            'cortex_kg_column_efficiency',
            'Cortical column efficiency',
            ['column'],
            registry=self.registry
        )
    
    def collect_metrics(self):
        try:
            # Get health data
            health_response = requests.get(f"{self.endpoint}/health", timeout=10)
            health_data = health_response.json()
            
            self.health_score.set(health_data.get('health_score', 0))
            
            # Get detailed stats
            stats_response = requests.post(f"{self.endpoint}/mcp", json={
                'method': 'get_memory_stats',
                'params': {'detailed_breakdown': True},
                'id': f'metrics_{int(time.time())}'
            }, timeout=30)
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json().get('result', {})
                
                # Column efficiency metrics
                for column_stat in stats_data.get('column_statistics', []):
                    efficiency = (column_stat['successful_allocations'] / 
                                max(column_stat['activations_count'], 1))
                    self.column_efficiency.labels(
                        column=column_stat['column_type']
                    ).set(efficiency)
            
            # Push to Prometheus pushgateway
            push_to_gateway(
                self.pushgateway_url,
                job='cortex-kg-custom-metrics',
                registry=self.registry
            )
            
        except Exception as e:
            print(f"Metrics collection failed: {e}")

# Usage
collector = CortexKGMetricsCollector(
    'http://localhost:8080',
    'http://pushgateway:9091'
)

while True:
    collector.collect_metrics()
    time.sleep(60)  # Collect every minute
```

### 2. Performance Baseline

```bash
# performance_baseline.sh - Establish performance baselines

echo "📊 CortexKG Performance Baseline"
echo "================================"

# Warm up the system
echo "Warming up system..."
for i in {1..10}; do
    curl -s http://localhost:8080/health > /dev/null
done

# Measure baseline performance
echo "Measuring baseline performance..."

# Simple memory storage
echo "Testing memory storage..."
STORE_TIME=$(curl -w "%{time_total}" -s -o /dev/null -X POST \
  http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "store_memory", "params": {"content": "Test baseline memory"}}')

echo "Memory storage time: ${STORE_TIME}s"

# Memory retrieval
echo "Testing memory retrieval..."
RETRIEVE_TIME=$(curl -w "%{time_total}" -s -o /dev/null -X POST \
  http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "retrieve_memory", "params": {"query": "test"}}')

echo "Memory retrieval time: ${RETRIEVE_TIME}s"

# System stats
echo "Testing system stats..."
STATS_TIME=$(curl -w "%{time_total}" -s -o /dev/null -X POST \
  http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "get_memory_stats", "params": {}}')

echo "System stats time: ${STATS_TIME}s"

echo ""
echo "✅ Baseline measurements complete"
echo "Store: ${STORE_TIME}s | Retrieve: ${RETRIEVE_TIME}s | Stats: ${STATS_TIME}s"
```

This comprehensive documentation provides everything needed to successfully deploy, monitor, and maintain the CortexKG MCP server in production environments.
```

**Success Criteria**:
- Complete deployment guide for multiple environments
- Comprehensive monitoring and observability setup
- Health check implementation and automation
- Performance baseline establishment and tracking
- Security hardening and operational procedures

### Task 9.4: Final Documentation Assembly
**Estimated Time**: 15 minutes  
**File**: `docs/README.md`

```markdown
# CortexKG MCP Server Documentation

Welcome to the comprehensive documentation for the CortexKG Model Context Protocol (MCP) server. This neuromorphic memory system provides AI agents with biologically-inspired memory capabilities through a standardized protocol interface.

## 📖 Documentation Overview

### Quick Start
- [**Installation Guide**](installation.md) - Get up and running quickly
- [**Basic Usage Examples**](examples/basic_usage.md) - Simple integration patterns
- [**Configuration Reference**](configuration.md) - Server configuration options

### API Reference
- [**API Documentation**](api/README.md) - Complete tool reference and schemas
- [**Neuromorphic Concepts**](api/neuromorphic_concepts.md) - Understanding the brain-inspired architecture
- [**Authentication Guide**](api/authentication.md) - OAuth 2.1 and JWT implementation

### Examples & Tutorials
- [**Basic Usage**](examples/basic_usage.md) - Simple memory operations
- [**Advanced Integration**](examples/advanced_integration.py) - High-performance async patterns
- [**Jupyter Integration**](examples/jupyter_memory.ipynb) - Notebook-enhanced workflows
- [**Web Application Example**](examples/flask_integration.py) - REST API integration

### Operations & Deployment
- [**Deployment Guide**](operations/deployment_guide.md) - Production deployment strategies
- [**Monitoring Guide**](operations/monitoring_guide.md) - Observability and alerting
- [**Security Guide**](operations/security.md) - Hardening and best practices
- [**Troubleshooting**](operations/troubleshooting.md) - Common issues and solutions

### Development
- [**Architecture Overview**](development/architecture.md) - System design and components
- [**Contributing Guide**](development/contributing.md) - How to contribute
- [**Testing Guide**](development/testing.md) - Test suite and quality assurance

## 🚀 Quick Start

### Installation

```bash
# Clone and build
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server
cargo build --release

# Configure
cp config/development.toml config/local.toml
# Edit config/local.toml with your settings

# Generate JWT secret
openssl rand -hex 32 > config/jwt.key

# Start server
./target/release/cortex_kg_mcp_server --config config/local.toml
```

### First Memory Operation

```bash
# Health check
curl http://localhost:8080/health

# Store a memory (requires authentication in production)
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "store_memory",
    "params": {
      "content": "The Pacific Ocean is the largest ocean on Earth",
      "tags": ["geography", "ocean"]
    },
    "id": "test_001"
  }'

# Search for memories
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "retrieve_memory", 
    "params": {
      "query": "largest ocean",
      "limit": 5
    },
    "id": "test_002"
  }'
```

## 🧠 Neuromorphic Architecture

### Core Concepts

The CortexKG MCP server implements a **4-column cortical architecture** inspired by neuroscience:

1. **Semantic Column**: Processes conceptual meaning and relationships
2. **Structural Column**: Optimizes graph topology and connections  
3. **Temporal Column**: Handles time-based patterns and sequences
4. **Exception Column**: Detects contradictions and anomalies

### Processing Pipeline

```
Input → TTFS Encoding → 4 Cortical Columns → Lateral Inhibition → Consensus → Memory Allocation
```

Each memory operation involves:
1. **Encoding**: Convert input to Time-To-First-Spike (TTFS) patterns
2. **Parallel Processing**: All columns analyze simultaneously  
3. **Competition**: Lateral inhibition determines winning strategy
4. **Learning**: STDP updates strengthen successful pathways

### Performance Characteristics

- **Response Time**: <100ms average, <200ms P95
- **Throughput**: >1,000 operations/minute
- **Memory Efficiency**: <2GB for 1M memories
- **Accuracy**: >95% allocation success rate

## 🛠 Available Tools

### Core Memory Operations

| Tool | Purpose | Required Permission |
|------|---------|-------------------|
| `store_memory` | Store new information | `cortex_kg:write` |
| `retrieve_memory` | Search and retrieve | `cortex_kg:read` |
| `update_memory` | Modify existing | `cortex_kg:write` |
| `delete_memory` | Remove memories | `cortex_kg:write` |

### Analysis & Management

| Tool | Purpose | Required Permission |
|------|---------|-------------------|
| `analyze_memory_graph` | Graph analysis | `cortex_kg:read` |
| `get_memory_stats` | System metrics | `cortex_kg:read` |
| `configure_learning` | Tune parameters | `cortex_kg:admin` |

### Retrieval Strategies

- **SemanticSimilarity**: Concept-based matching
- **StructuralMatching**: Graph topology analysis
- **TemporalProximity**: Time-based relevance
- **HybridApproach**: Multi-column consensus (default)
- **SpreadingActivation**: Neural propagation

## 🔐 Security & Authentication

### OAuth 2.1 Flow

1. **Authorization**: Get authorization URL
2. **User Consent**: User authorizes application
3. **Token Exchange**: Exchange code for access token
4. **API Access**: Use Bearer token for requests

### Permission Levels

- **Read**: Access retrieval and analysis tools
- **Write**: All read permissions plus memory modifications
- **Admin**: All permissions plus system configuration

### Rate Limiting

| Tier | Requests/min | Memory Ops/min | Burst Limit |
|------|-------------|----------------|-------------|
| Standard | 60 | 30 | 10 |
| Premium | 120 | 100 | 20 |
| Enterprise | 300 | 500 | 50 |

## 📊 Monitoring & Observability

### Health Endpoint

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "health_score": 0.92,
  "response_time_ms": 45.2,
  "throughput_ops_per_min": 1250.0,
  "error_rate_percent": 0.5,
  "memory_usage_mb": 512.8,
  "uptime_seconds": 86400
}
```

### Prometheus Metrics

The server exposes metrics at `/metrics`:
- Response times and throughput
- Cortical column performance
- Memory usage and cache efficiency  
- Error rates and success ratios

### Grafana Dashboards

Pre-built dashboards available for:
- System overview and performance
- Neuromorphic analysis and learning
- Resource utilization and health

## 🔧 Configuration

### Environment-Specific Configs

- **Development**: `config/development.toml` - Local testing
- **Staging**: `config/staging.toml` - Pre-production validation
- **Production**: `config/production.toml` - Optimized for scale

### Key Configuration Sections

```toml
[server]
host = "0.0.0.0"
port = 8080
max_connections = 1000

[neuromorphic]
ttfs_precision_ms = 0.1
cortical_columns = 4
stdp_learning_rate = 0.01

[performance]
enable_simd = true
cache_size_mb = 1024
connection_pool_size = 200

[security]
enable_oauth = true
jwt_secret_path = "/etc/cortex-kg/jwt.key"
rate_limit_per_minute = 1000
```

## 🚢 Deployment Options

### Docker
```bash
docker run -d \
  --name cortex-kg-mcp \
  -p 8080:8080 \
  -v ./config:/app/config \
  ghcr.io/your-org/cortex-kg-mcp:latest
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Systemd Service
```bash
sudo systemctl enable cortex-kg-mcp
sudo systemctl start cortex-kg-mcp
```

## 🧪 Testing

### Run Test Suite
```bash
# All tests
cargo test

# Specific categories
cargo test --test unit_tests
cargo test --test integration_tests
cargo test --test performance_tests

# With test runner
./target/debug/test_runner
```

### Performance Benchmarks
```bash
# Establish baseline
./scripts/performance_baseline.sh

# Load testing
./scripts/load_test.sh --concurrent 50 --duration 300
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for:
- Development setup
- Code standards and review process
- Testing requirements
- Documentation guidelines

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/cortex-kg-mcp-server
cd cortex-kg-mcp-server

# Install dependencies
cargo build

# Run tests
cargo test

# Start development server
cargo run -- --config config/development.toml
```

## 📋 Changelog

### Version 1.0.0 (Production Release)
- ✅ Complete neuromorphic 4-column architecture
- ✅ All 7 MCP tools implemented
- ✅ OAuth 2.1 authentication and authorization
- ✅ SIMD acceleration and performance optimization
- ✅ Comprehensive monitoring and observability
- ✅ Production deployment configurations
- ✅ Complete documentation and examples

## 📞 Support

### Documentation
- [API Reference](api/README.md)
- [Operations Guide](operations/deployment_guide.md)
- [Troubleshooting](operations/troubleshooting.md)

### Community
- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and share ideas
- Documentation Wiki: Community-maintained guides

### Professional Support
- Enterprise support available
- Custom integration assistance
- Performance optimization consulting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**CortexKG MCP Server** - Bringing neuromorphic memory to AI agents through standardized protocols.

*Built with 🧠 for the future of AI memory systems.*
```

**Success Criteria**:
- Complete documentation overview with clear navigation
- Quick start guide for immediate usage
- Comprehensive coverage of all features and capabilities
- Clear examples and usage patterns
- Production deployment guidance
- Troubleshooting and support information

## Validation Checklist

- [ ] Complete API documentation with all 7 tools
- [ ] Neuromorphic concepts explained clearly for developers
- [ ] Comprehensive usage examples from basic to advanced
- [ ] Production deployment guide with multiple platforms
- [ ] Monitoring and observability setup documented
- [ ] Security and authentication properly explained
- [ ] Performance characteristics and optimization covered
- [ ] Troubleshooting guides for common issues
- [ ] Contributing guidelines and development setup
- [ ] Professional documentation structure and navigation

## Next Phase Dependencies

This phase completes the documentation for:
- Production deployment with confidence
- Developer adoption and integration
- Operational maintenance and monitoring
- Community contribution and support
- Long-term project sustainability

---

The Phase 8 micro-phases breakdown is now complete! This comprehensive structure provides 450+ detailed, AI-actionable tasks organized into 9 logical phases that can be executed systematically to build the complete CortexKG MCP server. Each micro-phase has clear dependencies, success criteria, and validation checklists to ensure quality and completeness.