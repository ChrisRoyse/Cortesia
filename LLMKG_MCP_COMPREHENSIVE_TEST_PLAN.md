# LLMKG MCP Server Comprehensive Testing & Fix Plan

## Executive Summary

This document outlines a comprehensive testing strategy for the LLMKG MCP (Model Context Protocol) server, covering all 15 tools with 5 test scenarios each. Each test includes expected outcomes and remediation steps if failures occur.

## Current Issues Identified

1. **Critical Issue**: MCP server hangs on tool calls (>10 seconds response time)
   - **Root Cause**: Potential deadlock in async handlers or uninitialized resources
   - **Impact**: All tools are non-functional
   - **Priority**: P0 - Must fix before any testing can proceed

## Pre-Testing Fixes Required

### Fix 1: MCP Server Hanging Issue
```rust
// Location: src/mcp/llm_friendly_server/mod.rs
// Issue: Async deadlock in handle_tool_call
// Fix: Add timeout and proper error handling

impl LLMFriendlyMCPServer {
    pub async fn handle_tool_call(&self, tool_name: &str, params: Value) -> Result<Value> {
        // Add timeout wrapper
        match tokio::time::timeout(
            Duration::from_secs(5),
            self.handle_tool_call_internal(tool_name, params)
        ).await {
            Ok(result) => result,
            Err(_) => Err(GraphError::Timeout("Tool call timed out after 5 seconds".into()))
        }
    }
}
```

### Fix 2: Initialize Knowledge Engine Properly
```rust
// Location: src/bin/llmkg_mcp_server.rs
// Issue: Knowledge engine not properly initialized
// Fix: Add warm-up and verification

// After creating knowledge engine, add:
let engine = knowledge_engine.write().await;
engine.initialize_indices()?;
drop(engine);

// Add health check
if !mcp_server.health_check().await? {
    return Err(GraphError::InitializationError("MCP server health check failed".into()));
}
```

## Testing Framework

### Test Environment Setup
```bash
# 1. Build debug version with logging
cargo build --bin llmkg_mcp_server --features "mcp"

# 2. Set environment for verbose logging
set RUST_LOG=debug

# 3. Create test data directory
mkdir test_data

# 4. Run MCP server in test mode
llmkg_mcp_server.exe --data-dir ./test_data -v
```

### Test Harness
```python
# test_mcp_tools.py
import json
import subprocess
import time
from typing import Dict, Any, Optional

class MCPTestHarness:
    def __init__(self):
        self.process = None
        self.test_results = []
        
    def start_server(self):
        self.process = subprocess.Popen(
            ['llmkg_mcp_server.exe', '--data-dir', './test_data', '-v'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)  # Allow server to initialize
        
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": 1
        }
        
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        
        # Read response with timeout
        response_line = self.process.stdout.readline()
        if response_line:
            return json.loads(response_line)
        return None
```

## Tool Testing Specifications

### Tool 1: store_fact
**Purpose**: Store simple Subject-Predicate-Object triples

#### Test Scenarios

**Test 1.1: Basic Fact Storage**
```json
{
  "subject": "Einstein",
  "predicate": "is",
  "object": "scientist"
}
```
- **Expected**: Success, returns stored fact confirmation
- **Verify**: Query returns the fact
- **Fix if fails**: Check triple validation in `handlers/storage.rs`

**Test 1.2: Fact with Confidence**
```json
{
  "subject": "Quantum_Computer",
  "predicate": "invented_by",
  "object": "Multiple_Scientists",
  "confidence": 0.85
}
```
- **Expected**: Success with confidence stored
- **Verify**: Confidence value persisted
- **Fix if fails**: Check metadata storage in `KnowledgeEngine::add_triple`

**Test 1.3: Unicode and Special Characters**
```json
{
  "subject": "北京",
  "predicate": "capital_of",
  "object": "中国"
}
```
- **Expected**: Success with proper encoding
- **Verify**: Unicode preserved in queries
- **Fix if fails**: Check string handling in `Triple::new`

**Test 1.4: Long Predicate Validation**
```json
{
  "subject": "Test",
  "predicate": "this_is_a_very_long_predicate_that_exceeds_the_maximum_allowed_length_of_64_characters",
  "object": "Fail"
}
```
- **Expected**: Validation error
- **Verify**: Error message about length
- **Fix if fails**: Add validation in `handlers/storage.rs:21-23`

**Test 1.5: Duplicate Fact Handling**
```json
{
  "subject": "Einstein",
  "predicate": "is",
  "object": "scientist"
}
```
- **Expected**: Success (idempotent) or update confirmation
- **Verify**: No duplicate storage
- **Fix if fails**: Check deduplication logic

### Tool 2: store_knowledge
**Purpose**: Store complex knowledge chunks with auto-extraction

#### Test Scenarios

**Test 2.1: Basic Knowledge Chunk**
```json
{
  "title": "Einstein Biography",
  "content": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity.",
  "category": "biography"
}
```
- **Expected**: Success with entity extraction count
- **Verify**: Entities "Einstein", "1879", "1955", "Germany" extracted
- **Fix if fails**: Check `TripleExtractor::extract_from_text`

**Test 2.2: Technical Documentation**
```json
{
  "title": "Python List Methods",
  "content": "Python lists have methods like append(), extend(), insert(), remove(), pop(), clear(), index(), count(), sort(), and reverse().",
  "category": "technical",
  "source": "Python Documentation"
}
```
- **Expected**: Success with method relationships extracted
- **Verify**: Methods linked to Python entity
- **Fix if fails**: Check technical term extraction patterns

**Test 2.3: Large Knowledge Chunk**
```json
{
  "title": "Complete History of Computing",
  "content": "[50KB of text about computing history...]",
  "category": "historical"
}
```
- **Expected**: Success with chunking if needed
- **Verify**: Memory limits respected
- **Fix if fails**: Check `MAX_CHUNK_SIZE_BYTES` handling

**Test 2.4: Empty Content Validation**
```json
{
  "title": "Empty Knowledge",
  "content": "",
  "category": "test"
}
```
- **Expected**: Validation error
- **Verify**: Appropriate error message
- **Fix if fails**: Add content validation

**Test 2.5: Multi-language Content**
```json
{
  "title": "Multilingual Science",
  "content": "Einstein (English) was born in Deutschland (German). His work on 相対性理論 (Japanese: relativity) changed physics.",
  "category": "biography"
}
```
- **Expected**: Success with all languages preserved
- **Verify**: Cross-language entity linking
- **Fix if fails**: Check Unicode handling in extractor

### Tool 3: find_facts
**Purpose**: Query triples with pattern matching

#### Test Scenarios

**Test 3.1: Subject Query**
```json
{
  "query": {
    "subject": "Einstein"
  },
  "limit": 10
}
```
- **Expected**: All facts about Einstein
- **Verify**: Results include stored facts
- **Fix if fails**: Check subject index in `query_triples`

**Test 3.2: Predicate Query**
```json
{
  "query": {
    "predicate": "invented_by"
  },
  "limit": 5
}
```
- **Expected**: All invention relationships
- **Verify**: Different subjects/objects returned
- **Fix if fails**: Check predicate index

**Test 3.3: Combined Query**
```json
{
  "query": {
    "subject": "Einstein",
    "predicate": "is"
  },
  "limit": 10
}
```
- **Expected**: Specific facts matching both
- **Verify**: Intersection logic works
- **Fix if fails**: Check query combination logic

**Test 3.4: Empty Result Query**
```json
{
  "query": {
    "subject": "NonexistentEntity"
  }
}
```
- **Expected**: Empty results with suggestions
- **Verify**: Helpful suggestions provided
- **Fix if fails**: Check empty result handling

**Test 3.5: Large Result Set**
```json
{
  "query": {
    "predicate": "is"
  },
  "limit": 100
}
```
- **Expected**: Max 100 results
- **Verify**: Pagination works
- **Fix if fails**: Check limit enforcement

### Tool 4: ask_question
**Purpose**: Natural language question answering

#### Test Scenarios

**Test 4.1: Simple Question**
```json
{
  "question": "Who is Einstein?",
  "max_results": 5
}
```
- **Expected**: Relevant facts and knowledge chunks
- **Verify**: Natural answer format
- **Fix if fails**: Check NL processing in `handle_ask_question`

**Test 4.2: Complex Question with Context**
```json
{
  "question": "What did Einstein invent?",
  "context": "focusing on physics theories",
  "max_results": 3
}
```
- **Expected**: Theory-related results
- **Verify**: Context influences results
- **Fix if fails**: Check context processing

**Test 4.3: Relationship Question**
```json
{
  "question": "How are Einstein and Nobel Prize related?"
}
```
- **Expected**: Connection path or fact
- **Verify**: Relationship extraction
- **Fix if fails**: Check relationship query logic

**Test 4.4: Question About Missing Information**
```json
{
  "question": "What is Einstein's favorite food?"
}
```
- **Expected**: No results with suggestions
- **Verify**: Admits lack of knowledge
- **Fix if fails**: Check unknown handling

**Test 4.5: Multi-entity Question**
```json
{
  "question": "Compare Einstein and Newton contributions"
}
```
- **Expected**: Facts about both entities
- **Verify**: Multi-entity extraction
- **Fix if fails**: Check entity extraction

### Tool 5: explore_connections
**Purpose**: Find paths between entities

#### Test Scenarios

**Test 5.1: Direct Connection**
```json
{
  "start_entity": "Einstein",
  "end_entity": "Nobel_Prize",
  "max_depth": 1
}
```
- **Expected**: Direct path if exists
- **Verify**: Shortest path returned
- **Fix if fails**: Check BFS implementation

**Test 5.2: Multi-hop Connection**
```json
{
  "start_entity": "Einstein",
  "end_entity": "Quantum_Mechanics",
  "max_depth": 3
}
```
- **Expected**: Path through intermediate entities
- **Verify**: All paths within depth
- **Fix if fails**: Check depth limiting

**Test 5.3: All Connections from Entity**
```json
{
  "start_entity": "Python",
  "max_depth": 2
}
```
- **Expected**: All entities within 2 hops
- **Verify**: Breadth-first exploration
- **Fix if fails**: Check exploration without target

**Test 5.4: Filtered Connection Search**
```json
{
  "start_entity": "Einstein",
  "end_entity": "Germany",
  "relationship_types": ["born_in", "lived_in"],
  "max_depth": 2
}
```
- **Expected**: Only paths with specified relations
- **Verify**: Filter applied correctly
- **Fix if fails**: Check relationship filtering

**Test 5.5: No Connection Exists**
```json
{
  "start_entity": "Einstein",
  "end_entity": "Unrelated_Entity",
  "max_depth": 4
}
```
- **Expected**: No paths found message
- **Verify**: Doesn't hang or timeout
- **Fix if fails**: Check termination condition

### Tool 6: get_suggestions
**Purpose**: Intelligent suggestions for graph building

#### Test Scenarios

**Test 6.1: Missing Facts Suggestions**
```json
{
  "suggestion_type": "missing_facts",
  "focus_area": "Einstein",
  "limit": 5
}
```
- **Expected**: List of potential missing facts
- **Verify**: Relevant to existing knowledge
- **Fix if fails**: Check pattern analysis

**Test 6.2: Interesting Questions**
```json
{
  "suggestion_type": "interesting_questions",
  "limit": 3
}
```
- **Expected**: Thought-provoking questions
- **Verify**: Based on graph content
- **Fix if fails**: Check question generation

**Test 6.3: Potential Connections**
```json
{
  "suggestion_type": "potential_connections",
  "focus_area": "Physics"
}
```
- **Expected**: Missing link suggestions
- **Verify**: Logical connections
- **Fix if fails**: Check connection inference

**Test 6.4: Knowledge Gaps**
```json
{
  "suggestion_type": "knowledge_gaps"
}
```
- **Expected**: Areas needing more info
- **Verify**: Gap identification logic
- **Fix if fails**: Check completeness analysis

**Test 6.5: Focused Suggestions**
```json
{
  "suggestion_type": "missing_facts",
  "focus_area": "NonexistentArea"
}
```
- **Expected**: General suggestions or empty
- **Verify**: Handles unknown areas
- **Fix if fails**: Check fallback logic

### Tool 7: get_stats
**Purpose**: Graph statistics and health metrics

#### Test Scenarios

**Test 7.1: Basic Statistics**
```json
{
  "include_details": false
}
```
- **Expected**: Core metrics returned
- **Verify**: Accurate counts
- **Fix if fails**: Fix hanging issue first!

**Test 7.2: Detailed Statistics**
```json
{
  "include_details": true
}
```
- **Expected**: Category breakdowns included
- **Verify**: Additional detail sections
- **Fix if fails**: Check detailed stats collection

**Test 7.3: Empty Graph Stats**
```json
{
  "include_details": false
}
```
- **Expected**: Zero counts, 100% efficiency
- **Verify**: No division by zero
- **Fix if fails**: Check empty graph handling

**Test 7.4: Large Graph Stats**
```json
{
  "include_details": true
}
```
- **Expected**: Performance metrics scale
- **Verify**: Calculation efficiency
- **Fix if fails**: Check performance calculations

**Test 7.5: Stats After Operations**
```json
{
  "include_details": false
}
```
- **Expected**: Updated operation counts
- **Verify**: Usage tracking works
- **Fix if fails**: Check stats updates

### Tool 8: generate_graph_query
**Purpose**: Convert natural language to query languages

#### Test Scenarios

**Test 8.1: Simple Cypher Query**
```json
{
  "natural_query": "Find all scientists",
  "query_language": "cypher"
}
```
- **Expected**: Valid Cypher syntax
- **Verify**: Query is executable
- **Fix if fails**: Check Cypher generation

**Test 8.2: SPARQL Generation**
```json
{
  "natural_query": "What did Einstein invent?",
  "query_language": "sparql"
}
```
- **Expected**: Valid SPARQL syntax
- **Verify**: Proper PREFIX statements
- **Fix if fails**: Check SPARQL templates

**Test 8.3: Complex Gremlin Query**
```json
{
  "natural_query": "Find shortest path between Einstein and Nobel Prize",
  "query_language": "gremlin",
  "include_explanation": true
}
```
- **Expected**: Gremlin traversal with explanation
- **Verify**: Path query syntax
- **Fix if fails**: Check Gremlin generation

**Test 8.4: Unsupported Query Pattern**
```json
{
  "natural_query": "Calculate standard deviation of all facts",
  "query_language": "cypher"
}
```
- **Expected**: Best effort or error
- **Verify**: Appropriate response
- **Fix if fails**: Check pattern matching

**Test 8.5: Multi-condition Query**
```json
{
  "natural_query": "Find scientists who won Nobel Prize and lived in Germany",
  "query_language": "cypher"
}
```
- **Expected**: Multiple MATCH clauses
- **Verify**: AND logic correct
- **Fix if fails**: Check condition combining

### Tool 9: hybrid_search
**Purpose**: Multi-modal advanced search

#### Test Scenarios

**Test 9.1: Hybrid Search**
```json
{
  "query": "quantum physics theories",
  "search_type": "hybrid"
}
```
- **Expected**: Combined semantic and structural results
- **Verify**: Multiple ranking factors
- **Fix if fails**: Check fusion algorithm

**Test 9.2: Semantic Search Only**
```json
{
  "query": "revolutionary scientific discoveries",
  "search_type": "semantic",
  "limit": 5
}
```
- **Expected**: Conceptually similar results
- **Verify**: Embedding-based ranking
- **Fix if fails**: Check embedding search

**Test 9.3: Structural Search**
```json
{
  "query": "highly connected entities",
  "search_type": "structural"
}
```
- **Expected**: Hub entities returned
- **Verify**: Connection count ranking
- **Fix if fails**: Check graph metrics

**Test 9.4: Filtered Search**
```json
{
  "query": "physics",
  "filters": {
    "entity_types": ["person", "theory"],
    "min_confidence": 0.8
  }
}
```
- **Expected**: Only matching types
- **Verify**: Filters applied
- **Fix if fails**: Check filter logic

**Test 9.5: Keyword Search**
```json
{
  "query": "Einstein relativity 1905",
  "search_type": "keyword"
}
```
- **Expected**: Exact term matches
- **Verify**: All keywords found
- **Fix if fails**: Check text search

### Tool 10: validate_knowledge
**Purpose**: Quality assurance for stored knowledge

#### Test Scenarios

**Test 10.1: Full Validation**
```json
{
  "validation_type": "all"
}
```
- **Expected**: Complete validation report
- **Verify**: All check types run
- **Fix if fails**: Check validation pipeline

**Test 10.2: Consistency Check**
```json
{
  "validation_type": "consistency",
  "entity": "Einstein"
}
```
- **Expected**: Reference validation
- **Verify**: Broken links found
- **Fix if fails**: Check consistency logic

**Test 10.3: Conflict Detection**
```json
{
  "validation_type": "conflicts"
}
```
- **Expected**: Contradicting facts found
- **Verify**: Conflict identification
- **Fix if fails**: Check conflict detection

**Test 10.4: Auto-fix Issues**
```json
{
  "validation_type": "consistency",
  "fix_issues": true
}
```
- **Expected**: Issues fixed, report generated
- **Verify**: Fixes applied correctly
- **Fix if fails**: Check fix logic

**Test 10.5: Quality Scoring**
```json
{
  "validation_type": "quality"
}
```
- **Expected**: Quality metrics and score
- **Verify**: Scoring algorithm
- **Fix if fails**: Check quality metrics

## Additional MCP Tools (11-15)

### Tool 11: knowledge_search (Base Server)
**Test**: Similar to hybrid_search but simpler interface

### Tool 12: entity_lookup (Base Server)  
**Test**: Direct entity retrieval by name

### Tool 13: find_connections (Base Server)
**Test**: Similar to explore_connections

### Tool 14: expand_concept (Base Server)
**Test**: Concept expansion and related terms

### Tool 15: graph_statistics (Base Server)
**Test**: Similar to get_stats but different format

## Test Execution Plan

### Phase 1: Fix Critical Issues (Priority: P0)
1. Fix MCP server hanging issue
2. Add proper timeout handling
3. Implement health checks
4. Add request/response logging

### Phase 2: Basic Tool Testing (Priority: P1)
1. Test store_fact with simple data
2. Test find_facts to verify storage
3. Test get_stats to ensure no hangs
4. Verify basic round-trip functionality

### Phase 3: Complex Tool Testing (Priority: P2)
1. Test all knowledge extraction tools
2. Test multi-hop explorations
3. Test search algorithms
4. Test validation and fixes

### Phase 4: Performance Testing (Priority: P3)
1. Load test with 10K facts
2. Concurrent request handling
3. Memory efficiency validation
4. Response time benchmarks

### Phase 5: Integration Testing (Priority: P3)
1. Full workflow scenarios
2. Claude Code integration
3. Error recovery testing
4. Edge case validation

## Success Metrics

1. **Functionality**: All 15 tools pass 5 test scenarios each (75 total)
2. **Performance**: All tools respond within 100ms for basic operations
3. **Reliability**: No hangs, timeouts, or crashes in 1000 consecutive operations
4. **Quality**: 95% accuracy in entity extraction and relationship inference
5. **Scalability**: Handle 100K triples with <1GB memory usage

## Remediation Matrix

| Issue | Severity | Fix Location | Test to Verify |
|-------|----------|--------------|----------------|
| Server hangs | P0 | `handle_tool_call` | Test 7.1 |
| Timeout missing | P0 | All handlers | All tests |
| No validation | P1 | Input handlers | Test 1.4, 2.4 |
| Poor performance | P2 | Query methods | Test 3.5, 9.* |
| Memory leaks | P2 | Engine lifecycle | Long-running tests |
| Unicode issues | P3 | String handling | Test 1.3, 2.5 |

## Conclusion

This comprehensive plan covers all aspects of testing and fixing the LLMKG MCP server. The prioritized approach ensures critical issues are resolved first, enabling progressive validation of all functionality. Each test scenario includes specific inputs, expected outputs, and remediation steps, providing a complete quality assurance framework.