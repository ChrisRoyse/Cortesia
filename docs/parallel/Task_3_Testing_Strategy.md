# Task 3: Testing Strategy for LLM-Agnostic LLMKG

## Overview
This document outlines how to test LLMKG as an MCP tool that works with any LLM, using DeepSeek as one test client among others.

## Testing Layers

### Layer 1: Core Graph Operations (No LLM Required)
These tests verify the knowledge graph and cognitive algorithms work correctly.

```rust
// tests/core_graph_tests.rs
#[cfg(test)]
mod graph_tests {
    use llmkg::core::brain_enhanced_graph::BrainEnhancedGraph;
    
    #[tokio::test]
    async fn test_graph_insert_and_retrieve() {
        let graph = BrainEnhancedGraph::new().await.unwrap();
        let entity = create_test_entity("TestConcept");
        let key = graph.insert_entity(entity).await.unwrap();
        
        let retrieved = graph.get_entity(key).await.unwrap();
        assert_eq!(retrieved.name, "TestConcept");
    }
    
    #[tokio::test]
    async fn test_relationship_creation() {
        let graph = BrainEnhancedGraph::new().await.unwrap();
        let entity_a = graph.insert_entity(create_test_entity("A")).await.unwrap();
        let entity_b = graph.insert_entity(create_test_entity("B")).await.unwrap();
        
        let relationship = create_relationship(entity_a, entity_b, "relates_to");
        graph.insert_relationship(relationship).await.unwrap();
        
        let relations = graph.get_relationships(entity_a, None).await.unwrap();
        assert_eq!(relations.len(), 1);
    }
}
```

### Layer 2: Cognitive Pattern Tests (Graph Algorithms Only)
Test each cognitive pattern as a pure graph algorithm.

```rust
// tests/cognitive_pattern_tests.rs
#[cfg(test)]
mod convergent_tests {
    use llmkg::cognitive::convergent::ConvergentThinking;
    
    #[tokio::test]
    async fn test_convergent_finds_common_patterns() {
        // Create a graph with known convergence
        let graph = create_convergence_test_graph().await;
        
        let convergent = ConvergentThinking::new(Arc::new(graph));
        let concepts = vec![
            get_entity_key("Branch1"),
            get_entity_key("Branch2"),
            get_entity_key("Branch3"),
        ];
        
        let result = convergent.find_convergence(concepts).await.unwrap();
        
        assert_eq!(result.convergence_point, get_entity_key("CommonRoot"));
        assert!(result.confidence > 0.8);
        assert!(!result.supporting_patterns.is_empty());
    }
}

#[cfg(test)]
mod divergent_tests {
    use llmkg::cognitive::divergent::DivergentThinking;
    
    #[tokio::test]
    async fn test_divergent_generates_ideas() {
        let graph = create_divergent_test_graph().await;
        let divergent = DivergentThinking::new(Arc::new(graph));
        
        let seed = get_entity_key("CreativeSeed");
        let result = divergent.generate_ideas(seed).await.unwrap();
        
        assert!(result.ideas.len() >= 5);
        assert!(result.novelty_score > 0.0);
        
        // Verify ideas come from graph exploration
        for idea in &result.ideas {
            assert!(graph.entity_exists(idea.concept).await);
        }
    }
}
```

### Layer 3: MCP Protocol Tests (Mock MCP Client)
Test that cognitive patterns are properly exposed as MCP tools.

```rust
// tests/mcp_protocol_tests.rs
use llmkg::mcp::server::LLMKGMCPServer;
use mcp::protocol::{ToolRequest, ToolResponse};

#[tokio::test]
async fn test_mcp_tool_discovery() {
    let server = LLMKGMCPServer::new().await.unwrap();
    let tools = server.list_tools().await.unwrap();
    
    // Verify all cognitive patterns are exposed
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"convergent_thinking"));
    assert!(tool_names.contains(&"divergent_thinking"));
    assert!(tool_names.contains(&"lateral_thinking"));
    assert!(tool_names.contains(&"systems_thinking"));
    assert!(tool_names.contains(&"critical_thinking"));
    assert!(tool_names.contains(&"abstract_thinking"));
    assert!(tool_names.contains(&"adaptive_thinking"));
}

#[tokio::test]
async fn test_mcp_convergent_tool() {
    let server = LLMKGMCPServer::new().await.unwrap();
    
    // Create test data in graph
    setup_convergence_test_data(&server.graph).await;
    
    let request = ToolRequest {
        tool: "convergent_thinking".to_string(),
        arguments: json!({
            "concepts": ["quantum", "computing", "theory"],
            "depth": 3
        }),
    };
    
    let response = server.handle_tool_request(request).await.unwrap();
    
    match response {
        ToolResponse::Success { result } => {
            assert!(result.contains("convergence"));
            assert!(result.contains("common"));
        }
        _ => panic!("Expected success response"),
    }
}
```

### Layer 4: Integration Tests with Real LLMs
Test LLMKG with different LLMs as MCP clients.

#### DeepSeek Integration Test
```python
# tests/integration/test_deepseek_mcp.py
import pytest
from deepseek import DeepSeekClient
from llmkg_mcp_client import LLMKGMCPClient

@pytest.fixture
def llmkg_server():
    """Start LLMKG MCP server for testing"""
    server = start_llmkg_mcp_server(port=8080)
    yield server
    server.stop()

@pytest.fixture
def deepseek_client():
    """Create DeepSeek client with MCP capability"""
    return DeepSeekClient(
        api_key=os.environ.get("DEEPSEEK_TEST_API_KEY"),
        tools_endpoint="http://localhost:8080/mcp"
    )

def test_deepseek_uses_convergent_thinking(llmkg_server, deepseek_client):
    """Test that DeepSeek can use LLMKG's convergent thinking tool"""
    
    # Seed LLMKG with test data
    llmkg_client = LLMKGMCPClient("http://localhost:8080")
    llmkg_client.add_knowledge([
        ("AI", "relates_to", "Machine Learning"),
        ("AI", "relates_to", "Neural Networks"),
        ("Machine Learning", "uses", "Statistics"),
        ("Neural Networks", "uses", "Statistics"),
    ])
    
    # Ask DeepSeek to find convergence
    response = deepseek_client.chat(
        messages=[{
            "role": "user",
            "content": "What do Machine Learning and Neural Networks have in common?"
        }],
        tools=["llmkg.convergent_thinking"]
    )
    
    # Verify DeepSeek used the tool and found convergence
    assert "Statistics" in response.content
    assert response.tool_calls[0].tool == "convergent_thinking"
```

#### Claude Integration Test (via MCP)
```python
# tests/integration/test_claude_mcp.py
def test_claude_uses_lateral_thinking(llmkg_server):
    """Test that Claude can use LLMKG's lateral thinking tool"""
    
    # This would run in an environment where Claude has MCP access
    # to the LLMKG server
    
    # Setup test data
    setup_lateral_thinking_test_data(llmkg_server)
    
    # In practice, this would be tested through Claude's UI
    # or API with MCP enabled
    expected_mcp_request = {
        "tool": "lateral_thinking",
        "arguments": {
            "concept": "photosynthesis",
            "target_domain": "computing"
        }
    }
    
    # Verify the MCP protocol
    response = llmkg_server.handle_mcp_request(expected_mcp_request)
    assert "solar panels" in response.result
    assert "energy conversion" in response.result
```

#### Generic LLM Test Template
```python
# tests/integration/test_generic_llm.py
class GenericLLMTest:
    """Template for testing any LLM with LLMKG"""
    
    def test_llm_cognitive_pattern(self, llm_client, pattern_name):
        # 1. Setup LLMKG with test data
        setup_test_knowledge_graph()
        
        # 2. Create prompt that should trigger tool use
        prompt = self.get_test_prompt_for_pattern(pattern_name)
        
        # 3. Send to LLM with LLMKG tools available
        response = llm_client.query_with_tools(
            prompt=prompt,
            available_tools=[f"llmkg.{pattern_name}"]
        )
        
        # 4. Verify tool was used
        assert any(call.tool.startswith("llmkg.") for call in response.tool_calls)
        
        # 5. Verify response quality
        assert self.validate_response_for_pattern(response, pattern_name)
```

### Layer 5: Performance and Load Tests

```python
# tests/performance/test_load.py
import asyncio
import time

async def test_concurrent_mcp_requests():
    """Test LLMKG handles multiple concurrent MCP requests"""
    server = await start_llmkg_server()
    
    async def make_request(pattern, query):
        start = time.time()
        response = await server.handle_tool_request({
            "tool": pattern,
            "arguments": {"query": query}
        })
        return time.time() - start
    
    # Create 100 concurrent requests
    patterns = ["convergent_thinking", "divergent_thinking", "lateral_thinking"]
    tasks = []
    for i in range(100):
        pattern = patterns[i % len(patterns)]
        task = make_request(pattern, f"test_query_{i}")
        tasks.append(task)
    
    times = await asyncio.gather(*tasks)
    
    # Verify performance
    assert max(times) < 1.0  # No request takes more than 1 second
    assert sum(times) / len(times) < 0.1  # Average under 100ms
```

## Test Data Generators

### Graph Structure Generators
```rust
// tests/helpers/graph_generators.rs

pub async fn create_convergence_test_graph() -> BrainEnhancedGraph {
    let graph = BrainEnhancedGraph::new().await.unwrap();
    
    // Create tree structure with common root
    //       Root
    //      /  |  \
    //     A   B   C
    //    / \ / \ / \
    //   D  E F  G H  I
    
    let root = graph.insert_entity(create_entity("Root")).await.unwrap();
    let a = graph.insert_entity(create_entity("A")).await.unwrap();
    let b = graph.insert_entity(create_entity("B")).await.unwrap();
    let c = graph.insert_entity(create_entity("C")).await.unwrap();
    
    graph.insert_relationship(create_relation(root, a)).await.unwrap();
    graph.insert_relationship(create_relation(root, b)).await.unwrap();
    graph.insert_relationship(create_relation(root, c)).await.unwrap();
    
    // Add leaves...
    
    graph
}

pub async fn create_systems_test_graph() -> BrainEnhancedGraph {
    let graph = BrainEnhancedGraph::new().await.unwrap();
    
    // Create system with feedback loops
    // A -> B -> C -> D
    // ^              |
    // |              |
    // +--------------+
    
    // Implementation...
    
    graph
}
```

## Test Execution Plan

### Phase 1: Core Tests (No External Dependencies)
```bash
# Run only core graph and algorithm tests
cargo test --test core_graph_tests
cargo test --test cognitive_pattern_tests
```

### Phase 2: MCP Protocol Tests
```bash
# Test MCP server functionality
cargo test --test mcp_protocol_tests
```

### Phase 3: Integration Tests (Requires LLM Access)
```bash
# Set up test environment
export DEEPSEEK_TEST_API_KEY="test_key"
export OPENAI_TEST_API_KEY="test_key"

# Run integration tests
pytest tests/integration/
```

### Phase 4: Performance Tests
```bash
# Run load tests
cargo test --test performance_tests --release
```

## CI/CD Pipeline Configuration

```yaml
# .github/workflows/test.yml
name: LLMKG Tests

on: [push, pull_request]

jobs:
  core-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Core Tests
        run: cargo test --test core_graph_tests --test cognitive_pattern_tests
        
  mcp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run MCP Tests
        run: cargo test --test mcp_protocol_tests
        
  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Run Integration Tests
        env:
          DEEPSEEK_TEST_API_KEY: ${{ secrets.DEEPSEEK_TEST_API_KEY }}
        run: |
          cargo build --release
          pytest tests/integration/ -v
```

## Mock Clients for Testing

### Mock LLM Client
```python
# tests/mocks/mock_llm_client.py
class MockLLMClient:
    """Mock LLM client for testing MCP integration"""
    
    def __init__(self, mcp_endpoint):
        self.mcp_endpoint = mcp_endpoint
        self.tool_calls = []
    
    async def query_with_tools(self, prompt, tools):
        # Simulate LLM choosing appropriate tool
        if "common" in prompt or "convergence" in prompt:
            tool = "convergent_thinking"
        elif "ideas" in prompt or "creative" in prompt:
            tool = "divergent_thinking"
        else:
            tool = tools[0]
        
        # Call MCP endpoint
        response = await self.call_mcp_tool(tool, {"query": prompt})
        
        return MockResponse(
            content=f"Based on {tool}: {response}",
            tool_calls=[MockToolCall(tool, response)]
        )
```

## Success Metrics

1. **Core Tests**: 100% pass rate, < 5s total execution time
2. **MCP Tests**: All tools properly exposed, < 100ms response time
3. **Integration Tests**: Works with at least 3 different LLMs
4. **Performance Tests**: Handles 100+ concurrent requests
5. **No LLM Dependencies**: Core tests run without any API keys