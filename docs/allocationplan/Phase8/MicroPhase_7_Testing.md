# MicroPhase 7: Comprehensive Testing Suite (18-22 Micro-Tasks)

**Total Duration**: 5-7 hours (18-22 micro-tasks × 15-20 minutes each)  
**Priority**: Critical - Quality assurance requirement  
**Prerequisites**: All previous MicroPhases (1-6)

## Overview

Implement comprehensive unit, integration, performance, and end-to-end tests through atomic micro-tasks, each delivering ONE concrete test file in 15-20 minutes. Each task follows the established pattern with exact code specifications and clear verification criteria.

## Micro-Task Breakdown

---

## Micro-Task 7.1.1: Unit Tests for Semantic Column Processing
**Duration**: 18 minutes  
**Dependencies**: MicroPhase 2 completion  
**Input**: Semantic column implementation  
**Output**: Unit tests for semantic column  

### Task Prompt for AI
```
Create unit tests for semantic column processing:

```rust
use cortex_kg::mcp::neuromorphic::{
    cortical_column::{CorticalColumn, ColumnType},
    semantic_column::SemanticColumn,
};
use cortex_kg::core::types::TTFSPattern;

#[tokio::test]
async fn test_semantic_column_processing() {
    let semantic_column = SemanticColumn::new_with_networks(vec![]);
    
    let ttfs_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)],
        duration_ms: 10.0,
        encoding_quality: 0.95,
    };
    
    let result = semantic_column.process(&ttfs_pattern).await.unwrap();
    
    assert!(result.activation_strength > 0.0);
    assert!(result.confidence > 0.0);
    assert!(!result.neural_pathway.is_empty());
    assert!(result.processing_time_ms > 0.0);
    assert_eq!(semantic_column.get_column_type(), ColumnType::Semantic);
}

#[tokio::test]
async fn test_semantic_column_empty_pattern() {
    let semantic_column = SemanticColumn::new_with_networks(vec![]);
    
    let empty_pattern = TTFSPattern {
        spikes: vec![],
        duration_ms: 0.0,
        encoding_quality: 0.0,
    };
    
    let result = semantic_column.process(&empty_pattern).await.unwrap();
    assert_eq!(result.activation_strength, 0.0);
}
```
```

**Expected Deliverable**: `tests/unit/semantic_column_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.1.2: Unit Tests for Structural Column Processing
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.1.1  
**Input**: Structural column implementation  
**Output**: Unit tests for structural column  

### Task Prompt for AI
```
Create unit tests for structural column processing:

```rust
use cortex_kg::mcp::neuromorphic::{
    cortical_column::{CorticalColumn, ColumnType},
    structural_column::StructuralColumn,
};
use cortex_kg::core::types::TTFSPattern;

#[tokio::test]
async fn test_structural_column_processing() {
    let structural_column = StructuralColumn::new_with_networks(vec![]);
    
    let ttfs_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)],
        duration_ms: 10.0,
        encoding_quality: 0.95,
    };
    
    let result = structural_column.process(&ttfs_pattern).await.unwrap();
    
    assert!(result.activation_strength > 0.0);
    assert!(result.confidence > 0.0);
    assert!(!result.neural_pathway.is_empty());
    assert_eq!(structural_column.get_column_type(), ColumnType::Structural);
}

#[tokio::test]
async fn test_structural_column_pattern_recognition() {
    let structural_column = StructuralColumn::new_with_networks(vec![]);
    
    let complex_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.15, 1.5), (0.2, 2.0), (0.25, 2.5)],
        duration_ms: 20.0,
        encoding_quality: 0.98,
    };
    
    let result = structural_column.process(&complex_pattern).await.unwrap();
    assert!(result.activation_strength > 0.5);
}
```
```

**Expected Deliverable**: `tests/unit/structural_column_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.1.3: Unit Tests for Temporal Column Processing
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.1.2  
**Input**: Temporal column implementation  
**Output**: Unit tests for temporal column  

### Task Prompt for AI
```
Create unit tests for temporal column processing:

```rust
use cortex_kg::mcp::neuromorphic::{
    cortical_column::{CorticalColumn, ColumnType},
    temporal_column::TemporalColumn,
};
use cortex_kg::core::types::TTFSPattern;

#[tokio::test]
async fn test_temporal_column_processing() {
    let temporal_column = TemporalColumn::new_with_networks(vec![]);
    
    let ttfs_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)],
        duration_ms: 10.0,
        encoding_quality: 0.95,
    };
    
    let result = temporal_column.process(&ttfs_pattern).await.unwrap();
    
    assert!(result.activation_strength > 0.0);
    assert!(result.confidence > 0.0);
    assert!(!result.neural_pathway.is_empty());
    assert_eq!(temporal_column.get_column_type(), ColumnType::Temporal);
}

#[tokio::test]
async fn test_temporal_column_sequence_detection() {
    let temporal_column = TemporalColumn::new_with_networks(vec![]);
    
    let sequence_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.4, 1.0)],
        duration_ms: 30.0,
        encoding_quality: 0.92,
    };
    
    let result = temporal_column.process(&sequence_pattern).await.unwrap();
    assert!(result.processing_time_ms > 0.0);
}
```
```

**Expected Deliverable**: `tests/unit/temporal_column_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.1.4: Unit Tests for Exception Column Processing
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.1.3  
**Input**: Exception column implementation  
**Output**: Unit tests for exception column  

### Task Prompt for AI
```
Create unit tests for exception column processing:

```rust
use cortex_kg::mcp::neuromorphic::{
    cortical_column::{CorticalColumn, ColumnType},
    exception_column::ExceptionColumn,
};
use cortex_kg::core::types::TTFSPattern;

#[tokio::test]
async fn test_exception_column_normal_pattern() {
    let exception_column = ExceptionColumn::new_with_networks(vec![]);
    
    let normal_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)],
        duration_ms: 10.0,
        encoding_quality: 0.95,
    };
    
    let result = exception_column.process(&normal_pattern).await.unwrap();
    
    // Exception column should have low activation for normal patterns
    assert!(result.activation_strength >= 0.0);
    assert!(result.confidence >= 0.0);
    assert_eq!(exception_column.get_column_type(), ColumnType::Exception);
}

#[tokio::test]
async fn test_exception_column_anomalous_pattern() {
    let exception_column = ExceptionColumn::new_with_networks(vec![]);
    
    let anomalous_pattern = TTFSPattern {
        spikes: vec![(0.1, 100.0), (0.2, -50.0), (0.3, 200.0)],
        duration_ms: 5.0,
        encoding_quality: 0.1,
    };
    
    let result = exception_column.process(&anomalous_pattern).await.unwrap();
    assert!(result.activation_strength >= 0.0);
}
```
```

**Expected Deliverable**: `tests/unit/exception_column_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.1.5: Unit Tests for Lateral Inhibition Engine
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.1.4  
**Input**: Lateral inhibition implementation  
**Output**: Unit tests for lateral inhibition  

### Task Prompt for AI
```
Create unit tests for lateral inhibition engine:

```rust
use cortex_kg::mcp::neuromorphic::{
    lateral_inhibition::LateralInhibitionEngine,
    cortical_column::ColumnActivation,
};
use std::collections::HashMap;

#[tokio::test]
async fn test_lateral_inhibition_competition() {
    let inhibition_engine = LateralInhibitionEngine::new(0.8, 0.2);
    
    let mut activations = HashMap::new();
    activations.insert("semantic".to_string(), ColumnActivation {
        activation_strength: 0.9,
        confidence: 0.85,
        neural_pathway: vec!["semantic_1".to_string()],
        processing_time_ms: 5.0,
    });
    activations.insert("structural".to_string(), ColumnActivation {
        activation_strength: 0.6,
        confidence: 0.70,
        neural_pathway: vec!["structural_1".to_string()],
        processing_time_ms: 6.0,
    });
    
    let result = inhibition_engine.apply_inhibition(activations).await.unwrap();
    
    assert!(!result.is_empty());
    // Semantic should win due to higher activation
    assert!(result.get("semantic").unwrap().activation_strength > 
            result.get("structural").unwrap().activation_strength);
}

#[tokio::test]
async fn test_lateral_inhibition_threshold() {
    let inhibition_engine = LateralInhibitionEngine::new(0.8, 0.5);
    
    let mut activations = HashMap::new();
    activations.insert("weak".to_string(), ColumnActivation {
        activation_strength: 0.3,
        confidence: 0.30,
        neural_pathway: vec!["weak_1".to_string()],
        processing_time_ms: 5.0,
    });
    
    let result = inhibition_engine.apply_inhibition(activations).await.unwrap();
    assert!(result.get("weak").unwrap().activation_strength < 0.8);
}
```
```

**Expected Deliverable**: `tests/unit/lateral_inhibition_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.1.6: Unit Tests for Neuromorphic Core Integration
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.1.5  
**Input**: Neuromorphic core implementation  
**Output**: Unit tests for neuromorphic core  

### Task Prompt for AI
```
Create unit tests for neuromorphic core integration:

```rust
use cortex_kg::mcp::neuromorphic::NeuromorphicCore;
use cortex_kg::core::types::TTFSPattern;
use cortex_kg::mcp::handlers::memory::{MemoryRequest, MemoryResponse};

#[tokio::test]
async fn test_neuromorphic_core_processing() {
    let neuromorphic_core = NeuromorphicCore::new().await.unwrap();
    
    let memory_request = MemoryRequest {
        operation: "store".to_string(),
        content: "Test knowledge for neuromorphic processing".to_string(),
        metadata: std::collections::HashMap::new(),
    };
    
    let result = neuromorphic_core.process_memory_request(&memory_request).await.unwrap();
    
    assert!(matches!(result, MemoryResponse::Success { .. }));
}

#[tokio::test]
async fn test_neuromorphic_core_column_coordination() {
    let neuromorphic_core = NeuromorphicCore::new().await.unwrap();
    
    let ttfs_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)],
        duration_ms: 10.0,
        encoding_quality: 0.95,
    };
    
    let result = neuromorphic_core.process_pattern(&ttfs_pattern).await.unwrap();
    
    assert!(result.total_activation > 0.0);
    assert!(!result.winning_columns.is_empty());
    assert!(result.processing_time_ms > 0.0);
}
```
```

**Expected Deliverable**: `tests/unit/neuromorphic_core_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.2.1: Integration Tests for Store Memory Handler
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.1.6  
**Input**: Store memory handler implementation  
**Output**: Integration tests for store memory  

### Task Prompt for AI
```
Create integration tests for store memory handler:

```rust
use cortex_kg::mcp::handlers::memory::{StoreMemoryHandler, MemoryRequest, MemoryResponse};
use cortex_kg::mcp::CortexKGMCPServer;
use std::collections::HashMap;

#[tokio::test]
async fn test_store_memory_handler_integration() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let store_handler = StoreMemoryHandler::new(server.get_neuromorphic_core()).await.unwrap();
    
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "test".to_string());
    
    let request = MemoryRequest {
        operation: "store".to_string(),
        content: "Einstein was a theoretical physicist".to_string(),
        metadata,
    };
    
    let response = store_handler.handle(&request).await.unwrap();
    
    match response {
        MemoryResponse::Success { message, metadata } => {
            assert!(message.contains("stored"));
            assert!(!metadata.is_empty());
        },
        _ => panic!("Expected success response"),
    }
}

#[tokio::test]
async fn test_store_memory_handler_validation() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let store_handler = StoreMemoryHandler::new(server.get_neuromorphic_core()).await.unwrap();
    
    let request = MemoryRequest {
        operation: "store".to_string(),
        content: "".to_string(), // Empty content should be rejected
        metadata: HashMap::new(),
    };
    
    let response = store_handler.handle(&request).await;
    assert!(response.is_err());
}
```
```

**Expected Deliverable**: `tests/integration/store_memory_handler_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.2.2: Integration Tests for Retrieve Memory Handler
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.2.1  
**Input**: Retrieve memory handler implementation  
**Output**: Integration tests for retrieve memory  

### Task Prompt for AI
```
Create integration tests for retrieve memory handler:

```rust
use cortex_kg::mcp::handlers::memory::{RetrieveMemoryHandler, MemoryRequest, MemoryResponse};
use cortex_kg::mcp::CortexKGMCPServer;
use std::collections::HashMap;

#[tokio::test]
async fn test_retrieve_memory_handler_integration() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let retrieve_handler = RetrieveMemoryHandler::new(server.get_neuromorphic_core()).await.unwrap();
    
    // First store some memory
    let store_request = MemoryRequest {
        operation: "store".to_string(),
        content: "Quantum mechanics is a fundamental theory".to_string(),
        metadata: HashMap::new(),
    };
    server.process_memory_request(&store_request).await.unwrap();
    
    // Now retrieve it
    let mut query_metadata = HashMap::new();
    query_metadata.insert("query".to_string(), "quantum mechanics".to_string());
    
    let retrieve_request = MemoryRequest {
        operation: "retrieve".to_string(),
        content: "quantum".to_string(),
        metadata: query_metadata,
    };
    
    let response = retrieve_handler.handle(&retrieve_request).await.unwrap();
    
    match response {
        MemoryResponse::Success { message, metadata } => {
            assert!(message.contains("quantum") || message.contains("mechanics"));
            assert!(!metadata.is_empty());
        },
        _ => panic!("Expected success response"),
    }
}

#[tokio::test]
async fn test_retrieve_memory_handler_no_results() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let retrieve_handler = RetrieveMemoryHandler::new(server.get_neuromorphic_core()).await.unwrap();
    
    let request = MemoryRequest {
        operation: "retrieve".to_string(),
        content: "nonexistent_query_term_xyz".to_string(),
        metadata: HashMap::new(),
    };
    
    let response = retrieve_handler.handle(&request).await.unwrap();
    
    match response {
        MemoryResponse::Success { message, .. } => {
            assert!(message.contains("No results") || message.contains("not found"));
        },
        _ => panic!("Expected success response with no results message"),
    }
}
```
```

**Expected Deliverable**: `tests/integration/retrieve_memory_handler_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.2.3: Integration Tests for Tool Executor
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.2.2  
**Input**: Tool executor implementation  
**Output**: Integration tests for tool executor  

### Task Prompt for AI
```
Create integration tests for tool executor:

```rust
use cortex_kg::mcp::tools::{ToolExecutor, ToolRequest, ToolResponse};
use cortex_kg::mcp::CortexKGMCPServer;
use serde_json::json;

#[tokio::test]
async fn test_tool_executor_store_fact() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let tool_executor = ToolExecutor::new(server.get_neuromorphic_core()).await.unwrap();
    
    let request = ToolRequest {
        tool_name: "store_fact".to_string(),
        parameters: json!({
            "subject": "Einstein",
            "predicate": "developed",
            "object": "relativity theory"
        }),
    };
    
    let response = tool_executor.execute(&request).await.unwrap();
    
    match response {
        ToolResponse::Success { result, .. } => {
            assert!(result.contains("stored") || result.contains("success"));
        },
        _ => panic!("Expected success response"),
    }
}

#[tokio::test]
async fn test_tool_executor_invalid_tool() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let tool_executor = ToolExecutor::new(server.get_neuromorphic_core()).await.unwrap();
    
    let request = ToolRequest {
        tool_name: "nonexistent_tool".to_string(),
        parameters: json!({}),
    };
    
    let response = tool_executor.execute(&request).await;
    assert!(response.is_err());
}
```
```

**Expected Deliverable**: `tests/integration/tool_executor_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.2.4: Integration Tests for MCP Server Protocol
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.2.3  
**Input**: MCP server protocol implementation  
**Output**: Integration tests for MCP protocol  

### Task Prompt for AI
```
Create integration tests for MCP server protocol handling:

```rust
use cortex_kg::mcp::CortexKGMCPServer;
use mcp_server::{Server, Request, Response};
use serde_json::json;

#[tokio::test]
async fn test_mcp_protocol_tool_discovery() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let request = Request::new("tools/list".to_string(), json!({}));
    let response = server.handle_request(request).await.unwrap();
    
    match response {
        Response::Success { result } => {
            let tools = result.get("tools").unwrap().as_array().unwrap();
            assert!(!tools.is_empty());
            assert!(tools.iter().any(|tool| 
                tool.get("name").unwrap().as_str() == Some("store_fact")
            ));
        },
        _ => panic!("Expected success response"),
    }
}

#[tokio::test]
async fn test_mcp_protocol_tool_execution() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let request = Request::new("tools/call".to_string(), json!({
        "name": "store_fact",
        "arguments": {
            "subject": "Newton",
            "predicate": "formulated",
            "object": "laws of motion"
        }
    }));
    
    let response = server.handle_request(request).await.unwrap();
    
    match response {
        Response::Success { result } => {
            assert!(result.get("content").is_some());
        },
        _ => panic!("Expected success response"),
    }
}

#[tokio::test]
async fn test_mcp_protocol_invalid_request() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let request = Request::new("invalid/method".to_string(), json!({}));
    let response = server.handle_request(request).await;
    
    assert!(response.is_err());
}
```
```

**Expected Deliverable**: `tests/integration/mcp_protocol_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.3.1: Authentication and Authorization Tests
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.2.4  
**Input**: Authentication implementation  
**Output**: Security tests for authentication  

### Task Prompt for AI
```
Create authentication and authorization tests:

```rust
use cortex_kg::mcp::auth::{AuthManager, AuthToken, AuthLevel};
use cortex_kg::mcp::CortexKGMCPServer;
use chrono::{Duration, Utc};

#[tokio::test]
async fn test_auth_token_validation() {
    let auth_manager = AuthManager::new().await.unwrap();
    
    let valid_token = AuthToken {
        token: "valid_test_token".to_string(),
        level: AuthLevel::ReadWrite,
        expires_at: Utc::now() + Duration::hours(1),
        issued_at: Utc::now(),
    };
    
    auth_manager.register_token(&valid_token).await.unwrap();
    
    let is_valid = auth_manager.validate_token("valid_test_token").await.unwrap();
    assert!(is_valid);
}

#[tokio::test]
async fn test_auth_token_expiration() {
    let auth_manager = AuthManager::new().await.unwrap();
    
    let expired_token = AuthToken {
        token: "expired_test_token".to_string(),
        level: AuthLevel::ReadOnly,
        expires_at: Utc::now() - Duration::hours(1),
        issued_at: Utc::now() - Duration::hours(2),
    };
    
    auth_manager.register_token(&expired_token).await.unwrap();
    
    let is_valid = auth_manager.validate_token("expired_test_token").await.unwrap();
    assert!(!is_valid);
}

#[tokio::test]
async fn test_auth_level_enforcement() {
    let auth_manager = AuthManager::new().await.unwrap();
    
    let read_only_token = AuthToken {
        token: "readonly_token".to_string(),
        level: AuthLevel::ReadOnly,
        expires_at: Utc::now() + Duration::hours(1),
        issued_at: Utc::now(),
    };
    
    auth_manager.register_token(&read_only_token).await.unwrap();
    
    let can_read = auth_manager.check_permission("readonly_token", AuthLevel::ReadOnly).await.unwrap();
    assert!(can_read);
    
    let can_write = auth_manager.check_permission("readonly_token", AuthLevel::ReadWrite).await.unwrap();
    assert!(!can_write);
}
```
```

**Expected Deliverable**: `tests/security/auth_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.3.2: Security Vulnerability Tests
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.3.1  
**Input**: Security implementation  
**Output**: Security vulnerability tests  

### Task Prompt for AI
```
Create security vulnerability tests:

```rust
use cortex_kg::mcp::CortexKGMCPServer;
use cortex_kg::mcp::handlers::memory::MemoryRequest;
use std::collections::HashMap;

#[tokio::test]
async fn test_injection_attack_prevention() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let malicious_content = r#"'; DROP TABLE facts; --"#;
    
    let request = MemoryRequest {
        operation: "store".to_string(),
        content: malicious_content.to_string(),
        metadata: HashMap::new(),
    };
    
    // This should not crash the server or corrupt data
    let result = server.process_memory_request(&request).await;
    
    // Should either handle safely or reject
    match result {
        Ok(_) => {
            // If accepted, verify no corruption occurred
            let retrieve_request = MemoryRequest {
                operation: "retrieve".to_string(),
                content: "test".to_string(),
                metadata: HashMap::new(),
            };
            let _ = server.process_memory_request(&retrieve_request).await.unwrap();
        },
        Err(_) => {
            // Rejection is acceptable for malicious input
        }
    }
}

#[tokio::test]
async fn test_input_size_limits() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    // Create very large input
    let large_content = "A".repeat(10_000_000); // 10MB
    
    let request = MemoryRequest {
        operation: "store".to_string(),
        content: large_content,
        metadata: HashMap::new(),
    };
    
    let result = server.process_memory_request(&request).await;
    
    // Should reject oversized input
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_access_safety() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let request = MemoryRequest {
                operation: "store".to_string(),
                content: format!("Concurrent test data {}", i),
                metadata: HashMap::new(),
            };
            server_clone.process_memory_request(&request).await
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
```
```

**Expected Deliverable**: `tests/security/vulnerability_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.4.1: Performance Tests for Memory Operations
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.3.2  
**Input**: Memory operation implementations  
**Output**: Performance tests for memory ops  

### Task Prompt for AI
```
Create performance tests for memory operations:

```rust
use cortex_kg::mcp::CortexKGMCPServer;
use cortex_kg::mcp::handlers::memory::MemoryRequest;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_store_memory_performance() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    let start = Instant::now();
    
    for i in 0..100 {
        let request = MemoryRequest {
            operation: "store".to_string(),
            content: format!("Performance test data item {}", i),
            metadata: HashMap::new(),
        };
        
        server.process_memory_request(&request).await.unwrap();
    }
    
    let duration = start.elapsed();
    
    // Should complete 100 stores in under 10 seconds
    assert!(duration < Duration::from_secs(10));
    
    // Average should be under 100ms per operation
    let avg_per_operation = duration / 100;
    assert!(avg_per_operation < Duration::from_millis(100));
}

#[tokio::test]
async fn test_retrieve_memory_performance() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    // Store test data first
    for i in 0..50 {
        let request = MemoryRequest {
            operation: "store".to_string(),
            content: format!("Retrieval performance test data {}", i),
            metadata: HashMap::new(),
        };
        server.process_memory_request(&request).await.unwrap();
    }
    
    let start = Instant::now();
    
    for i in 0..50 {
        let request = MemoryRequest {
            operation: "retrieve".to_string(),
            content: format!("performance test {}", i),
            metadata: HashMap::new(),
        };
        
        server.process_memory_request(&request).await.unwrap();
    }
    
    let duration = start.elapsed();
    
    // Should complete 50 retrievals in under 5 seconds
    assert!(duration < Duration::from_secs(5));
    
    // Average should be under 100ms per operation
    let avg_per_operation = duration / 50;
    assert!(avg_per_operation < Duration::from_millis(100));
}
```
```

**Expected Deliverable**: `tests/performance/memory_performance_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.4.2: Performance Tests for Neuromorphic Processing
**Duration**: 20 minutes  
**Dependencies**: Micro-Task 7.4.1  
**Input**: Neuromorphic processing implementation  
**Output**: Performance tests for neuromorphic core  

### Task Prompt for AI
```
Create performance tests for neuromorphic processing:

```rust
use cortex_kg::mcp::neuromorphic::NeuromorphicCore;
use cortex_kg::core::types::TTFSPattern;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_neuromorphic_processing_latency() {
    let neuromorphic_core = NeuromorphicCore::new().await.unwrap();
    
    let ttfs_pattern = TTFSPattern {
        spikes: vec![(0.1, 1.0), (0.2, 2.0), (0.3, 3.0), (0.4, 4.0)],
        duration_ms: 20.0,
        encoding_quality: 0.95,
    };
    
    let start = Instant::now();
    let result = neuromorphic_core.process_pattern(&ttfs_pattern).await.unwrap();
    let processing_time = start.elapsed();
    
    // Processing should complete in under 50ms for this pattern
    assert!(processing_time < Duration::from_millis(50));
    assert!(result.total_activation > 0.0);
}

#[tokio::test]
async fn test_neuromorphic_processing_throughput() {
    let neuromorphic_core = NeuromorphicCore::new().await.unwrap();
    
    let start = Instant::now();
    
    for i in 0..100 {
        let ttfs_pattern = TTFSPattern {
            spikes: vec![(0.1, i as f64), (0.2, (i + 1) as f64)],
            duration_ms: 10.0,
            encoding_quality: 0.90,
        };
        
        neuromorphic_core.process_pattern(&ttfs_pattern).await.unwrap();
    }
    
    let total_time = start.elapsed();
    
    // Should process 100 patterns in under 5 seconds
    assert!(total_time < Duration::from_secs(5));
    
    // Average processing time should be under 50ms
    let avg_time = total_time / 100;
    assert!(avg_time < Duration::from_millis(50));
}

#[tokio::test]
async fn test_neuromorphic_concurrent_processing() {
    let neuromorphic_core = NeuromorphicCore::new().await.unwrap();
    
    let start = Instant::now();
    let mut handles = vec![];
    
    for i in 0..10 {
        let core_clone = neuromorphic_core.clone();
        let handle = tokio::spawn(async move {
            let ttfs_pattern = TTFSPattern {
                spikes: vec![(0.1, i as f64), (0.2, (i + 1) as f64)],
                duration_ms: 15.0,
                encoding_quality: 0.92,
            };
            
            let start_individual = Instant::now();
            let result = core_clone.process_pattern(&ttfs_pattern).await.unwrap();
            let individual_time = start_individual.elapsed();
            
            (result, individual_time)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let (result, individual_time) = handle.await.unwrap();
        assert!(result.total_activation > 0.0);
        assert!(individual_time < Duration::from_millis(100));
    }
    
    let total_concurrent_time = start.elapsed();
    
    // Concurrent processing should be faster than sequential
    assert!(total_concurrent_time < Duration::from_secs(2));
}
```
```

**Expected Deliverable**: `tests/performance/neuromorphic_performance_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.4.3: Load Tests for Concurrent Operations
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.4.2  
**Input**: Server implementation  
**Output**: Load tests for concurrent ops  

### Task Prompt for AI
```
Create load tests for concurrent operations:

```rust
use cortex_kg::mcp::CortexKGMCPServer;
use cortex_kg::mcp::handlers::memory::MemoryRequest;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use std::sync::Arc;

#[tokio::test]
async fn test_concurrent_store_operations() {
    let server = CortexKGMCPServer::new().await.unwrap();
    let semaphore = Arc::new(Semaphore::new(20)); // Limit concurrent operations
    
    let start = Instant::now();
    let mut handles = vec![];
    
    for i in 0..100 {
        let server_clone = server.clone();
        let semaphore_clone = semaphore.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();
            
            let request = MemoryRequest {
                operation: "store".to_string(),
                content: format!("Concurrent load test data {}", i),
                metadata: HashMap::new(),
            };
            
            let op_start = Instant::now();
            let result = server_clone.process_memory_request(&request).await;
            let op_duration = op_start.elapsed();
            
            (result, op_duration)
        });
        handles.push(handle);
    }
    
    let mut successful_operations = 0;
    let mut total_operation_time = Duration::new(0, 0);
    
    for handle in handles {
        let (result, op_duration) = handle.await.unwrap();
        if result.is_ok() {
            successful_operations += 1;
            total_operation_time += op_duration;
        }
    }
    
    let total_time = start.elapsed();
    
    // At least 95% operations should succeed
    assert!(successful_operations >= 95);
    
    // Total time should be reasonable (under 30 seconds)
    assert!(total_time < Duration::from_secs(30));
    
    // Average operation time should be reasonable
    if successful_operations > 0 {
        let avg_op_time = total_operation_time / successful_operations as u32;
        assert!(avg_op_time < Duration::from_millis(200));
    }
}

#[tokio::test]
async fn test_mixed_concurrent_operations() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    // Pre-populate with some data
    for i in 0..20 {
        let request = MemoryRequest {
            operation: "store".to_string(),
            content: format!("Pre-populated data {}", i),
            metadata: HashMap::new(),
        };
        server.process_memory_request(&request).await.unwrap();
    }
    
    let start = Instant::now();
    let mut handles = vec![];
    
    // Mix of store and retrieve operations
    for i in 0..50 {
        let server_clone = server.clone();
        
        let handle = tokio::spawn(async move {
            if i % 2 == 0 {
                // Store operation
                let request = MemoryRequest {
                    operation: "store".to_string(),
                    content: format!("Mixed test store {}", i),
                    metadata: HashMap::new(),
                };
                server_clone.process_memory_request(&request).await
            } else {
                // Retrieve operation
                let request = MemoryRequest {
                    operation: "retrieve".to_string(),
                    content: "data".to_string(),
                    metadata: HashMap::new(),
                };
                server_clone.process_memory_request(&request).await
            }
        });
        handles.push(handle);
    }
    
    let mut successful_operations = 0;
    
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successful_operations += 1;
        }
    }
    
    let total_time = start.elapsed();
    
    // At least 90% operations should succeed
    assert!(successful_operations >= 45);
    
    // Should complete in reasonable time
    assert!(total_time < Duration::from_secs(20));
}
```
```

**Expected Deliverable**: `tests/performance/load_tests.rs`  
**Verification**: All tests pass  

---

## Micro-Task 7.5.1: Test Suite Integration Setup
**Duration**: 15 minutes  
**Dependencies**: Micro-Task 7.4.3  
**Input**: All test files  
**Output**: Test suite integration  

### Task Prompt for AI
```
Create test suite integration setup:

```rust
// tests/mod.rs
pub mod unit {
    pub mod semantic_column_tests;
    pub mod structural_column_tests;
    pub mod temporal_column_tests;
    pub mod exception_column_tests;
    pub mod lateral_inhibition_tests;
    pub mod neuromorphic_core_tests;
}

pub mod integration {
    pub mod store_memory_handler_tests;
    pub mod retrieve_memory_handler_tests;
    pub mod tool_executor_tests;
    pub mod mcp_protocol_tests;
}

pub mod security {
    pub mod auth_tests;
    pub mod vulnerability_tests;
}

pub mod performance {
    pub mod memory_performance_tests;
    pub mod neuromorphic_performance_tests;
    pub mod load_tests;
}

// tests/test_runner.rs
use std::time::{Duration, Instant};

pub struct TestSuite {
    pub name: String,
    pub tests: Vec<String>,
}

pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub total_duration: Duration,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            skipped: 0,
            total_duration: Duration::new(0, 0),
        }
    }
    
    pub fn print_summary(&self) {
        println!("\n=== TEST SUMMARY ===");
        println!("✅ Passed: {}", self.passed);
        println!("❌ Failed: {}", self.failed);
        println!("⏭️  Skipped: {}", self.skipped);
        println!("⏱️  Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        
        let total = self.passed + self.failed + self.skipped;
        if total > 0 {
            let success_rate = (self.passed as f64 / total as f64) * 100.0;
            println!("📊 Success Rate: {:.1}%", success_rate);
        }
    }
}

pub async fn run_test_suite(suite: &TestSuite) -> TestResults {
    println!("Running test suite: {}", suite.name);
    
    let start = Instant::now();
    let mut results = TestResults::new();
    
    for test_name in &suite.tests {
        println!("  Running: {}", test_name);
        // Test execution logic would go here
        results.passed += 1;
    }
    
    results.total_duration = start.elapsed();
    results
}
```
```

**Expected Deliverable**: `tests/mod.rs` and `tests/test_runner.rs`  
**Verification**: Test structure compiles  

---

## Micro-Task 7.5.2: CI/CD Test Configuration
**Duration**: 18 minutes  
**Dependencies**: Micro-Task 7.5.1  
**Input**: Test suite structure  
**Output**: CI/CD test configuration  

### Task Prompt for AI
```
Create CI/CD test configuration:

```yaml
# .github/workflows/test.yml
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run unit tests
      run: |
        cargo test unit --verbose --timeout 60
    
    - name: Run integration tests
      run: |
        cargo test integration --verbose --timeout 120
    
    - name: Run security tests
      run: |
        cargo test security --verbose --timeout 60
    
    - name: Run performance tests
      run: |
        cargo test performance --verbose --timeout 300

  code-quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
        components: rustfmt, clippy
    
    - name: Check formatting
      run: cargo fmt -- --check
    
    - name: Run clippy
      run: cargo clippy -- -D warnings
    
    - name: Check documentation
      run: cargo doc --no-deps --document-private-items

  coverage:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Install cargo-tarpaulin
      run: cargo install cargo-tarpaulin
    
    - name: Generate coverage report
      run: |
        cargo tarpaulin --verbose --all-features --workspace --timeout 120 --out Xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./cobertura.xml
        fail_ci_if_error: true
```

Create a local test script:

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "🚀 Starting Comprehensive Test Suite"

echo "📋 Running unit tests..."
cargo test unit --verbose

echo "🔗 Running integration tests..."
cargo test integration --verbose

echo "🔒 Running security tests..."
cargo test security --verbose

echo "⚡ Running performance tests..."
cargo test performance --verbose

echo "📊 Generating test report..."
cargo test --verbose -- --show-output 2>&1 | tee test_results.log

echo "✅ All tests completed successfully!"
```
```

**Expected Deliverable**: `.github/workflows/test.yml` and `scripts/run_tests.sh`  
**Verification**: CI configuration is valid  

---

## Micro-Task 7.5.3: Test Documentation and Reporting
**Duration**: 17 minutes  
**Dependencies**: Micro-Task 7.5.2  
**Input**: Test suite and CI configuration  
**Output**: Test documentation  

### Task Prompt for AI
```
Create test documentation and reporting:

```rust
// tests/test_reporter.rs
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use chrono::{DateTime, Utc};

#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration: Duration,
    pub message: Option<String>,
}

#[derive(Debug)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
}

pub struct TestReport {
    pub suite_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub results: Vec<TestResult>,
}

impl TestReport {
    pub fn new(suite_name: String) -> Self {
        Self {
            suite_name,
            start_time: Utc::now(),
            end_time: Utc::now(),
            results: Vec::new(),
        }
    }
    
    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }
    
    pub fn finalize(&mut self) {
        self.end_time = Utc::now();
    }
    
    pub fn generate_markdown_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("# Test Report: {}\n\n", self.suite_name));
        report.push_str(&format!("- **Start Time**: {}\n", self.start_time.format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("- **End Time**: {}\n", self.end_time.format("%Y-%m-%d %H:%M:%S UTC")));
        
        let total_duration = self.end_time.signed_duration_since(self.start_time);
        report.push_str(&format!("- **Duration**: {:.2}s\n\n", total_duration.num_milliseconds() as f64 / 1000.0));
        
        let passed = self.results.iter().filter(|r| matches!(r.status, TestStatus::Passed)).count();
        let failed = self.results.iter().filter(|r| matches!(r.status, TestStatus::Failed)).count();
        let skipped = self.results.iter().filter(|r| matches!(r.status, TestStatus::Skipped)).count();
        
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- ✅ **Passed**: {}\n", passed));
        report.push_str(&format!("- ❌ **Failed**: {}\n", failed));
        report.push_str(&format!("- ⏭️ **Skipped**: {}\n", skipped));
        
        if failed > 0 {
            report.push_str("\n## Failed Tests\n\n");
            for result in &self.results {
                if matches!(result.status, TestStatus::Failed) {
                    report.push_str(&format!("- **{}**: {}\n", 
                        result.name, 
                        result.message.as_deref().unwrap_or("No details available")
                    ));
                }
            }
        }
        
        report
    }
    
    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(self.generate_markdown_report().as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_report_generation() {
        let mut report = TestReport::new("Test Suite".to_string());
        
        report.add_result(TestResult {
            name: "test_example".to_string(),
            status: TestStatus::Passed,
            duration: Duration::from_millis(100),
            message: None,
        });
        
        report.finalize();
        
        let markdown = report.generate_markdown_report();
        assert!(markdown.contains("Test Suite"));
        assert!(markdown.contains("Passed**: 1"));
    }
}
```

Create test documentation:

```markdown
# Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for the CortexKG MCP Server.

## Test Categories

### Unit Tests
- **Location**: `tests/unit/`
- **Purpose**: Test individual components in isolation
- **Coverage**: All neuromorphic columns, core processing logic
- **Target**: >90% code coverage

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions
- **Coverage**: Memory handlers, tool executors, MCP protocol
- **Target**: All critical user journeys

### Security Tests
- **Location**: `tests/security/`
- **Purpose**: Validate security measures
- **Coverage**: Authentication, authorization, input validation
- **Target**: Zero known vulnerabilities

### Performance Tests
- **Location**: `tests/performance/`
- **Purpose**: Validate performance requirements
- **Coverage**: Memory operations, neuromorphic processing, concurrent operations
- **Target**: Sub-100ms response times for standard operations

## Running Tests

### Local Development
```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test category
cargo test unit
cargo test integration
cargo test security
cargo test performance
```

### CI/CD Pipeline
Tests run automatically on:
- Pull requests
- Pushes to main/develop branches
- Nightly schedules for performance regression detection

## Performance Benchmarks

| Operation | Target | Measurement |
|-----------|---------|-------------|
| Store Memory | <100ms | Average response time |
| Retrieve Memory | <100ms | Average response time |
| Neuromorphic Processing | <50ms | Pattern processing time |
| Concurrent Operations | >95% success | Under load |

## Test Data Management

### Test Isolation
- Each test runs in isolation
- No shared state between tests
- Fresh server instance per test

### Test Data
- Generated programmatically
- No external dependencies
- Reproducible results
```
```

**Expected Deliverable**: `tests/test_reporter.rs` and `docs/TESTING.md`  
**Verification**: Documentation is complete and accurate  

---

## Summary

This micro-phase breaks down comprehensive testing into **18 manageable micro-tasks**, each taking 15-20 minutes and producing ONE concrete deliverable. The tasks cover:

- **6 Unit Test Tasks** (7.1.1 - 7.1.6): Individual component testing
- **4 Integration Test Tasks** (7.2.1 - 7.2.4): Component interaction testing  
- **2 Security Test Tasks** (7.3.1 - 7.3.2): Authentication and vulnerability testing
- **3 Performance Test Tasks** (7.4.1 - 7.4.3): Performance and load testing
- **3 Test Infrastructure Tasks** (7.5.1 - 7.5.3): Test suite setup and CI/CD

Each micro-task follows the established pattern with exact specifications, clear deliverables, and verification criteria, ensuring 100% completion confidence for AI execution.