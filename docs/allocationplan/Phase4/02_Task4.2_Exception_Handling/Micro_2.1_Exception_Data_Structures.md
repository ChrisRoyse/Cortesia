# Micro Phase 2.1: Exception Data Structures

**Estimated Time**: 30 minutes
**Dependencies**: Task 4.1 Complete
**Objective**: Create the core data structures for storing and managing property exceptions

## Task Description

Design and implement the foundational data structures that will store exceptions when inherited properties are overridden by local values.

## Deliverables

Create `src/exceptions/store.rs` with:

1. **Exception struct**: Core exception data with source tracking
2. **ExceptionSource enum**: Track how exceptions were created
3. **ExceptionStore struct**: Efficient storage and retrieval
4. **Exception statistics**: Memory usage and count tracking
5. **Serialization support**: For persistence and network transfer

## Success Criteria

- [ ] Exception struct stores inherited vs actual values
- [ ] ExceptionSource tracks creation method (detected, explicit, promoted)
- [ ] ExceptionStore provides O(1) lookup by (node, property)
- [ ] Memory usage per exception < 200 bytes
- [ ] Serialization roundtrip preserves all data
- [ ] Thread-safe concurrent access

## Implementation Requirements

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Exception {
    pub inherited_value: PropertyValue,
    pub actual_value: PropertyValue,
    pub reason: String,
    pub source: ExceptionSource,
    pub created_at: Instant,
    pub confidence: f32, // 0.0-1.0, how certain we are this is an exception
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExceptionSource {
    Explicit,        // User-defined exception
    Detected,        // Auto-detected during insertion
    Promoted,        // Created during property promotion
    Conflict,        // Multiple inheritance conflict resolution
}

pub struct ExceptionStore {
    exceptions: DashMap<(NodeId, String), Exception>,
    index_by_property: DashMap<String, HashSet<NodeId>>,
    index_by_source: DashMap<ExceptionSource, HashSet<(NodeId, String)>>,
    stats: ExceptionStats,
}

#[derive(Debug, Default)]
pub struct ExceptionStats {
    pub total_exceptions: AtomicUsize,
    pub total_bytes: AtomicUsize,
    pub by_source: HashMap<ExceptionSource, AtomicUsize>,
}

impl ExceptionStore {
    pub fn new() -> Self;
    pub fn add_exception(&self, node: NodeId, property: String, exception: Exception);
    pub fn get_exception(&self, node: NodeId, property: &str) -> Option<Exception>;
    pub fn remove_exception(&self, node: NodeId, property: &str) -> Option<Exception>;
    pub fn get_node_exceptions(&self, node: NodeId) -> HashMap<String, Exception>;
    pub fn get_property_exceptions(&self, property: &str) -> Vec<(NodeId, Exception)>;
    pub fn get_stats(&self) -> ExceptionStats;
}
```

## Test Requirements

Must pass exception storage tests:
```rust
#[test]
fn test_exception_creation_and_storage() {
    let store = ExceptionStore::new();
    let node = NodeId(1);
    
    let exception = Exception {
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "Penguin cannot fly".to_string(),
        source: ExceptionSource::Explicit,
        created_at: Instant::now(),
        confidence: 1.0,
    };
    
    store.add_exception(node, "can_fly".to_string(), exception.clone());
    
    let retrieved = store.get_exception(node, "can_fly");
    assert_eq!(retrieved, Some(exception));
    
    let stats = store.get_stats();
    assert_eq!(stats.total_exceptions.load(Ordering::Relaxed), 1);
}

#[test]
fn test_exception_indexing() {
    let store = ExceptionStore::new();
    
    // Add exceptions for same property across different nodes
    for i in 0..10 {
        let exception = Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: format!("Exception {}", i),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        store.add_exception(NodeId(i), "can_fly".to_string(), exception);
    }
    
    let property_exceptions = store.get_property_exceptions("can_fly");
    assert_eq!(property_exceptions.len(), 10);
}

#[test]
fn test_memory_efficiency() {
    let store = ExceptionStore::new();
    
    // Add 1000 exceptions
    for i in 0..1000 {
        let exception = Exception {
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("special_{}", i)),
            reason: "Test exception".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.9,
        };
        store.add_exception(NodeId(i), "test_prop".to_string(), exception);
    }
    
    let stats = store.get_stats();
    let avg_bytes_per_exception = stats.total_bytes.load(Ordering::Relaxed) / 1000;
    assert!(avg_bytes_per_exception < 200); // < 200 bytes per exception
}
```

## File Location
`src/exceptions/store.rs`

## Next Micro Phase
After completion, proceed to Micro 2.2: Exception Detection Engine