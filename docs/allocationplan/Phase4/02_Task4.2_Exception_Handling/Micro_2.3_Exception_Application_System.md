# Micro Phase 2.3: Exception Application System

**Estimated Time**: 35 minutes
**Dependencies**: Micro 2.2 (Exception Detection Engine)
**Objective**: Implement the exception handler that applies exceptions during property resolution and lookups

## Task Description

Create a system that applies stored exceptions when properties are being resolved, ensuring that exceptional values override inherited values during property lookups. This handler integrates with the inheritance system to provide the correct property values when exceptions exist.

The system must efficiently handle exception application during inheritance traversal while maintaining performance for both exception and non-exception cases.

## Deliverables

Create `src/exceptions/handler.rs` with:

1. **ExceptionHandler struct**: Core handler for applying exceptions during resolution
2. **Property resolution**: Override inherited values with exceptional values
3. **Exception caching**: Cache frequently accessed exceptions for performance
4. **Conflict resolution**: Handle multiple applicable exceptions
5. **Performance monitoring**: Track application performance and hit rates

## Success Criteria

- [ ] Exceptions correctly override inherited values during property resolution
- [ ] Handler integrates seamlessly with existing inheritance system
- [ ] Exception lookup performs in < 100μs per property
- [ ] Cache hit rate > 90% for frequently accessed properties
- [ ] Conflict resolution follows predictable precedence rules
- [ ] Thread-safe concurrent exception application

## Implementation Requirements

```rust
pub struct ExceptionHandler {
    exception_store: Arc<ExceptionStore>,
    cache: DashMap<(NodeId, String), CachedException>,
    cache_stats: CacheStatistics,
    conflict_resolver: ConflictResolver,
}

#[derive(Debug, Clone)]
pub struct CachedException {
    exception: Exception,
    cached_at: Instant,
    access_count: AtomicU64,
}

#[derive(Debug, Clone)]
pub enum ExceptionConflict {
    MultipleExceptions(Vec<Exception>),
    SourceConflict(Exception, Exception),
    ValueTypeConflict(Exception, PropertyValue),
}

pub struct ConflictResolver {
    precedence_rules: Vec<ConflictRule>,
}

#[derive(Debug, Clone)]
pub struct ConflictRule {
    sources: Vec<ExceptionSource>,
    resolution_strategy: ResolutionStrategy,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    HighestConfidence,
    MostRecent,
    ExplicitFirst,
    AverageValues,
}

impl ExceptionHandler {
    pub fn new(exception_store: Arc<ExceptionStore>) -> Self;
    
    pub fn resolve_property(
        &self,
        node: NodeId,
        property: &str,
        inherited_value: Option<PropertyValue>
    ) -> Option<PropertyValue>;
    
    pub fn apply_exceptions_to_node(
        &self,
        node: NodeId,
        properties: &mut HashMap<String, PropertyValue>
    );
    
    pub fn handle_exception_conflict(
        &self,
        conflicts: Vec<Exception>
    ) -> Exception;
    
    pub fn cache_exception(&self, node: NodeId, property: String, exception: Exception);
    
    pub fn get_cached_exception(&self, node: NodeId, property: &str) -> Option<Exception>;
    
    pub fn clear_cache_for_node(&self, node: NodeId);
    
    pub fn get_handler_statistics(&self) -> HandlerStatistics;
}

#[derive(Debug, Default)]
pub struct HandlerStatistics {
    pub total_resolutions: AtomicU64,
    pub exceptions_applied: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub conflicts_resolved: AtomicU64,
    pub avg_resolution_time_micros: AtomicU64,
}

#[derive(Debug, Default)]
pub struct CacheStatistics {
    pub size: AtomicUsize,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
}
```

## Test Requirements

Must pass exception application tests:
```rust
#[test]
fn test_basic_exception_application() {
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(store.clone());
    
    // Setup exception
    let exception = Exception {
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "Penguin cannot fly".to_string(),
        source: ExceptionSource::Explicit,
        created_at: Instant::now(),
        confidence: 1.0,
    };
    
    store.add_exception(NodeId(1), "can_fly".to_string(), exception);
    
    // Test property resolution
    let result = handler.resolve_property(
        NodeId(1),
        "can_fly",
        Some(PropertyValue::Boolean(true))
    );
    
    assert_eq!(result, Some(PropertyValue::Boolean(false)));
}

#[test]
fn test_exception_application_to_node() {
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(store.clone());
    
    // Add multiple exceptions
    let exceptions = vec![
        ("can_fly", PropertyValue::Boolean(false)),
        ("color", PropertyValue::String("black".to_string())),
    ];
    
    for (prop, value) in &exceptions {
        let exception = Exception {
            inherited_value: PropertyValue::String("inherited".to_string()),
            actual_value: value.clone(),
            reason: "Override".to_string(),
            source: ExceptionSource::Explicit,
            created_at: Instant::now(),
            confidence: 1.0,
        };
        store.add_exception(NodeId(1), prop.to_string(), exception);
    }
    
    let mut properties = HashMap::new();
    properties.insert("can_fly".to_string(), PropertyValue::Boolean(true));
    properties.insert("color".to_string(), PropertyValue::String("inherited".to_string()));
    properties.insert("size".to_string(), PropertyValue::String("medium".to_string()));
    
    handler.apply_exceptions_to_node(NodeId(1), &mut properties);
    
    assert_eq!(properties.get("can_fly"), Some(&PropertyValue::Boolean(false)));
    assert_eq!(properties.get("color"), Some(&PropertyValue::String("black".to_string())));
    assert_eq!(properties.get("size"), Some(&PropertyValue::String("medium".to_string()))); // Unchanged
}

#[test]
fn test_exception_caching() {
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(store.clone());
    
    let exception = Exception {
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "Test".to_string(),
        source: ExceptionSource::Explicit,
        created_at: Instant::now(),
        confidence: 1.0,
    };
    
    store.add_exception(NodeId(1), "test_prop".to_string(), exception.clone());
    
    // First access - cache miss
    let result1 = handler.resolve_property(NodeId(1), "test_prop", Some(PropertyValue::Boolean(true)));
    
    // Second access - cache hit
    let result2 = handler.resolve_property(NodeId(1), "test_prop", Some(PropertyValue::Boolean(true)));
    
    assert_eq!(result1, result2);
    
    let stats = handler.get_handler_statistics();
    assert!(stats.cache_hits.load(Ordering::Relaxed) >= 1);
}

#[test]
fn test_conflict_resolution() {
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(store.clone());
    
    // Create conflicting exceptions
    let exception1 = Exception {
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "First reason".to_string(),
        source: ExceptionSource::Explicit,
        created_at: Instant::now() - Duration::from_secs(10),
        confidence: 0.8,
    };
    
    let exception2 = Exception {
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "Second reason".to_string(),
        source: ExceptionSource::Detected,
        created_at: Instant::now(),
        confidence: 0.9,
    };
    
    let resolved = handler.handle_exception_conflict(vec![exception1, exception2]);
    
    // Should resolve to higher confidence exception
    assert_eq!(resolved.confidence, 0.9);
    assert_eq!(resolved.reason, "Second reason");
}

#[test]
fn test_handler_performance() {
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(store.clone());
    
    // Add many exceptions
    for i in 0..1000 {
        let exception = Exception {
            inherited_value: PropertyValue::String("inherited".to_string()),
            actual_value: PropertyValue::String(format!("exception_{}", i)),
            reason: "Test".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        store.add_exception(NodeId(i), "test_prop".to_string(), exception);
    }
    
    // Test resolution performance
    let start = Instant::now();
    for i in 0..1000 {
        handler.resolve_property(
            NodeId(i),
            "test_prop",
            Some(PropertyValue::String("inherited".to_string()))
        );
    }
    let elapsed = start.elapsed();
    
    let avg_micros = elapsed.as_micros() / 1000;
    assert!(avg_micros < 100); // < 100μs per resolution
}
```

## File Location
`src/exceptions/handler.rs`

## Next Micro Phase
After completion, proceed to Micro 2.4: Exception Pattern Learning