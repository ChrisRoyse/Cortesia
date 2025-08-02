# AI Prompt: Micro Phase 2.1 - Exception Data Structures

You are tasked with creating the core data structures for storing and managing property exceptions in the inheritance system. Your goal is to create `src/exceptions/store.rs` with efficient exception storage that tracks when inherited properties are overridden by local values.

## Your Task
Implement the foundational data structures that store exceptions when inherited properties are overridden, including source tracking, efficient storage/retrieval, and comprehensive statistics.

## Specific Requirements
1. Create `src/exceptions/store.rs` with Exception and ExceptionStore structs
2. Implement ExceptionSource enum to track how exceptions were created
3. Provide O(1) lookup performance by (node, property) key
4. Keep memory usage per exception under 200 bytes
5. Add serialization support for persistence and network transfer
6. Ensure thread-safe concurrent access using appropriate data structures
7. Implement comprehensive statistics tracking for performance monitoring

## Expected Code Structure
You must implement these exact signatures:

```rust
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exception {
    pub inherited_value: PropertyValue,
    pub actual_value: PropertyValue,
    pub reason: String,
    pub source: ExceptionSource,
    pub created_at: Instant,
    pub confidence: f32, // 0.0-1.0, how certain we are this is an exception
}

impl Exception {
    pub fn new(
        inherited_value: PropertyValue,
        actual_value: PropertyValue,
        reason: String,
        source: ExceptionSource,
        confidence: f32,
    ) -> Self {
        Self {
            inherited_value,
            actual_value,
            reason,
            source,
            created_at: Instant::now(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
    
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
    
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
    
    pub fn memory_size(&self) -> usize {
        // Estimate memory usage for this exception
        std::mem::size_of::<Self>()
            + self.inherited_value.size_estimate()
            + self.actual_value.size_estimate()
            + self.reason.capacity()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExceptionSource {
    Explicit,        // User-defined exception
    Detected,        // Auto-detected during insertion
    Promoted,        // Created during property promotion
    Conflict,        // Multiple inheritance conflict resolution
    Imported,        // Loaded from external source
}

impl ExceptionSource {
    pub fn is_automatic(&self) -> bool {
        matches!(self, ExceptionSource::Detected | ExceptionSource::Conflict)
    }
    
    pub fn priority(&self) -> u8 {
        match self {
            ExceptionSource::Explicit => 100,
            ExceptionSource::Promoted => 80,
            ExceptionSource::Conflict => 60,
            ExceptionSource::Detected => 40,
            ExceptionSource::Imported => 20,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ExceptionKey {
    node_id: NodeId,
    property_name: String,
}

impl ExceptionKey {
    fn new(node_id: NodeId, property_name: &str) -> Self {
        Self {
            node_id,
            property_name: property_name.to_string(),
        }
    }
}

pub struct ExceptionStore {
    exceptions: DashMap<ExceptionKey, Exception>,
    index_by_property: DashMap<String, HashSet<NodeId>>,
    index_by_source: DashMap<ExceptionSource, HashSet<ExceptionKey>>,
    index_by_node: DashMap<NodeId, HashSet<String>>,
    stats: ExceptionStats,
}

impl ExceptionStore {
    pub fn new() -> Self {
        Self {
            exceptions: DashMap::new(),
            index_by_property: DashMap::new(),
            index_by_source: DashMap::new(),
            index_by_node: DashMap::new(),
            stats: ExceptionStats::default(),
        }
    }
    
    pub fn add_exception(&self, node: NodeId, property: String, exception: Exception) {
        let key = ExceptionKey::new(node, &property);
        
        // Check if this replaces an existing exception
        let is_replacement = self.exceptions.contains_key(&key);
        
        // Insert the exception
        self.exceptions.insert(key.clone(), exception.clone());
        
        // Update indices
        self.index_by_property
            .entry(property.clone())
            .or_insert_with(HashSet::new)
            .insert(node);
        
        self.index_by_source
            .entry(exception.source.clone())
            .or_insert_with(HashSet::new)
            .insert(key.clone());
        
        self.index_by_node
            .entry(node)
            .or_insert_with(HashSet::new)
            .insert(property);
        
        // Update statistics
        if is_replacement {
            self.stats.record_update();
        } else {
            self.stats.record_addition();
        }
        
        self.stats.update_memory_usage(exception.memory_size() as i64);
    }
    
    pub fn get_exception(&self, node: NodeId, property: &str) -> Option<Exception> {
        let key = ExceptionKey::new(node, property);
        self.exceptions.get(&key).map(|entry| {
            self.stats.record_access();
            entry.clone()
        })
    }
    
    pub fn remove_exception(&self, node: NodeId, property: &str) -> Option<Exception> {
        let key = ExceptionKey::new(node, property);
        
        if let Some((_, exception)) = self.exceptions.remove(&key) {
            // Update indices
            if let Some(mut property_index) = self.index_by_property.get_mut(property) {
                property_index.remove(&node);
            }
            
            if let Some(mut source_index) = self.index_by_source.get_mut(&exception.source) {
                source_index.remove(&key);
            }
            
            if let Some(mut node_index) = self.index_by_node.get_mut(&node) {
                node_index.remove(property);
            }
            
            self.stats.record_removal();
            self.stats.update_memory_usage(-(exception.memory_size() as i64));
            
            Some(exception)
        } else {
            None
        }
    }
    
    pub fn get_node_exceptions(&self, node: NodeId) -> HashMap<String, Exception> {
        let mut result = HashMap::new();
        
        if let Some(properties) = self.index_by_node.get(&node) {
            for property in properties.iter() {
                if let Some(exception) = self.get_exception(node, property) {
                    result.insert(property.clone(), exception);
                }
            }
        }
        
        result
    }
    
    pub fn get_property_exceptions(&self, property: &str) -> Vec<(NodeId, Exception)> {
        let mut result = Vec::new();
        
        if let Some(nodes) = self.index_by_property.get(property) {
            for &node in nodes.iter() {
                if let Some(exception) = self.get_exception(node, property) {
                    result.push((node, exception));
                }
            }
        }
        
        result
    }
    
    pub fn get_exceptions_by_source(&self, source: &ExceptionSource) -> Vec<(NodeId, String, Exception)> {
        let mut result = Vec::new();
        
        if let Some(keys) = self.index_by_source.get(source) {
            for key in keys.iter() {
                if let Some(exception) = self.exceptions.get(key) {
                    result.push((key.node_id, key.property_name.clone(), exception.clone()));
                }
            }
        }
        
        result
    }
    
    pub fn get_high_confidence_exceptions(&self, min_confidence: f32) -> Vec<(NodeId, String, Exception)> {
        self.exceptions
            .iter()
            .filter(|kv| kv.value().confidence >= min_confidence)
            .map(|kv| (kv.key().node_id, kv.key().property_name.clone(), kv.value().clone()))
            .collect()
    }
    
    pub fn clear_node_exceptions(&self, node: NodeId) {
        let properties: Vec<String> = self.index_by_node
            .get(&node)
            .map(|props| props.iter().cloned().collect())
            .unwrap_or_default();
        
        for property in properties {
            self.remove_exception(node, &property);
        }
    }
    
    pub fn clear_property_exceptions(&self, property: &str) {
        let nodes: Vec<NodeId> = self.index_by_property
            .get(property)
            .map(|nodes| nodes.iter().cloned().collect())
            .unwrap_or_default();
        
        for node in nodes {
            self.remove_exception(node, property);
        }
    }
    
    pub fn get_stats(&self) -> ExceptionStats {
        // Update live statistics
        let mut stats = self.stats.clone();
        stats.total_exceptions.store(self.exceptions.len(), Ordering::Relaxed);
        stats
    }
    
    pub fn size(&self) -> usize {
        self.exceptions.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.exceptions.is_empty()
    }
    
    pub fn compact(&self) {
        // Remove empty index entries
        self.index_by_property.retain(|_, nodes| !nodes.is_empty());
        self.index_by_source.retain(|_, keys| !keys.is_empty());
        self.index_by_node.retain(|_, properties| !properties.is_empty());
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExceptionStats {
    pub total_exceptions: AtomicUsize,
    pub total_bytes: AtomicUsize,
    pub by_source: HashMap<ExceptionSource, AtomicUsize>,
    pub additions: AtomicU64,
    pub removals: AtomicU64,
    pub updates: AtomicU64,
    pub accesses: AtomicU64,
}

impl ExceptionStats {
    fn record_addition(&self) {
        self.additions.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_removal(&self) {
        self.removals.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_access(&self) {
        self.accesses.fetch_add(1, Ordering::Relaxed);
    }
    
    fn update_memory_usage(&self, delta: i64) {
        if delta > 0 {
            self.total_bytes.fetch_add(delta as usize, Ordering::Relaxed);
        } else {
            self.total_bytes.fetch_sub((-delta) as usize, Ordering::Relaxed);
        }
    }
    
    pub fn average_exception_size(&self) -> f64 {
        let total = self.total_exceptions.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.total_bytes.load(Ordering::Relaxed) as f64 / total as f64
        }
    }
    
    pub fn operations_count(&self) -> u64 {
        self.additions.load(Ordering::Relaxed)
            + self.removals.load(Ordering::Relaxed)
            + self.updates.load(Ordering::Relaxed)
            + self.accesses.load(Ordering::Relaxed)
    }
}

impl Default for ExceptionStore {
    fn default() -> Self {
        Self::new()
    }
}

// Utility functions for exception analysis
impl ExceptionStore {
    pub fn analyze_patterns(&self) -> ExceptionAnalysis {
        let mut analysis = ExceptionAnalysis::default();
        
        for kv in self.exceptions.iter() {
            let exception = kv.value();
            
            // Analyze by source
            *analysis.source_distribution.entry(exception.source.clone()).or_insert(0) += 1;
            
            // Analyze by confidence
            if exception.confidence >= 0.9 {
                analysis.high_confidence_count += 1;
            } else if exception.confidence >= 0.7 {
                analysis.medium_confidence_count += 1;
            } else {
                analysis.low_confidence_count += 1;
            }
            
            // Analyze by age
            let age = exception.age();
            if age < std::time::Duration::from_secs(3600) {
                analysis.recent_exceptions += 1;
            }
        }
        
        analysis
    }
}

#[derive(Debug, Default)]
pub struct ExceptionAnalysis {
    pub source_distribution: HashMap<ExceptionSource, usize>,
    pub high_confidence_count: usize,
    pub medium_confidence_count: usize,
    pub low_confidence_count: usize,
    pub recent_exceptions: usize,
}
```

## Success Criteria (You must verify these)
- [ ] Exception struct stores inherited vs actual values correctly
- [ ] ExceptionSource tracks creation method (detected, explicit, promoted, conflict, imported)
- [ ] ExceptionStore provides O(1) lookup by (node, property) key
- [ ] Memory usage per exception < 200 bytes (verified through testing)
- [ ] Serialization roundtrip preserves all data exactly
- [ ] Thread-safe concurrent access without data races or deadlocks
- [ ] Comprehensive indexing supports efficient queries by node, property, and source
- [ ] Statistics tracking provides accurate performance metrics
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_exception_creation_and_storage() {
        let store = ExceptionStore::new();
        let node = NodeId(1);
        
        let exception = Exception::new(
            PropertyValue::Boolean(true),
            PropertyValue::Boolean(false),
            "Penguin cannot fly".to_string(),
            ExceptionSource::Explicit,
            1.0,
        );
        
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
            let exception = Exception::new(
                PropertyValue::Boolean(true),
                PropertyValue::Boolean(false),
                format!("Exception {}", i),
                ExceptionSource::Detected,
                0.8,
            );
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
            let exception = Exception::new(
                PropertyValue::String("default".to_string()),
                PropertyValue::String(format!("special_{}", i)),
                "Test exception".to_string(),
                ExceptionSource::Detected,
                0.9,
            );
            store.add_exception(NodeId(i), "test_prop".to_string(), exception);
        }
        
        let stats = store.get_stats();
        let avg_bytes_per_exception = stats.average_exception_size();
        assert!(avg_bytes_per_exception < 200.0); // < 200 bytes per exception
    }

    #[test]
    fn test_exception_source_filtering() {
        let store = ExceptionStore::new();
        
        // Add exceptions with different sources
        let sources = [
            ExceptionSource::Explicit,
            ExceptionSource::Detected,
            ExceptionSource::Promoted,
            ExceptionSource::Conflict,
        ];
        
        for (i, source) in sources.iter().enumerate() {
            let exception = Exception::new(
                PropertyValue::Integer(0),
                PropertyValue::Integer(i as i64),
                "Test".to_string(),
                source.clone(),
                0.8,
            );
            store.add_exception(NodeId(i as u64), "value".to_string(), exception);
        }
        
        let explicit_exceptions = store.get_exceptions_by_source(&ExceptionSource::Explicit);
        assert_eq!(explicit_exceptions.len(), 1);
        
        let detected_exceptions = store.get_exceptions_by_source(&ExceptionSource::Detected);
        assert_eq!(detected_exceptions.len(), 1);
    }

    #[test]
    fn test_high_confidence_filtering() {
        let store = ExceptionStore::new();
        
        // Add exceptions with varying confidence
        let confidences = [0.5, 0.7, 0.9, 0.95, 1.0];
        
        for (i, &confidence) in confidences.iter().enumerate() {
            let exception = Exception::new(
                PropertyValue::Float(0.0),
                PropertyValue::Float(confidence),
                "Test".to_string(),
                ExceptionSource::Detected,
                confidence,
            );
            store.add_exception(NodeId(i as u64), "confidence".to_string(), exception);
        }
        
        let high_confidence = store.get_high_confidence_exceptions(0.8);
        assert_eq!(high_confidence.len(), 3); // 0.9, 0.95, 1.0
    }

    #[test]
    fn test_concurrent_access() {
        let store = Arc::new(ExceptionStore::new());
        let num_threads = 8;
        let exceptions_per_thread = 100;
        
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let store = store.clone();
            thread::spawn(move || {
                for i in 0..exceptions_per_thread {
                    let exception = Exception::new(
                        PropertyValue::Integer(0),
                        PropertyValue::Integer(i),
                        format!("Thread {} exception {}", thread_id, i),
                        ExceptionSource::Detected,
                        0.8,
                    );
                    
                    let node = NodeId(thread_id as u64 * 1000 + i as u64);
                    store.add_exception(node, "test_prop".to_string(), exception);
                    
                    // Immediately try to retrieve it
                    let retrieved = store.get_exception(node, "test_prop");
                    assert!(retrieved.is_some());
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(store.size(), num_threads * exceptions_per_thread);
    }

    #[test]
    fn test_serialization() {
        let exception = Exception::new(
            PropertyValue::String("inherited".to_string()),
            PropertyValue::String("actual".to_string()),
            "Test serialization".to_string(),
            ExceptionSource::Explicit,
            0.95,
        );
        
        let serialized = serde_json::to_string(&exception).unwrap();
        let deserialized: Exception = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(exception.inherited_value, deserialized.inherited_value);
        assert_eq!(exception.actual_value, deserialized.actual_value);
        assert_eq!(exception.reason, deserialized.reason);
        assert_eq!(exception.source, deserialized.source);
        assert_eq!(exception.confidence, deserialized.confidence);
    }

    #[test]
    fn test_exception_analysis() {
        let store = ExceptionStore::new();
        
        // Add diverse exceptions
        for i in 0..50 {
            let source = match i % 4 {
                0 => ExceptionSource::Explicit,
                1 => ExceptionSource::Detected,
                2 => ExceptionSource::Promoted,
                _ => ExceptionSource::Conflict,
            };
            
            let confidence = match i % 3 {
                0 => 0.95, // High
                1 => 0.75, // Medium
                _ => 0.5,  // Low
            };
            
            let exception = Exception::new(
                PropertyValue::Integer(0),
                PropertyValue::Integer(i),
                "Analysis test".to_string(),
                source,
                confidence,
            );
            
            store.add_exception(NodeId(i as u64), "test".to_string(), exception);
        }
        
        let analysis = store.analyze_patterns();
        assert!(!analysis.source_distribution.is_empty());
        assert!(analysis.high_confidence_count > 0);
        assert!(analysis.medium_confidence_count > 0);
        assert!(analysis.low_confidence_count > 0);
    }

    #[test]
    fn test_exception_removal_and_cleanup() {
        let store = ExceptionStore::new();
        let node = NodeId(1);
        
        let exception = Exception::new(
            PropertyValue::Boolean(true),
            PropertyValue::Boolean(false),
            "Test removal".to_string(),
            ExceptionSource::Explicit,
            1.0,
        );
        
        store.add_exception(node, "test_prop".to_string(), exception.clone());
        assert_eq!(store.size(), 1);
        
        let removed = store.remove_exception(node, "test_prop");
        assert_eq!(removed, Some(exception));
        assert_eq!(store.size(), 0);
        
        // Verify indices are cleaned up
        assert!(store.get_node_exceptions(node).is_empty());
        assert!(store.get_property_exceptions("test_prop").is_empty());
    }
}
```

## File to Create
Create exactly this file: `src/exceptions/store.rs`

## Dependencies Required
You may need to add dependencies to Cargo.toml:
```toml
[dependencies]
dashmap = "5.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## When Complete
Respond with "MICRO PHASE 2.1 COMPLETE" and a brief summary of what you implemented, including:
- Exception storage strategy used
- Indexing approach for efficient queries
- Memory efficiency techniques employed
- Thread safety mechanisms
- Confirmation that all tests pass