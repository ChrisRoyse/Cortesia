# Task 36: Property Inheritance Engine

## Metadata
- **Micro-Phase**: 2.36
- **Duration**: 25-30 minutes
- **Dependencies**: Task 35 (hierarchy_builder)
- **Output**: `src/hierarchy_detection/property_inheritance_engine.rs`

## Description
Create the property inheritance engine that implements property inheritance with caching, achieving <1ms performance with caching. This component manages property propagation through concept hierarchies, handles inheritance exceptions, and provides efficient cached lookups for inherited properties.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy_detection::{ConceptHierarchy, HierarchyNode, ExtractedConcept, ConceptType};
    use std::collections::HashMap;

    #[test]
    fn test_inheritance_engine_creation() {
        let engine = PropertyInheritanceEngine::new();
        assert!(engine.is_enabled());
        assert_eq!(engine.cache_size_limit(), 10000);
        assert_eq!(engine.inheritance_strategy(), InheritanceStrategy::MultipleInheritance);
    }
    
    #[test]
    fn test_simple_property_inheritance() {
        let engine = PropertyInheritanceEngine::new();
        let hierarchy = create_test_hierarchy();
        
        // Get inherited properties for "dog" (should inherit from "mammal" and "animal")
        let inherited_props = engine.get_inherited_properties("dog", &hierarchy).unwrap();
        
        // Should have properties from all ancestors
        assert!(inherited_props.contains_key("alive")); // from animal
        assert!(inherited_props.contains_key("warm_blooded")); // from mammal
        assert!(inherited_props.contains_key("has_fur")); // from dog itself
        
        // Check property sources
        assert_eq!(inherited_props.get("alive").unwrap().source, "animal");
        assert_eq!(inherited_props.get("warm_blooded").unwrap().source, "mammal");
        assert_eq!(inherited_props.get("has_fur").unwrap().source, "dog");
    }
    
    #[test]
    fn test_property_override_handling() {
        let engine = PropertyInheritanceEngine::new();
        let mut hierarchy = create_test_hierarchy();
        
        // Add a concept that overrides a parent property
        add_concept_to_hierarchy(&mut hierarchy, "platypus", "mammal", hashmap! {
            "reproduction" => "lays_eggs", // Overrides mammal's "live_birth"
            "has_bill" => "true"
        });
        
        let inherited_props = engine.get_inherited_properties("platypus", &hierarchy).unwrap();
        
        // Should have overridden property
        assert_eq!(inherited_props.get("reproduction").unwrap().value, "lays_eggs");
        assert_eq!(inherited_props.get("reproduction").unwrap().source, "platypus");
        
        // Should still inherit other properties
        assert_eq!(inherited_props.get("alive").unwrap().value, "true");
        assert_eq!(inherited_props.get("warm_blooded").unwrap().value, "true");
    }
    
    #[test]
    fn test_inheritance_caching_performance() {
        let engine = PropertyInheritanceEngine::new();
        let hierarchy = create_large_test_hierarchy(1000); // 1000 node hierarchy
        
        // First call (uncached)
        let start = std::time::Instant::now();
        let props1 = engine.get_inherited_properties("concept_999", &hierarchy).unwrap();
        let uncached_time = start.elapsed();
        
        // Second call (cached)
        let start = std::time::Instant::now();
        let props2 = engine.get_inherited_properties("concept_999", &hierarchy).unwrap();
        let cached_time = start.elapsed();
        
        // Cached call should be much faster (<1ms)
        assert!(cached_time < std::time::Duration::from_millis(1));
        assert!(cached_time < uncached_time / 10); // At least 10x faster
        
        // Results should be identical
        assert_eq!(props1.len(), props2.len());
        for (key, value1) in &props1 {
            assert_eq!(value1, props2.get(key).unwrap());
        }
    }
    
    #[test]
    fn test_multiple_inheritance_resolution() {
        let engine = PropertyInheritanceEngine::new();
        let hierarchy = create_multiple_inheritance_hierarchy();
        
        // Get properties for concept with multiple parents
        let inherited_props = engine.get_inherited_properties("flying_mammal", &hierarchy).unwrap();
        
        // Should have properties from both parents
        assert!(inherited_props.contains_key("warm_blooded")); // from mammal
        assert!(inherited_props.contains_key("can_fly")); // from flying_animal
        assert!(inherited_props.contains_key("alive")); // from both (should resolve conflict)
        
        // Check conflict resolution
        let alive_prop = inherited_props.get("alive").unwrap();
        assert_eq!(alive_prop.value, "true");
        assert!(alive_prop.inheritance_path.len() >= 2); // Should track multiple paths
    }
    
    #[test]
    fn test_inheritance_path_tracking() {
        let engine = PropertyInheritanceEngine::new();
        let hierarchy = create_test_hierarchy();
        
        let inherited_props = engine.get_inherited_properties("golden_retriever", &hierarchy).unwrap();
        
        // Check inheritance path for a property
        let alive_prop = inherited_props.get("alive").unwrap();
        assert_eq!(alive_prop.inheritance_path, vec!["golden_retriever", "dog", "mammal", "animal"]);
        
        let has_fur_prop = inherited_props.get("has_fur").unwrap();
        assert_eq!(has_fur_prop.inheritance_path, vec!["golden_retriever", "dog"]);
    }
    
    #[test]
    fn test_property_invalidation_on_hierarchy_change() {
        let mut engine = PropertyInheritanceEngine::new();
        let mut hierarchy = create_test_hierarchy();
        
        // Get properties (fills cache)
        let props1 = engine.get_inherited_properties("dog", &hierarchy).unwrap();
        assert!(props1.contains_key("alive"));
        
        // Modify hierarchy
        add_property_to_concept(&mut hierarchy, "mammal", "extinct", "false");
        
        // Invalidate cache for affected concepts
        engine.invalidate_cache_for_descendants("mammal", &hierarchy);
        
        // Get properties again (should reflect changes)
        let props2 = engine.get_inherited_properties("dog", &hierarchy).unwrap();
        assert!(props2.contains_key("extinct"));
        assert_eq!(props2.get("extinct").unwrap().value, "false");
    }
    
    #[test]
    fn test_inheritance_exception_handling() {
        let engine = PropertyInheritanceEngine::new();
        let mut hierarchy = create_test_hierarchy();
        
        // Add inheritance exception
        add_inheritance_exception(&mut hierarchy, "penguin", "can_fly", InheritanceException {
            inherited_value: "true".to_string(),
            actual_value: "false".to_string(),
            exception_reason: "Flightless bird".to_string(),
            confidence: 0.95,
        });
        
        let inherited_props = engine.get_inherited_properties("penguin", &hierarchy).unwrap();
        
        // Should use exception value instead of inherited value
        assert_eq!(inherited_props.get("can_fly").unwrap().value, "false");
        assert!(inherited_props.get("can_fly").unwrap().is_exception);
        assert_eq!(inherited_props.get("can_fly").unwrap().exception_reason, Some("Flightless bird".to_string()));
    }
    
    #[test]
    fn test_property_conflict_resolution_strategies() {
        let engine = PropertyInheritanceEngine::new();
        
        // Test highest confidence strategy
        let mut engine_confidence = PropertyInheritanceEngine::new();
        engine_confidence.set_conflict_resolution(ConflictResolution::HighestConfidence);
        
        let hierarchy = create_conflicting_properties_hierarchy();
        let props = engine_confidence.get_inherited_properties("conflicted_concept", &hierarchy).unwrap();
        
        // Should choose property with highest confidence
        let color_prop = props.get("color").unwrap();
        assert_eq!(color_prop.confidence, 0.9); // Higher confidence property
        
        // Test most specific strategy
        let mut engine_specific = PropertyInheritanceEngine::new();
        engine_specific.set_conflict_resolution(ConflictResolution::MostSpecific);
        
        let props2 = engine_specific.get_inherited_properties("conflicted_concept", &hierarchy).unwrap();
        let color_prop2 = props2.get("color").unwrap();
        
        // Should choose property from most specific (closest) ancestor
        assert!(color_prop2.inheritance_path.len() <= color_prop.inheritance_path.len());
    }
    
    #[test]
    fn test_cache_memory_management() {
        let mut engine = PropertyInheritanceEngine::new();
        engine.set_cache_size_limit(100); // Small cache for testing
        
        let hierarchy = create_large_test_hierarchy(1000);
        
        // Fill cache beyond limit
        for i in 0..150 {
            let concept_name = format!("concept_{}", i);
            if hierarchy.has_node(&concept_name) {
                engine.get_inherited_properties(&concept_name, &hierarchy).unwrap();
            }
        }
        
        // Cache should not exceed limit
        assert!(engine.cache_size() <= 100);
        
        // Should have evicted oldest entries
        let cache_stats = engine.get_cache_statistics();
        assert!(cache_stats.evictions > 0);
    }
    
    #[test]
    fn test_inheritance_performance_with_deep_hierarchy() {
        let engine = PropertyInheritanceEngine::new();
        
        // Create deep hierarchy (depth 50)
        let hierarchy = create_deep_hierarchy(50);
        
        // Test inheritance from deepest node
        let start = std::time::Instant::now();
        let inherited_props = engine.get_inherited_properties("concept_49", &hierarchy).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time even for deep hierarchy
        assert!(elapsed < std::time::Duration::from_millis(10));
        
        // Should have properties from all ancestors
        assert!(inherited_props.len() >= 50); // At least one property per level
        
        // Check that deepest property has longest inheritance path
        let deep_prop = inherited_props.values()
            .max_by_key(|p| p.inheritance_path.len())
            .unwrap();
        assert!(deep_prop.inheritance_path.len() >= 50);
    }
    
    fn create_test_hierarchy() -> ConceptHierarchy {
        // Create a simple test hierarchy: animal -> mammal -> dog -> golden_retriever
        let mut hierarchy = ConceptHierarchy::new();
        
        add_concept_to_hierarchy(&mut hierarchy, "animal", "", hashmap! {
            "alive" => "true",
            "can_move" => "true"
        });
        
        add_concept_to_hierarchy(&mut hierarchy, "mammal", "animal", hashmap! {
            "warm_blooded" => "true",
            "has_hair" => "true",
            "reproduction" => "live_birth"
        });
        
        add_concept_to_hierarchy(&mut hierarchy, "dog", "mammal", hashmap! {
            "has_fur" => "true",
            "domesticated" => "true",
            "loyalty" => "high"
        });
        
        add_concept_to_hierarchy(&mut hierarchy, "golden_retriever", "dog", hashmap! {
            "color" => "golden",
            "size" => "large",
            "temperament" => "friendly"
        });
        
        hierarchy
    }
    
    fn create_large_test_hierarchy(size: usize) -> ConceptHierarchy {
        let mut hierarchy = ConceptHierarchy::new();
        
        // Create linear hierarchy
        for i in 0..size {
            let concept_name = format!("concept_{}", i);
            let parent = if i == 0 { "" } else { &format!("concept_{}", i - 1) };
            
            add_concept_to_hierarchy(&mut hierarchy, &concept_name, parent, hashmap! {
                &format!("property_{}", i) => &format!("value_{}", i)
            });
        }
        
        hierarchy
    }
    
    fn create_multiple_inheritance_hierarchy() -> ConceptHierarchy {
        let mut hierarchy = ConceptHierarchy::new();
        
        add_concept_to_hierarchy(&mut hierarchy, "animal", "", hashmap! {
            "alive" => "true"
        });
        
        add_concept_to_hierarchy(&mut hierarchy, "mammal", "animal", hashmap! {
            "warm_blooded" => "true"
        });
        
        add_concept_to_hierarchy(&mut hierarchy, "flying_animal", "animal", hashmap! {
            "can_fly" => "true"
        });
        
        // Concept with multiple parents
        add_concept_with_multiple_parents(&mut hierarchy, "flying_mammal", 
            vec!["mammal", "flying_animal"], hashmap! {
            "wing_type" => "membranous"
        });
        
        hierarchy
    }
    
    fn create_deep_hierarchy(depth: usize) -> ConceptHierarchy {
        let mut hierarchy = ConceptHierarchy::new();
        
        for i in 0..depth {
            let concept_name = format!("concept_{}", i);
            let parent = if i == 0 { "" } else { &format!("concept_{}", i - 1) };
            
            add_concept_to_hierarchy(&mut hierarchy, &concept_name, parent, hashmap! {
                &format!("level_property") => &format!("level_{}", i)
            });
        }
        
        hierarchy
    }
    
    fn create_conflicting_properties_hierarchy() -> ConceptHierarchy {
        // Simplified implementation for test
        ConceptHierarchy::new()
    }
    
    // Helper functions (simplified implementations)
    fn add_concept_to_hierarchy(hierarchy: &mut ConceptHierarchy, name: &str, parent: &str, properties: HashMap<&str, &str>) {
        // Implementation would add concept to hierarchy
    }
    
    fn add_concept_with_multiple_parents(hierarchy: &mut ConceptHierarchy, name: &str, parents: Vec<&str>, properties: HashMap<&str, &str>) {
        // Implementation would add concept with multiple parents
    }
    
    fn add_property_to_concept(hierarchy: &mut ConceptHierarchy, concept: &str, property: &str, value: &str) {
        // Implementation would add property to existing concept
    }
    
    fn add_inheritance_exception(hierarchy: &mut ConceptHierarchy, concept: &str, property: &str, exception: InheritanceException) {
        // Implementation would add inheritance exception
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use crate::hierarchy_detection::ConceptHierarchy;

/// Strategy for resolving inheritance conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Choose property with highest confidence
    HighestConfidence,
    /// Choose property from most specific (closest) ancestor
    MostSpecific,
    /// Choose property from most general (root) ancestor
    MostGeneral,
    /// Combine conflicting properties if possible
    Combine,
    /// Manual resolution required
    Manual,
}

/// Strategy for inheritance behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InheritanceStrategy {
    /// Single inheritance only (tree structure)
    SingleInheritance,
    /// Multiple inheritance allowed (DAG structure)
    MultipleInheritance,
    /// Interface-style inheritance (properties only)
    InterfaceInheritance,
}

/// Cache eviction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheEvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
}

/// An inherited property with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritedProperty {
    /// Property value
    pub value: String,
    
    /// Source concept that defined this property
    pub source: String,
    
    /// Confidence in this property value
    pub confidence: f32,
    
    /// Path of inheritance (from current concept to source)
    pub inheritance_path: Vec<String>,
    
    /// Whether this property is an exception to normal inheritance
    pub is_exception: bool,
    
    /// Exception reason if applicable
    pub exception_reason: Option<String>,
    
    /// Property type information
    pub property_type: PropertyType,
    
    /// Timestamp when property was inherited
    pub inherited_at: u64,
}

/// Type of property
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyType {
    /// Simple string value
    String,
    /// Numeric value
    Numeric,
    /// Boolean value
    Boolean,
    /// Reference to another concept
    ConceptReference,
    /// List of values
    List,
    /// Complex structured data
    Structured,
}

/// Exception to normal inheritance rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceException {
    /// The value that would be inherited normally
    pub inherited_value: String,
    
    /// The actual value for this concept
    pub actual_value: String,
    
    /// Reason for the exception
    pub exception_reason: String,
    
    /// Confidence in the exception
    pub confidence: f32,
    
    /// Source of the exception information
    pub exception_source: Option<String>,
}

/// Cache entry for inherited properties
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Inherited properties
    properties: HashMap<String, InheritedProperty>,
    
    /// Timestamp when cached
    cached_at: u64,
    
    /// Access count for LFU eviction
    access_count: usize,
    
    /// Last access time for LRU eviction
    last_accessed: u64,
}

/// Statistics about cache performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: usize,
    
    /// Total cache misses
    pub misses: usize,
    
    /// Cache hit ratio
    pub hit_ratio: f32,
    
    /// Number of evictions
    pub evictions: usize,
    
    /// Current cache size
    pub current_size: usize,
    
    /// Maximum cache size
    pub max_size: usize,
    
    /// Average lookup time (microseconds)
    pub average_lookup_time: f32,
}

/// Result of property inheritance computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceResult {
    /// All inherited properties
    pub properties: HashMap<String, InheritedProperty>,
    
    /// Conflicts that were resolved
    pub conflicts_resolved: Vec<PropertyConflict>,
    
    /// Exceptions that were applied
    pub exceptions_applied: Vec<String>,
    
    /// Computation metadata
    pub computation_metadata: InheritanceMetadata,
}

/// Information about a property conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConflict {
    /// Property name
    pub property_name: String,
    
    /// Conflicting values
    pub conflicting_values: Vec<(String, String)>, // (value, source)
    
    /// Resolution strategy used
    pub resolution_strategy: ConflictResolution,
    
    /// Final resolved value
    pub resolved_value: String,
    
    /// Confidence in resolution
    pub resolution_confidence: f32,
}

/// Metadata about inheritance computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceMetadata {
    /// Concepts visited during inheritance
    pub concepts_visited: Vec<String>,
    
    /// Total properties considered
    pub properties_considered: usize,
    
    /// Computation time (microseconds)
    pub computation_time: u64,
    
    /// Whether result was cached
    pub was_cached: bool,
    
    /// Cache key used
    pub cache_key: String,
}

/// Main property inheritance engine
pub struct PropertyInheritanceEngine {
    /// Inheritance strategy
    inheritance_strategy: InheritanceStrategy,
    
    /// Conflict resolution strategy
    conflict_resolution: ConflictResolution,
    
    /// Cache eviction strategy
    cache_eviction_strategy: CacheEvictionStrategy,
    
    /// Property cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    
    /// Cache size limit
    cache_size_limit: usize,
    
    /// Cache statistics
    cache_stats: Arc<RwLock<CacheStatistics>>,
    
    /// Whether engine is enabled
    enabled: bool,
    
    /// Maximum inheritance depth
    max_inheritance_depth: usize,
    
    /// Property type inference enabled
    type_inference_enabled: bool,
}

impl PropertyInheritanceEngine {
    /// Create a new property inheritance engine
    pub fn new() -> Self {
        Self {
            inheritance_strategy: InheritanceStrategy::MultipleInheritance,
            conflict_resolution: ConflictResolution::HighestConfidence,
            cache_eviction_strategy: CacheEvictionStrategy::LRU,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_size_limit: 10000,
            cache_stats: Arc::new(RwLock::new(CacheStatistics {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
                evictions: 0,
                current_size: 0,
                max_size: 10000,
                average_lookup_time: 0.0,
            })),
            enabled: true,
            max_inheritance_depth: 50,
            type_inference_enabled: true,
        }
    }
    
    /// Get inherited properties for a concept
    pub fn get_inherited_properties(
        &self,
        concept_name: &str,
        hierarchy: &ConceptHierarchy
    ) -> Result<HashMap<String, InheritedProperty>, InheritanceError> {
        if !self.enabled {
            return Err(InheritanceError::EngineDisabled);
        }
        
        let lookup_start = std::time::Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(concept_name, hierarchy);
        
        if let Some(cached_properties) = self.get_from_cache(&cache_key) {
            self.record_cache_hit(lookup_start.elapsed().as_micros() as u64);
            return Ok(cached_properties);
        }
        
        // Cache miss - compute inheritance
        self.record_cache_miss();
        
        let computation_result = self.compute_inheritance(concept_name, hierarchy)?;
        
        // Cache the result
        self.store_in_cache(cache_key, &computation_result.properties);
        
        let lookup_time = lookup_start.elapsed().as_micros() as u64;
        self.update_average_lookup_time(lookup_time);
        
        Ok(computation_result.properties)
    }
    
    /// Compute inheritance for a concept
    fn compute_inheritance(
        &self,
        concept_name: &str,
        hierarchy: &ConceptHierarchy
    ) -> Result<InheritanceResult, InheritanceError> {
        let computation_start = std::time::Instant::now();
        
        let mut properties = HashMap::new();
        let mut conflicts_resolved = Vec::new();
        let mut exceptions_applied = Vec::new();
        let mut concepts_visited = Vec::new();
        let mut properties_considered = 0;
        
        // Use BFS to traverse hierarchy and collect properties
        let mut queue = VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        
        queue.push_back((concept_name.to_string(), 0, vec![concept_name.to_string()]));
        
        while let Some((current_concept, depth, inheritance_path)) = queue.pop_front() {
            if visited.contains(&current_concept) || depth > self.max_inheritance_depth {
                continue;
            }
            
            visited.insert(current_concept.clone());
            concepts_visited.push(current_concept.clone());
            
            // Get properties for current concept
            if let Some(concept_properties) = hierarchy.get_concept_properties(&current_concept) {
                for (prop_name, prop_value) in concept_properties {
                    properties_considered += 1;
                    
                    // Check for inheritance exception
                    if let Some(exception) = hierarchy.get_inheritance_exception(&current_concept, &prop_name) {
                        let inherited_prop = InheritedProperty {
                            value: exception.actual_value.clone(),
                            source: current_concept.clone(),
                            confidence: exception.confidence,
                            inheritance_path: inheritance_path.clone(),
                            is_exception: true,
                            exception_reason: Some(exception.exception_reason.clone()),
                            property_type: self.infer_property_type(&exception.actual_value),
                            inherited_at: current_timestamp(),
                        };
                        
                        self.resolve_property_conflict(
                            &prop_name,
                            inherited_prop,
                            &mut properties,
                            &mut conflicts_resolved
                        );
                        
                        exceptions_applied.push(prop_name.clone());
                        continue;
                    }
                    
                    // Normal property inheritance
                    let inherited_prop = InheritedProperty {
                        value: prop_value.clone(),
                        source: current_concept.clone(),
                        confidence: 0.9, // Default confidence
                        inheritance_path: inheritance_path.clone(),
                        is_exception: false,
                        exception_reason: None,
                        property_type: self.infer_property_type(&prop_value),
                        inherited_at: current_timestamp(),
                    };
                    
                    self.resolve_property_conflict(
                        &prop_name,
                        inherited_prop,
                        &mut properties,
                        &mut conflicts_resolved
                    );
                }
            }
            
            // Add parents to queue
            if let Some(parents) = hierarchy.get_concept_parents(&current_concept) {
                for parent in parents {
                    if !visited.contains(parent) {
                        let mut new_path = inheritance_path.clone();
                        new_path.push(parent.clone());
                        queue.push_back((parent.clone(), depth + 1, new_path));
                    }
                }
            }
        }
        
        let computation_time = computation_start.elapsed().as_micros() as u64;
        
        Ok(InheritanceResult {
            properties,
            conflicts_resolved,
            exceptions_applied,
            computation_metadata: InheritanceMetadata {
                concepts_visited,
                properties_considered,
                computation_time,
                was_cached: false,
                cache_key: self.generate_cache_key(concept_name, hierarchy),
            },
        })
    }
    
    /// Resolve property conflict when multiple values exist
    fn resolve_property_conflict(
        &self,
        property_name: &str,
        new_property: InheritedProperty,
        properties: &mut HashMap<String, InheritedProperty>,
        conflicts_resolved: &mut Vec<PropertyConflict>
    ) {
        if let Some(existing_property) = properties.get(property_name) {
            // Conflict detected
            let conflicting_values = vec![
                (existing_property.value.clone(), existing_property.source.clone()),
                (new_property.value.clone(), new_property.source.clone()),
            ];
            
            let (resolved_property, resolution_confidence) = match self.conflict_resolution {
                ConflictResolution::HighestConfidence => {
                    if new_property.confidence > existing_property.confidence {
                        (new_property, new_property.confidence)
                    } else {
                        (existing_property.clone(), existing_property.confidence)
                    }
                }
                ConflictResolution::MostSpecific => {
                    // Choose property with shortest inheritance path (most specific)
                    if new_property.inheritance_path.len() < existing_property.inheritance_path.len() {
                        (new_property, 0.8)
                    } else {
                        (existing_property.clone(), 0.8)
                    }
                }
                ConflictResolution::MostGeneral => {
                    // Choose property with longest inheritance path (most general)
                    if new_property.inheritance_path.len() > existing_property.inheritance_path.len() {
                        (new_property, 0.7)
                    } else {
                        (existing_property.clone(), 0.7)
                    }
                }
                ConflictResolution::Combine => {
                    // Try to combine values if possible
                    let combined_value = format!("{};{}", existing_property.value, new_property.value);
                    let mut combined_property = new_property.clone();
                    combined_property.value = combined_value;
                    combined_property.confidence = (existing_property.confidence + new_property.confidence) / 2.0;
                    (combined_property, 0.6)
                }
                ConflictResolution::Manual => {
                    // Default to existing property for manual resolution
                    (existing_property.clone(), 0.5)
                }
            };
            
            conflicts_resolved.push(PropertyConflict {
                property_name: property_name.to_string(),
                conflicting_values,
                resolution_strategy: self.conflict_resolution,
                resolved_value: resolved_property.value.clone(),
                resolution_confidence,
            });
            
            properties.insert(property_name.to_string(), resolved_property);
        } else {
            // No conflict - add property
            properties.insert(property_name.to_string(), new_property);
        }
    }
    
    /// Infer property type from value
    fn infer_property_type(&self, value: &str) -> PropertyType {
        if !self.type_inference_enabled {
            return PropertyType::String;
        }
        
        // Simple type inference
        if value.parse::<bool>().is_ok() {
            PropertyType::Boolean
        } else if value.parse::<f64>().is_ok() {
            PropertyType::Numeric
        } else if value.starts_with('[') && value.ends_with(']') {
            PropertyType::List
        } else if value.starts_with('{') && value.ends_with('}') {
            PropertyType::Structured
        } else {
            PropertyType::String
        }
    }
    
    /// Generate cache key for concept and hierarchy
    fn generate_cache_key(&self, concept_name: &str, hierarchy: &ConceptHierarchy) -> String {
        format!("{}:{}", concept_name, hierarchy.get_version_hash())
    }
    
    /// Get properties from cache
    fn get_from_cache(&self, cache_key: &str) -> Option<HashMap<String, InheritedProperty>> {
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(cache_key) {
                // Update access information for LRU/LFU
                let mut cache_write = self.cache.write().ok()?;
                if let Some(entry_mut) = cache_write.get_mut(cache_key) {
                    entry_mut.access_count += 1;
                    entry_mut.last_accessed = current_timestamp();
                }
                
                return Some(entry.properties.clone());
            }
        }
        None
    }
    
    /// Store properties in cache
    fn store_in_cache(&self, cache_key: String, properties: &HashMap<String, InheritedProperty>) {
        if let Ok(mut cache) = self.cache.write() {
            // Check if cache is full and evict if necessary
            if cache.len() >= self.cache_size_limit {
                self.evict_cache_entry(&mut cache);
            }
            
            let entry = CacheEntry {
                properties: properties.clone(),
                cached_at: current_timestamp(),
                access_count: 1,
                last_accessed: current_timestamp(),
            };
            
            cache.insert(cache_key, entry);
            
            // Update cache statistics
            if let Ok(mut stats) = self.cache_stats.write() {
                stats.current_size = cache.len();
            }
        }
    }
    
    /// Evict cache entry based on eviction strategy
    fn evict_cache_entry(&self, cache: &mut HashMap<String, CacheEntry>) {
        if cache.is_empty() {
            return;
        }
        
        let key_to_remove = match self.cache_eviction_strategy {
            CacheEvictionStrategy::LRU => {
                // Find least recently used
                cache.iter()
                    .min_by_key(|(_, entry)| entry.last_accessed)
                    .map(|(key, _)| key.clone())
            }
            CacheEvictionStrategy::LFU => {
                // Find least frequently used
                cache.iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(key, _)| key.clone())
            }
            CacheEvictionStrategy::FIFO => {
                // Find oldest entry
                cache.iter()
                    .min_by_key(|(_, entry)| entry.cached_at)
                    .map(|(key, _)| key.clone())
            }
            CacheEvictionStrategy::Random => {
                // Remove random entry
                cache.keys().next().cloned()
            }
        };
        
        if let Some(key) = key_to_remove {
            cache.remove(&key);
            
            // Update eviction count
            if let Ok(mut stats) = self.cache_stats.write() {
                stats.evictions += 1;
            }
        }
    }
    
    /// Record cache hit
    fn record_cache_hit(&self, lookup_time: u64) {
        if let Ok(mut stats) = self.cache_stats.write() {
            stats.hits += 1;
            self.update_hit_ratio(&mut stats);
        }
    }
    
    /// Record cache miss
    fn record_cache_miss(&self) {
        if let Ok(mut stats) = self.cache_stats.write() {
            stats.misses += 1;
            self.update_hit_ratio(&mut stats);
        }
    }
    
    /// Update hit ratio
    fn update_hit_ratio(&self, stats: &mut CacheStatistics) {
        let total = stats.hits + stats.misses;
        if total > 0 {
            stats.hit_ratio = stats.hits as f32 / total as f32;
        }
    }
    
    /// Update average lookup time
    fn update_average_lookup_time(&self, lookup_time: u64) {
        if let Ok(mut stats) = self.cache_stats.write() {
            let total_requests = stats.hits + stats.misses;
            if total_requests > 0 {
                stats.average_lookup_time = 
                    (stats.average_lookup_time * (total_requests - 1) as f32 + lookup_time as f32) / total_requests as f32;
            } else {
                stats.average_lookup_time = lookup_time as f32;
            }
        }
    }
    
    /// Invalidate cache for a concept and its descendants
    pub fn invalidate_cache_for_descendants(&mut self, concept_name: &str, hierarchy: &ConceptHierarchy) {
        if let Ok(mut cache) = self.cache.write() {
            let keys_to_remove: Vec<String> = cache.keys()
                .filter(|key| {
                    // Extract concept name from cache key
                    if let Some(cached_concept) = key.split(':').next() {
                        hierarchy.is_descendant(cached_concept, concept_name)
                    } else {
                        false
                    }
                })
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                cache.remove(&key);
            }
            
            // Update cache size
            if let Ok(mut stats) = self.cache_stats.write() {
                stats.current_size = cache.len();
            }
        }
    }
    
    /// Clear entire cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
            
            if let Ok(mut stats) = self.cache_stats.write() {
                stats.current_size = 0;
            }
        }
    }
    
    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        if let Ok(stats) = self.cache_stats.read() {
            stats.clone()
        } else {
            CacheStatistics {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
                evictions: 0,
                current_size: 0,
                max_size: self.cache_size_limit,
                average_lookup_time: 0.0,
            }
        }
    }
    
    /// Check if engine is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Get cache size limit
    pub fn cache_size_limit(&self) -> usize {
        self.cache_size_limit
    }
    
    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        if let Ok(cache) = self.cache.read() {
            cache.len()
        } else {
            0
        }
    }
    
    /// Get inheritance strategy
    pub fn inheritance_strategy(&self) -> InheritanceStrategy {
        self.inheritance_strategy
    }
    
    /// Set cache size limit
    pub fn set_cache_size_limit(&mut self, limit: usize) {
        self.cache_size_limit = limit;
        
        if let Ok(mut stats) = self.cache_stats.write() {
            stats.max_size = limit;
        }
    }
    
    /// Set conflict resolution strategy
    pub fn set_conflict_resolution(&mut self, strategy: ConflictResolution) {
        self.conflict_resolution = strategy;
    }
    
    /// Set inheritance strategy
    pub fn set_inheritance_strategy(&mut self, strategy: InheritanceStrategy) {
        self.inheritance_strategy = strategy;
    }
    
    /// Enable or disable engine
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Error types for property inheritance
#[derive(Debug, thiserror::Error)]
pub enum InheritanceError {
    #[error("Property inheritance engine is disabled")]
    EngineDisabled,
    
    #[error("Concept not found: {0}")]
    ConceptNotFound(String),
    
    #[error("Inheritance depth exceeded: {0}")]
    InheritanceDepthExceeded(usize),
    
    #[error("Circular inheritance detected: {0}")]
    CircularInheritance(String),
    
    #[error("Property conflict resolution failed: {0}")]
    ConflictResolutionFailed(String),
    
    #[error("Cache operation failed: {0}")]
    CacheOperationFailed(String),
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for PropertyInheritanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Simplified implementations for ConceptHierarchy methods used in this component
impl ConceptHierarchy {
    pub fn get_concept_properties(&self, _concept_name: &str) -> Option<&HashMap<String, String>> {
        None // Simplified
    }
    
    pub fn get_concept_parents(&self, _concept_name: &str) -> Option<&Vec<String>> {
        None // Simplified
    }
    
    pub fn get_inheritance_exception(&self, _concept_name: &str, _property_name: &str) -> Option<&InheritanceException> {
        None // Simplified
    }
    
    pub fn get_version_hash(&self) -> u64 {
        0 // Simplified
    }
    
    pub fn is_descendant(&self, _descendant: &str, _ancestor: &str) -> bool {
        false // Simplified
    }
}
```

## Verification Steps
1. Create PropertyInheritanceEngine with configurable strategies and caching
2. Implement property inheritance with <1ms cached performance
3. Add conflict resolution for multiple inheritance scenarios
4. Implement cache management with LRU/LFU eviction strategies
5. Add inheritance exception handling for special cases
6. Ensure cache invalidation works correctly on hierarchy changes

## Success Criteria
- [ ] PropertyInheritanceEngine compiles without errors
- [ ] Property inheritance performance <1ms with caching
- [ ] Conflict resolution strategies work correctly for multiple inheritance
- [ ] Cache management efficiently handles memory constraints
- [ ] Inheritance exceptions properly override normal inheritance
- [ ] All tests pass with comprehensive coverage