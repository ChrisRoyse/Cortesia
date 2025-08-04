# Task 38: Hierarchy Validation Tests

## Metadata
- **Micro-Phase**: 2.38
- **Duration**: 30-35 minutes
- **Dependencies**: Task 34 (concept_extraction_core), Task 35 (hierarchy_builder), Task 36 (property_inheritance_engine), Task 37 (exception_detection_system)
- **Output**: `src/hierarchy_detection/hierarchy_validation_tests.rs`

## Description
Create comprehensive validation and testing for the complete hierarchy detection system, ensuring integration of all components (concept extraction, hierarchy building, property inheritance, and exception detection) with end-to-end testing scenarios and performance validation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy_detection::{
        ConceptExtractionCore, HierarchyBuilder, PropertyInheritanceEngine, 
        ExceptionDetectionSystem, ConceptHierarchy, ValidatedFact
    };
    use crate::quality_integration::{FactContent, ConfidenceComponents};
    use std::collections::HashMap;

    #[test]
    fn test_complete_hierarchy_system_creation() {
        let system = CompleteHierarchySystem::new();
        assert!(system.is_enabled());
        assert!(system.concept_extractor.is_enabled());
        assert!(system.hierarchy_builder.is_enabled());
        assert!(system.inheritance_engine.is_enabled());
        assert!(system.exception_detector.is_enabled());
    }
    
    #[test]
    fn test_end_to_end_animal_hierarchy_processing() {
        let system = CompleteHierarchySystem::new();
        
        let animal_facts = vec![
            "Animals are living organisms that can move",
            "Mammals are warm-blooded animals that give live birth",
            "Birds are animals that have feathers and most can fly",
            "Dogs are domesticated mammals that are loyal companions",
            "Penguins are birds that cannot fly but are excellent swimmers",
            "Golden retrievers are large, friendly dogs with golden fur",
            "Platypus is a mammal that lays eggs and has a bill like a duck"
        ];
        
        let result = system.process_facts_to_hierarchy(animal_facts).unwrap();
        
        // Verify hierarchy structure
        assert!(result.hierarchy.has_node("animal"));
        assert!(result.hierarchy.has_node("mammal"));
        assert!(result.hierarchy.has_node("bird"));
        assert!(result.hierarchy.has_node("dog"));
        assert!(result.hierarchy.has_node("penguin"));
        assert!(result.hierarchy.has_node("golden_retriever"));
        assert!(result.hierarchy.has_node("platypus"));
        
        // Verify hierarchical relationships
        assert!(result.hierarchy.is_ancestor("animal", "golden_retriever"));
        assert!(result.hierarchy.is_ancestor("mammal", "dog"));
        assert!(result.hierarchy.is_ancestor("bird", "penguin"));
        
        // Verify exceptions were detected
        assert!(result.detected_exceptions.iter().any(|e| 
            e.concept_name == "penguin" && e.property_name == "can_fly"));
        assert!(result.detected_exceptions.iter().any(|e| 
            e.concept_name == "platypus" && e.property_name == "reproduction"));
        
        // Verify overall quality
        assert!(result.processing_confidence > 0.8);
        assert!(result.hierarchy_quality_score > 0.85);
    }
    
    #[test]
    fn test_botanical_classification_with_exceptions() {
        let system = CompleteHierarchySystem::new();
        
        let botanical_facts = vec![
            "Plants are living organisms that produce their own food",
            "Fruits are the seed-bearing parts of plants",
            "Vegetables are plant parts used in cooking",
            "Tomatoes are botanically fruits but culinarily used as vegetables",
            "Carrots are root vegetables that are orange",
            "Strawberries are aggregate fruits with seeds on the outside",
            "Rhubarb is botanically a vegetable but used as a fruit in cooking"
        ];
        
        let result = system.process_facts_to_hierarchy(botanical_facts).unwrap();
        
        // Verify complex classification
        assert!(result.hierarchy.has_node("plant"));
        assert!(result.hierarchy.has_node("fruit"));
        assert!(result.hierarchy.has_node("vegetable"));
        assert!(result.hierarchy.has_node("tomato"));
        assert!(result.hierarchy.has_node("rhubarb"));
        
        // Verify classification exceptions
        let tomato_exceptions: Vec<_> = result.detected_exceptions.iter()
            .filter(|e| e.concept_name == "tomato")
            .collect();
        assert!(!tomato_exceptions.is_empty());
        
        let rhubarb_exceptions: Vec<_> = result.detected_exceptions.iter()
            .filter(|e| e.concept_name == "rhubarb")
            .collect();
        assert!(!rhubarb_exceptions.is_empty());
    }
    
    #[test]
    fn test_vehicle_hierarchy_with_inheritance() {
        let system = CompleteHierarchySystem::new();
        
        let vehicle_facts = vec![
            "Vehicles are machines used for transportation",
            "Cars are motor vehicles with four wheels",
            "Boats are vehicles that travel on water",
            "Airplanes are vehicles that fly through the air",
            "Sports cars are high-performance cars designed for speed",
            "Sailboats are boats powered by wind",
            "Submarines are underwater boats that can dive"
        ];
        
        let result = system.process_facts_to_hierarchy(vehicle_facts).unwrap();
        
        // Test property inheritance
        let sports_car_properties = system.inheritance_engine
            .get_inherited_properties("sports_car", &result.hierarchy)
            .unwrap();
        
        // Should inherit vehicle properties
        assert!(sports_car_properties.contains_key("transportation") || 
                sports_car_properties.contains_key("machine"));
        
        // Should inherit car properties
        assert!(sports_car_properties.contains_key("wheels") || 
                sports_car_properties.contains_key("motor"));
        
        // Test submarine exception (underwater vs water surface)
        let submarine_exceptions: Vec<_> = result.detected_exceptions.iter()
            .filter(|e| e.concept_name == "submarine")
            .collect();
        
        if !submarine_exceptions.is_empty() {
            let habitat_exception = submarine_exceptions.iter()
                .find(|e| e.property_name.contains("habitat") || e.property_name.contains("environment"));
            assert!(habitat_exception.is_some());
        }
    }
    
    #[test]
    fn test_performance_with_large_fact_set() {
        let system = CompleteHierarchySystem::new();
        
        // Generate 100 facts about a complex domain
        let mut large_fact_set = Vec::new();
        
        // Base categories
        large_fact_set.push("Living things are organisms that are alive".to_string());
        large_fact_set.push("Animals are living things that can move".to_string());
        large_fact_set.push("Plants are living things that make their own food".to_string());
        
        // Generate mammal hierarchy (30 facts)
        for i in 0..10 {
            large_fact_set.push(format!("Mammal species {} is a warm-blooded animal", i));
            large_fact_set.push(format!("Mammal species {} gives live birth", i));
            large_fact_set.push(format!("Mammal species {} has hair or fur", i));
        }
        
        // Generate bird hierarchy (30 facts)
        for i in 0..10 {
            large_fact_set.push(format!("Bird species {} has feathers and can fly", i));
            large_fact_set.push(format!("Bird species {} lays eggs", i));
            large_fact_set.push(format!("Bird species {} has a beak", i));
        }
        
        // Generate exceptions (10 facts)
        large_fact_set.push("Penguin species 0 is a bird that cannot fly".to_string());
        large_fact_set.push("Penguin species 0 is an excellent swimmer".to_string());
        large_fact_set.push("Ostrich species 1 is a bird that cannot fly".to_string());
        large_fact_set.push("Bat species 0 is a mammal that can fly".to_string());
        large_fact_set.push("Platypus species 0 is a mammal that lays eggs".to_string());
        large_fact_set.push("Whale species 0 is a mammal that lives in water".to_string());
        large_fact_set.push("Dolphin species 1 is a mammal that lives in water".to_string());
        large_fact_set.push("Echidna species 0 is a mammal that lays eggs".to_string());
        large_fact_set.push("Kiwi species 0 is a bird that cannot fly".to_string());
        large_fact_set.push("Flying fish species 0 is a fish that can glide".to_string());
        
        // Remaining facts to reach 100
        for i in 10..27 {
            large_fact_set.push(format!("Species {} has various characteristics", i));
        }
        
        assert_eq!(large_fact_set.len(), 100);
        
        let start = std::time::Instant::now();
        let result = system.process_facts_to_hierarchy(large_fact_set).unwrap();
        let elapsed = start.elapsed();
        
        // Performance requirements
        assert!(elapsed < std::time::Duration::from_secs(5)); // Should complete in under 5 seconds
        
        // Quality requirements
        assert!(result.hierarchy.node_count() >= 50); // Should extract many concepts
        assert!(result.detected_exceptions.len() >= 5); // Should detect exceptions
        assert!(result.processing_confidence > 0.7); // Should maintain quality
        
        println!("Processed 100 facts in {:?}, created {} nodes, detected {} exceptions", 
                elapsed, result.hierarchy.node_count(), result.detected_exceptions.len());
    }
    
    #[test]
    fn test_concurrent_processing() {
        use std::sync::Arc;
        use std::thread;
        
        let system = Arc::new(CompleteHierarchySystem::new());
        
        let fact_sets = vec![
            vec!["Dogs are mammals", "Cats are mammals", "Mammals are animals"],
            vec!["Cars have wheels", "Bikes have wheels", "Vehicles have wheels"],
            vec!["Apples are fruits", "Oranges are fruits", "Fruits are plant parts"],
            vec!["Birds can fly", "Penguins cannot fly", "Penguins are birds"],
        ];
        
        let handles: Vec<_> = fact_sets.into_iter().enumerate().map(|(i, facts)| {
            let system_clone = Arc::clone(&system);
            thread::spawn(move || {
                let result = system_clone.process_facts_to_hierarchy(facts);
                (i, result)
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // All should succeed
        for (i, result) in results {
            assert!(result.is_ok(), "Thread {} failed: {:?}", i, result.err());
            let hierarchy_result = result.unwrap();
            assert!(hierarchy_result.hierarchy.node_count() > 0);
        }
    }
    
    #[test]
    fn test_error_handling_and_recovery() {
        let system = CompleteHierarchySystem::new();
        
        // Test with malformed facts
        let malformed_facts = vec![
            "", // Empty fact
            "This is not a proper fact about concepts", // Unclear fact
            "Something something darkside", // Nonsensical
            "Dogs are good boys", // Valid but informal
            "Cats are independent animals", // Valid
        ];
        
        let result = system.process_facts_to_hierarchy(malformed_facts);
        
        // Should handle errors gracefully
        assert!(result.is_ok());
        let hierarchy_result = result.unwrap();
        
        // Should extract valid concepts despite errors
        assert!(hierarchy_result.hierarchy.node_count() >= 2); // At least cats and dogs
        
        // Should report processing issues
        assert!(hierarchy_result.processing_warnings.len() > 0);
    }
    
    #[test]
    fn test_incremental_hierarchy_updates() {
        let mut system = CompleteHierarchySystem::new();
        
        // Initial facts
        let initial_facts = vec![
            "Animals are living things",
            "Mammals are animals",
            "Dogs are mammals"
        ];
        
        let initial_result = system.process_facts_to_hierarchy(initial_facts).unwrap();
        assert_eq!(initial_result.hierarchy.node_count(), 3);
        
        // Additional facts
        let additional_facts = vec![
            "Golden retrievers are dogs",
            "Labradors are dogs",
            "Birds are animals",
            "Penguins are birds that cannot fly"
        ];
        
        let update_result = system.update_hierarchy_with_facts(additional_facts).unwrap();
        
        // Should have more nodes
        assert!(update_result.hierarchy.node_count() > initial_result.hierarchy.node_count());
        
        // Should maintain consistency
        assert!(update_result.hierarchy.is_ancestor("animal", "golden_retriever"));
        assert!(update_result.hierarchy.is_ancestor("dog", "labrador"));
        
        // Should detect new exceptions
        let penguin_exceptions: Vec<_> = update_result.detected_exceptions.iter()
            .filter(|e| e.concept_name == "penguin")
            .collect();
        assert!(!penguin_exceptions.is_empty());
    }
    
    #[test]
    fn test_hierarchy_consistency_validation() {
        let system = CompleteHierarchySystem::new();
        
        let facts = vec![
            "A is a B",
            "B is a C", 
            "C is a D",
            "D is an A" // This creates a cycle
        ];
        
        let result = system.process_facts_to_hierarchy(facts).unwrap();
        
        // System should detect and resolve cycles
        assert!(!result.hierarchy.has_cycles());
        assert!(result.cycles_detected > 0);
        assert!(result.cycles_resolved > 0);
        
        // Should still create valid hierarchy
        assert!(result.hierarchy.node_count() >= 4);
        assert!(result.hierarchy_quality_score > 0.5); // Should maintain reasonable quality
    }
    
    #[test]
    fn test_memory_usage_and_cleanup() {
        let system = CompleteHierarchySystem::new();
        
        // Process multiple fact sets and ensure memory is managed
        for i in 0..10 {
            let facts = vec![
                format!("Category {} is a type of thing", i),
                format!("Subcategory {} is part of category {}", i, i),
                format!("Item {} belongs to subcategory {}", i, i),
            ];
            
            let _result = system.process_facts_to_hierarchy(facts).unwrap();
            
            // Check cache sizes don't grow unbounded
            let cache_stats = system.inheritance_engine.get_cache_statistics();
            assert!(cache_stats.current_size <= cache_stats.max_size);
        }
        
        // Test manual cleanup
        system.cleanup_caches();
        let cache_stats = system.inheritance_engine.get_cache_statistics();
        assert_eq!(cache_stats.current_size, 0);
    }
    
    #[test]
    fn test_confidence_score_accuracy() {
        let system = CompleteHierarchySystem::new();
        
        // High-quality facts should produce high confidence
        let high_quality_facts = vec![
            "Mammals are warm-blooded vertebrate animals",
            "Dogs are domesticated mammals in the family Canidae",
            "Golden retrievers are a breed of dog known for their friendly temperament"
        ];
        
        let high_quality_result = system.process_facts_to_hierarchy(high_quality_facts).unwrap();
        assert!(high_quality_result.processing_confidence > 0.8);
        assert!(high_quality_result.hierarchy_quality_score > 0.85);
        
        // Lower-quality facts should produce lower confidence
        let low_quality_facts = vec![
            "Things exist",
            "Some things are other things",
            "Stuff happens with things"
        ];
        
        let low_quality_result = system.process_facts_to_hierarchy(low_quality_facts).unwrap();
        assert!(low_quality_result.processing_confidence < high_quality_result.processing_confidence);
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::hierarchy_detection::{
    ConceptExtractionCore, HierarchyBuilder, PropertyInheritanceEngine, 
    ExceptionDetectionSystem, ConceptHierarchy, ExtractedConcept, 
    ConceptRelationship, DetectedException
};
use crate::quality_integration::ValidatedFact;

/// Result of complete hierarchy processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyProcessingResult {
    /// The constructed hierarchy
    pub hierarchy: ConceptHierarchy,
    
    /// All extracted concepts
    pub extracted_concepts: Vec<ExtractedConcept>,
    
    /// All detected relationships
    pub concept_relationships: Vec<ConceptRelationship>,
    
    /// All detected exceptions
    pub detected_exceptions: Vec<ConceptException>,
    
    /// Overall processing confidence (0.0-1.0)
    pub processing_confidence: f32,
    
    /// Hierarchy quality score (0.0-1.0)
    pub hierarchy_quality_score: f32,
    
    /// Number of cycles detected and resolved
    pub cycles_detected: usize,
    pub cycles_resolved: usize,
    
    /// Processing warnings and issues
    pub processing_warnings: Vec<String>,
    
    /// Performance metrics
    pub performance_metrics: ProcessingPerformanceMetrics,
    
    /// Processing metadata
    pub processing_metadata: ProcessingMetadata,
}

/// Exception detected in hierarchy with additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptException {
    /// Concept name that has the exception
    pub concept_name: String,
    
    /// Property name with exception
    pub property_name: String,
    
    /// Inherited value
    pub inherited_value: String,
    
    /// Actual value
    pub actual_value: String,
    
    /// Exception confidence
    pub confidence: f32,
    
    /// Exception reason
    pub reason: String,
    
    /// Source of the exception
    pub source: String,
}

/// Performance metrics for hierarchy processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPerformanceMetrics {
    /// Total processing time (milliseconds)
    pub total_processing_time: u64,
    
    /// Concept extraction time
    pub concept_extraction_time: u64,
    
    /// Hierarchy building time
    pub hierarchy_building_time: u64,
    
    /// Property inheritance time
    pub property_inheritance_time: u64,
    
    /// Exception detection time
    pub exception_detection_time: u64,
    
    /// Facts processed per second
    pub facts_per_second: f32,
    
    /// Concepts extracted per second
    pub concepts_per_second: f32,
}

/// Metadata about processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Number of input facts
    pub input_facts_count: usize,
    
    /// Number of valid facts processed
    pub processed_facts_count: usize,
    
    /// Number of concepts extracted
    pub concepts_extracted: usize,
    
    /// Number of relationships found
    pub relationships_found: usize,
    
    /// Number of exceptions detected
    pub exceptions_detected: usize,
    
    /// Processing algorithm version
    pub algorithm_version: String,
    
    /// Processing timestamp
    pub processed_at: u64,
}

/// Configuration for hierarchy processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyProcessingConfig {
    /// Minimum confidence for concept extraction
    pub min_concept_confidence: f32,
    
    /// Minimum confidence for relationships
    pub min_relationship_confidence: f32,
    
    /// Minimum confidence for exception detection
    pub min_exception_confidence: f32,
    
    /// Maximum processing time (milliseconds)
    pub max_processing_time: u64,
    
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    
    /// Cache intermediate results
    pub enable_caching: bool,
    
    /// Validate hierarchy consistency
    pub validate_consistency: bool,
}

/// Update result for incremental processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyUpdateResult {
    /// Updated hierarchy
    pub hierarchy: ConceptHierarchy,
    
    /// New concepts added
    pub new_concepts: Vec<String>,
    
    /// New relationships added
    pub new_relationships: Vec<ConceptRelationship>,
    
    /// New exceptions detected
    pub new_exceptions: Vec<ConceptException>,
    
    /// Concepts that were updated
    pub updated_concepts: Vec<String>,
    
    /// Update confidence
    pub update_confidence: f32,
    
    /// Update performance
    pub update_time: u64,
}

/// Complete hierarchy processing system
pub struct CompleteHierarchySystem {
    /// Concept extractor
    pub concept_extractor: ConceptExtractionCore,
    
    /// Hierarchy builder
    pub hierarchy_builder: HierarchyBuilder,
    
    /// Property inheritance engine
    pub property_inheritance_engine: PropertyInheritanceEngine,
    
    /// Exception detector
    pub exception_detector: ExceptionDetectionSystem,
    
    /// Processing configuration
    config: HierarchyProcessingConfig,
    
    /// Current hierarchy state
    current_hierarchy: Arc<Mutex<Option<ConceptHierarchy>>>,
    
    /// Performance monitor
    performance_monitor: ProcessingPerformanceMonitor,
    
    /// System enabled flag
    enabled: bool,
}

impl CompleteHierarchySystem {
    /// Create a new complete hierarchy system
    pub fn new() -> Self {
        Self {
            concept_extractor: ConceptExtractionCore::new(),
            hierarchy_builder: HierarchyBuilder::new(),
            property_inheritance_engine: PropertyInheritanceEngine::new(),
            exception_detector: ExceptionDetectionSystem::new(),
            config: HierarchyProcessingConfig::default(),
            current_hierarchy: Arc::new(Mutex::new(None)),
            performance_monitor: ProcessingPerformanceMonitor::new(),
            enabled: true,
        }
    }
    
    /// Process facts to create complete hierarchy
    pub fn process_facts_to_hierarchy(&self, facts: Vec<String>) -> Result<HierarchyProcessingResult, HierarchyProcessingError> {
        if !self.enabled {
            return Err(HierarchyProcessingError::SystemDisabled);
        }
        
        let processing_start = std::time::Instant::now();
        let mut processing_warnings = Vec::new();
        
        // Convert facts to ValidatedFacts (simplified for testing)
        let validated_facts: Vec<ValidatedFact> = facts.iter()
            .enumerate()
            .filter_map(|(i, fact)| {
                if fact.trim().is_empty() {
                    processing_warnings.push(format!("Empty fact at index {}", i));
                    None
                } else {
                    Some(self.create_validated_fact(fact))
                }
            })
            .collect();
        
        if validated_facts.is_empty() {
            return Err(HierarchyProcessingError::NoValidFacts);
        }
        
        // Step 1: Extract concepts from all facts
        let extraction_start = std::time::Instant::now();
        let mut all_concepts = Vec::new();
        let mut all_relationships = Vec::new();
        
        for fact in &validated_facts {
            match self.concept_extractor.extract_concepts(fact) {
                Ok(extraction_result) => {
                    all_concepts.extend(extraction_result.extracted_concepts);
                    all_relationships.extend(extraction_result.concept_relationships);
                }
                Err(e) => {
                    processing_warnings.push(format!("Concept extraction failed: {:?}", e));
                }
            }
        }
        
        let concept_extraction_time = extraction_start.elapsed().as_millis() as u64;
        
        if all_concepts.is_empty() {
            return Err(HierarchyProcessingError::NoConceptsExtracted);
        }
        
        // Step 2: Build hierarchy
        let hierarchy_start = std::time::Instant::now();
        let hierarchy_result = self.hierarchy_builder.build_hierarchy(all_concepts.clone(), all_relationships.clone())
            .map_err(HierarchyProcessingError::HierarchyBuildingFailed)?;
        
        let hierarchy_building_time = hierarchy_start.elapsed().as_millis() as u64;
        
        // Step 3: Detect exceptions
        let exception_start = std::time::Instant::now();
        let mut all_exceptions = Vec::new();
        
        for concept in &all_concepts {
            // Get inherited properties for concept
            if let Ok(inherited_props) = self.property_inheritance_engine.get_inherited_properties(&concept.name, &hierarchy_result.hierarchy) {
                // Convert facts related to this concept
                let concept_facts: Vec<String> = facts.iter()
                    .filter(|fact| fact.to_lowercase().contains(&concept.name.to_lowercase()))
                    .cloned()
                    .collect();
                
                if !concept_facts.is_empty() {
                    match self.exception_detector.detect_exceptions(&concept.name, &inherited_props, &concept_facts, &hierarchy_result.hierarchy) {
                        Ok(exceptions) => {
                            for exception in exceptions {
                                all_exceptions.push(ConceptException {
                                    concept_name: concept.name.clone(),
                                    property_name: exception.property_name,
                                    inherited_value: exception.inherited_value,
                                    actual_value: exception.actual_value,
                                    confidence: exception.confidence,
                                    reason: exception.exception_reason,
                                    source: exception.detection_strategy,
                                });
                            }
                        }
                        Err(e) => {
                            processing_warnings.push(format!("Exception detection failed for {}: {:?}", concept.name, e));
                        }
                    }
                }
            }
        }
        
        let exception_detection_time = exception_start.elapsed().as_millis() as u64;
        
        // Calculate quality scores
        let processing_confidence = self.calculate_processing_confidence(&all_concepts, &hierarchy_result, &all_exceptions);
        let hierarchy_quality_score = self.calculate_hierarchy_quality(&hierarchy_result, &all_concepts);
        
        let total_processing_time = processing_start.elapsed().as_millis() as u64;
        
        // Update current hierarchy
        if let Ok(mut current) = self.current_hierarchy.lock() {
            *current = Some(hierarchy_result.hierarchy.clone());
        }
        
        Ok(HierarchyProcessingResult {
            hierarchy: hierarchy_result.hierarchy,
            extracted_concepts: all_concepts.clone(),
            concept_relationships: all_relationships,
            detected_exceptions: all_exceptions,
            processing_confidence,
            hierarchy_quality_score,
            cycles_detected: hierarchy_result.cycles_detected.len(),
            cycles_resolved: hierarchy_result.cycles_resolved.len(),
            processing_warnings,
            performance_metrics: ProcessingPerformanceMetrics {
                total_processing_time,
                concept_extraction_time,
                hierarchy_building_time,
                property_inheritance_time: 0, // Included in exception detection
                exception_detection_time,
                facts_per_second: if total_processing_time > 0 {
                    facts.len() as f32 / (total_processing_time as f32 / 1000.0)
                } else {
                    0.0
                },
                concepts_per_second: if total_processing_time > 0 {
                    all_concepts.len() as f32 / (total_processing_time as f32 / 1000.0)
                } else {
                    0.0
                },
            },
            processing_metadata: ProcessingMetadata {
                input_facts_count: facts.len(),
                processed_facts_count: validated_facts.len(),
                concepts_extracted: all_concepts.len(),
                relationships_found: all_relationships.len(),
                exceptions_detected: all_exceptions.len(),
                algorithm_version: "1.0.0".to_string(),
                processed_at: current_timestamp(),
            },
        })
    }
    
    /// Update existing hierarchy with new facts
    pub fn update_hierarchy_with_facts(&mut self, new_facts: Vec<String>) -> Result<HierarchyUpdateResult, HierarchyProcessingError> {
        let update_start = std::time::Instant::now();
        
        // Get current hierarchy
        let current_hierarchy = if let Ok(hierarchy_guard) = self.current_hierarchy.lock() {
            hierarchy_guard.clone()
        } else {
            return Err(HierarchyProcessingError::SystemError("Could not access current hierarchy".to_string()));
        };
        
        if current_hierarchy.is_none() {
            // No existing hierarchy, process from scratch
            let result = self.process_facts_to_hierarchy(new_facts)?;
            return Ok(HierarchyUpdateResult {
                hierarchy: result.hierarchy,
                new_concepts: result.extracted_concepts.iter().map(|c| c.name.clone()).collect(),
                new_relationships: result.concept_relationships,
                new_exceptions: result.detected_exceptions,
                updated_concepts: Vec::new(),
                update_confidence: result.processing_confidence,
                update_time: update_start.elapsed().as_millis() as u64,
            });
        }
        
        let mut hierarchy = current_hierarchy.unwrap();
        
        // Extract concepts from new facts
        let validated_facts: Vec<ValidatedFact> = new_facts.iter()
            .map(|fact| self.create_validated_fact(fact))
            .collect();
        
        let mut new_concepts = Vec::new();
        let mut new_relationships = Vec::new();
        
        for fact in &validated_facts {
            if let Ok(extraction_result) = self.concept_extractor.extract_concepts(fact) {
                new_concepts.extend(extraction_result.extracted_concepts);
                new_relationships.extend(extraction_result.concept_relationships);
            }
        }
        
        // Update hierarchy incrementally
        if let Ok(update_result) = self.hierarchy_builder.update_hierarchy_incremental(&mut hierarchy, new_concepts.clone(), new_relationships.clone()) {
            // Detect new exceptions
            let mut new_exceptions = Vec::new();
            for concept in &new_concepts {
                if let Ok(inherited_props) = self.property_inheritance_engine.get_inherited_properties(&concept.name, &hierarchy) {
                    let concept_facts: Vec<String> = new_facts.iter()
                        .filter(|fact| fact.to_lowercase().contains(&concept.name.to_lowercase()))
                        .cloned()
                        .collect();
                    
                    if !concept_facts.is_empty() {
                        if let Ok(exceptions) = self.exception_detector.detect_exceptions(&concept.name, &inherited_props, &concept_facts, &hierarchy) {
                            for exception in exceptions {
                                new_exceptions.push(ConceptException {
                                    concept_name: concept.name.clone(),
                                    property_name: exception.property_name,
                                    inherited_value: exception.inherited_value,
                                    actual_value: exception.actual_value,
                                    confidence: exception.confidence,
                                    reason: exception.exception_reason,
                                    source: exception.detection_strategy,
                                });
                            }
                        }
                    }
                }
            }
            
            // Update current hierarchy
            if let Ok(mut current) = self.current_hierarchy.lock() {
                *current = Some(hierarchy.clone());
            }
            
            Ok(HierarchyUpdateResult {
                hierarchy,
                new_concepts: update_result.nodes_added,
                new_relationships: update_result.relationships_added.into_iter()
                    .map(|(source, target)| ConceptRelationship {
                        source_concept: source,
                        target_concept: target,
                        relationship_type: crate::hierarchy_detection::RelationshipType::IsA,
                        confidence: 0.8,
                        evidence: "incremental_update".to_string(),
                        properties: HashMap::new(),
                    })
                    .collect(),
                new_exceptions,
                updated_concepts: Vec::new(), // Could track which concepts were modified
                update_confidence: 0.8, // Could calculate based on update success
                update_time: update_start.elapsed().as_millis() as u64,
            })
        } else {
            Err(HierarchyProcessingError::UpdateFailed("Incremental update failed".to_string()))
        }
    }
    
    /// Calculate processing confidence
    fn calculate_processing_confidence(
        &self,
        concepts: &[ExtractedConcept],
        hierarchy_result: &crate::hierarchy_detection::HierarchyBuildResult,
        exceptions: &[ConceptException]
    ) -> f32 {
        if concepts.is_empty() {
            return 0.0;
        }
        
        let concept_confidence_avg = concepts.iter()
            .map(|c| c.confidence)
            .sum::<f32>() / concepts.len() as f32;
        
        let hierarchy_confidence = hierarchy_result.construction_confidence;
        
        let exception_quality = if exceptions.is_empty() {
            0.8 // Neutral score if no exceptions
        } else {
            exceptions.iter().map(|e| e.confidence).sum::<f32>() / exceptions.len() as f32
        };
        
        (concept_confidence_avg * 0.4 + hierarchy_confidence * 0.4 + exception_quality * 0.2).min(1.0)
    }
    
    /// Calculate hierarchy quality score
    fn calculate_hierarchy_quality(
        &self,
        hierarchy_result: &crate::hierarchy_detection::HierarchyBuildResult,
        concepts: &[ExtractedConcept]
    ) -> f32 {
        let mut quality_score = 0.0;
        
        // Base score from hierarchy construction
        quality_score += hierarchy_result.construction_confidence * 0.4;
        
        // Validation score
        if hierarchy_result.validation_result.is_well_formed {
            quality_score += 0.2;
        }
        if hierarchy_result.validation_result.has_consistent_types {
            quality_score += 0.2;
        }
        if !hierarchy_result.validation_result.has_cycles {
            quality_score += 0.1;
        }
        
        // Concept coverage score
        if !concepts.is_empty() {
            let coverage = hierarchy_result.hierarchy.node_count() as f32 / concepts.len() as f32;
            quality_score += coverage.min(1.0) * 0.1;
        }
        
        quality_score.min(1.0)
    }
    
    /// Create a simplified ValidatedFact for testing
    fn create_validated_fact(&self, fact_text: &str) -> ValidatedFact {
        use crate::quality_integration::{FactContent, ConfidenceComponents};
        
        let fact_content = FactContent::new(fact_text);
        let confidence = ConfidenceComponents::new(0.85, 0.8, 0.82);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        validated_fact.mark_fully_validated();
        validated_fact
    }
    
    /// Clean up all caches
    pub fn cleanup_caches(&self) {
        // This would clear caches in all components
        // Implementation depends on the actual cache interfaces
    }
    
    /// Check if system is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && 
        self.concept_extractor.is_enabled() &&
        self.hierarchy_builder.is_enabled() &&
        self.property_inheritance_engine.is_enabled() &&
        self.exception_detector.is_enabled()
    }
    
    /// Enable or disable system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Get current hierarchy
    pub fn get_current_hierarchy(&self) -> Option<ConceptHierarchy> {
        if let Ok(hierarchy_guard) = self.current_hierarchy.lock() {
            hierarchy_guard.clone()
        } else {
            None
        }
    }
    
    /// Get processing statistics
    pub fn get_processing_statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("enabled".to_string(), self.enabled.to_string());
        stats.insert("concept_extractor_enabled".to_string(), self.concept_extractor.is_enabled().to_string());
        stats.insert("hierarchy_builder_enabled".to_string(), self.hierarchy_builder.is_enabled().to_string());
        stats.insert("inheritance_engine_enabled".to_string(), self.property_inheritance_engine.is_enabled().to_string());
        stats.insert("exception_detector_enabled".to_string(), self.exception_detector.is_enabled().to_string());
        
        if let Some(hierarchy) = self.get_current_hierarchy() {
            stats.insert("current_hierarchy_nodes".to_string(), hierarchy.node_count().to_string());
            stats.insert("current_hierarchy_depth".to_string(), hierarchy.depth().to_string());
        }
        
        stats
    }
}

/// Error types for hierarchy processing
#[derive(Debug, thiserror::Error)]
pub enum HierarchyProcessingError {
    #[error("Hierarchy processing system is disabled")]
    SystemDisabled,
    
    #[error("No valid facts provided")]
    NoValidFacts,
    
    #[error("No concepts could be extracted")]
    NoConceptsExtracted,
    
    #[error("Hierarchy building failed: {0}")]
    HierarchyBuildingFailed(#[from] crate::hierarchy_detection::HierarchyBuildError),
    
    #[error("Update failed: {0}")]
    UpdateFailed(String),
    
    #[error("System error: {0}")]
    SystemError(String),
    
    #[error("Processing timeout")]
    ProcessingTimeout,
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Performance monitoring for complete system
pub struct ProcessingPerformanceMonitor {
    processing_times: Vec<u64>,
    average_processing_time: f32,
}

impl ProcessingPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            processing_times: Vec::new(),
            average_processing_time: 0.0,
        }
    }
}

impl Default for HierarchyProcessingConfig {
    fn default() -> Self {
        Self {
            min_concept_confidence: 0.7,
            min_relationship_confidence: 0.6,
            min_exception_confidence: 0.7,
            max_processing_time: 30000, // 30 seconds
            enable_parallel_processing: true,
            enable_caching: true,
            validate_consistency: true,
        }
    }
}

impl Default for CompleteHierarchySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// Extension trait to add missing methods to ConceptHierarchy
impl ConceptHierarchy {
    pub fn has_cycles(&self) -> bool {
        // This would implement cycle detection
        false // Simplified for now
    }
}
```

## Verification Steps
1. Create CompleteHierarchySystem integrating all hierarchy detection components
2. Implement comprehensive end-to-end testing with real-world examples
3. Add performance testing for large fact sets (100+ facts)
4. Implement concurrent processing validation
5. Add error handling and recovery testing
6. Create incremental update and consistency validation tests

## Success Criteria
- [ ] CompleteHierarchySystem compiles without errors
- [ ] End-to-end processing works for complex real-world scenarios
- [ ] Performance meets requirements (5 seconds for 100 facts)
- [ ] Exception detection integrates correctly with inheritance
- [ ] Concurrent processing handles multiple fact sets safely
- [ ] All integration tests pass with comprehensive coverage
- [ ] System maintains consistency and handles errors gracefully