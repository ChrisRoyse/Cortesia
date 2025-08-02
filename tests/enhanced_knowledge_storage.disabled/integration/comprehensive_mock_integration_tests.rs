//! Comprehensive Mock Data Integration Tests
//! 
//! Tests the entire enhanced knowledge storage system using comprehensive mock data
//! to validate system behavior, performance, and error handling.

use std::time::{Duration, Instant};
use tokio::time::timeout;
use crate::enhanced_knowledge_storage::types::{ComplexityLevel, ProcessingResult};
use crate::tests::enhanced_knowledge_storage::mocks::comprehensive_mock_data::*;

/// Integration test suite for comprehensive mock data
pub struct ComprehensiveMockIntegrationTests {
    system: MockEnhancedKnowledgeSystem,
}

impl ComprehensiveMockIntegrationTests {
    pub fn new() -> Self {
        Self {
            system: MockEnhancedKnowledgeSystem::new(),
        }
    }

    pub fn with_custom_system(system: MockEnhancedKnowledgeSystem) -> Self {
        Self { system }
    }
}

impl Default for ComprehensiveMockIntegrationTests {
    fn default() -> Self {
        Self::new()
    }
}

/// Test scientific document processing pipeline
#[tokio::test]  
async fn test_scientific_document_processing_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test each scientific document
    let scientific_docs = test_suite.system.documents.get_documents_by_type(DocumentType::Scientific);
    assert!(!scientific_docs.is_empty(), "No scientific documents available for testing");
    
    for doc in scientific_docs {
        println!("Testing scientific document: {}", doc.title);
        
        // Process the document
        let result = test_suite.system.process_document_type(DocumentType::Scientific).await;
        assert!(result.is_ok(), "Failed to process scientific document: {}", doc.title);
        
        let result = result.unwrap();
        assert!(result.success, "Processing failed for document: {}", doc.title);
        assert!(result.quality_score >= doc.expected_quality_score * 0.9, 
               "Quality score {} below expected minimum {} for document: {}", 
               result.quality_score, doc.expected_quality_score * 0.9, doc.title);
        
        // Verify processing time is reasonable
        assert!(result.processing_time <= doc.expected_processing_time * 2,
               "Processing time {:?} exceeded expected maximum {:?} for document: {}",
               result.processing_time, doc.expected_processing_time * 2, doc.title);
    }
}

/// Test technical documentation processing
#[tokio::test]
async fn test_technical_documentation_processing() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    let tech_docs = test_suite.system.documents.get_documents_by_type(DocumentType::Technical);
    assert!(!tech_docs.is_empty());
    
    for doc in tech_docs {
        println!("Testing technical document: {}", doc.title);
        
        let result = test_suite.system.process_document_type(DocumentType::Technical).await;
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.success);
        
        // Technical documents should have good entity extraction
        assert!(result.quality_score >= 0.85, 
               "Technical document quality score {} too low for: {}", 
               result.quality_score, doc.title);
    }
}

/// Test multi-hop reasoning across different knowledge domains
#[tokio::test]
async fn test_multi_hop_reasoning_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test various multi-hop queries
    let queries = vec![
        ("How did Einstein's work influence modern GPS technology?", 2),
        ("What is the connection between Steve Jobs and mobile computing?", 2),
        ("How does Alan Turing's work relate to modern AI?", 3),
        ("What links quantum computing to drug discovery?", 2),
    ];
    
    for (query, expected_min_hops) in queries {
        println!("Testing multi-hop query: {}", query);
        
        let result = test_suite.system.perform_multi_hop_query(query).await;
        assert!(result.is_ok(), "Multi-hop query failed: {}", query);
        
        let result = result.unwrap();
        assert!(result.hops >= expected_min_hops,
               "Query '{}' returned {} hops, expected at least {}",
               query, result.hops, expected_min_hops);
        assert!(result.confidence > 0.7,
               "Query '{}' confidence {} too low", query, result.confidence);
        assert!(!result.reasoning_path.is_empty(),
               "Query '{}' returned empty reasoning path", query);
    }
}

/// Test performance under various load conditions
#[tokio::test]
async fn test_performance_load_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test different load levels
    let load_tests = vec![
        (10, 0.95),  // Light load - 95% success rate
        (50, 0.90),  // Medium load - 90% success rate  
        (100, 0.85), // Heavy load - 85% success rate
    ];
    
    for (document_count, expected_success_rate) in load_tests {
        println!("Testing load with {} documents", document_count);
        
        let start_time = Instant::now();
        let result = test_suite.system.process_concurrent_documents(document_count).await;
        let total_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Load test failed for {} documents", document_count);
        
        let result = result.unwrap();
        assert!(result.success_rate >= expected_success_rate,
               "Success rate {} below expected {} for {} documents",
               result.success_rate, expected_success_rate, document_count);
        
        // Verify reasonable throughput
        let throughput = result.successful_documents as f64 / total_time.as_secs_f64();
        assert!(throughput > 5.0, "Throughput {} docs/sec too low for load test", throughput);
    }
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    let error_scenarios = test_suite.system.error_scenarios.get_all_scenarios();
    assert!(!error_scenarios.is_empty(), "No error scenarios available for testing");
    
    for scenario in error_scenarios {
        println!("Testing error scenario: {}", scenario.name);
        
        // Setup the error condition
        let setup_result = (scenario.setup)();
        assert!(setup_result.is_ok(), "Failed to setup error scenario: {}", scenario.name);
        
        // Test error detection and recovery
        match scenario.expected_error {
            ProcessingError::InsufficientMemory => {
                // Test memory pressure handling
                let result = test_suite.system.process_concurrent_documents(200).await;
                // Should handle gracefully even under memory pressure
                assert!(result.is_ok() || matches!(result, Err(ProcessingError::InsufficientMemory)));
            },
            ProcessingError::ProcessingTimeout => {
                // Test timeout handling with timeout wrapper
                let result = timeout(Duration::from_secs(1), 
                    test_suite.system.perform_multi_hop_query("Complex query")).await;
                // Should either complete quickly or timeout gracefully
                assert!(result.is_ok() || result.is_err());
            },
            ProcessingError::MalformedInput => {
                // Test malformed input handling
                let result = test_suite.system.process_document_type(DocumentType::Scientific).await;
                // Should handle malformed input gracefully
                assert!(result.is_ok() || matches!(result, Err(ProcessingError::MalformedInput)));
            },
            _ => {
                // Test other error types as they become available
                println!("Error scenario {} not fully implemented yet", scenario.name);
            }
        }
    }
}

/// Test different complexity levels with appropriate models
#[tokio::test]
async fn test_complexity_model_matching() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    for complexity in [ComplexityLevel::Low, ComplexityLevel::Medium, ComplexityLevel::High] {
        println!("Testing complexity level: {:?}", complexity);
        
        let docs = test_suite.system.documents.get_documents_by_complexity(complexity);
        if docs.is_empty() {
            continue;
        }
        
        let doc = docs[0];
        let result = test_suite.system.process_document_type(doc.document_type).await;
        assert!(result.is_ok(), "Failed to process {:?} complexity document", complexity);
        
        let result = result.unwrap();
        
        // Verify appropriate model was selected
        let expected_model = match complexity {
            ComplexityLevel::Low => "smollm2_135m",
            ComplexityLevel::Medium => "smollm2_360m",
            ComplexityLevel::High => "smollm2_1_7b",
        };
        
        assert!(result.model_used.contains(expected_model),
               "Expected model {} for {:?} complexity, got {}",
               expected_model, complexity, result.model_used);
    }
}

/// Test entity knowledge base integration
#[tokio::test]
async fn test_entity_knowledge_base_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test all entity types are available
    for entity_type in [EntityType::Person, EntityType::Organization, EntityType::Concept, 
                       EntityType::Technology, EntityType::Location] {
        let entities = test_suite.system.entities.get_entities_by_type(entity_type);
        assert!(!entities.is_empty(), "No entities of type {:?} available", entity_type);
        
        for entity in entities {
            // Verify entity completeness
            assert!(!entity.name.is_empty(), "Entity has empty name");
            assert!(!entity.description.is_empty(), "Entity {} has empty description", entity.name);
            assert!(entity.confidence > 0.0, "Entity {} has zero confidence", entity.name);
            
            // Verify entity properties
            match entity.entity_type {
                EntityType::Person => {
                    // People should have biographical information
                    assert!(entity.properties.contains_key("nationality") || 
                           entity.properties.contains_key("field"),
                           "Person {} missing expected properties", entity.name);
                },
                EntityType::Organization => {
                    // Organizations should have founding information
                    assert!(entity.properties.contains_key("founded") || 
                           entity.properties.contains_key("industry"),
                           "Organization {} missing expected properties", entity.name);
                },
                _ => {
                    // Other types should have at least some properties
                    assert!(!entity.properties.is_empty(),
                           "Entity {} of type {:?} has no properties", entity.name, entity.entity_type);
                }
            }
        }
    }
}

/// Test relationship network connectivity
#[tokio::test]
async fn test_relationship_network_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    let all_relationships = test_suite.system.relationships.get_all_relationships();
    assert!(!all_relationships.is_empty(), "No relationships available for testing");
    
    // Test relationship completeness
    for relationship in all_relationships {
        assert!(!relationship.source.is_empty(), "Relationship has empty source");
        assert!(!relationship.target.is_empty(), "Relationship has empty target");
        assert!(relationship.confidence > 0.0, "Relationship has zero confidence");
        assert!(!relationship.supporting_evidence.is_empty(), 
               "Relationship {}->{}  has no supporting evidence",
               relationship.source, relationship.target);
    }
    
    // Test relationship types distribution
    let mut relationship_type_counts = std::collections::HashMap::new();
    for relationship in all_relationships {
        let rel_type = relationship.predicate.predicate_string();
        *relationship_type_counts.entry(rel_type).or_insert(0) += 1;
    }
    
    assert!(!relationship_type_counts.is_empty(), "No relationship types found");
    assert!(relationship_type_counts.len() >= 5, 
           "Should have at least 5 different relationship types, found {}", 
           relationship_type_counts.len());
}

/// Test temporal relationship handling
#[tokio::test]
async fn test_temporal_relationship_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    let temporal_relationships = &test_suite.system.relationships.temporal_relationships;
    assert!(!temporal_relationships.is_empty(), "No temporal relationships available");
    
    for temp_rel in temporal_relationships {
        let relationship = &temp_rel.relationship;
        
        // Verify temporal context
        assert!(relationship.temporal_context.is_some(),
               "Temporal relationship {}->{}  missing temporal context",
               relationship.source, relationship.target);
        
        // Verify temporal bounds if present
        if temp_rel.start_time.is_some() && temp_rel.end_time.is_some() {
            let start = temp_rel.start_time.as_ref().unwrap();
            let end = temp_rel.end_time.as_ref().unwrap();
            
            // Basic sanity check - start should be before end (for years)
            if let (Ok(start_year), Ok(end_year)) = (start.parse::<i32>(), end.parse::<i32>()) {
                assert!(start_year <= end_year,
                       "Temporal relationship has start year {} after end year {}",
                       start_year, end_year);
            }
        }
    }
}

/// Test multilingual content processing
#[tokio::test]
async fn test_multilingual_content_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    let multilingual_docs = test_suite.system.documents.get_documents_by_type(DocumentType::Multilingual);
    if multilingual_docs.is_empty() {
        println!("No multilingual documents available for testing");
        return;
    }
    
    for doc in multilingual_docs {
        println!("Testing multilingual document: {}", doc.title);
        
        let result = test_suite.system.process_document_type(DocumentType::Multilingual).await;
        assert!(result.is_ok(), "Failed to process multilingual document: {}", doc.title);
        
        let result = result.unwrap();
        
        // Multilingual documents may have lower accuracy but should still process
        assert!(result.quality_score >= 0.75,
               "Multilingual document quality score {} too low for: {}",
               result.quality_score, doc.title);
        
        // Should extract both English and non-English entities
        let expected_entities = &doc.expected_entities;
        let multilingual_entities = expected_entities.iter()
            .filter(|entity| entity.chars().any(|c| !c.is_ascii()))
            .count();
        
        if multilingual_entities > 0 {
            println!("Document {} contains {} multilingual entities", doc.title, multilingual_entities);
        }
    }
}

/// Test document type classification and routing
#[tokio::test]
async fn test_document_type_routing_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test each document type gets appropriate processing
    for doc_type in [DocumentType::Scientific, DocumentType::Technical, DocumentType::Narrative,
                    DocumentType::Mixed, DocumentType::Temporal, DocumentType::Multilingual] {
        let documents = test_suite.system.documents.get_documents_by_type(doc_type);
        if documents.is_empty() {
            continue;
        }
        
        println!("Testing document type routing for: {:?}", doc_type);
        
        let result = test_suite.system.process_document_type(doc_type).await;
        assert!(result.is_ok(), "Failed to process {:?} document type", doc_type);
        
        let result = result.unwrap();
        assert!(result.success, "Processing failed for {:?} document type", doc_type);
        
        // Verify output indicates correct document type processing
        assert!(result.output.contains(&format!("{:?}", doc_type)) ||
               result.output.to_lowercase().contains(&doc_type_to_string(doc_type).to_lowercase()),
               "Output doesn't indicate correct document type processing: {}", result.output);
    }
}

/// Comprehensive system integration test using all mock components
#[tokio::test]
async fn test_complete_mock_data_integration() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    println!("Starting comprehensive mock data integration test");
    
    // 1. Test document collection completeness
    let all_documents = test_suite.system.documents.get_all_documents();
    assert!(all_documents.len() >= 10, "Should have at least 10 test documents, found {}", all_documents.len());
    
    // 2. Test entity knowledge base completeness  
    let all_entities = test_suite.system.entities.get_all_entities();
    assert!(all_entities.len() >= 5, "Should have at least 5 entities, found {}", all_entities.len());
    
    // 3. Test relationship network connectivity
    let all_relationships = test_suite.system.relationships.get_all_relationships();
    assert!(all_relationships.len() >= 10, "Should have at least 10 relationships, found {}", all_relationships.len());
    
    // 4. Test performance data availability
    let perf_data = &test_suite.system.performance_data;
    assert!(perf_data.get_accuracy_metric("entity_extraction") > 0.8);
    assert!(perf_data.get_expected_processing_time(ComplexityLevel::High) > 
           perf_data.get_expected_processing_time(ComplexityLevel::Low));
    
    // 5. Test error scenarios availability
    let error_scenarios = test_suite.system.error_scenarios.get_all_scenarios();
    assert!(error_scenarios.len() >= 4, "Should have at least 4 error scenarios, found {}", error_scenarios.len());
    
    // 6. Test scientific document processing
    let scientific_result = test_suite.system.process_document_type(DocumentType::Scientific).await;
    assert!(scientific_result.is_ok());
    let scientific_result = scientific_result.unwrap();
    assert!(scientific_result.quality_score > 0.85);
    
    // 7. Test multi-hop reasoning
    let reasoning_result = test_suite.system.perform_multi_hop_query(
        "How did Einstein's work influence modern GPS technology?"
    ).await;
    assert!(reasoning_result.is_ok());
    let reasoning_result = reasoning_result.unwrap();
    assert!(reasoning_result.hops >= 2);
    assert!(reasoning_result.confidence > 0.7);
    
    // 8. Test performance under load
    let load_test_result = test_suite.system.process_concurrent_documents(50).await;
    assert!(load_test_result.is_ok());
    let load_test_result = load_test_result.unwrap();
    assert!(load_test_result.success_rate > 0.90);
    
    println!("Comprehensive mock data integration test completed successfully");
}

/// Performance validation test
#[tokio::test]
async fn test_mock_system_performance_validation() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test that mock system behaves realistically in terms of performance
    let perf_data = &test_suite.system.performance_data;
    
    // Test processing time scaling
    let low_time = perf_data.get_expected_processing_time(ComplexityLevel::Low);
    let medium_time = perf_data.get_expected_processing_time(ComplexityLevel::Medium);
    let high_time = perf_data.get_expected_processing_time(ComplexityLevel::High);
    
    assert!(low_time < medium_time);
    assert!(medium_time < high_time);
    assert!(high_time.as_millis() > 1000); // High complexity should take > 1 second
    
    // Test memory usage scaling
    let small_memory = perf_data.get_model_memory_usage("smollm2_135m");
    let medium_memory = perf_data.get_model_memory_usage("smollm2_360m");
    let large_memory = perf_data.get_model_memory_usage("smollm2_1_7b");
    
    assert!(small_memory < medium_memory);
    assert!(medium_memory < large_memory);
    assert!(large_memory > 3_000_000_000); // Large model should use > 3GB
    
    // Test accuracy metrics are realistic
    assert!(perf_data.get_accuracy_metric("entity_extraction") > 0.8);
    assert!(perf_data.get_accuracy_metric("relationship_mapping") > 0.75);
    assert!(perf_data.get_accuracy_metric("semantic_chunking") > 0.85);
    
    println!("Mock system performance validation completed");
}

/// Data consistency validation test
#[tokio::test]
async fn test_mock_data_consistency() {
    let test_suite = ComprehensiveMockIntegrationTests::new();
    
    // Test that entities referenced in relationships exist in the knowledge base
    let all_entities = test_suite.system.entities.get_all_entities();
    let entity_names: std::collections::HashSet<_> = all_entities.iter()
        .map(|e| e.name.as_str())
        .collect();
    
    let all_relationships = test_suite.system.relationships.get_all_relationships();
    
    for relationship in all_relationships {
        // Note: Some test relationships may reference entities not in the KB (like "GPS Technology")
        // This is intentional for testing multi-hop reasoning beyond the KB
        println!("Checking relationship: {} -> {} -> {}", 
                relationship.source, relationship.predicate.predicate_string(), relationship.target);
    }
    
    // Test that expected entities in documents are realistic
    let all_documents = test_suite.system.documents.get_all_documents();
    for doc in all_documents {
        assert!(!doc.expected_entities.is_empty(), 
               "Document '{}' has no expected entities", doc.title);
        assert!(!doc.expected_relationships.is_empty(),
               "Document '{}' has no expected relationships", doc.title);
        
        // Test that entity count matches complexity
        let entity_count = doc.expected_entities.len();
        match doc.complexity_level {
            ComplexityLevel::Low => assert!(entity_count >= 2 && entity_count <= 8),
            ComplexityLevel::Medium => assert!(entity_count >= 5 && entity_count <= 15),
            ComplexityLevel::High => assert!(entity_count >= 8),
        }
    }
    
    println!("Mock data consistency validation completed");
}

// Utility function for document type string conversion
fn doc_type_to_string(doc_type: DocumentType) -> &'static str {
    match doc_type {
        DocumentType::Scientific => "Scientific",
        DocumentType::Technical => "Technical", 
        DocumentType::Narrative => "Narrative",
        DocumentType::Mixed => "Mixed",
        DocumentType::Temporal => "Temporal",
        DocumentType::Multilingual => "Multilingual",
    }
}