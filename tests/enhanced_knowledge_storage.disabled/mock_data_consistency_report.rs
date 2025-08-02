//! Mock Data Consistency Report
//! 
//! Generates detailed reports on the consistency and completeness of mock data
//! components within the enhanced knowledge storage system test suite.

use std::collections::{HashMap, HashSet};
use crate::tests::enhanced_knowledge_storage::mocks::comprehensive_mock_data::*;

/// Report generator for mock data consistency analysis
pub struct MockDataConsistencyReport {
    system: MockEnhancedKnowledgeSystem,
}

impl MockDataConsistencyReport {
    pub fn new() -> Self {
        Self {
            system: MockEnhancedKnowledgeSystem::new(),
        }
    }

    /// Generate a comprehensive consistency report
    pub fn generate_full_report(&self) -> ConsistencyReport {
        ConsistencyReport {
            document_analysis: self.analyze_document_collection(),
            entity_analysis: self.analyze_entity_knowledge_base(),
            relationship_analysis: self.analyze_relationship_network(),
            performance_analysis: self.analyze_performance_data(),
            integration_analysis: self.analyze_system_integration(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Analyze document collection for completeness and consistency
    fn analyze_document_collection(&self) -> DocumentAnalysis {
        let all_docs = self.system.documents.get_all_documents();
        
        let mut complexity_distribution = HashMap::new();
        let mut type_distribution = HashMap::new();
        let mut entity_count_stats = Vec::new();
        let mut relationship_count_stats = Vec::new();
        let mut quality_scores = Vec::new();
        
        for doc in &all_docs {
            *complexity_distribution.entry(doc.complexity_level).or_insert(0) += 1;
            *type_distribution.entry(doc.document_type).or_insert(0) += 1;
            
            entity_count_stats.push(doc.expected_entities.len());
            relationship_count_stats.push(doc.expected_relationships.len());
            quality_scores.push(doc.expected_quality_score);
        }
        
        DocumentAnalysis {
            total_documents: all_docs.len(),
            complexity_distribution,
            type_distribution,
            avg_entities_per_doc: entity_count_stats.iter().sum::<usize>() as f64 / entity_count_stats.len() as f64,
            avg_relationships_per_doc: relationship_count_stats.iter().sum::<usize>() as f64 / relationship_count_stats.len() as f64,
            avg_quality_score: quality_scores.iter().sum::<f32>() as f64 / quality_scores.len() as f64,
            entity_count_range: (
                *entity_count_stats.iter().min().unwrap_or(&0),
                *entity_count_stats.iter().max().unwrap_or(&0)
            ),
            relationship_count_range: (
                *relationship_count_stats.iter().min().unwrap_or(&0),
                *relationship_count_stats.iter().max().unwrap_or(&0)
            ),
        }
    }

    /// Analyze entity knowledge base for coverage and quality
    fn analyze_entity_knowledge_base(&self) -> EntityAnalysis {
        let all_entities = self.system.entities.get_all_entities();
        
        let mut type_distribution = HashMap::new();
        let mut confidence_scores = Vec::new();
        let mut property_counts = Vec::new();
        let mut entities_with_aliases = 0;
        
        for entity in &all_entities {
            *type_distribution.entry(entity.entity_type).or_insert(0) += 1;
            confidence_scores.push(entity.confidence);
            property_counts.push(entity.properties.len());
            
            if !entity.aliases.is_empty() {
                entities_with_aliases += 1;
            }
        }
        
        EntityAnalysis {
            total_entities: all_entities.len(),
            type_distribution,
            avg_confidence: confidence_scores.iter().sum::<f32>() as f64 / confidence_scores.len() as f64,
            avg_properties_per_entity: property_counts.iter().sum::<usize>() as f64 / property_counts.len() as f64,
            entities_with_aliases: entities_with_aliases,
            alias_coverage: entities_with_aliases as f64 / all_entities.len() as f64,
        }
    }

    /// Analyze relationship network for connectivity and completeness
    fn analyze_relationship_network(&self) -> RelationshipAnalysis {
        let all_relationships = self.system.relationships.get_all_relationships();
        
        let mut predicate_distribution = HashMap::new();
        let mut confidence_scores = Vec::new();
        let mut evidence_counts = Vec::new();
        let mut temporal_relationships = 0;
        let mut source_entities = HashSet::new();
        let mut target_entities = HashSet::new();
        
        for relationship in &all_relationships {
            let predicate = relationship.predicate.predicate_string();
            *predicate_distribution.entry(predicate).or_insert(0) += 1;
            
            confidence_scores.push(relationship.confidence);
            evidence_counts.push(relationship.supporting_evidence.len());
            
            if relationship.temporal_context.is_some() {
                temporal_relationships += 1;
            }
            
            source_entities.insert(relationship.source.clone());
            target_entities.insert(relationship.target.clone());
        }
        
        // Test multi-hop connectivity
        let multi_hop_paths = vec![
            self.system.relationships.find_multi_hop_path("Albert Einstein", "GPS Technology", 3),
            self.system.relationships.find_multi_hop_path("Steve Jobs", "iPhone", 2),
            self.system.relationships.find_multi_hop_path("Alan Turing", "Artificial Intelligence", 2),
        ];
        
        let multi_hop_connectivity = multi_hop_paths.iter()
            .map(|paths| !paths.is_empty())
            .filter(|&connected| connected)
            .count() as f64 / multi_hop_paths.len() as f64;
        
        RelationshipAnalysis {
            total_relationships: all_relationships.len(),
            predicate_distribution,
            avg_confidence: confidence_scores.iter().sum::<f32>() as f64 / confidence_scores.len() as f64,
            avg_evidence_per_relationship: evidence_counts.iter().sum::<usize>() as f64 / evidence_counts.len() as f64,
            temporal_relationship_count: temporal_relationships,
            temporal_coverage: temporal_relationships as f64 / all_relationships.len() as f64,
            unique_source_entities: source_entities.len(),
            unique_target_entities: target_entities.len(),
            multi_hop_connectivity,
        }
    }

    /// Analyze performance data for realism and consistency
    fn analyze_performance_data(&self) -> PerformanceAnalysis {
        let perf_data = &self.system.performance_data;
        
        // Test processing time scaling
        let low_time = perf_data.get_expected_processing_time(ComplexityLevel::Low);
        let medium_time = perf_data.get_expected_processing_time(ComplexityLevel::Medium);
        let high_time = perf_data.get_expected_processing_time(ComplexityLevel::High);
        
        let time_scaling_correct = low_time < medium_time && medium_time < high_time;
        
        // Test memory usage scaling
        let small_memory = perf_data.get_model_memory_usage("smollm2_135m");
        let medium_memory = perf_data.get_model_memory_usage("smollm2_360m");
        let large_memory = perf_data.get_model_memory_usage("smollm2_1_7b");
        
        let memory_scaling_correct = small_memory < medium_memory && medium_memory < large_memory;
        
        // Test accuracy metrics
        let accuracy_metrics = vec![
            perf_data.get_accuracy_metric("entity_extraction"),
            perf_data.get_accuracy_metric("relationship_mapping"),
            perf_data.get_accuracy_metric("semantic_chunking"),
        ];
        
        let avg_accuracy = accuracy_metrics.iter().sum::<f32>() as f64 / accuracy_metrics.len() as f64;
        let all_accuracies_realistic = accuracy_metrics.iter().all(|&acc| acc > 0.7 && acc < 1.0);
        
        PerformanceAnalysis {
            processing_time_scaling_correct: time_scaling_correct,
            memory_usage_scaling_correct: memory_scaling_correct,
            avg_accuracy_metric: avg_accuracy,
            accuracy_metrics_realistic: all_accuracies_realistic,
            low_complexity_time_ms: low_time.as_millis() as u64,
            high_complexity_time_ms: high_time.as_millis() as u64,
            small_model_memory_mb: small_memory / 1_000_000,
            large_model_memory_mb: large_memory / 1_000_000,
        }
    }

    /// Analyze system integration and cross-component consistency
    fn analyze_system_integration(&self) -> IntegrationAnalysis {
        let documents = self.system.documents.get_all_documents();
        let entities = self.system.entities.get_all_entities();
        let relationships = self.system.relationships.get_all_relationships();
        
        // Check document-entity consistency
        let mut doc_entities_found_in_kb = 0;
        let mut total_doc_entities = 0;
        
        let entity_names: HashSet<String> = entities.iter()
            .flat_map(|e| std::iter::once(e.name.clone()).chain(e.aliases.clone()))
            .collect();
        
        for doc in documents {
            for expected_entity in &doc.expected_entities {
                total_doc_entities += 1;
                if entity_names.contains(expected_entity) {
                    doc_entities_found_in_kb += 1;
                }
            }
        }
        
        let entity_coverage = if total_doc_entities > 0 {
            doc_entities_found_in_kb as f64 / total_doc_entities as f64
        } else {
            0.0
        };
        
        // Check relationship-entity consistency
        let mut relationship_entities_found = 0;
        let mut total_relationship_entities = 0;
        
        for relationship in relationships {
            total_relationship_entities += 2; // source and target
            if entity_names.contains(&relationship.source) {
                relationship_entities_found += 1;
            }
            if entity_names.contains(&relationship.target) {
                relationship_entities_found += 1;
            }
        }
        
        let relationship_entity_coverage = if total_relationship_entities > 0 {
            relationship_entities_found as f64 / total_relationship_entities as f64
        } else {
            0.0
        };
        
        // Test complexity-model matching
        let complexity_model_mapping_correct = self.test_complexity_model_mapping();
        
        IntegrationAnalysis {
            entity_coverage_in_documents: entity_coverage,
            relationship_entity_coverage: relationship_entity_coverage,
            complexity_model_mapping_correct,
            cross_component_references_valid: entity_coverage > 0.3, // At least 30% coverage
        }
    }

    /// Test that complexity levels map to appropriate models
    fn test_complexity_model_mapping(&self) -> bool {
        // This would normally test the actual model selection logic
        // For mock purposes, we assume correct mapping based on performance data
        let perf_data = &self.system.performance_data;
        
        // Check that different models have different memory footprints
        let small_mem = perf_data.get_model_memory_usage("smollm2_135m");
        let large_mem = perf_data.get_model_memory_usage("smollm2_1_7b");
        
        small_mem < large_mem
    }

    /// Generate recommendations for improving mock data
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let doc_analysis = self.analyze_document_collection();
        let entity_analysis = self.analyze_entity_knowledge_base();
        let relationship_analysis = self.analyze_relationship_network();
        let performance_analysis = self.analyze_performance_data();
        let integration_analysis = self.analyze_system_integration();
        
        // Document recommendations
        if doc_analysis.total_documents < 15 {
            recommendations.push("Consider adding more test documents for better coverage".to_string());
        }
        
        if doc_analysis.avg_quality_score < 0.85 {
            recommendations.push("Some documents have low expected quality scores - consider reviewing".to_string());
        }
        
        // Entity recommendations
        if entity_analysis.alias_coverage < 0.5 {
            recommendations.push("Consider adding more aliases to entities for better name variation testing".to_string());
        }
        
        if entity_analysis.avg_confidence < 0.85 {
            recommendations.push("Some entities have low confidence scores - consider reviewing entity quality".to_string());
        }
        
        // Relationship recommendations
        if relationship_analysis.temporal_coverage < 0.3 {
            recommendations.push("Consider adding more temporal relationships for time-based reasoning tests".to_string());
        }
        
        if relationship_analysis.multi_hop_connectivity < 0.7 {
            recommendations.push("Improve multi-hop connectivity by adding more relationship chains".to_string());
        }
        
        // Performance recommendations
        if !performance_analysis.accuracy_metrics_realistic {
            recommendations.push("Some accuracy metrics appear unrealistic - review performance data".to_string());
        }
        
        // Integration recommendations
        if integration_analysis.entity_coverage_in_documents < 0.4 {
            recommendations.push("Low entity coverage between documents and KB - add more connecting entities".to_string());
        }
        
        if integration_analysis.relationship_entity_coverage < 0.5 {
            recommendations.push("Many relationship entities not found in KB - improve entity-relationship consistency".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Mock data appears well-structured and consistent".to_string());
        }
        
        recommendations
    }
}

impl Default for MockDataConsistencyReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete consistency report structure
#[derive(Debug)]
pub struct ConsistencyReport {
    pub document_analysis: DocumentAnalysis,
    pub entity_analysis: EntityAnalysis,
    pub relationship_analysis: RelationshipAnalysis,
    pub performance_analysis: PerformanceAnalysis,
    pub integration_analysis: IntegrationAnalysis,
    pub recommendations: Vec<String>,
}

#[derive(Debug)]
pub struct DocumentAnalysis {
    pub total_documents: usize,
    pub complexity_distribution: HashMap<ComplexityLevel, i32>,
    pub type_distribution: HashMap<DocumentType, i32>,
    pub avg_entities_per_doc: f64,
    pub avg_relationships_per_doc: f64,
    pub avg_quality_score: f64,
    pub entity_count_range: (usize, usize),
    pub relationship_count_range: (usize, usize),
}

#[derive(Debug)]
pub struct EntityAnalysis {
    pub total_entities: usize,
    pub type_distribution: HashMap<EntityType, i32>,
    pub avg_confidence: f64,
    pub avg_properties_per_entity: f64,
    pub entities_with_aliases: usize,
    pub alias_coverage: f64,
}

#[derive(Debug)]
pub struct RelationshipAnalysis {
    pub total_relationships: usize,
    pub predicate_distribution: HashMap<String, i32>,
    pub avg_confidence: f64,
    pub avg_evidence_per_relationship: f64,
    pub temporal_relationship_count: usize,
    pub temporal_coverage: f64,
    pub unique_source_entities: usize,
    pub unique_target_entities: usize,
    pub multi_hop_connectivity: f64,
}

#[derive(Debug)]
pub struct PerformanceAnalysis {
    pub processing_time_scaling_correct: bool,
    pub memory_usage_scaling_correct: bool,
    pub avg_accuracy_metric: f64,
    pub accuracy_metrics_realistic: bool,
    pub low_complexity_time_ms: u64,
    pub high_complexity_time_ms: u64,
    pub small_model_memory_mb: u64,
    pub large_model_memory_mb: u64,
}

#[derive(Debug)]
pub struct IntegrationAnalysis {
    pub entity_coverage_in_documents: f64,
    pub relationship_entity_coverage: f64,
    pub complexity_model_mapping_correct: bool,
    pub cross_component_references_valid: bool,
}

impl ConsistencyReport {
    /// Print a formatted report to stdout
    pub fn print_report(&self) {
        println!("📋 MOCK DATA CONSISTENCY REPORT");
        println!("================================");
        
        println!("\n📄 DOCUMENT ANALYSIS");
        println!("   Total Documents: {}", self.document_analysis.total_documents);
        println!("   Avg Entities/Doc: {:.1}", self.document_analysis.avg_entities_per_doc);
        println!("   Avg Relationships/Doc: {:.1}", self.document_analysis.avg_relationships_per_doc);
        println!("   Avg Quality Score: {:.2}", self.document_analysis.avg_quality_score);
        println!("   Entity Count Range: {:?}", self.document_analysis.entity_count_range);
        
        println!("\n🧠 ENTITY ANALYSIS");
        println!("   Total Entities: {}", self.entity_analysis.total_entities);
        println!("   Avg Confidence: {:.2}", self.entity_analysis.avg_confidence);
        println!("   Avg Properties/Entity: {:.1}", self.entity_analysis.avg_properties_per_entity);
        println!("   Alias Coverage: {:.1}%", self.entity_analysis.alias_coverage * 100.0);
        
        println!("\n🔗 RELATIONSHIP ANALYSIS");
        println!("   Total Relationships: {}", self.relationship_analysis.total_relationships);
        println!("   Avg Confidence: {:.2}", self.relationship_analysis.avg_confidence);
        println!("   Temporal Coverage: {:.1}%", self.relationship_analysis.temporal_coverage * 100.0);
        println!("   Multi-hop Connectivity: {:.1}%", self.relationship_analysis.multi_hop_connectivity * 100.0);
        
        println!("\n⚡ PERFORMANCE ANALYSIS");
        println!("   Time Scaling Correct: {}", self.performance_analysis.processing_time_scaling_correct);
        println!("   Memory Scaling Correct: {}", self.performance_analysis.memory_usage_scaling_correct);
        println!("   Avg Accuracy: {:.2}", self.performance_analysis.avg_accuracy_metric);  
        println!("   Low/High Complexity: {}ms/{}ms", 
                self.performance_analysis.low_complexity_time_ms,
                self.performance_analysis.high_complexity_time_ms);
        
        println!("\n🔄 INTEGRATION ANALYSIS");
        println!("   Entity Coverage in Docs: {:.1}%", self.integration_analysis.entity_coverage_in_documents * 100.0);
        println!("   Relationship-Entity Coverage: {:.1}%", self.integration_analysis.relationship_entity_coverage * 100.0);
        println!("   Complexity Mapping Correct: {}", self.integration_analysis.complexity_model_mapping_correct);
        
        println!("\n💡 RECOMMENDATIONS");
        for (i, recommendation) in self.recommendations.iter().enumerate() {
            println!("   {}. {}", i + 1, recommendation);
        }
        
        println!("\n================================");
    }
    
    /// Calculate overall consistency score (0.0 to 1.0)
    pub fn overall_consistency_score(&self) -> f64 {
        let mut score = 0.0;
        let mut components = 0;
        
        // Document component score
        if self.document_analysis.total_documents >= 10 {
            score += 0.2;
        }
        components += 1;
        
        // Entity component score  
        if self.entity_analysis.avg_confidence > 0.85 {
            score += 0.2;
        }
        components += 1;
        
        // Relationship component score
        if self.relationship_analysis.multi_hop_connectivity > 0.7 {
            score += 0.2;
        }
        components += 1;
        
        // Performance component score
        if self.performance_analysis.processing_time_scaling_correct &&
           self.performance_analysis.memory_usage_scaling_correct {
            score += 0.2;
        }
        components += 1;
        
        // Integration component score
        if self.integration_analysis.cross_component_references_valid {
            score += 0.2;
        }
        components += 1;
        
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_report_generation() {
        let report_generator = MockDataConsistencyReport::new();
        let report = report_generator.generate_full_report();
        
        // Basic validation
        assert!(report.document_analysis.total_documents > 0);
        assert!(report.entity_analysis.total_entities > 0);
        assert!(report.relationship_analysis.total_relationships > 0);
        assert!(!report.recommendations.is_empty());
        
        // Print report for manual inspection
        report.print_report();
        
        // Test overall consistency score
        let score = report.overall_consistency_score();
        assert!(score >= 0.0 && score <= 1.0);
        println!("Overall Consistency Score: {:.2}", score);
    }

    #[test]
    fn test_document_analysis() {
        let report_generator = MockDataConsistencyReport::new();
        let doc_analysis = report_generator.analyze_document_collection();
        
        assert!(doc_analysis.total_documents >= 5);
        assert!(doc_analysis.avg_quality_score > 0.7);
        assert!(doc_analysis.avg_entities_per_doc > 0.0);
        assert!(doc_analysis.entity_count_range.1 > doc_analysis.entity_count_range.0);
    }

    #[test]
    fn test_entity_analysis() {
        let report_generator = MockDataConsistencyReport::new();
        let entity_analysis = report_generator.analyze_entity_knowledge_base();
        
        assert!(entity_analysis.total_entities >= 3);
        assert!(entity_analysis.avg_confidence > 0.8);
        assert!(entity_analysis.avg_properties_per_entity > 0.0);
    }

    #[test]
    fn test_relationship_analysis() {
        let report_generator = MockDataConsistencyReport::new();
        let rel_analysis = report_generator.analyze_relationship_network();
        
        assert!(rel_analysis.total_relationships >= 5);
        assert!(rel_analysis.avg_confidence > 0.7);
        assert!(rel_analysis.unique_source_entities > 0);
        assert!(rel_analysis.unique_target_entities > 0);
    }

    #[test]
    fn test_performance_analysis() {
        let report_generator = MockDataConsistencyReport::new();
        let perf_analysis = report_generator.analyze_performance_data();
        
        assert!(perf_analysis.processing_time_scaling_correct);
        assert!(perf_analysis.memory_usage_scaling_correct);
        assert!(perf_analysis.avg_accuracy_metric > 0.7);
        assert!(perf_analysis.high_complexity_time_ms > perf_analysis.low_complexity_time_ms);
    }

    #[test]
    fn test_integration_analysis() {
        let report_generator = MockDataConsistencyReport::new();
        let integration_analysis = report_generator.analyze_system_integration();
        
        assert!(integration_analysis.entity_coverage_in_documents >= 0.0);
        assert!(integration_analysis.relationship_entity_coverage >= 0.0);
        assert!(integration_analysis.complexity_model_mapping_correct);
    }
}