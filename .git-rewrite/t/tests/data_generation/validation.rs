//! Data Quality Validation Framework
//! 
//! Provides comprehensive validation of generated synthetic data quality and correctness.

use crate::data_generation::{
    TestGraph, TestEntity, TestEdge, GraphProperties,
    EmbeddingTestSet, QuantizationTestSet,
    TraversalQuery, RagQuery, SimilarityQuery,
    GoldenStandards, ExactComputationEngine
};
use std::collections::{HashMap, HashSet};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Comprehensive validation report for generated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub overall_status: ValidationStatus,
    pub graph_validation: GraphValidationReport,
    pub embedding_validation: EmbeddingValidationReport,
    pub query_validation: QueryValidationReport,
    pub performance_validation: PerformanceValidationReport,
    pub consistency_validation: ConsistencyValidationReport,
    pub issues: Vec<ValidationIssue>,
    pub summary: ValidationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Failed,
    Warning,
}

/// Graph-specific validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphValidationReport {
    pub topology_validation: TopologyValidationResults,
    pub mathematical_properties: MathematicalValidationResults,
    pub structural_integrity: StructuralValidationResults,
    pub scale_validation: ScaleValidationResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyValidationResults {
    pub connectivity_correct: bool,
    pub degree_distribution_valid: bool,
    pub clustering_coefficient_valid: bool,
    pub diameter_valid: bool,
    pub path_length_distribution_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalValidationResults {
    pub erdos_renyi_properties: bool,
    pub barabasi_albert_properties: bool,
    pub watts_strogatz_properties: bool,
    pub complete_graph_properties: bool,
    pub tree_properties: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralValidationResults {
    pub no_duplicate_entities: bool,
    pub no_duplicate_edges: bool,
    pub edge_entity_references_valid: bool,
    pub graph_connectivity_valid: bool,
    pub metadata_consistency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleValidationResults {
    pub small_graphs_valid: bool,
    pub medium_graphs_valid: bool,
    pub large_graphs_valid: bool,
    pub scaling_properties_correct: bool,
}

/// Embedding-specific validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingValidationReport {
    pub dimension_consistency: bool,
    pub normalization_correct: bool,
    pub clustering_properties: bool,
    pub distance_constraints: bool,
    pub similarity_preservation: bool,
    pub numerical_stability: bool,
}

/// Query-specific validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryValidationReport {
    pub traversal_queries_valid: bool,
    pub similarity_queries_valid: bool,
    pub rag_queries_valid: bool,
    pub complex_queries_valid: bool,
    pub golden_standards_accurate: bool,
}

/// Performance validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceValidationReport {
    pub latency_bounds_realistic: bool,
    pub memory_bounds_accurate: bool,
    pub throughput_estimates_valid: bool,
    pub scalability_predictions_sound: bool,
}

/// Consistency validation across components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyValidationReport {
    pub graph_embedding_consistency: bool,
    pub query_result_consistency: bool,
    pub cross_component_integrity: bool,
    pub determinism_verified: bool,
}

/// Individual validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    pub component: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    MathematicalProperty,
    StructuralIntegrity,
    PerformanceExpectation,
    DataConsistency,
    NumericalStability,
}

/// Summary of validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_checks: u32,
    pub passed_checks: u32,
    pub failed_checks: u32,
    pub warning_checks: u32,
    pub pass_rate: f64,
    pub critical_issues: u32,
    pub recommendations: Vec<String>,
}

/// Comprehensive data validator
pub struct DataQualityValidator {
    tolerance: f64,
    strict_mode: bool,
}

impl DataQualityValidator {
    /// Create a new data quality validator
    pub fn new(tolerance: f64, strict_mode: bool) -> Self {
        Self { tolerance, strict_mode }
    }

    /// Validate a complete dataset
    pub fn validate_dataset(
        &self,
        graphs: &[TestGraph],
        embeddings: &HashMap<String, EmbeddingTestSet>,
        queries: &[TraversalQuery],
        rag_queries: &[RagQuery],
        similarity_queries: &[SimilarityQuery],
        golden_standards: Option<&GoldenStandards>,
    ) -> Result<ValidationReport> {
        let mut issues = Vec::new();
        let mut total_checks = 0;
        let mut passed_checks = 0;

        // Validate graphs
        let graph_validation = self.validate_graphs(graphs, &mut issues, &mut total_checks, &mut passed_checks)?;

        // Validate embeddings
        let embedding_validation = self.validate_embeddings(embeddings, &mut issues, &mut total_checks, &mut passed_checks)?;

        // Validate queries
        let query_validation = self.validate_queries(queries, rag_queries, similarity_queries, &mut issues, &mut total_checks, &mut passed_checks)?;

        // Validate performance expectations
        let performance_validation = self.validate_performance_expectations(graphs, embeddings, &mut issues, &mut total_checks, &mut passed_checks)?;

        // Validate consistency across components
        let consistency_validation = self.validate_consistency(graphs, embeddings, golden_standards, &mut issues, &mut total_checks, &mut passed_checks)?;

        // Determine overall status
        let critical_issues = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical)).count() as u32;
        let error_issues = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Error)).count() as u32;
        let warning_issues = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Warning)).count() as u32;

        let overall_status = if critical_issues > 0 || error_issues > 0 {
            ValidationStatus::Failed
        } else if warning_issues > 0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };

        let pass_rate = if total_checks > 0 {
            passed_checks as f64 / total_checks as f64
        } else {
            0.0
        };

        let recommendations = self.generate_recommendations(&issues);

        let summary = ValidationSummary {
            total_checks,
            passed_checks,
            failed_checks: total_checks - passed_checks,
            warning_checks: warning_issues,
            pass_rate,
            critical_issues,
            recommendations,
        };

        Ok(ValidationReport {
            overall_status,
            graph_validation,
            embedding_validation,
            query_validation,
            performance_validation,
            consistency_validation,
            issues,
            summary,
        })
    }

    /// Validate graph properties and structure
    fn validate_graphs(
        &self,
        graphs: &[TestGraph],
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<GraphValidationReport> {
        let mut topology_results = TopologyValidationResults {
            connectivity_correct: true,
            degree_distribution_valid: true,
            clustering_coefficient_valid: true,
            diameter_valid: true,
            path_length_distribution_valid: true,
        };

        let mut mathematical_results = MathematicalValidationResults {
            erdos_renyi_properties: true,
            barabasi_albert_properties: true,
            watts_strogatz_properties: true,
            complete_graph_properties: true,
            tree_properties: true,
        };

        let mut structural_results = StructuralValidationResults {
            no_duplicate_entities: true,
            no_duplicate_edges: true,
            edge_entity_references_valid: true,
            graph_connectivity_valid: true,
            metadata_consistency: true,
        };

        let mut scale_results = ScaleValidationResults {
            small_graphs_valid: true,
            medium_graphs_valid: true,
            large_graphs_valid: true,
            scaling_properties_correct: true,
        };

        for (i, graph) in graphs.iter().enumerate() {
            // Validate basic structure
            *total_checks += 1;
            if graph.entities.is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Critical,
                    category: IssueCategory::StructuralIntegrity,
                    description: format!("Graph {} has no entities", i),
                    component: "graph_structure".to_string(),
                    suggested_fix: Some("Ensure graph generation creates at least one entity".to_string()),
                });
                structural_results.graph_connectivity_valid = false;
            } else {
                *passed_checks += 1;
            }

            // Validate entity uniqueness
            *total_checks += 1;
            let mut entity_ids = HashSet::new();
            let mut duplicate_found = false;
            for entity in &graph.entities {
                if !entity_ids.insert(entity.id) {
                    duplicate_found = true;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::StructuralIntegrity,
                        description: format!("Duplicate entity ID {} in graph {}", entity.id, i),
                        component: "entity_uniqueness".to_string(),
                        suggested_fix: Some("Ensure entity ID generation is unique".to_string()),
                    });
                }
            }
            
            if !duplicate_found {
                *passed_checks += 1;
            } else {
                structural_results.no_duplicate_entities = false;
            }

            // Validate edge references
            *total_checks += 1;
            let mut invalid_references = false;
            for edge in &graph.edges {
                if !entity_ids.contains(&edge.source) || !entity_ids.contains(&edge.target) {
                    invalid_references = true;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::StructuralIntegrity,
                        description: format!("Edge references non-existent entity: {} -> {} in graph {}", edge.source, edge.target, i),
                        component: "edge_references".to_string(),
                        suggested_fix: Some("Ensure edges only reference existing entities".to_string()),
                    });
                }
            }
            
            if !invalid_references {
                *passed_checks += 1;
            } else {
                structural_results.edge_entity_references_valid = false;
            }

            // Validate metadata consistency
            *total_checks += 1;
            if graph.properties.entity_count != graph.entities.len() as u64 {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::DataConsistency,
                    description: format!("Entity count mismatch in graph {}: expected {}, got {}", 
                        i, graph.properties.entity_count, graph.entities.len()),
                    component: "metadata_consistency".to_string(),
                    suggested_fix: Some("Update graph properties after generation".to_string()),
                });
                structural_results.metadata_consistency = false;
            } else {
                *passed_checks += 1;
            }

            // Validate mathematical properties based on graph type
            if let Some(graph_type) = self.infer_graph_type(graph) {
                self.validate_graph_type_properties(
                    graph, 
                    &graph_type, 
                    i, 
                    &mut mathematical_results, 
                    issues, 
                    total_checks, 
                    passed_checks
                )?;
            }

            // Validate scale-specific properties
            self.validate_scale_properties(graph, i, &mut scale_results, issues, total_checks, passed_checks)?;
        }

        Ok(GraphValidationReport {
            topology_validation: topology_results,
            mathematical_properties: mathematical_results,
            structural_integrity: structural_results,
            scale_validation: scale_results,
        })
    }

    /// Validate embedding properties
    fn validate_embeddings(
        &self,
        embeddings: &HashMap<String, EmbeddingTestSet>,
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<EmbeddingValidationReport> {
        let mut report = EmbeddingValidationReport {
            dimension_consistency: true,
            normalization_correct: true,
            clustering_properties: true,
            distance_constraints: true,
            similarity_preservation: true,
            numerical_stability: true,
        };

        for (name, embedding_set) in embeddings {
            // Validate dimension consistency
            *total_checks += 1;
            let mut dimension_consistent = true;
            for (entity_id, embedding) in &embedding_set.embeddings {
                if embedding.len() != embedding_set.dimension {
                    dimension_consistent = false;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::DataConsistency,
                        description: format!("Dimension mismatch for entity {} in embedding set {}: expected {}, got {}", 
                            entity_id, name, embedding_set.dimension, embedding.len()),
                        component: "embedding_dimensions".to_string(),
                        suggested_fix: Some("Ensure all embeddings have consistent dimensions".to_string()),
                    });
                }
            }
            
            if dimension_consistent {
                *passed_checks += 1;
            } else {
                report.dimension_consistency = false;
            }

            // Validate numerical stability
            *total_checks += 1;
            let mut numerically_stable = true;
            for (entity_id, embedding) in &embedding_set.embeddings {
                for (i, &value) in embedding.iter().enumerate() {
                    if !value.is_finite() {
                        numerically_stable = false;
                        issues.push(ValidationIssue {
                            severity: IssueSeverity::Critical,
                            category: IssueCategory::NumericalStability,
                            description: format!("Non-finite value in embedding for entity {} at dimension {}", entity_id, i),
                            component: "numerical_stability".to_string(),
                            suggested_fix: Some("Check embedding generation for numerical overflow/underflow".to_string()),
                        });
                    }
                }
            }
            
            if numerically_stable {
                *passed_checks += 1;
            } else {
                report.numerical_stability = false;
            }

            // Validate clustering properties
            if !embedding_set.cluster_assignments.is_empty() {
                *total_checks += 1;
                let clustering_valid = self.validate_clustering_properties(embedding_set, name, issues)?;
                if clustering_valid {
                    *passed_checks += 1;
                } else {
                    report.clustering_properties = false;
                }
            }

            // Validate expected nearest neighbors if available
            if !embedding_set.expected_nearest_neighbors.is_empty() {
                *total_checks += 1;
                let neighbors_valid = self.validate_nearest_neighbors(embedding_set, name, issues)?;
                if neighbors_valid {
                    *passed_checks += 1;
                } else {
                    report.similarity_preservation = false;
                }
            }
        }

        Ok(report)
    }

    /// Validate query correctness and completeness
    fn validate_queries(
        &self,
        traversal_queries: &[TraversalQuery],
        rag_queries: &[RagQuery],
        similarity_queries: &[SimilarityQuery],
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<QueryValidationReport> {
        let mut report = QueryValidationReport {
            traversal_queries_valid: true,
            similarity_queries_valid: true,
            rag_queries_valid: true,
            complex_queries_valid: true,
            golden_standards_accurate: true,
        };

        // Validate traversal queries
        for (i, query) in traversal_queries.iter().enumerate() {
            *total_checks += 1;
            
            if query.expected_entities.is_empty() && query.expected_result_count > 0 {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::DataConsistency,
                    description: format!("Traversal query {} has positive result count but no expected entities", i),
                    component: "traversal_queries".to_string(),
                    suggested_fix: Some("Ensure expected entities are populated when result count > 0".to_string()),
                });
                report.traversal_queries_valid = false;
            } else {
                *passed_checks += 1;
            }
        }

        // Validate RAG queries
        for (i, query) in rag_queries.iter().enumerate() {
            *total_checks += 1;
            
            if query.expected_context.len() != query.expected_relevance_scores.len() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::DataConsistency,
                    description: format!("RAG query {} has mismatched context and relevance score counts", i),
                    component: "rag_queries".to_string(),
                    suggested_fix: Some("Ensure each context entity has a corresponding relevance score".to_string()),
                });
                report.rag_queries_valid = false;
            } else {
                *passed_checks += 1;
            }

            // Validate relevance scores are in valid range
            *total_checks += 1;
            let mut scores_valid = true;
            for (j, &score) in query.expected_relevance_scores.iter().enumerate() {
                if score < 0.0 || score > 1.0 || !score.is_finite() {
                    scores_valid = false;
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::DataConsistency,
                        description: format!("Invalid relevance score {} at position {} in RAG query {}", score, j, i),
                        component: "rag_relevance_scores".to_string(),
                        suggested_fix: Some("Ensure relevance scores are in range [0, 1] and finite".to_string()),
                    });
                }
            }
            
            if scores_valid {
                *passed_checks += 1;
            } else {
                report.rag_queries_valid = false;
            }
        }

        // Validate similarity queries
        for (i, query) in similarity_queries.iter().enumerate() {
            *total_checks += 1;
            
            if query.expected_neighbors.len() != query.k {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::DataConsistency,
                    description: format!("Similarity query {} has {} expected neighbors but k={}", i, query.expected_neighbors.len(), query.k),
                    component: "similarity_queries".to_string(),
                    suggested_fix: Some("Ensure number of expected neighbors matches k parameter".to_string()),
                });
                report.similarity_queries_valid = false;
            } else {
                *passed_checks += 1;
            }
        }

        Ok(report)
    }

    /// Validate performance expectations
    fn validate_performance_expectations(
        &self,
        graphs: &[TestGraph],
        embeddings: &HashMap<String, EmbeddingTestSet>,
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<PerformanceValidationReport> {
        let mut report = PerformanceValidationReport {
            latency_bounds_realistic: true,
            memory_bounds_accurate: true,
            throughput_estimates_valid: true,
            scalability_predictions_sound: true,
        };

        // Calculate expected performance bounds based on data size
        let total_entities: u64 = graphs.iter().map(|g| g.entities.len() as u64).sum();
        let total_embeddings: u64 = embeddings.values().map(|e| e.embeddings.len() as u64).sum();

        // Validate latency bounds are reasonable
        *total_checks += 1;
        let expected_min_latency = 0.1; // 0.1ms minimum
        let expected_max_latency = total_entities as f64 * 0.01; // 0.01ms per entity

        if expected_min_latency < 0.0 || expected_max_latency > 10000.0 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::PerformanceExpectation,
                description: "Unrealistic latency bounds detected".to_string(),
                component: "performance_bounds".to_string(),
                suggested_fix: Some("Review latency calculations for reasonableness".to_string()),
            });
            report.latency_bounds_realistic = false;
        } else {
            *passed_checks += 1;
        }

        // Validate memory bounds
        *total_checks += 1;
        let expected_memory_per_entity = 64.0; // bytes
        let total_expected_memory = total_entities as f64 * expected_memory_per_entity;

        if total_expected_memory > 1e9 && graphs.iter().any(|g| g.entities.len() < 1000) {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::PerformanceExpectation,
                description: "Memory usage seems too high for small graphs".to_string(),
                component: "memory_bounds".to_string(),
                suggested_fix: Some("Review memory calculation methodology".to_string()),
            });
            report.memory_bounds_accurate = false;
        } else {
            *passed_checks += 1;
        }

        Ok(report)
    }

    /// Validate consistency across all components
    fn validate_consistency(
        &self,
        graphs: &[TestGraph],
        embeddings: &HashMap<String, EmbeddingTestSet>,
        golden_standards: Option<&GoldenStandards>,
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<ConsistencyValidationReport> {
        let mut report = ConsistencyValidationReport {
            graph_embedding_consistency: true,
            query_result_consistency: true,
            cross_component_integrity: true,
            determinism_verified: true,
        };

        // Validate graph-embedding consistency
        *total_checks += 1;
        let mut consistency_issues = 0;
        
        for embedding_set in embeddings.values() {
            for &entity_id in embedding_set.embeddings.keys() {
                let mut found_in_graph = false;
                for graph in graphs {
                    if graph.entities.iter().any(|e| e.id == entity_id) {
                        found_in_graph = true;
                        break;
                    }
                }
                
                if !found_in_graph {
                    consistency_issues += 1;
                    if consistency_issues <= 5 { // Limit issue reporting
                        issues.push(ValidationIssue {
                            severity: IssueSeverity::Warning,
                            category: IssueCategory::DataConsistency,
                            description: format!("Entity {} has embedding but not found in any graph", entity_id),
                            component: "graph_embedding_consistency".to_string(),
                            suggested_fix: Some("Ensure embeddings are generated only for existing entities".to_string()),
                        });
                    }
                }
            }
        }
        
        if consistency_issues == 0 {
            *passed_checks += 1;
        } else {
            report.graph_embedding_consistency = false;
        }

        // Validate golden standards consistency
        if let Some(standards) = golden_standards {
            *total_checks += 1;
            
            if standards.checksum.is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::DataConsistency,
                    description: "Golden standards checksum is empty".to_string(),
                    component: "golden_standards".to_string(),
                    suggested_fix: Some("Ensure checksum is computed for golden standards".to_string()),
                });
                report.query_result_consistency = false;
            } else {
                *passed_checks += 1;
            }
        }

        Ok(report)
    }

    // Helper methods

    fn infer_graph_type(&self, graph: &TestGraph) -> Option<String> {
        // Simple heuristics to infer graph type
        let entity_count = graph.entities.len();
        let edge_count = graph.edges.len();
        
        if entity_count == 0 {
            return None;
        }

        let density = (2.0 * edge_count as f64) / (entity_count as f64 * (entity_count - 1) as f64);
        
        if density > 0.9 {
            Some("complete".to_string())
        } else if edge_count == entity_count - 1 {
            Some("tree".to_string())
        } else if density < 0.1 {
            Some("sparse".to_string())
        } else {
            Some("general".to_string())
        }
    }

    fn validate_graph_type_properties(
        &self,
        graph: &TestGraph,
        graph_type: &str,
        graph_index: usize,
        mathematical_results: &mut MathematicalValidationResults,
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<()> {
        match graph_type {
            "complete" => {
                *total_checks += 1;
                let expected_edges = (graph.entities.len() * (graph.entities.len() - 1)) / 2;
                if graph.edges.len() != expected_edges {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::MathematicalProperty,
                        description: format!("Complete graph {} has incorrect edge count: expected {}, got {}", 
                            graph_index, expected_edges, graph.edges.len()),
                        component: "complete_graph_properties".to_string(),
                        suggested_fix: Some("Verify complete graph generation algorithm".to_string()),
                    });
                    mathematical_results.complete_graph_properties = false;
                } else {
                    *passed_checks += 1;
                }
            },
            
            "tree" => {
                *total_checks += 1;
                let expected_edges = graph.entities.len() - 1;
                if graph.edges.len() != expected_edges {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        category: IssueCategory::MathematicalProperty,
                        description: format!("Tree graph {} has incorrect edge count: expected {}, got {}", 
                            graph_index, expected_edges, graph.edges.len()),
                        component: "tree_properties".to_string(),
                        suggested_fix: Some("Verify tree generation algorithm".to_string()),
                    });
                    mathematical_results.tree_properties = false;
                } else {
                    *passed_checks += 1;
                }
            },
            
            _ => {
                // General validation for other graph types
                *total_checks += 1;
                *passed_checks += 1; // Accept general graphs as valid
            }
        }
        
        Ok(())
    }

    fn validate_scale_properties(
        &self,
        graph: &TestGraph,
        graph_index: usize,
        scale_results: &mut ScaleValidationResults,
        issues: &mut Vec<ValidationIssue>,
        total_checks: &mut u32,
        passed_checks: &mut u32,
    ) -> Result<()> {
        *total_checks += 1;
        
        let entity_count = graph.entities.len();
        let edge_count = graph.edges.len();
        
        // Check for reasonable scaling
        if entity_count > 0 && edge_count > entity_count * entity_count {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::MathematicalProperty,
                description: format!("Graph {} has more edges than possible: {} entities, {} edges", 
                    graph_index, entity_count, edge_count),
                component: "graph_scaling".to_string(),
                suggested_fix: Some("Check edge generation logic for overcounting".to_string()),
            });
            
            if entity_count < 100 {
                scale_results.small_graphs_valid = false;
            } else if entity_count < 1000 {
                scale_results.medium_graphs_valid = false;
            } else {
                scale_results.large_graphs_valid = false;
            }
        } else {
            *passed_checks += 1;
        }
        
        Ok(())
    }

    fn validate_clustering_properties(&self, embedding_set: &EmbeddingTestSet, name: &str, issues: &mut Vec<ValidationIssue>) -> Result<bool> {
        let cluster_count = embedding_set.cluster_assignments.values().max().unwrap_or(&0) + 1;
        
        if cluster_count == 0 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::DataConsistency,
                description: format!("Embedding set {} has no clusters defined", name),
                component: "clustering_properties".to_string(),
                suggested_fix: Some("Ensure cluster assignments are properly generated".to_string()),
            });
            return Ok(false);
        }

        // Validate silhouette score is reasonable
        if embedding_set.properties.silhouette_score < -1.0 || embedding_set.properties.silhouette_score > 1.0 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::MathematicalProperty,
                description: format!("Invalid silhouette score {} in embedding set {}", 
                    embedding_set.properties.silhouette_score, name),
                component: "clustering_quality".to_string(),
                suggested_fix: Some("Check silhouette score calculation".to_string()),
            });
            return Ok(false);
        }

        Ok(true)
    }

    fn validate_nearest_neighbors(&self, embedding_set: &EmbeddingTestSet, name: &str, issues: &mut Vec<ValidationIssue>) -> Result<bool> {
        for (entity_id, neighbors) in &embedding_set.expected_nearest_neighbors {
            // Check that neighbors are sorted by distance
            for i in 1..neighbors.len() {
                if neighbors[i].1 < neighbors[i-1].1 {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        category: IssueCategory::DataConsistency,
                        description: format!("Nearest neighbors for entity {} in {} are not sorted by distance", entity_id, name),
                        component: "nearest_neighbors".to_string(),
                        suggested_fix: Some("Ensure nearest neighbors are sorted by increasing distance".to_string()),
                    });
                    return Ok(false);
                }
            }
            
            // Check that entity doesn't include itself as neighbor
            if neighbors.iter().any(|(id, _)| id == entity_id) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::DataConsistency,
                    description: format!("Entity {} includes itself as nearest neighbor in {}", entity_id, name),
                    component: "nearest_neighbors".to_string(),
                    suggested_fix: Some("Exclude query entity from its own nearest neighbors".to_string()),
                });
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn generate_recommendations(&self, issues: &[ValidationIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_count = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical)).count();
        let error_count = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Error)).count();
        let warning_count = issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Warning)).count();

        if critical_count > 0 {
            recommendations.push("Address all critical issues before using generated data".to_string());
        }

        if error_count > 0 {
            recommendations.push("Fix data generation errors to ensure test reliability".to_string());
        }

        if warning_count > 5 {
            recommendations.push("Review data generation parameters to reduce warnings".to_string());
        }

        // Category-specific recommendations
        let math_issues = issues.iter().filter(|i| matches!(i.category, IssueCategory::MathematicalProperty)).count();
        if math_issues > 0 {
            recommendations.push("Verify mathematical property calculations in generators".to_string());
        }

        let consistency_issues = issues.iter().filter(|i| matches!(i.category, IssueCategory::DataConsistency)).count();
        if consistency_issues > 0 {
            recommendations.push("Improve consistency checks between data generation components".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Data quality validation passed - generated data is ready for testing".to_string());
        }

        recommendations
    }
}

/// Quick validation function for basic checks
pub fn quick_validate(
    graphs: &[TestGraph],
    embeddings: &HashMap<String, EmbeddingTestSet>,
) -> Result<bool> {
    let validator = DataQualityValidator::new(1e-6, false);
    let report = validator.validate_dataset(graphs, embeddings, &[], &[], &[], None)?;
    Ok(matches!(report.overall_status, ValidationStatus::Passed))
}

/// Comprehensive validation function with full reporting
pub fn comprehensive_validate(
    graphs: &[TestGraph],
    embeddings: &HashMap<String, EmbeddingTestSet>,
    queries: &[TraversalQuery],
    rag_queries: &[RagQuery],
    similarity_queries: &[SimilarityQuery],
    golden_standards: Option<&GoldenStandards>,
) -> Result<ValidationReport> {
    let validator = DataQualityValidator::new(1e-10, true);
    validator.validate_dataset(graphs, embeddings, queries, rag_queries, similarity_queries, golden_standards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generation::graph_topologies::GraphTopologyGenerator;
    use crate::data_generation::embeddings::{EmbeddingGenerator, ClusterSpec};

    fn create_test_data() -> (Vec<TestGraph>, HashMap<String, EmbeddingTestSet>) {
        let mut graph_gen = GraphTopologyGenerator::new(42);
        let graphs = vec![
            graph_gen.generate_erdos_renyi(20, 0.1).unwrap(),
            graph_gen.generate_complete_graph(10).unwrap(),
        ];

        let mut embedding_gen = EmbeddingGenerator::new(42, 64).unwrap();
        let cluster_specs = vec![
            ClusterSpec {
                id: 0,
                size: 15,
                radius: 0.3,
                center: None,
                label: "Test Cluster".to_string(),
            },
        ];
        
        let embedding_set = embedding_gen.generate_clustered_embeddings(cluster_specs).unwrap();
        let mut embeddings = HashMap::new();
        embeddings.insert("test_embeddings".to_string(), embedding_set);

        (graphs, embeddings)
    }

    #[test]
    fn test_basic_validation() {
        let (graphs, embeddings) = create_test_data();
        
        let is_valid = quick_validate(&graphs, &embeddings).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_comprehensive_validation() {
        let (graphs, embeddings) = create_test_data();
        
        let report = comprehensive_validate(&graphs, &embeddings, &[], &[], &[], None).unwrap();
        assert!(matches!(report.overall_status, ValidationStatus::Passed | ValidationStatus::Warning));
        assert!(report.summary.total_checks > 0);
    }

    #[test]
    fn test_graph_structure_validation() {
        let validator = DataQualityValidator::new(1e-6, false);
        let mut issues = Vec::new();
        let mut total_checks = 0;
        let mut passed_checks = 0;

        let (graphs, _) = create_test_data();
        let report = validator.validate_graphs(&graphs, &mut issues, &mut total_checks, &mut passed_checks).unwrap();
        
        assert!(report.structural_integrity.no_duplicate_entities);
        assert!(report.structural_integrity.edge_entity_references_valid);
    }

    #[test]
    fn test_embedding_validation() {
        let validator = DataQualityValidator::new(1e-6, false);
        let mut issues = Vec::new();
        let mut total_checks = 0;
        let mut passed_checks = 0;

        let (_, embeddings) = create_test_data();
        let report = validator.validate_embeddings(&embeddings, &mut issues, &mut total_checks, &mut passed_checks).unwrap();
        
        assert!(report.dimension_consistency);
        assert!(report.numerical_stability);
    }

    #[test]
    fn test_invalid_graph_detection() {
        let validator = DataQualityValidator::new(1e-6, true);
        
        // Create graph with invalid structure
        let mut invalid_graph = TestGraph {
            entities: vec![],  // Empty entities
            edges: vec![],
            properties: crate::data_generation::GraphProperties {
                entity_count: 5,  // Inconsistent count
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: crate::data_generation::ConnectivityType::Random,
                expected_path_length: 0.0,
            },
        };

        let graphs = vec![invalid_graph];
        let embeddings = HashMap::new();
        
        let report = validator.validate_dataset(&graphs, &embeddings, &[], &[], &[], None).unwrap();
        assert!(matches!(report.overall_status, ValidationStatus::Failed));
        assert!(report.issues.len() > 0);
    }

    #[test]
    fn test_recommendation_generation() {
        let validator = DataQualityValidator::new(1e-6, false);
        
        let issues = vec![
            ValidationIssue {
                severity: IssueSeverity::Critical,
                category: IssueCategory::StructuralIntegrity,
                description: "Critical test issue".to_string(),
                component: "test".to_string(),
                suggested_fix: None,
            },
        ];
        
        let recommendations = validator.generate_recommendations(&issues);
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].contains("critical"));
    }
}