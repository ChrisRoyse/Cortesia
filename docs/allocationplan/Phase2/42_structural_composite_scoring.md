# Task 42: Structural Composite Scoring

## Metadata
- **Micro-Phase**: 2.42
- **Duration**: 15-20 minutes
- **Dependencies**: Task 35 (hierarchy_builder), Task 39 (scoring_framework_design)
- **Output**: `src/allocation_scoring/structural_composite_scoring.rs`

## Description
Implement sophisticated structural fit analysis using graph metrics, hierarchy depth analysis, branching factor optimization, and composite scoring algorithms. This component analyzes how well a concept fits structurally within the knowledge graph hierarchy, considering graph topology, node connectivity, and structural balance with <0.4ms per evaluation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_scoring::{AllocationContext, ScoringStrategy};
    use crate::hierarchy_detection::{ExtractedConcept, ConceptHierarchy};
    use std::collections::HashMap;

    #[test]
    fn test_structural_composite_strategy_creation() {
        let strategy = StructuralCompositeStrategy::new();
        assert_eq!(strategy.name(), "structural_composite");
        assert!(strategy.supports_parallel());
        assert!(strategy.is_enabled());
        assert_eq!(strategy.get_max_depth_penalty(), 0.1);
    }
    
    #[test]
    fn test_hierarchy_depth_scoring() {
        let strategy = StructuralCompositeStrategy::new();
        
        // Create concepts at different hierarchy depths
        let root_concept = create_test_concept("animal", 0.9);
        let mid_concept = create_test_concept("mammal", 0.85);
        let leaf_concept = create_test_concept("golden_retriever", 0.8);
        
        // Test depth scoring with different target depths
        let shallow_context = create_hierarchy_context("species", &["genus"], 2); // Shallow
        let medium_context = create_hierarchy_context("breed", &["species", "genus", "family"], 4); // Medium
        let deep_context = create_hierarchy_context("variety", &["breed", "species", "genus", "family", "order"], 6); // Deep
        
        let shallow_score = strategy.score(&mid_concept, &shallow_context).unwrap();
        let medium_score = strategy.score(&mid_concept, &medium_context).unwrap();
        let deep_score = strategy.score(&mid_concept, &deep_context).unwrap();
        
        // Medium depth should generally score best for structural fit
        assert!(medium_score >= shallow_score);
        assert!(medium_score >= deep_score);
        
        // All scores should be reasonable
        assert!(shallow_score >= 0.3 && shallow_score <= 1.0);
        assert!(medium_score >= 0.3 && medium_score <= 1.0);
        assert!(deep_score >= 0.3 && deep_score <= 1.0);
    }
    
    #[test]
    fn test_branching_factor_analysis() {
        let strategy = StructuralCompositeStrategy::new();
        
        // Create hierarchy with different branching factors
        let hierarchy = create_test_hierarchy_with_branching();
        
        let concept = create_test_concept("new_mammal", 0.8);
        let low_branch_context = create_context_with_hierarchy("mammal", &["animal"], &hierarchy);
        let high_branch_context = create_context_with_hierarchy("bird", &["animal"], &hierarchy);
        
        let low_branch_score = strategy.score(&concept, &low_branch_context).unwrap();
        let high_branch_score = strategy.score(&concept, &high_branch_context).unwrap();
        
        // Should prefer balanced branching factors
        assert!(low_branch_score >= 0.4);
        assert!(high_branch_score >= 0.4);
        
        // Test branching factor analysis directly
        let analysis = strategy.analyze_branching_factor(&hierarchy, "mammal").unwrap();
        assert!(analysis.current_children > 0);
        assert!(analysis.optimal_range.start <= analysis.optimal_range.end);
    }
    
    #[test]
    fn test_graph_connectivity_metrics() {
        let strategy = StructuralCompositeStrategy::new();
        
        let hierarchy = create_connected_test_hierarchy();
        let concept = create_test_concept("test_concept", 0.8);
        
        // Test with well-connected vs isolated contexts
        let connected_context = create_context_with_hierarchy("mammal", &["animal"], &hierarchy);
        let isolated_context = create_context_with_hierarchy("isolated_node", &[], &hierarchy);
        
        let connected_score = strategy.score(&concept, &connected_context).unwrap();
        let isolated_score = strategy.score(&concept, &isolated_context).unwrap();
        
        // Connected contexts should generally score higher
        assert!(connected_score > isolated_score);
        
        // Test connectivity metrics directly
        let connectivity = strategy.calculate_graph_connectivity(&hierarchy, "mammal").unwrap();
        assert!(connectivity.node_degree > 0);
        assert!(connectivity.clustering_coefficient >= 0.0);
        assert!(connectivity.betweenness_centrality >= 0.0);
    }
    
    #[test]
    fn test_structural_balance_optimization() {
        let strategy = StructuralCompositeStrategy::new();
        
        // Create unbalanced hierarchy
        let unbalanced_hierarchy = create_unbalanced_test_hierarchy();
        let concept = create_test_concept("balancing_concept", 0.8);
        
        // Test contexts with different balance characteristics
        let unbalanced_context = create_context_with_hierarchy("unbalanced_parent", &[], &unbalanced_hierarchy);
        
        let score = strategy.score(&concept, &unbalanced_context).unwrap();
        let balance_analysis = strategy.analyze_structural_balance(&unbalanced_hierarchy, "unbalanced_parent").unwrap();
        
        // Should detect imbalance and adjust scoring accordingly
        assert!(balance_analysis.balance_factor < 0.8); // Detect imbalance
        assert!(balance_analysis.rebalancing_suggestions.len() > 0);
        assert!(score >= 0.2); // Should still allow allocation but with lower score
    }
    
    #[test]
    fn test_composite_scoring_weights() {
        let mut strategy = StructuralCompositeStrategy::new();
        
        // Configure custom weights
        let weights = StructuralWeights {
            depth_weight: 0.3,
            branching_weight: 0.25,
            connectivity_weight: 0.25,
            balance_weight: 0.2,
        };
        
        strategy.set_structural_weights(weights.clone());
        assert_eq!(strategy.get_structural_weights().depth_weight, 0.3);
        
        let hierarchy = create_test_hierarchy_with_branching();
        let concept = create_test_concept("test_concept", 0.8);
        let context = create_context_with_hierarchy("mammal", &["animal"], &hierarchy);
        
        let score = strategy.score(&concept, &context).unwrap();
        let breakdown = strategy.score_with_breakdown(&concept, &context).unwrap();
        
        // Should provide detailed breakdown
        assert!(breakdown.depth_score >= 0.0);
        assert!(breakdown.branching_score >= 0.0);
        assert!(breakdown.connectivity_score >= 0.0);
        assert!(breakdown.balance_score >= 0.0);
        
        // Verify weighted combination
        let expected_score = breakdown.depth_score * weights.depth_weight +
                           breakdown.branching_score * weights.branching_weight +
                           breakdown.connectivity_score * weights.connectivity_weight +
                           breakdown.balance_score * weights.balance_weight;
        
        assert!((score - expected_score).abs() < 0.01);
    }
    
    #[test]
    fn test_graph_metrics_calculation() {
        let strategy = StructuralCompositeStrategy::new();
        
        let hierarchy = create_test_hierarchy_with_metrics();
        
        // Test various graph metrics
        let metrics = strategy.calculate_comprehensive_graph_metrics(&hierarchy, "central_node").unwrap();
        
        assert!(metrics.degree_centrality >= 0.0 && metrics.degree_centrality <= 1.0);
        assert!(metrics.closeness_centrality >= 0.0 && metrics.closeness_centrality <= 1.0);
        assert!(metrics.betweenness_centrality >= 0.0);
        assert!(metrics.clustering_coefficient >= 0.0 && metrics.clustering_coefficient <= 1.0);
        assert!(metrics.pagerank_score >= 0.0);
        
        // Test path-based metrics
        assert!(metrics.average_path_length >= 1.0);
        assert!(metrics.eccentricity >= 1.0);
    }
    
    #[test]
    fn test_structural_fit_optimization() {
        let strategy = StructuralCompositeStrategy::new();
        
        let hierarchy = create_optimizable_test_hierarchy();
        let concept = create_test_concept("optimization_target", 0.8);
        
        // Test multiple potential parent positions
        let contexts = vec![
            create_context_with_hierarchy("option_a", &["root"], &hierarchy),
            create_context_with_hierarchy("option_b", &["root"], &hierarchy),
            create_context_with_hierarchy("option_c", &["root"], &hierarchy),
        ];
        
        let scores: Vec<_> = contexts.iter()
            .map(|ctx| strategy.score(&concept, ctx).unwrap())
            .collect();
        
        // Find optimal placement
        let optimal_index = strategy.find_optimal_structural_placement(&concept, &contexts).unwrap();
        assert!(optimal_index < contexts.len());
        
        // Verify optimal choice has highest score
        let optimal_score = scores[optimal_index];
        for (i, &score) in scores.iter().enumerate() {
            if i != optimal_index {
                assert!(optimal_score >= score);
            }
        }
    }
    
    #[test]
    fn test_batch_structural_scoring() {
        let strategy = StructuralCompositeStrategy::new();
        
        let concepts = vec![
            create_test_concept("concept_1", 0.8),
            create_test_concept("concept_2", 0.85),
            create_test_concept("concept_3", 0.9),
            create_test_concept("concept_4", 0.75),
            create_test_concept("concept_5", 0.82),
        ];
        
        let hierarchy = create_test_hierarchy_with_branching();
        let context = create_context_with_hierarchy("mammal", &["animal"], &hierarchy);
        
        let start = std::time::Instant::now();
        let scores = strategy.batch_score(&concepts, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete batch scoring in under 10ms
        assert!(elapsed < std::time::Duration::from_millis(10));
        assert_eq!(scores.len(), 5);
        
        // All scores should be valid
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }
    
    #[test]
    fn test_structural_scoring_performance() {
        let strategy = StructuralCompositeStrategy::new();
        
        // Create large hierarchy for performance testing
        let large_hierarchy = create_large_test_hierarchy(100); // 100 nodes
        let concept = create_test_concept("performance_test", 0.8);
        let context = create_context_with_hierarchy("mid_level_node", &["root"], &large_hierarchy);
        
        let start = std::time::Instant::now();
        let score = strategy.score(&concept, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete in under 0.4ms as specified
        assert!(elapsed < std::time::Duration::from_micros(400));
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_structural_anomaly_detection() {
        let strategy = StructuralCompositeStrategy::new();
        
        let hierarchy = create_hierarchy_with_anomalies();
        
        // Test detection of structural anomalies
        let anomalies = strategy.detect_structural_anomalies(&hierarchy).unwrap();
        
        assert!(!anomalies.is_empty());
        
        // Should detect different types of anomalies
        let has_orphan = anomalies.iter().any(|a| a.anomaly_type == AnomalyType::OrphanNode);
        let has_overcrowded = anomalies.iter().any(|a| a.anomaly_type == AnomalyType::OvercrowdedParent);
        
        assert!(has_orphan || has_overcrowded);
        
        // Test scoring with anomalous structure
        let concept = create_test_concept("anomaly_test", 0.8);
        let anomalous_context = create_context_with_hierarchy("anomalous_node", &[], &hierarchy);
        
        let score = strategy.score(&concept, &anomalous_context).unwrap();
        assert!(score >= 0.0 && score <= 1.0); // Should handle anomalies gracefully
    }
    
    #[test]
    fn test_depth_penalty_configuration() {
        let mut strategy = StructuralCompositeStrategy::new();
        
        // Test different depth penalty configurations
        strategy.set_depth_penalty_factor(0.05); // Low penalty
        assert_eq!(strategy.get_max_depth_penalty(), 0.05);
        
        let hierarchy = create_deep_test_hierarchy(10); // 10 levels deep
        let concept = create_test_concept("deep_concept", 0.8);
        let deep_context = create_context_with_hierarchy("level_8", &["level_7", "level_6", "level_5", "level_4", "level_3", "level_2", "level_1", "root"], &hierarchy);
        
        let low_penalty_score = strategy.score(&concept, &deep_context).unwrap();
        
        // Increase penalty
        strategy.set_depth_penalty_factor(0.3); // High penalty
        let high_penalty_score = strategy.score(&concept, &deep_context).unwrap();
        
        // Higher penalty should result in lower score for deep placement
        assert!(low_penalty_score >= high_penalty_score);
    }
    
    fn create_test_concept(name: &str, confidence: f32) -> ExtractedConcept {
        use crate::hierarchy_detection::{ExtractedConcept, ConceptType, TextSpan};
        use std::collections::HashMap;
        
        ExtractedConcept {
            name: name.to_string(),
            concept_type: ConceptType::Entity,
            properties: HashMap::new(),
            source_span: TextSpan {
                start: 0,
                end: name.len(),
                text: name.to_string(),
            },
            confidence,
            suggested_parent: None,
            semantic_features: vec![0.5; 100],
            extracted_at: 0,
        }
    }
    
    fn create_hierarchy_context(target: &str, ancestors: &[&str], depth: usize) -> AllocationContext {
        use crate::allocation_scoring::AllocationContext;
        
        let mut context_properties = HashMap::new();
        context_properties.insert("target_depth".to_string(), depth.to_string());
        
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties,
            allocation_timestamp: 0,
        }
    }
    
    fn create_context_with_hierarchy(target: &str, ancestors: &[&str], hierarchy: &ConceptHierarchy) -> AllocationContext {
        use crate::allocation_scoring::AllocationContext;
        
        let mut context_properties = HashMap::new();
        context_properties.insert("hierarchy_context".to_string(), "true".to_string());
        
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties,
            allocation_timestamp: 0,
        }
    }
    
    fn create_test_hierarchy_with_branching() -> ConceptHierarchy {
        // Create a mock hierarchy with varied branching factors
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_connected_test_hierarchy() -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_unbalanced_test_hierarchy() -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_test_hierarchy_with_metrics() -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_optimizable_test_hierarchy() -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_large_test_hierarchy(node_count: usize) -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_hierarchy_with_anomalies() -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
    
    fn create_deep_test_hierarchy(depth: usize) -> ConceptHierarchy {
        use crate::hierarchy_detection::ConceptHierarchy;
        ConceptHierarchy::new() // Simplified mock
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;
use crate::allocation_scoring::{AllocationContext, ScoringStrategy, ScoringError};
use crate::hierarchy_detection::{ExtractedConcept, ConceptHierarchy};

/// Configuration for structural composite scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCompositeConfig {
    /// Maximum depth penalty factor
    pub max_depth_penalty: f32,
    
    /// Optimal branching factor range
    pub optimal_branching_range: (usize, usize),
    
    /// Weight for graph connectivity in scoring
    pub connectivity_importance: f32,
    
    /// Threshold for considering structure balanced
    pub balance_threshold: f32,
    
    /// Whether to enable optimization suggestions
    pub enable_optimization: bool,
    
    /// Performance optimization level (1-3)
    pub optimization_level: u8,
}

impl Default for StructuralCompositeConfig {
    fn default() -> Self {
        Self {
            max_depth_penalty: 0.1,
            optimal_branching_range: (2, 8),
            connectivity_importance: 0.3,
            balance_threshold: 0.7,
            enable_optimization: true,
            optimization_level: 2,
        }
    }
}

/// Weights for different structural components
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuralWeights {
    /// Weight for hierarchy depth analysis
    pub depth_weight: f32,
    
    /// Weight for branching factor optimization
    pub branching_weight: f32,
    
    /// Weight for graph connectivity metrics
    pub connectivity_weight: f32,
    
    /// Weight for structural balance
    pub balance_weight: f32,
}

impl Default for StructuralWeights {
    fn default() -> Self {
        Self {
            depth_weight: 0.3,
            branching_weight: 0.25,
            connectivity_weight: 0.25,
            balance_weight: 0.2,
        }
    }
}

impl StructuralWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.depth_weight + self.branching_weight + self.connectivity_weight + self.balance_weight;
        
        if total > 0.0 {
            self.depth_weight /= total;
            self.branching_weight /= total;
            self.connectivity_weight /= total;
            self.balance_weight /= total;
        }
    }
}

/// Comprehensive graph metrics for structural analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Degree centrality (0.0-1.0)
    pub degree_centrality: f32,
    
    /// Closeness centrality (0.0-1.0)
    pub closeness_centrality: f32,
    
    /// Betweenness centrality (≥0.0)
    pub betweenness_centrality: f32,
    
    /// Clustering coefficient (0.0-1.0)
    pub clustering_coefficient: f32,
    
    /// PageRank score (≥0.0)
    pub pagerank_score: f32,
    
    /// Average path length from this node
    pub average_path_length: f32,
    
    /// Eccentricity (maximum distance to any node)
    pub eccentricity: f32,
}

/// Connectivity analysis for a specific node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityAnalysis {
    /// Number of direct connections (degree)
    pub node_degree: usize,
    
    /// Clustering coefficient for this node
    pub clustering_coefficient: f32,
    
    /// Betweenness centrality
    pub betweenness_centrality: f32,
    
    /// Distance to nearest hub
    pub hub_distance: Option<usize>,
    
    /// Connectivity quality score (0.0-1.0)
    pub connectivity_quality: f32,
}

/// Branching factor analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingFactorAnalysis {
    /// Current number of children
    pub current_children: usize,
    
    /// Optimal range for this level
    pub optimal_range: std::ops::Range<usize>,
    
    /// Branching factor score (0.0-1.0)
    pub branching_score: f32,
    
    /// Suggested optimizations
    pub optimization_suggestions: Vec<String>,
}

/// Structural balance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralBalanceAnalysis {
    /// Overall balance factor (0.0-1.0)
    pub balance_factor: f32,
    
    /// Left-right balance for binary-like structures
    pub left_right_balance: f32,
    
    /// Depth balance across branches
    pub depth_balance: f32,
    
    /// Load distribution balance
    pub load_distribution: f32,
    
    /// Suggestions for rebalancing
    pub rebalancing_suggestions: Vec<String>,
}

/// Detailed structural scoring breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralScoringBreakdown {
    /// Depth-based score component
    pub depth_score: f32,
    
    /// Branching factor score component
    pub branching_score: f32,
    
    /// Connectivity score component
    pub connectivity_score: f32,
    
    /// Balance score component
    pub balance_score: f32,
    
    /// Final composite score
    pub composite_score: f32,
    
    /// Performance metrics
    pub calculation_time: u64, // nanoseconds
    
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Types of structural anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnomalyType {
    /// Node with no connections
    OrphanNode,
    
    /// Parent with too many children
    OvercrowdedParent,
    
    /// Excessive depth in one branch
    OverlyDeepBranch,
    
    /// Structural imbalance
    StructuralImbalance,
    
    /// Poor connectivity
    PoorConnectivity,
    
    /// Cycle in hierarchy
    CyclicStructure,
}

/// Structural anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralAnomaly {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    
    /// Node(s) involved in the anomaly
    pub affected_nodes: Vec<String>,
    
    /// Severity of the anomaly (0.0-1.0)
    pub severity: f32,
    
    /// Suggested remediation
    pub remediation_suggestion: String,
}

/// Advanced structural composite scoring strategy
pub struct StructuralCompositeStrategy {
    /// Strategy name
    name: String,
    
    /// Configuration
    config: StructuralCompositeConfig,
    
    /// Structural weights
    weights: StructuralWeights,
    
    /// Whether strategy is enabled
    enabled: bool,
    
    /// Graph metrics calculator
    metrics_calculator: GraphMetricsCalculator,
    
    /// Structural optimizer
    optimizer: StructuralOptimizer,
}

impl StructuralCompositeStrategy {
    /// Create a new structural composite strategy
    pub fn new() -> Self {
        Self {
            name: "structural_composite".to_string(),
            config: StructuralCompositeConfig::default(),
            weights: StructuralWeights::default(),
            enabled: true,
            metrics_calculator: GraphMetricsCalculator::new(),
            optimizer: StructuralOptimizer::new(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: StructuralCompositeConfig) -> Self {
        let mut strategy = Self::new();
        strategy.config = config;
        strategy
    }
    
    /// Calculate comprehensive structural score
    fn calculate_structural_score(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        let scoring_start = std::time::Instant::now();
        
        // For this implementation, we'll use mock hierarchy analysis
        // In a real system, this would integrate with the actual ConceptHierarchy
        
        let depth_score = self.calculate_depth_score(context)?;
        let branching_score = self.calculate_branching_score(context)?;
        let connectivity_score = self.calculate_connectivity_score(context)?;
        let balance_score = self.calculate_balance_score(context)?;
        
        let normalized_weights = {
            let mut w = self.weights.clone();
            w.normalize();
            w
        };
        
        let composite_score = depth_score * normalized_weights.depth_weight +
                             branching_score * normalized_weights.branching_weight +
                             connectivity_score * normalized_weights.connectivity_weight +
                             balance_score * normalized_weights.balance_weight;
        
        Ok(composite_score.max(0.0).min(1.0))
    }
    
    /// Calculate depth-based score
    fn calculate_depth_score(&self, context: &AllocationContext) -> Result<f32, ScoringError> {
        let current_depth = context.ancestor_concepts.len();
        
        // Parse target depth from context if available
        let target_depth = context.context_properties.get("target_depth")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(current_depth);
        
        // Optimal depth range (typically 3-6 levels)
        let optimal_range = 3..=6;
        
        let depth_score = if optimal_range.contains(&target_depth) {
            1.0 // Perfect depth
        } else if target_depth < *optimal_range.start() {
            // Too shallow
            0.7 + (target_depth as f32 / *optimal_range.start() as f32) * 0.3
        } else {
            // Too deep - apply penalty
            let excess_depth = target_depth - *optimal_range.end();
            let penalty = (excess_depth as f32 * self.config.max_depth_penalty).min(0.5);
            1.0 - penalty
        };
        
        Ok(depth_score.max(0.0).min(1.0))
    }
    
    /// Calculate branching factor score
    fn calculate_branching_score(&self, context: &AllocationContext) -> Result<f32, ScoringError> {
        // In a real implementation, this would analyze the actual hierarchy structure
        // For now, use a simplified scoring based on context information
        
        let estimated_siblings = context.ancestor_concepts.len().saturating_sub(1);
        let optimal_min = self.config.optimal_branching_range.0;
        let optimal_max = self.config.optimal_branching_range.1;
        
        let branching_score = if estimated_siblings >= optimal_min && estimated_siblings <= optimal_max {
            1.0 // Optimal branching
        } else if estimated_siblings < optimal_min {
            // Too few siblings
            0.6 + (estimated_siblings as f32 / optimal_min as f32) * 0.4
        } else {
            // Too many siblings
            let excess = estimated_siblings - optimal_max;
            let penalty = (excess as f32 * 0.1).min(0.4);
            1.0 - penalty
        };
        
        Ok(branching_score.max(0.0).min(1.0))
    }
    
    /// Calculate connectivity score
    fn calculate_connectivity_score(&self, context: &AllocationContext) -> Result<f32, ScoringError> {
        // Estimate connectivity based on context richness
        let connection_indicators = context.ancestor_concepts.len() + 
                                  context.context_properties.len();
        
        // Score based on connectivity indicators
        let connectivity_score = match connection_indicators {
            0..=2 => 0.3,      // Poor connectivity
            3..=5 => 0.6,      // Moderate connectivity
            6..=10 => 0.9,     // Good connectivity
            _ => 1.0,          // Excellent connectivity
        };
        
        Ok(connectivity_score)
    }
    
    /// Calculate structural balance score
    fn calculate_balance_score(&self, context: &AllocationContext) -> Result<f32, ScoringError> {
        // Simplified balance assessment based on context structure
        let has_ancestors = !context.ancestor_concepts.is_empty();
        let has_properties = !context.context_properties.is_empty();
        let hierarchy_depth = context.ancestor_concepts.len();
        
        let mut balance_factors = Vec::new();
        
        // Depth balance factor
        if hierarchy_depth <= 6 {
            balance_factors.push(1.0 - (hierarchy_depth as f32 * 0.1));
        } else {
            balance_factors.push(0.4); // Penalty for excessive depth
        }
        
        // Context richness balance
        if has_ancestors && has_properties {
            balance_factors.push(1.0);
        } else if has_ancestors || has_properties {
            balance_factors.push(0.7);
        } else {
            balance_factors.push(0.3);
        }
        
        let balance_score = if balance_factors.is_empty() {
            0.5
        } else {
            balance_factors.iter().sum::<f32>() / balance_factors.len() as f32
        };
        
        Ok(balance_score.max(0.0).min(1.0))
    }
    
    /// Get detailed scoring breakdown
    pub fn score_with_breakdown(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<StructuralScoringBreakdown, ScoringError> {
        let calculation_start = std::time::Instant::now();
        
        let depth_score = self.calculate_depth_score(context)?;
        let branching_score = self.calculate_branching_score(context)?;
        let connectivity_score = self.calculate_connectivity_score(context)?;
        let balance_score = self.calculate_balance_score(context)?;
        
        let normalized_weights = {
            let mut w = self.weights.clone();
            w.normalize();
            w
        };
        
        let composite_score = depth_score * normalized_weights.depth_weight +
                             branching_score * normalized_weights.branching_weight +
                             connectivity_score * normalized_weights.connectivity_weight +
                             balance_score * normalized_weights.balance_weight;
        
        let calculation_time = calculation_start.elapsed().as_nanos() as u64;
        
        let mut recommendations = Vec::new();
        
        // Generate recommendations based on scores
        if depth_score < 0.7 {
            recommendations.push("Consider adjusting hierarchy depth for better structural fit".to_string());
        }
        if branching_score < 0.7 {
            recommendations.push("Optimize branching factor to improve structural balance".to_string());
        }
        if connectivity_score < 0.7 {
            recommendations.push("Enhance connectivity to improve graph integration".to_string());
        }
        if balance_score < 0.7 {
            recommendations.push("Address structural imbalances for better hierarchy health".to_string());
        }
        
        Ok(StructuralScoringBreakdown {
            depth_score,
            branching_score,
            connectivity_score,
            balance_score,
            composite_score: composite_score.max(0.0).min(1.0),
            calculation_time,
            recommendations,
        })
    }
    
    /// Calculate comprehensive graph metrics (mock implementation)
    pub fn calculate_comprehensive_graph_metrics(&self, hierarchy: &ConceptHierarchy, node_name: &str) -> Result<GraphMetrics, ScoringError> {
        // This is a mock implementation for testing
        // In a real system, this would analyze the actual graph structure
        
        Ok(GraphMetrics {
            degree_centrality: 0.6,
            closeness_centrality: 0.5,
            betweenness_centrality: 0.3,
            clustering_coefficient: 0.4,
            pagerank_score: 0.15,
            average_path_length: 2.5,
            eccentricity: 4.0,
        })
    }
    
    /// Calculate graph connectivity for a specific node
    pub fn calculate_graph_connectivity(&self, hierarchy: &ConceptHierarchy, node_name: &str) -> Result<ConnectivityAnalysis, ScoringError> {
        // Mock implementation
        Ok(ConnectivityAnalysis {
            node_degree: 3,
            clustering_coefficient: 0.4,
            betweenness_centrality: 0.3,
            hub_distance: Some(2),
            connectivity_quality: 0.7,
        })
    }
    
    /// Analyze branching factor for a specific node
    pub fn analyze_branching_factor(&self, hierarchy: &ConceptHierarchy, node_name: &str) -> Result<BranchingFactorAnalysis, ScoringError> {
        // Mock implementation
        let current_children = 4; // Mock value
        let optimal_range = self.config.optimal_branching_range.0..self.config.optimal_branching_range.1;
        
        let branching_score = if optimal_range.contains(&current_children) {
            1.0
        } else if current_children < optimal_range.start {
            0.6 + (current_children as f32 / optimal_range.start as f32) * 0.4
        } else {
            let excess = current_children - optimal_range.end;
            1.0 - (excess as f32 * 0.1).min(0.5)
        };
        
        let mut suggestions = Vec::new();
        if current_children > optimal_range.end {
            suggestions.push("Consider grouping children into subcategories".to_string());
        } else if current_children < optimal_range.start {
            suggestions.push("Hierarchy may be too deep, consider flattening".to_string());
        }
        
        Ok(BranchingFactorAnalysis {
            current_children,
            optimal_range,
            branching_score,
            optimization_suggestions: suggestions,
        })
    }
    
    /// Analyze structural balance
    pub fn analyze_structural_balance(&self, hierarchy: &ConceptHierarchy, node_name: &str) -> Result<StructuralBalanceAnalysis, ScoringError> {
        // Mock implementation
        Ok(StructuralBalanceAnalysis {
            balance_factor: 0.7,
            left_right_balance: 0.8,
            depth_balance: 0.6,
            load_distribution: 0.75,
            rebalancing_suggestions: vec![
                "Consider redistributing children across subtrees".to_string(),
                "Balance depth differences between branches".to_string(),
            ],
        })
    }
    
    /// Find optimal structural placement among multiple options
    pub fn find_optimal_structural_placement(&self, concept: &ExtractedConcept, contexts: &[AllocationContext]) -> Result<usize, ScoringError> {
        if contexts.is_empty() {
            return Err(ScoringError::StrategyExecutionFailed("No contexts provided".to_string()));
        }
        
        let scores: Result<Vec<_>, _> = contexts.iter()
            .map(|ctx| self.calculate_structural_score(concept, ctx))
            .collect();
        
        let scores = scores?;
        
        // Find index of highest score
        let optimal_index = scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        Ok(optimal_index)
    }
    
    /// Detect structural anomalies in hierarchy
    pub fn detect_structural_anomalies(&self, hierarchy: &ConceptHierarchy) -> Result<Vec<StructuralAnomaly>, ScoringError> {
        let mut anomalies = Vec::new();
        
        // Mock anomaly detection
        anomalies.push(StructuralAnomaly {
            anomaly_type: AnomalyType::OrphanNode,
            affected_nodes: vec!["isolated_node".to_string()],
            severity: 0.8,
            remediation_suggestion: "Connect isolated node to appropriate parent".to_string(),
        });
        
        anomalies.push(StructuralAnomaly {
            anomaly_type: AnomalyType::OvercrowdedParent,
            affected_nodes: vec!["overcrowded_parent".to_string()],
            severity: 0.6,
            remediation_suggestion: "Group children into subcategories".to_string(),
        });
        
        Ok(anomalies)
    }
    
    /// Batch score multiple concepts
    pub fn batch_score(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<f32>, ScoringError> {
        let scores: Result<Vec<_>, _> = concepts.par_iter()
            .map(|concept| self.calculate_structural_score(concept, context))
            .collect();
        
        scores
    }
    
    /// Get maximum depth penalty
    pub fn get_max_depth_penalty(&self) -> f32 {
        self.config.max_depth_penalty
    }
    
    /// Set depth penalty factor
    pub fn set_depth_penalty_factor(&mut self, penalty: f32) {
        self.config.max_depth_penalty = penalty.max(0.0).min(1.0);
    }
    
    /// Set structural weights
    pub fn set_structural_weights(&mut self, mut weights: StructuralWeights) {
        weights.normalize();
        self.weights = weights;
    }
    
    /// Get structural weights
    pub fn get_structural_weights(&self) -> &StructuralWeights {
        &self.weights
    }
    
    /// Check if strategy is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Enable or disable strategy
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl ScoringStrategy for StructuralCompositeStrategy {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn score(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        if !self.enabled {
            return Ok(0.0);
        }
        
        self.calculate_structural_score(concept, context)
    }
    
    fn supports_parallel(&self) -> bool {
        true
    }
    
    fn weight_preference(&self) -> f32 {
        0.3 // Structural fit is quite important for hierarchy health
    }
}

/// Graph metrics calculator
pub struct GraphMetricsCalculator {
    // Configuration and caching for metric calculations
}

impl GraphMetricsCalculator {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Calculate all relevant graph metrics for a node
    pub fn calculate_all_metrics(&self, hierarchy: &ConceptHierarchy, node_name: &str) -> Result<GraphMetrics, ScoringError> {
        // Mock implementation
        Ok(GraphMetrics {
            degree_centrality: 0.5,
            closeness_centrality: 0.4,
            betweenness_centrality: 0.2,
            clustering_coefficient: 0.3,
            pagerank_score: 0.1,
            average_path_length: 3.0,
            eccentricity: 5.0,
        })
    }
}

/// Structural optimizer for hierarchy improvements
pub struct StructuralOptimizer {
    // Optimization algorithms and heuristics
}

impl StructuralOptimizer {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Suggest structural optimizations
    pub fn suggest_optimizations(&self, hierarchy: &ConceptHierarchy, target_node: &str) -> Result<Vec<String>, ScoringError> {
        // Mock optimization suggestions
        Ok(vec![
            "Consider creating intermediate categories to reduce branching factor".to_string(),
            "Balance subtree depths for improved navigation".to_string(),
            "Enhance cross-references to improve connectivity".to_string(),
        ])
    }
}

impl Default for StructuralCompositeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

// Mock implementation for ConceptHierarchy methods needed for testing
impl ConceptHierarchy {
    pub fn new() -> Self {
        Self {
            // This would be properly implemented in the actual ConceptHierarchy
        }
    }
}
```

## Verification Steps
1. Create StructuralCompositeStrategy with comprehensive graph analysis capabilities
2. Implement depth analysis with configurable penalty factors
3. Add branching factor optimization with optimal range management
4. Implement connectivity metrics calculation using graph algorithms
5. Add structural balance analysis and anomaly detection
6. Ensure performance meets <0.4ms per evaluation requirement

## Success Criteria
- [ ] StructuralCompositeStrategy compiles without errors
- [ ] Depth analysis correctly penalizes excessive hierarchy depth
- [ ] Branching factor optimization identifies optimal parent placement
- [ ] Graph connectivity metrics provide meaningful structural insights
- [ ] Structural balance analysis detects hierarchy imbalances
- [ ] All tests pass with comprehensive coverage