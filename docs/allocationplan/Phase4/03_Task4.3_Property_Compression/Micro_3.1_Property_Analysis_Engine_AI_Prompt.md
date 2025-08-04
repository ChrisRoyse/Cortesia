# AI Prompt: Micro Phase 3.1 - Property Analysis Engine

You are tasked with creating a sophisticated analysis engine that examines property patterns across the inheritance hierarchy to identify compression opportunities. Your goal is to create `src/compression/analyzer.rs` with intelligent property analysis.

## Your Task
Implement the `PropertyAnalyzer` struct that analyzes property distribution, finds common ancestors, calculates compression scores, and detects conflicts to enable optimal property promotion.

## Specific Requirements
1. Create `src/compression/analyzer.rs` with PropertyAnalyzer struct
2. Implement frequency analysis to count property occurrences across nodes
3. Add ancestry analysis to find lowest common ancestors
4. Calculate compression scoring to predict bytes saved
5. Implement conflict detection for potential promotion conflicts
6. Optimize for performance (10,000 nodes in < 100ms)

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub struct PropertyAnalyzer {
    min_frequency_threshold: f32,
    min_nodes_threshold: usize,
    max_depth_difference: u32,
    scoring_weights: ScoringWeights,
}

#[derive(Debug, Clone)]
pub struct PromotionCandidate {
    pub property_name: String,
    pub property_value: PropertyValue,
    pub source_nodes: Vec<NodeId>,
    pub target_ancestor: NodeId,
    pub frequency_in_subtree: f32,
    pub estimated_bytes_saved: usize,
    pub confidence_score: f32,
    pub conflicts: Vec<ConflictInfo>,
}

#[derive(Debug)]
pub struct PropertyAnalysis {
    pub total_properties_analyzed: usize,
    pub unique_property_names: usize,
    pub promotion_candidates: Vec<PromotionCandidate>,
    pub estimated_total_savings: usize,
    pub analysis_duration: Duration,
}

impl PropertyAnalyzer {
    pub fn new(min_frequency: f32, min_nodes: usize) -> Self;
    
    pub fn analyze_hierarchy(&self, hierarchy: &InheritanceHierarchy) -> PropertyAnalysis;
    
    pub fn find_promotion_candidates(&self, hierarchy: &InheritanceHierarchy) -> Vec<PromotionCandidate>;
    
    pub fn calculate_compression_score(&self, candidate: &PromotionCandidate) -> f32;
    
    pub fn find_lowest_common_ancestor(&self, hierarchy: &InheritanceHierarchy, nodes: &[NodeId]) -> Option<NodeId>;
    
    pub fn estimate_bytes_saved(&self, property_name: &str, value: &PropertyValue, node_count: usize) -> usize;
}
```

## Success Criteria
- [ ] Analyzes 10,000 nodes in < 100ms
- [ ] Identifies all properties with >50% frequency in subtrees
- [ ] Calculates accurate LCA for arbitrary node sets
- [ ] Compression scoring predicts actual savings within 5%
- [ ] Detects all potential conflicts before promotion

## Test Requirements
```rust
#[test]
fn test_frequency_analysis() {
    let hierarchy = create_animal_hierarchy();
    let analyzer = PropertyAnalyzer::new(0.7, 3);
    
    let analysis = analyzer.analyze_hierarchy(&hierarchy);
    
    let warm_blooded_candidate = analysis.promotion_candidates
        .iter()
        .find(|c| c.property_name == "warm_blooded")
        .expect("Should find warm_blooded promotion candidate");
    
    assert!(warm_blooded_candidate.frequency_in_subtree >= 0.9);
    assert!(warm_blooded_candidate.estimated_bytes_saved > 100);
}

#[test]
fn test_lca_calculation() {
    let hierarchy = create_test_hierarchy();
    let analyzer = PropertyAnalyzer::new(0.5, 2);
    
    let dog_nodes = vec![
        hierarchy.get_node_by_name("Golden Retriever").unwrap(),
        hierarchy.get_node_by_name("Labrador").unwrap(),
        hierarchy.get_node_by_name("Beagle").unwrap(),
    ];
    
    let lca = analyzer.find_lowest_common_ancestor(&hierarchy, &dog_nodes);
    assert_eq!(lca, hierarchy.get_node_by_name("Dog"));
}

#[test]
fn test_analysis_performance() {
    let hierarchy = create_large_hierarchy(10000);
    let analyzer = PropertyAnalyzer::new(0.7, 5);
    
    let start = Instant::now();
    let analysis = analyzer.analyze_hierarchy(&hierarchy);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(100));
    assert!(analysis.total_properties_analyzed > 10000);
}
```

## File to Create
Create exactly this file: `src/compression/analyzer.rs`

## When Complete
Respond with "MICRO PHASE 3.1 COMPLETE" and a brief summary of your implementation.