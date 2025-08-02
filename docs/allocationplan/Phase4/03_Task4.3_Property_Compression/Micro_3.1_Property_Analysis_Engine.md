# Micro Phase 3.1: Property Analysis Engine

**Estimated Time**: 40 minutes
**Dependencies**: Task 4.2 Complete (Exception Handling)
**Objective**: Analyze property distribution across hierarchy to identify compression opportunities

## Task Description

Create a sophisticated analysis engine that examines property patterns across the inheritance hierarchy to identify which properties should be promoted to achieve maximum compression.

## Deliverables

Create `src/compression/analyzer.rs` with:

1. **PropertyAnalyzer struct**: Core analysis engine
2. **Frequency analysis**: Count property occurrences across nodes
3. **Ancestry analysis**: Find lowest common ancestors for property groups
4. **Compression scoring**: Calculate bytes saved for each promotion candidate
5. **Conflict detection**: Identify potential promotion conflicts

## Success Criteria

- [ ] Analyzes 10,000 nodes in < 100ms
- [ ] Identifies all properties with >50% frequency in subtrees
- [ ] Calculates accurate LCA for arbitrary node sets
- [ ] Compression scoring predicts actual savings within 5%
- [ ] Detects all potential conflicts before promotion
- [ ] Memory usage O(n) where n = number of unique properties

## Implementation Requirements

```rust
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

#[derive(Debug, Clone)]
struct ScoringWeights {
    frequency_weight: f32,
    depth_weight: f32,
    size_weight: f32,
    conflict_penalty: f32,
}
```

## Test Requirements

Must pass property analysis tests:
```rust
#[test]
fn test_frequency_analysis() {
    let hierarchy = create_animal_hierarchy();
    let analyzer = PropertyAnalyzer::new(0.7, 3);
    
    // All mammals should have "warm_blooded" = true
    let analysis = analyzer.analyze_hierarchy(&hierarchy);
    
    let warm_blooded_candidate = analysis.promotion_candidates
        .iter()
        .find(|c| c.property_name == "warm_blooded" && c.property_value == PropertyValue::Boolean(true))
        .expect("Should find warm_blooded promotion candidate");
    
    assert!(warm_blooded_candidate.frequency_in_subtree >= 0.9);
    assert!(warm_blooded_candidate.estimated_bytes_saved > 100);
}

#[test]
fn test_lca_calculation() {
    let hierarchy = create_test_hierarchy();
    let analyzer = PropertyAnalyzer::new(0.5, 2);
    
    // Find LCA of all dog breeds
    let dog_nodes = vec![
        hierarchy.get_node_by_name("Golden Retriever").unwrap(),
        hierarchy.get_node_by_name("Labrador").unwrap(),
        hierarchy.get_node_by_name("Beagle").unwrap(),
    ];
    
    let lca = analyzer.find_lowest_common_ancestor(&hierarchy, &dog_nodes);
    assert_eq!(lca, hierarchy.get_node_by_name("Dog"));
}

#[test]
fn test_compression_scoring() {
    let analyzer = PropertyAnalyzer::new(0.6, 3);
    
    let candidate = PromotionCandidate {
        property_name: "loyal".to_string(),
        property_value: PropertyValue::Boolean(true),
        source_nodes: (0..100).map(NodeId).collect(),
        target_ancestor: NodeId(1000),
        frequency_in_subtree: 0.95,
        estimated_bytes_saved: 2000,
        confidence_score: 0.9,
        conflicts: vec![],
    };
    
    let score = analyzer.calculate_compression_score(&candidate);
    assert!(score > 0.8); // High score for good candidate
}

#[test]
fn test_analysis_performance() {
    let hierarchy = create_large_hierarchy(10000);
    let analyzer = PropertyAnalyzer::new(0.7, 5);
    
    let start = Instant::now();
    let analysis = analyzer.analyze_hierarchy(&hierarchy);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(100)); // < 100ms for 10k nodes
    assert!(analysis.total_properties_analyzed > 10000);
    assert!(!analysis.promotion_candidates.is_empty());
}

#[test]
fn test_bytes_saved_estimation() {
    let analyzer = PropertyAnalyzer::new(0.6, 3);
    
    let property_name = "species";
    let value = PropertyValue::String("Canis lupus".to_string());
    let node_count = 50;
    
    let estimated = analyzer.estimate_bytes_saved(property_name, &value, node_count);
    
    // Should account for property name + value storage across nodes
    let expected_min = (property_name.len() + value.to_string().len()) * (node_count - 1);
    assert!(estimated >= expected_min);
}
```

## File Location
`src/compression/analyzer.rs`

## Next Micro Phase
After completion, proceed to Micro 3.2: Property Promotion Engine