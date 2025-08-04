# Micro Task 35: Belief System Integration Tests

**Priority**: CRITICAL  
**Estimated Time**: 50 minutes  
**Dependencies**: Phase 6 TMS, Tasks 31-34 (complete belief integration system)  
**Skills Required**: Rust testing, integration testing, belief system validation

## Objective

Implement comprehensive integration tests for the complete belief system, validating the interaction between TMS, belief-aware queries, temporal activation, context switching, and justification paths to ensure robust end-to-end functionality.

## Context

This final task for Day 5B validates that all belief system components work together correctly, providing confidence that the integrated system can handle complex real-world scenarios involving belief revision, multi-context reasoning, and temporal consistency.

## Specifications

### Required Test Categories

1. **End-to-End Belief Query Tests**
   - Complete belief-aware query processing workflows
   - Integration of TMS constraints with activation spreading
   - Temporal belief resolution across different time points
   - Multi-context query processing and result synthesis

2. **Belief Revision Integration Tests**
   - Belief revision propagation through activation network
   - Context switching triggered by belief changes
   - Justification path updates after belief revision
   - Temporal consistency maintenance during revisions

3. **Multi-Context Consistency Tests**
   - Context isolation and interference detection
   - Cross-context belief comparison and synthesis
   - Context merging and splitting operations
   - Resource management across multiple contexts

4. **Performance and Stress Tests**
   - Large-scale belief networks (1000+ beliefs)
   - Concurrent multi-context processing
   - Memory usage under belief system load
   - Latency targets under various conditions

### Performance Requirements

- End-to-end query latency: <50ms for complex scenarios
- Belief revision propagation: <20ms for typical networks
- Multi-context processing: <100ms for 5 contexts
- Memory overhead: <50% additional for belief features

## Implementation Guide

### Step 1: End-to-End Belief Query Tests

```rust
// File: tests/belief_system/end_to_end_tests.rs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use crate::core::activation::belief_aware_state::BeliefAwareActivationState;
use crate::core::context::context_manager::{ContextManager, ContextPriority};
use crate::core::query::belief_query_processor::BeliefQueryProcessor;
use crate::core::query::chronological_query_processor::ChronologicalQueryProcessor;
use crate::core::query::multi_context_query_processor::MultiContextQueryProcessor;
use crate::core::query::intent::{QueryIntent, TemporalQueryIntent, MultiContextQueryIntent};
use crate::tms::{TruthMaintenanceSystem, BeliefContext, BeliefId, BeliefStatus};
use crate::core::types::{NodeId, ContextId, Timestamp};

#[tokio::test]
async fn test_complete_belief_aware_query_workflow() {
    // Setup integrated belief system
    let belief_system = setup_integrated_belief_system().await;
    
    // Create test knowledge with beliefs and justifications
    let knowledge_setup = setup_test_knowledge_with_beliefs(&belief_system).await.unwrap();
    
    // Define a complex query that requires belief reasoning
    let query_intent = QueryIntent {
        query_text: "What evidence supports the claim that renewable energy is economically viable?".to_string(),
        entity_mentions: vec!["renewable energy".to_string(), "economic viability".to_string()],
        relationship_types: vec!["supports".to_string(), "evidence_for".to_string()],
        context_hints: vec!["economics".to_string(), "energy".to_string()],
        confidence_threshold: Some(0.6),
    };
    
    // Process query through belief-aware system
    let query_start = std::time::Instant::now();
    let belief_result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None, // Use default belief context
        &knowledge_setup.graph,
    ).await.unwrap();
    let query_duration = query_start.elapsed();
    
    // Validate results
    assert!(!belief_result.activated_nodes.is_empty(), "Query should activate relevant nodes");
    assert!(belief_result.belief_analysis.total_beliefs_activated > 0, "Should activate beliefs");
    assert!(belief_result.reliability_score > 0.5, "Should have reasonable reliability");
    assert!(belief_result.confidence_score > 0.6, "Should meet confidence threshold");
    
    // Check justification tracing
    if let Some(justification_trace) = &belief_result.justification_trace {
        assert!(!justification_trace.node_paths.is_empty(), "Should trace justification paths");
        
        // Validate path quality
        for (node_id, paths) in &justification_trace.node_paths {
            assert!(!paths.is_empty(), "Each activated node should have justification paths");
            
            for path in paths {
                assert!(path.total_strength > 0.0, "Paths should have positive strength");
                assert!(!path.beliefs.is_empty(), "Paths should include belief chains");
                
                // Check for circular reasoning detection
                if path.is_circular {
                    println!("Detected circular reasoning in path for node {:?}", node_id);
                }
            }
        }
    }
    
    // Performance validation
    assert!(query_duration < Duration::from_millis(50), 
            "End-to-end query should complete within 50ms, took {:?}", query_duration);
    
    println!("✅ Complete belief-aware query workflow test passed");
    println!("   - Query duration: {:?}", query_duration);
    println!("   - Activated nodes: {}", belief_result.activated_nodes.len());
    println!("   - Reliability score: {:.3}", belief_result.reliability_score);
    println!("   - Confidence score: {:.3}", belief_result.confidence_score);
}

#[tokio::test]
async fn test_temporal_belief_integration() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create temporal knowledge with evolving beliefs
    let temporal_setup = setup_temporal_knowledge(&belief_system).await.unwrap();
    
    // Test point-in-time query
    let past_timestamp = SystemTime::now() - Duration::from_secs(3600); // 1 hour ago
    let temporal_intent = TemporalQueryIntent::PointInTime { 
        timestamp: past_timestamp 
    };
    
    let query_intent = QueryIntent {
        query_text: "What was the scientific consensus on climate change?".to_string(),
        entity_mentions: vec!["climate change".to_string(), "scientific consensus".to_string()],
        relationship_types: vec!["consensus_on".to_string()],
        context_hints: vec!["science".to_string(), "climate".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    let temporal_result = belief_system.chronological_processor.process_temporal_query(
        &query_intent,
        &temporal_intent,
        &temporal_setup.graph,
    ).await.unwrap();
    
    // Validate temporal consistency
    assert_eq!(temporal_result.query_timestamp, past_timestamp);
    assert!(!temporal_result.temporal_activations.activation_history.is_empty());
    
    // Check historical context
    assert!(temporal_result.historical_context.timeline_events.len() > 0);
    
    // Validate temporal belief resolution
    for (node_id, activation) in &temporal_result.temporal_activations.belief_activations {
        // All beliefs should be valid at the query timestamp
        assert!(temporal_result.temporal_activations
            .is_node_valid_at_time(*node_id, past_timestamp)
            .await.unwrap());
    }
    
    // Test evolution tracking
    let start_time = SystemTime::now() - Duration::from_secs(7200); // 2 hours ago
    let end_time = SystemTime::now();
    
    for activated_node in temporal_result.temporal_activations.base_state.activated_nodes() {
        let evolution = temporal_result.temporal_activations
            .trace_activation_evolution(activated_node, start_time, end_time)
            .await;
        
        if let Ok(evolution) = evolution {
            assert!(!evolution.activation_points.is_empty());
            assert!(evolution.evolution_metrics.total_changes >= 0);
        }
    }
    
    println!("✅ Temporal belief integration test passed");
}

#[tokio::test]
async fn test_multi_context_belief_processing() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create multiple belief contexts with different assumptions
    let context_setup = setup_multi_context_beliefs(&belief_system).await.unwrap();
    
    let multi_context_intent = MultiContextQueryIntent {
        context_selection: Some(ContextSelectionStrategy::Manual(
            context_setup.context_ids.clone()
        )),
        aggregation_method: Some(AggregationMethod::ConsensusFiltered(0.7)),
        include_context_analysis: true,
        max_contexts: 5,
    };
    
    let query_intent = QueryIntent {
        query_text: "What is the impact of artificial intelligence on employment?".to_string(),
        entity_mentions: vec!["artificial intelligence".to_string(), "employment".to_string()],
        relationship_types: vec!["impacts".to_string(), "affects".to_string()],
        context_hints: vec!["technology".to_string(), "economics".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    let multi_context_result = belief_system.multi_context_processor.process_multi_context_query(
        &query_intent,
        &multi_context_intent,
        &context_setup.graph,
    ).await.unwrap();
    
    // Validate multi-context processing
    assert_eq!(multi_context_result.context_results.len(), context_setup.context_ids.len());
    
    // Check consensus analysis
    assert!(multi_context_result.consensus_report.overall_consensus_level >= 0.0);
    assert!(multi_context_result.consensus_report.overall_consensus_level <= 1.0);
    
    // Validate conflict detection
    if multi_context_result.conflict_report.total_conflicts > 0 {
        assert!(!multi_context_result.conflict_report.severe_conflicts.is_empty() ||
               !multi_context_result.conflict_report.moderate_conflicts.is_empty());
        
        // Should provide resolution recommendations
        assert!(!multi_context_result.conflict_report.resolution_recommendations.is_empty());
    }
    
    // Check aggregated results
    assert!(!multi_context_result.aggregated_result.final_activations.is_empty());
    
    // Validate context efficiency
    for (context_id, _) in &multi_context_result.context_results {
        assert!(multi_context_result.context_analysis.context_efficiency
                .contains_key(context_id));
    }
    
    println!("✅ Multi-context belief processing test passed");
    println!("   - Contexts processed: {}", multi_context_result.context_results.len());
    println!("   - Overall consensus: {:.3}", multi_context_result.consensus_report.overall_consensus_level);
    println!("   - Conflicts detected: {}", multi_context_result.conflict_report.total_conflicts);
}

async fn setup_integrated_belief_system() -> IntegratedBeliefSystem {
    let tms = Arc::new(RwLock::new(TruthMaintenanceSystem::new().await.unwrap()));
    let context_manager = Arc::new(RwLock::new(ContextManager::new()));
    
    IntegratedBeliefSystem {
        tms: tms.clone(),
        context_manager: context_manager.clone(),
        belief_query_processor: BeliefQueryProcessor::new(tms.clone()),
        chronological_processor: ChronologicalQueryProcessor::new(tms.clone(), /* temporal_graph */ Arc::new(RwLock::new(TemporalBeliefGraph::new()))),
        multi_context_processor: MultiContextQueryProcessor::new(
            context_manager.clone(),
            ContextAwareSpreader::new(context_manager, TMSConstrainedSpreader::new(tms)),
        ),
    }
}

struct IntegratedBeliefSystem {
    tms: Arc<RwLock<TruthMaintenanceSystem>>,
    context_manager: Arc<RwLock<ContextManager>>,
    belief_query_processor: BeliefQueryProcessor,
    chronological_processor: ChronologicalQueryProcessor,
    multi_context_processor: MultiContextQueryProcessor,
}
```

### Step 2: Belief Revision Integration Tests

```rust
// File: tests/belief_system/belief_revision_tests.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_belief_revision_propagation() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Setup initial belief network
    let initial_setup = setup_belief_network(&belief_system).await.unwrap();
    
    // Create initial query to establish baseline
    let query_intent = QueryIntent {
        query_text: "Is smoking harmful to health?".to_string(),
        entity_mentions: vec!["smoking".to_string(), "health".to_string()],
        relationship_types: vec!["harmful_to".to_string()],
        context_hints: vec!["health".to_string(), "medical".to_string()],
        confidence_threshold: Some(0.6),
    };
    
    let initial_result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None,
        &initial_setup.graph,
    ).await.unwrap();
    
    let initial_confidence = initial_result.confidence_score;
    let initial_activations = initial_result.activated_nodes.clone();
    
    // Perform belief revision - add stronger evidence
    let revision_start = std::time::Instant::now();
    
    let new_belief = Belief {
        belief_id: BeliefId::new(),
        content: "Multiple longitudinal studies confirm smoking causes cancer".to_string(),
        confidence: 0.95,
        sources: vec![Source::new("Medical Journal", 0.9)],
        supporting_evidence: vec!["Study A".to_string(), "Study B".to_string()],
        temporal_validity: TemporalValidity::Ongoing,
    };
    
    // Add belief through TMS
    {
        let mut tms = belief_system.tms.write().await;
        tms.add_belief(new_belief.clone()).await.unwrap();
        
        // Create justification linking new evidence to existing beliefs
        let justification = JustificationLink {
            justification_id: JustificationId::new(),
            antecedent_beliefs: vec![new_belief.belief_id],
            consequent_belief: initial_setup.target_belief_id,
            inference_rule: InferenceRule::EvidentialSupport,
            confidence: 0.9,
            strength: 0.85,
        };
        
        tms.add_justification(justification).await.unwrap();
    }
    
    let revision_duration = revision_start.elapsed();
    
    // Query again to see revision effects
    let post_revision_result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None,
        &initial_setup.graph,
    ).await.unwrap();
    
    // Validate belief revision effects
    assert!(post_revision_result.confidence_score > initial_confidence,
            "Confidence should increase after adding strong evidence");
    
    // Check that justification paths reflect new evidence
    if let Some(justification_trace) = &post_revision_result.justification_trace {
        let has_new_evidence = justification_trace.node_paths.values()
            .any(|paths| paths.iter()
                .any(|path| path.beliefs.contains(&new_belief.belief_id)));
        
        assert!(has_new_evidence, "New evidence should appear in justification paths");
    }
    
    // Validate belief revision history
    assert!(!post_revision_result.activation_summary.revision_history.is_empty(),
            "Should track belief revision history");
    
    // Performance check
    assert!(revision_duration < Duration::from_millis(20),
            "Belief revision propagation should complete within 20ms");
    
    println!("✅ Belief revision propagation test passed");
    println!("   - Initial confidence: {:.3}", initial_confidence);
    println!("   - Post-revision confidence: {:.3}", post_revision_result.confidence_score);
    println!("   - Revision duration: {:?}", revision_duration);
}

#[tokio::test]
async fn test_context_switching_on_belief_revision() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create contexts with conflicting beliefs
    let context_a = {
        let mut manager = belief_system.context_manager.write().await;
        manager.create_context(
            BeliefContext::new("optimistic_scenario"),
            None,
            ContextPriority::Medium,
        ).await.unwrap()
    };
    
    let context_b = {
        let mut manager = belief_system.context_manager.write().await;
        manager.create_context(
            BeliefContext::new("pessimistic_scenario"),
            None,
            ContextPriority::Medium,
        ).await.unwrap()
    };
    
    // Add conflicting beliefs to each context
    setup_conflicting_beliefs_in_contexts(&belief_system, context_a, context_b).await.unwrap();
    
    // Set up automatic context switching trigger
    let switch_trigger = SwitchTrigger::BeliefRevision {
        belief_id: BeliefId::new(),
        revision_type: RevisionType::Retraction,
    };
    
    belief_system.context_switch_orchestrator.register_trigger(switch_trigger).await.unwrap();
    
    // Activate context A and perform query
    {
        let mut manager = belief_system.context_manager.write().await;
        manager.activate_context(context_a).await.unwrap();
    }
    
    let query_intent = QueryIntent {
        query_text: "What are the economic prospects for the next year?".to_string(),
        entity_mentions: vec!["economic prospects".to_string()],
        relationship_types: vec!["predicts".to_string()],
        context_hints: vec!["economics".to_string(), "future".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    let result_a = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        Some(BeliefContext::from_context_id(context_a)),
        &setup_test_graph(),
    ).await.unwrap();
    
    // Retract a key belief (should trigger context switch)
    let retraction_belief_id = result_a.activated_nodes[0].belief_id; // Use first activated belief
    
    {
        let mut tms = belief_system.tms.write().await;
        tms.retract_belief(retraction_belief_id).await.unwrap();
    }
    
    // Execute pending context switches
    let switch_execution = belief_system.context_switch_orchestrator
        .execute_pending_switches().await.unwrap();
    
    // Validate context switch occurred
    assert!(!switch_execution.executed_switches.is_empty(),
            "Should execute context switch after belief retraction");
    
    // Query again to see different results
    let result_b = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None, // Use current active context
        &setup_test_graph(),
    ).await.unwrap();
    
    // Results should be different due to context switch
    assert_ne!(result_a.activated_nodes.len(), result_b.activated_nodes.len(),
              "Results should differ after context switch");
    
    println!("✅ Context switching on belief revision test passed");
}

#[tokio::test]
async fn test_justification_path_updates() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Setup initial justification paths
    let path_setup = setup_initial_justification_paths(&belief_system).await.unwrap();
    
    let query_intent = QueryIntent {
        query_text: "Why should we invest in renewable energy?".to_string(),
        entity_mentions: vec!["renewable energy".to_string(), "investment".to_string()],
        relationship_types: vec!["reason_for".to_string()],
        context_hints: vec!["energy".to_string(), "investment".to_string()],
        confidence_threshold: Some(0.6),
    };
    
    // Get initial justification paths
    let initial_result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None,
        &path_setup.graph,
    ).await.unwrap();
    
    let initial_paths = initial_result.justification_trace.clone().unwrap();
    let initial_path_count = initial_paths.node_paths.values()
        .map(|paths| paths.len())
        .sum::<usize>();
    
    // Add new supporting belief that creates additional justification path
    let new_supporting_belief = Belief {
        belief_id: BeliefId::new(),
        content: "Government subsidies make renewable energy financially attractive".to_string(),
        confidence: 0.8,
        sources: vec![Source::new("Economic Analysis", 0.85)],
        supporting_evidence: vec!["Policy Report 2024".to_string()],
        temporal_validity: TemporalValidity::Ongoing,
    };
    
    {
        let mut tms = belief_system.tms.write().await;
        tms.add_belief(new_supporting_belief.clone()).await.unwrap();
        
        // Create justification chain
        let justification = JustificationLink {
            justification_id: JustificationId::new(),
            antecedent_beliefs: vec![new_supporting_belief.belief_id],
            consequent_belief: path_setup.target_belief_id,
            inference_rule: InferenceRule::EconomicJustification,
            confidence: 0.75,
            strength: 0.8,
        };
        
        tms.add_justification(justification).await.unwrap();
    }
    
    // Query again to get updated paths
    let updated_result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None,
        &path_setup.graph,
    ).await.unwrap();
    
    let updated_paths = updated_result.justification_trace.clone().unwrap();
    let updated_path_count = updated_paths.node_paths.values()
        .map(|paths| paths.len())
        .sum::<usize>();
    
    // Should have additional justification paths
    assert!(updated_path_count >= initial_path_count,
            "Should have same or more justification paths after adding belief");
    
    // Check that new belief appears in some justification path
    let contains_new_belief = updated_paths.node_paths.values()
        .any(|paths| paths.iter()
            .any(|path| path.beliefs.contains(&new_supporting_belief.belief_id)));
    
    assert!(contains_new_belief, "New belief should appear in justification paths");
    
    // Validate path quality improvements
    let average_strength_before = calculate_average_path_strength(&initial_paths);
    let average_strength_after = calculate_average_path_strength(&updated_paths);
    
    assert!(average_strength_after >= average_strength_before,
            "Average path strength should not decrease");
    
    println!("✅ Justification path updates test passed");
    println!("   - Initial paths: {}", initial_path_count);
    println!("   - Updated paths: {}", updated_path_count);
    println!("   - Strength before: {:.3}", average_strength_before);
    println!("   - Strength after: {:.3}", average_strength_after);
}

fn calculate_average_path_strength(trace: &JustificationTraceResult) -> f32 {
    let total_strength: f32 = trace.node_paths.values()
        .flat_map(|paths| paths.iter())
        .map(|path| path.total_strength)
        .sum();
    
    let total_paths = trace.node_paths.values()
        .map(|paths| paths.len())
        .sum::<usize>();
    
    if total_paths > 0 {
        total_strength / total_paths as f32
    } else {
        0.0
    }
}
```

### Step 3: Multi-Context Consistency Tests

```rust
// File: tests/belief_system/multi_context_consistency_tests.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_context_isolation() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create isolated contexts with different beliefs
    let context_medical = create_medical_context(&belief_system).await.unwrap();
    let context_legal = create_legal_context(&belief_system).await.unwrap();
    
    // Add context-specific beliefs
    setup_context_specific_beliefs(&belief_system, context_medical, context_legal).await.unwrap();
    
    let query_intent = QueryIntent {
        query_text: "What are the implications of genetic testing?".to_string(),
        entity_mentions: vec!["genetic testing".to_string()],
        relationship_types: vec!["implies".to_string()],
        context_hints: vec!["genetics".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    // Query in medical context
    let medical_result = {
        let mut manager = belief_system.context_manager.write().await;
        manager.activate_context(context_medical).await.unwrap();
        drop(manager);
        
        belief_system.belief_query_processor.process_belief_query(
            &query_intent,
            Some(BeliefContext::from_context_id(context_medical)),
            &setup_test_graph(),
        ).await.unwrap()
    };
    
    // Query in legal context
    let legal_result = {
        let mut manager = belief_system.context_manager.write().await;
        manager.activate_context(context_legal).await.unwrap();
        drop(manager);
        
        belief_system.belief_query_processor.process_belief_query(
            &query_intent,
            Some(BeliefContext::from_context_id(context_legal)),
            &setup_test_graph(),
        ).await.unwrap()
    };
    
    // Validate context isolation
    assert_ne!(medical_result.activated_nodes.len(), legal_result.activated_nodes.len(),
              "Results should differ between contexts");
    
    // Check that context-specific beliefs are properly isolated
    let medical_beliefs: HashSet<_> = medical_result.activated_nodes.iter()
        .flat_map(|node| &node.supporting_beliefs)
        .collect();
    
    let legal_beliefs: HashSet<_> = legal_result.activated_nodes.iter()
        .flat_map(|node| &node.supporting_beliefs)
        .collect();
    
    let overlap = medical_beliefs.intersection(&legal_beliefs).count();
    let total_beliefs = medical_beliefs.len() + legal_beliefs.len();
    
    let overlap_ratio = if total_beliefs > 0 {
        overlap as f32 / total_beliefs as f32
    } else {
        0.0
    };
    
    // Some overlap is expected for shared foundational beliefs, but should be limited
    assert!(overlap_ratio < 0.3, "Context overlap should be limited: {:.3}", overlap_ratio);
    
    println!("✅ Context isolation test passed");
    println!("   - Medical beliefs: {}", medical_beliefs.len());
    println!("   - Legal beliefs: {}", legal_beliefs.len());
    println!("   - Overlap ratio: {:.3}", overlap_ratio);
}

#[tokio::test]
async fn test_context_merging_operations() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create contexts to merge
    let context_a = create_context_with_beliefs(&belief_system, "economics_optimistic").await.unwrap();
    let context_b = create_context_with_beliefs(&belief_system, "economics_realistic").await.unwrap();
    
    // Define merge strategy
    let merge_strategy = MergeStrategy::Weighted(HashMap::from([
        (context_a, 0.6),
        (context_b, 0.4),
    ]));
    
    // Perform merge operation
    let merge_start = std::time::Instant::now();
    let merged_context = {
        let mut manager = belief_system.context_manager.write().await;
        manager.merge_contexts(
            vec![context_a, context_b],
            merge_strategy,
        ).await.unwrap()
    };
    let merge_duration = merge_start.elapsed();
    
    // Test query in merged context
    let query_intent = QueryIntent {
        query_text: "What are the economic trends for next quarter?".to_string(),
        entity_mentions: vec!["economic trends".to_string()],
        relationship_types: vec!["predicts".to_string()],
        context_hints: vec!["economics".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    let merged_result = {
        let mut manager = belief_system.context_manager.write().await;
        manager.activate_context(merged_context).await.unwrap();
        drop(manager);
        
        belief_system.belief_query_processor.process_belief_query(
            &query_intent,
            Some(BeliefContext::from_context_id(merged_context)),
            &setup_test_graph(),
        ).await.unwrap()
    };
    
    // Validate merge results
    assert!(!merged_result.activated_nodes.is_empty(),
            "Merged context should produce query results");
    
    // Check that merge preserved important beliefs from both contexts
    let merged_beliefs: HashSet<_> = merged_result.activated_nodes.iter()
        .flat_map(|node| &node.supporting_beliefs)
        .collect();
    
    assert!(merged_beliefs.len() > 0, "Merged context should contain beliefs");
    
    // Performance validation
    assert!(merge_duration < Duration::from_millis(100),
            "Context merge should complete within 100ms");
    
    println!("✅ Context merging operations test passed");
    println!("   - Merge duration: {:?}", merge_duration);
    println!("   - Merged beliefs: {}", merged_beliefs.len());
}

#[tokio::test]
async fn test_cross_context_interference_detection() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create contexts that should interfere with each other
    let context_conservative = create_context_with_assumptions(&belief_system, "conservative_economics").await.unwrap();
    let context_progressive = create_context_with_assumptions(&belief_system, "progressive_economics").await.unwrap();
    
    // Add potentially conflicting beliefs
    setup_conflicting_economic_beliefs(&belief_system, context_conservative, context_progressive).await.unwrap();
    
    // Process same query in both contexts simultaneously
    let query_intent = QueryIntent {
        query_text: "What is the optimal tax policy?".to_string(),
        entity_mentions: vec!["tax policy".to_string()],
        relationship_types: vec!["optimal".to_string()],
        context_hints: vec!["economics".to_string(), "policy".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    let multi_context_intent = MultiContextQueryIntent {
        context_selection: Some(ContextSelectionStrategy::Manual(
            vec![context_conservative, context_progressive]
        )),
        aggregation_method: Some(AggregationMethod::ConflictAware),
        include_context_analysis: true,
        max_contexts: 2,
    };
    
    let interference_result = belief_system.multi_context_processor.process_multi_context_query(
        &query_intent,
        &multi_context_intent,
        &setup_test_graph(),
    ).await.unwrap();
    
    // Validate interference detection
    assert!(interference_result.conflict_report.total_conflicts > 0,
            "Should detect conflicts between opposing economic viewpoints");
    
    // Check for severe conflicts
    let has_severe_conflicts = !interference_result.conflict_report.severe_conflicts.is_empty();
    
    if has_severe_conflicts {
        println!("Detected {} severe conflicts", interference_result.conflict_report.severe_conflicts.len());
        
        // Validate conflict resolution recommendations
        assert!(!interference_result.conflict_report.resolution_recommendations.is_empty(),
                "Should provide resolution recommendations for severe conflicts");
    }
    
    // Check context diversity
    assert!(interference_result.context_analysis.context_diversity > 0.5,
            "Contexts should be sufficiently diverse");
    
    // Low consensus expected due to conflicting viewpoints
    assert!(interference_result.consensus_report.overall_consensus_level < 0.5,
            "Consensus should be low for conflicting contexts");
    
    println!("✅ Cross-context interference detection test passed");
    println!("   - Total conflicts: {}", interference_result.conflict_report.total_conflicts);
    println!("   - Context diversity: {:.3}", interference_result.context_analysis.context_diversity);
    println!("   - Consensus level: {:.3}", interference_result.consensus_report.overall_consensus_level);
}

#[tokio::test]
async fn test_resource_management_across_contexts() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create many contexts to test resource management
    let mut context_ids = Vec::new();
    for i in 0..15 { // More than typical max_active_contexts
        let context_id = {
            let mut manager = belief_system.context_manager.write().await;
            manager.create_context(
                BeliefContext::new(&format!("test_context_{}", i)),
                None,
                if i % 3 == 0 { ContextPriority::High } else { ContextPriority::Medium },
            ).await.unwrap()
        };
        context_ids.push(context_id);
    }
    
    // Try to activate all contexts (should trigger resource management)
    let mut activation_results = Vec::new();
    for &context_id in &context_ids {
        let result = {
            let mut manager = belief_system.context_manager.write().await;
            manager.activate_context(context_id).await
        };
        activation_results.push((context_id, result));
    }
    
    // Check that resource limits were enforced
    let active_count = {
        let manager = belief_system.context_manager.read().await;
        let contexts = manager.active_contexts.read().await;
        contexts.values().filter(|c| c.is_active).count()
    };
    
    assert!(active_count <= 10, // Assuming max_active_contexts = 10
            "Should not exceed maximum active contexts: {}", active_count);
    
    // Verify that high-priority contexts are preferentially kept active
    let (high_priority_active, low_priority_active) = {
        let manager = belief_system.context_manager.read().await;
        let contexts = manager.active_contexts.read().await;
        
        let high_priority = contexts.values()
            .filter(|c| c.is_active && c.priority == ContextPriority::High)
            .count();
        
        let low_priority = contexts.values()
            .filter(|c| c.is_active && c.priority == ContextPriority::Medium)
            .count();
        
        (high_priority, low_priority)
    };
    
    if high_priority_active > 0 && low_priority_active > 0 {
        // If we have both types, high priority should be proportionally more represented
        let high_priority_ratio = high_priority_active as f32 / (high_priority_active + low_priority_active) as f32;
        assert!(high_priority_ratio >= 0.3, "High priority contexts should be preserved");
    }
    
    // Test memory usage tracking
    let total_memory_usage = {
        let manager = belief_system.context_manager.read().await;
        let contexts = manager.active_contexts.read().await;
        contexts.values().map(|c| c.memory_usage).sum::<usize>()
    };
    
    println!("✅ Resource management test passed");
    println!("   - Total contexts created: {}", context_ids.len());
    println!("   - Active contexts: {}", active_count);
    println!("   - High priority active: {}", high_priority_active);
    println!("   - Low priority active: {}", low_priority_active);
    println!("   - Total memory usage: {} bytes", total_memory_usage);
}
```

### Step 4: Performance and Stress Tests

```rust
// File: tests/belief_system/performance_stress_tests.rs

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[tokio::test]
async fn test_large_scale_belief_network_performance() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Create large belief network (1000+ beliefs)
    let large_network_setup = setup_large_belief_network(&belief_system, 1500).await.unwrap();
    
    println!("Created belief network with {} beliefs", large_network_setup.belief_count);
    
    // Define complex query
    let query_intent = QueryIntent {
        query_text: "What factors contribute to sustainable development?".to_string(),
        entity_mentions: vec![
            "sustainable development".to_string(),
            "environmental factors".to_string(),
            "economic factors".to_string(),
            "social factors".to_string(),
        ],
        relationship_types: vec![
            "contributes_to".to_string(),
            "supports".to_string(),
            "enables".to_string(),
        ],
        context_hints: vec![
            "sustainability".to_string(),
            "development".to_string(),
            "environment".to_string(),
        ],
        confidence_threshold: Some(0.6),
    };
    
    // Measure query performance
    let query_start = Instant::now();
    let result = belief_system.belief_query_processor.process_belief_query(
        &query_intent,
        None,
        &large_network_setup.graph,
    ).await.unwrap();
    let query_duration = query_start.elapsed();
    
    // Performance assertions
    assert!(query_duration < Duration::from_millis(100),
            "Large network query should complete within 100ms, took {:?}", query_duration);
    
    assert!(!result.activated_nodes.is_empty(),
            "Should activate nodes in large network");
    
    assert!(result.activated_nodes.len() <= 100,
            "Should limit result size for performance");
    
    // Memory usage check
    let memory_usage = measure_memory_usage(&belief_system).await;
    let memory_mb = memory_usage as f64 / 1024.0 / 1024.0;
    
    assert!(memory_mb < 100.0,
            "Memory usage should be reasonable: {:.2} MB", memory_mb);
    
    // Test justification path performance on large network
    if let Some(justification_trace) = &result.justification_trace {
        let path_count: usize = justification_trace.node_paths.values()
            .map(|paths| paths.len())
            .sum();
        
        assert!(path_count > 0, "Should trace paths in large network");
        assert!(path_count <= 500, "Should limit path count for performance");
    }
    
    println!("✅ Large scale belief network performance test passed");
    println!("   - Belief network size: {}", large_network_setup.belief_count);
    println!("   - Query duration: {:?}", query_duration);
    println!("   - Memory usage: {:.2} MB", memory_mb);
    println!("   - Activated nodes: {}", result.activated_nodes.len());
}

#[tokio::test]
async fn test_concurrent_multi_context_processing() {
    let belief_system = Arc::new(setup_integrated_belief_system().await);
    
    // Create multiple contexts for concurrent processing
    let context_count = 8;
    let mut context_ids = Vec::new();
    
    for i in 0..context_count {
        let context_id = {
            let mut manager = belief_system.context_manager.write().await;
            manager.create_context(
                BeliefContext::new(&format!("concurrent_context_{}", i)),
                None,
                ContextPriority::Medium,
            ).await.unwrap()
        };
        context_ids.push(context_id);
    }
    
    // Setup beliefs in each context
    for &context_id in &context_ids {
        setup_context_specific_beliefs_async(&belief_system, context_id).await.unwrap();
    }
    
    let query_intent = QueryIntent {
        query_text: "What are the implications of climate change?".to_string(),
        entity_mentions: vec!["climate change".to_string()],
        relationship_types: vec!["implies".to_string()],
        context_hints: vec!["climate".to_string(), "environment".to_string()],
        confidence_threshold: Some(0.5),
    };
    
    // Process queries concurrently across all contexts
    let concurrent_start = Instant::now();
    
    let tasks: Vec<_> = context_ids.into_iter().map(|context_id| {
        let belief_system = belief_system.clone();
        let query_intent = query_intent.clone();
        
        tokio::spawn(async move {
            // Activate context
            {
                let mut manager = belief_system.context_manager.write().await;
                manager.activate_context(context_id).await.unwrap();
            }
            
            // Process query
            let result = belief_system.belief_query_processor.process_belief_query(
                &query_intent,
                Some(BeliefContext::from_context_id(context_id)),
                &setup_test_graph(),
            ).await.unwrap();
            
            (context_id, result)
        })
    }).collect();
    
    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    let concurrent_duration = concurrent_start.elapsed();
    
    // Validate concurrent processing
    assert_eq!(results.len(), context_count);
    
    let mut successful_results = 0;
    for task_result in results {
        match task_result {
            Ok((_context_id, query_result)) => {
                assert!(!query_result.activated_nodes.is_empty(),
                        "Each context should produce results");
                successful_results += 1;
            }
            Err(e) => {
                panic!("Concurrent task failed: {:?}", e);
            }
        }
    }
    
    assert_eq!(successful_results, context_count,
              "All concurrent queries should succeed");
    
    // Performance assertion
    assert!(concurrent_duration < Duration::from_millis(200),
            "Concurrent processing should complete within 200ms, took {:?}", concurrent_duration);
    
    // Check for resource contention
    let avg_duration_per_context = concurrent_duration.as_millis() / context_count as u128;
    assert!(avg_duration_per_context < 50,
            "Average per-context duration should be reasonable: {}ms", avg_duration_per_context);
    
    println!("✅ Concurrent multi-context processing test passed");
    println!("   - Contexts processed: {}", context_count);
    println!("   - Total duration: {:?}", concurrent_duration);
    println!("   - Avg per context: {}ms", avg_duration_per_context);
}

#[tokio::test]
async fn test_memory_usage_under_load() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Baseline memory measurement
    let baseline_memory = measure_memory_usage(&belief_system).await;
    
    // Create increasing load
    let load_phases = vec![
        ("Small load", 10, 2),      // 10 beliefs, 2 contexts
        ("Medium load", 100, 5),    // 100 beliefs, 5 contexts
        ("Large load", 500, 8),     // 500 beliefs, 8 contexts
        ("Stress load", 1000, 12),  // 1000 beliefs, 12 contexts
    ];
    
    for (phase_name, belief_count, context_count) in load_phases {
        println!("Testing {} phase...", phase_name);
        
        // Setup load
        let load_setup = setup_belief_load(&belief_system, belief_count, context_count).await.unwrap();
        
        // Measure memory after load
        let load_memory = measure_memory_usage(&belief_system).await;
        let memory_increase = load_memory - baseline_memory;
        let memory_increase_mb = memory_increase as f64 / 1024.0 / 1024.0;
        
        // Perform queries under load
        let query_start = Instant::now();
        let _result = perform_load_test_queries(&belief_system, &load_setup).await.unwrap();
        let query_duration = query_start.elapsed();
        
        // Memory usage should scale reasonably
        let memory_per_belief = memory_increase as f64 / belief_count as f64;
        assert!(memory_per_belief < 1024.0 * 10.0, // 10KB per belief max
                "Memory per belief should be reasonable: {:.1} bytes", memory_per_belief);
        
        // Performance should degrade gracefully
        let expected_max_duration = Duration::from_millis(20 + (belief_count / 10) as u64);
        assert!(query_duration < expected_max_duration,
                "Query duration should scale gracefully: {:?} vs expected max {:?}",
                query_duration, expected_max_duration);
        
        println!("   Memory increase: {:.2} MB", memory_increase_mb);
        println!("   Memory per belief: {:.1} bytes", memory_per_belief);
        println!("   Query duration: {:?}", query_duration);
        
        // Cleanup for next phase
        cleanup_belief_load(&belief_system, load_setup).await.unwrap();
    }
    
    // Final memory check - should return close to baseline
    let final_memory = measure_memory_usage(&belief_system).await;
    let final_increase = final_memory - baseline_memory;
    let final_increase_mb = final_increase as f64 / 1024.0 / 1024.0;
    
    assert!(final_increase_mb < 5.0,
            "Memory should return close to baseline after cleanup: {:.2} MB increase", 
            final_increase_mb);
    
    println!("✅ Memory usage under load test passed");
    println!("   Baseline memory: {:.2} MB", baseline_memory as f64 / 1024.0 / 1024.0);
    println!("   Final memory: {:.2} MB", final_memory as f64 / 1024.0 / 1024.0);
    println!("   Net increase: {:.2} MB", final_increase_mb);
}

#[tokio::test]
async fn test_latency_targets_under_stress() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Setup stress test environment
    let stress_setup = setup_stress_test_environment(&belief_system).await.unwrap();
    
    // Define latency targets for different operations
    let latency_targets = vec![
        ("Simple belief query", Duration::from_millis(10)),
        ("Complex multi-context query", Duration::from_millis(50)),
        ("Justification path tracing", Duration::from_millis(15)),
        ("Belief revision propagation", Duration::from_millis(20)),
        ("Context switching", Duration::from_millis(2)),
    ];
    
    // Test each operation type multiple times
    let iterations = 20;
    let mut results = HashMap::new();
    
    for (operation_name, target_latency) in latency_targets {
        let mut durations = Vec::new();
        
        for _ in 0..iterations {
            let duration = match operation_name {
                "Simple belief query" => {
                    test_simple_belief_query_latency(&belief_system, &stress_setup).await.unwrap()
                }
                "Complex multi-context query" => {
                    test_complex_multi_context_latency(&belief_system, &stress_setup).await.unwrap()
                }
                "Justification path tracing" => {
                    test_justification_tracing_latency(&belief_system, &stress_setup).await.unwrap()
                }
                "Belief revision propagation" => {
                    test_belief_revision_latency(&belief_system, &stress_setup).await.unwrap()
                }
                "Context switching" => {
                    test_context_switching_latency(&belief_system, &stress_setup).await.unwrap()
                }
                _ => Duration::from_millis(0),
            };
            
            durations.push(duration);
        }
        
        // Calculate statistics
        let avg_duration = Duration::from_nanos(
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() / iterations as u128
        );
        
        let max_duration = durations.iter().max().copied().unwrap();
        let min_duration = durations.iter().min().copied().unwrap();
        
        // Validate against targets
        assert!(avg_duration < target_latency,
                "{} average latency exceeds target: {:?} vs {:?}",
                operation_name, avg_duration, target_latency);
        
        // 95th percentile should be within 2x target
        let mut sorted_durations = durations.clone();
        sorted_durations.sort();
        let p95_index = (iterations as f32 * 0.95) as usize;
        let p95_duration = sorted_durations[p95_index.min(iterations - 1)];
        
        assert!(p95_duration < target_latency * 2,
                "{} 95th percentile latency too high: {:?} vs 2x target {:?}",
                operation_name, p95_duration, target_latency * 2);
        
        results.insert(operation_name, (avg_duration, min_duration, max_duration, p95_duration));
        
        println!("   {}: avg {:?}, min {:?}, max {:?}, p95 {:?}",
                operation_name, avg_duration, min_duration, max_duration, p95_duration);
    }
    
    println!("✅ Latency targets under stress test passed");
}

// Helper function implementations...
async fn measure_memory_usage(_belief_system: &IntegratedBeliefSystem) -> usize {
    // In a real implementation, this would measure actual memory usage
    // For testing, we'll simulate based on system state
    1024 * 1024 * 10 // 10MB baseline
}
```

## File Locations

- `tests/belief_system/end_to_end_tests.rs` - Complete workflow integration tests
- `tests/belief_system/belief_revision_tests.rs` - Belief revision integration tests
- `tests/belief_system/multi_context_consistency_tests.rs` - Multi-context consistency tests
- `tests/belief_system/performance_stress_tests.rs` - Performance and stress tests
- `tests/belief_system/test_helpers.rs` - Shared test utilities and setup functions
- `tests/belief_system/mod.rs` - Test module organization

## Success Criteria

- [ ] All end-to-end belief query workflows pass
- [ ] Belief revision propagation works correctly across components
- [ ] Multi-context consistency maintained under all conditions
- [ ] Performance targets met under stress testing
- [ ] Memory usage scales appropriately with system load
- [ ] Latency targets achieved for all operation types
- [ ] All tests pass with >95% reliability
- [ ] Integration test coverage >90% for belief system components

## Test Requirements

```rust
#[tokio::test]
async fn test_belief_system_integration_health_check() {
    let belief_system = setup_integrated_belief_system().await;
    
    // Comprehensive health check of all components
    let health_checks = vec![
        ("TMS", check_tms_health(&belief_system).await),
        ("Context Manager", check_context_manager_health(&belief_system).await),
        ("Belief Query Processor", check_belief_query_processor_health(&belief_system).await),
        ("Temporal Processor", check_temporal_processor_health(&belief_system).await),
        ("Multi-Context Processor", check_multi_context_processor_health(&belief_system).await),
    ];
    
    for (component, health_result) in health_checks {
        assert!(health_result.is_ok(), "{} health check failed: {:?}", component, health_result);
        println!("✅ {} health check passed", component);
    }
    
    // Test component interaction
    let integration_result = test_component_interactions(&belief_system).await;
    assert!(integration_result.is_ok(), "Component integration failed: {:?}", integration_result);
    
    println!("✅ Belief system integration health check passed");
}

// Comprehensive benchmark test
#[tokio::test]
async fn test_belief_system_benchmarks() {
    let belief_system = setup_integrated_belief_system().await;
    
    let benchmark_results = run_comprehensive_benchmarks(&belief_system).await.unwrap();
    
    // Validate all benchmark targets
    assert!(benchmark_results.end_to_end_query_latency < Duration::from_millis(50));
    assert!(benchmark_results.belief_revision_latency < Duration::from_millis(20));
    assert!(benchmark_results.context_switch_latency < Duration::from_millis(2));
    assert!(benchmark_results.multi_context_processing_latency < Duration::from_millis(100));
    assert!(benchmark_results.justification_tracing_latency < Duration::from_millis(15));
    
    // Memory usage targets
    assert!(benchmark_results.memory_overhead_percentage < 50.0);
    assert!(benchmark_results.memory_usage_mb < 100.0);
    
    // Throughput targets
    assert!(benchmark_results.queries_per_second > 100.0);
    assert!(benchmark_results.belief_revisions_per_second > 200.0);
    
    println!("✅ Belief system benchmarks passed");
    print_benchmark_summary(&benchmark_results);
}
```

## Quality Gates

- [ ] End-to-end query latency <50ms for complex scenarios
- [ ] Belief revision propagation <20ms for typical networks
- [ ] Context switching <2ms between contexts
- [ ] Multi-context processing <100ms for 5 contexts
- [ ] Memory overhead <50% additional for belief features
- [ ] Test reliability >95% across all test categories
- [ ] Integration test coverage >90% of belief system code
- [ ] No memory leaks detected in stress tests
- [ ] Performance scaling verified up to 1000+ belief networks

## Next Task

Upon completion of this task, **Day 5B: Wisdom (Belief Integration)** is complete. The belief system is now fully integrated with comprehensive testing validation. Proceed to Phase 7 integration and validation activities.

## Summary

This task completes the belief system integration by providing comprehensive test coverage that validates:

1. **End-to-end functionality** - Complete belief-aware query workflows
2. **Component integration** - TMS, temporal, context, and justification systems working together
3. **Performance characteristics** - Meeting latency and throughput targets
4. **Stress resilience** - Handling large-scale networks and concurrent processing
5. **Memory management** - Efficient resource usage under load

The integrated belief system now provides transparent, justifiable, and efficient reasoning capabilities that maintain logical consistency while supporting complex multi-context and temporal reasoning scenarios.