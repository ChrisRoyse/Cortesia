# Task 31: Cortical Voting System with Consensus Generation

## Metadata
- **Micro-Phase**: 2.31
- **Duration**: 45-50 minutes
- **Dependencies**: Task 29 (multi_column_processor), Task 30 (lateral_inhibition), Task 25-27 (column implementations)
- **Output**: `src/multi_column/cortical_voting.rs`

## Description
Implement neuromorphic cortical voting system that generates consensus decisions from multiple column responses after lateral inhibition. This system aggregates votes from semantic, structural, temporal, and exception columns using biologically-inspired voting mechanisms including weighted voting, confidence-based consensus, and dynamic threshold adjustment. The system must achieve >95% consensus agreement while maintaining processing times under 2ms and supporting both democratic and expertise-weighted voting modes.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_column::{ColumnVote, ColumnId, LateralInhibition};
    use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId};
    use std::time::{Duration, Instant};

    #[test]
    fn test_cortical_voting_initialization() {
        let voting_config = VotingConfig::default();
        let voting_system = CorticalVotingSystem::new(voting_config).unwrap();
        
        // Verify initialization
        assert!(voting_system.is_ready());
        assert_eq!(voting_system.get_voting_mode(), VotingMode::WeightedConsensus);
        assert_eq!(voting_system.get_consensus_threshold(), 0.6);
        assert_eq!(voting_system.get_total_voters(), 4); // 4 columns
        
        // Verify configuration
        let config = voting_system.get_configuration();
        assert!(config.enable_confidence_weighting);
        assert!(config.enable_expertise_adjustment);
        assert_eq!(config.quorum_threshold, 0.5);
        
        // Verify internal state
        let state = voting_system.get_voting_state();
        assert_eq!(state.total_votes_cast, 0);
        assert_eq!(state.consensus_history.len(), 0);
        assert!(state.column_expertise.len() == 4);
    }
    
    #[test]
    fn test_basic_consensus_voting() {
        let voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Create column votes with clear consensus
        let votes = vec![
            create_column_vote(ColumnId::Semantic, 0.9, 0.85, "dog"),
            create_column_vote(ColumnId::Structural, 0.8, 0.75, "dog"),
            create_column_vote(ColumnId::Temporal, 0.7, 0.65, "dog"),
            create_column_vote(ColumnId::Exception, 0.3, 0.2, "cat"), // Dissenting vote
        ];
        
        let start = Instant::now();
        let consensus = voting_system.generate_consensus(&votes).unwrap();
        let voting_time = start.elapsed();
        
        // Verify processing speed
        assert!(voting_time < Duration::from_millis(2), 
               "Voting took too long: {:?}", voting_time);
        
        // Verify consensus result
        assert_eq!(consensus.winning_concept, ConceptId::new("dog"));
        assert!(consensus.consensus_strength > 0.7, "Strong consensus expected");
        assert!(consensus.agreement_level > 0.75, "High agreement expected");
        assert_eq!(consensus.supporting_columns.len(), 3); // 3 columns support winner
        assert_eq!(consensus.dissenting_columns.len(), 1); // 1 column dissents
        
        // Verify voting metadata
        assert_eq!(consensus.total_votes, 4);
        assert!(consensus.voting_confidence > 0.8);
        assert!(consensus.processing_time < Duration::from_millis(2));
    }
    
    #[test]
    fn test_weighted_consensus_voting() {
        let config = VotingConfig {
            voting_mode: VotingMode::WeightedConsensus,
            enable_confidence_weighting: true,
            ..Default::default()
        };
        
        let voting_system = CorticalVotingSystem::new(config).unwrap();
        
        // Create votes with different confidence levels
        let votes = vec![
            create_column_vote(ColumnId::Semantic, 0.95, 0.9, "elephant"),    // High confidence
            create_column_vote(ColumnId::Structural, 0.6, 0.5, "elephant"),  // Medium confidence
            create_column_vote(ColumnId::Temporal, 0.4, 0.3, "mouse"),       // Low confidence, different concept
            create_column_vote(ColumnId::Exception, 0.7, 0.6, "elephant"),   // Medium-high confidence
        ];
        
        let consensus = voting_system.generate_consensus(&votes).unwrap();
        
        // High confidence votes should dominate
        assert_eq!(consensus.winning_concept, ConceptId::new("elephant"));
        
        // Verify weighted calculation
        assert!(consensus.consensus_strength > 0.75, "Weighted consensus should be strong");
        
        // Verify confidence influence
        let high_conf_influence = consensus.vote_breakdown.iter()
            .find(|v| v.column_id == ColumnId::Semantic)
            .unwrap()
            .effective_weight;
        
        let low_conf_influence = consensus.vote_breakdown.iter()
            .find(|v| v.column_id == ColumnId::Temporal)
            .unwrap()
            .effective_weight;
        
        assert!(high_conf_influence > low_conf_influence * 2.0, 
               "High confidence should have much more influence");
    }
    
    #[test]
    fn test_expertise_weighted_voting() {
        let config = VotingConfig {
            voting_mode: VotingMode::ExpertiseWeighted,
            enable_expertise_adjustment: true,
            ..Default::default()
        };
        
        let mut voting_system = CorticalVotingSystem::new(config).unwrap();
        
        // Set different expertise levels
        voting_system.set_column_expertise(ColumnId::Semantic, 0.9);     // High expertise
        voting_system.set_column_expertise(ColumnId::Structural, 0.6);   // Medium expertise
        voting_system.set_column_expertise(ColumnId::Temporal, 0.4);     // Low expertise
        voting_system.set_column_expertise(ColumnId::Exception, 0.8);    // High expertise
        
        // Create votes where expert columns agree
        let votes = vec![
            create_column_vote(ColumnId::Semantic, 0.8, 0.7, "car"),      // Expert agrees
            create_column_vote(ColumnId::Structural, 0.7, 0.6, "truck"),  // Non-expert disagrees
            create_column_vote(ColumnId::Temporal, 0.6, 0.5, "truck"),    // Low expertise disagrees
            create_column_vote(ColumnId::Exception, 0.85, 0.8, "car"),    // Expert agrees
        ];
        
        let consensus = voting_system.generate_consensus(&votes).unwrap();
        
        // Expert consensus should win despite numerical minority
        assert_eq!(consensus.winning_concept, ConceptId::new("car"));
        
        // Verify expertise influence
        let expert_influence: f32 = consensus.vote_breakdown.iter()
            .filter(|v| v.column_id == ColumnId::Semantic || v.column_id == ColumnId::Exception)
            .map(|v| v.effective_weight)
            .sum();
        
        let non_expert_influence: f32 = consensus.vote_breakdown.iter()
            .filter(|v| v.column_id == ColumnId::Structural || v.column_id == ColumnId::Temporal)
            .map(|v| v.effective_weight)
            .sum();
        
        assert!(expert_influence > non_expert_influence, 
               "Expert columns should have more total influence");
    }
    
    #[test]
    fn test_democratic_voting() {
        let config = VotingConfig {
            voting_mode: VotingMode::Democratic,
            enable_confidence_weighting: false,
            enable_expertise_adjustment: false,
            ..Default::default()
        };
        
        let voting_system = CorticalVotingSystem::new(config).unwrap();
        
        // Create votes where majority rules
        let votes = vec![
            create_column_vote(ColumnId::Semantic, 0.6, 0.5, "bird"),
            create_column_vote(ColumnId::Structural, 0.7, 0.6, "bird"),
            create_column_vote(ColumnId::Temporal, 0.65, 0.55, "bird"),
            create_column_vote(ColumnId::Exception, 0.9, 0.85, "plane"), // Higher confidence but minority
        ];
        
        let consensus = voting_system.generate_consensus(&votes).unwrap();
        
        // Majority should win regardless of confidence
        assert_eq!(consensus.winning_concept, ConceptId::new("bird"));
        
        // All votes should have equal weight
        for vote_info in &consensus.vote_breakdown {
            assert!((vote_info.effective_weight - 0.25).abs() < 0.01, // 1/4 = 0.25
                   "Democratic voting should have equal weights");
        }
    }
    
    #[test]
    fn test_adaptive_threshold_adjustment() {
        let config = VotingConfig {
            enable_adaptive_threshold: true,
            adaptation_sensitivity: 0.1,
            ..Default::default()
        };
        
        let mut voting_system = CorticalVotingSystem::new(config).unwrap();
        
        // Process several rounds with different consensus levels
        
        // Round 1: High consensus scenario
        let high_consensus_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.9, 0.85, "strong_consensus"),
            create_column_vote(ColumnId::Structural, 0.88, 0.83, "strong_consensus"),
            create_column_vote(ColumnId::Temporal, 0.85, 0.8, "strong_consensus"),
            create_column_vote(ColumnId::Exception, 0.2, 0.15, "outlier"),
        ];
        
        let initial_threshold = voting_system.get_consensus_threshold();
        let _consensus1 = voting_system.generate_consensus(&high_consensus_votes).unwrap();
        
        // Process multiple high consensus rounds
        for _ in 0..5 {
            let _consensus = voting_system.generate_consensus(&high_consensus_votes).unwrap();
        }
        
        let adapted_threshold_high = voting_system.get_consensus_threshold();
        
        // Round 2: Low consensus scenario
        let low_consensus_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.55, 0.5, "concept_a"),
            create_column_vote(ColumnId::Structural, 0.52, 0.47, "concept_b"),
            create_column_vote(ColumnId::Temporal, 0.58, 0.53, "concept_a"),
            create_column_vote(ColumnId::Exception, 0.54, 0.49, "concept_c"),
        ];
        
        // Process multiple low consensus rounds
        for _ in 0..5 {
            let _consensus = voting_system.generate_consensus(&low_consensus_votes).unwrap();
        }
        
        let adapted_threshold_low = voting_system.get_consensus_threshold();
        
        // Verify threshold adaptation
        assert_ne!(initial_threshold, adapted_threshold_high, 
                  "Threshold should adapt to high consensus");
        assert_ne!(adapted_threshold_high, adapted_threshold_low, 
                  "Threshold should adapt to consensus level");
        
        // High consensus should lead to higher threshold
        if adapted_threshold_high != initial_threshold {
            // Adaptation occurred
            let adaptation_stats = voting_system.get_adaptation_statistics();
            assert!(adaptation_stats.total_adaptations > 0);
        }
    }
    
    #[test]
    fn test_quorum_requirements() {
        let config = VotingConfig {
            quorum_threshold: 0.75, // Require 75% participation
            min_participating_columns: 3,
            ..Default::default()
        };
        
        let voting_system = CorticalVotingSystem::new(config).unwrap();
        
        // Test insufficient participation
        let insufficient_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.8, 0.7, "concept"),
            create_column_vote(ColumnId::Structural, 0.0, 0.0, ""), // Non-participating
        ];
        
        let result = voting_system.generate_consensus(&insufficient_votes);
        assert!(result.is_err(), "Should fail with insufficient quorum");
        
        if let Err(VotingError::InsufficientQuorum { required, actual }) = result {
            assert_eq!(required, 3);
            assert_eq!(actual, 1); // Only semantic column participating
        }
        
        // Test sufficient participation
        let sufficient_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.8, 0.7, "concept"),
            create_column_vote(ColumnId::Structural, 0.7, 0.6, "concept"),
            create_column_vote(ColumnId::Temporal, 0.6, 0.5, "concept"),
            create_column_vote(ColumnId::Exception, 0.0, 0.0, ""), // Non-participating but quorum met
        ];
        
        let consensus = voting_system.generate_consensus(&sufficient_votes).unwrap();
        assert_eq!(consensus.winning_concept, ConceptId::new("concept"));
        assert_eq!(consensus.participating_columns, 3);
    }
    
    #[test]
    fn test_tie_breaking_mechanisms() {
        let voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Create perfect tie scenario
        let tie_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.7, 0.65, "option_a"),
            create_column_vote(ColumnId::Structural, 0.7, 0.65, "option_b"),
            create_column_vote(ColumnId::Temporal, 0.7, 0.65, "option_a"),
            create_column_vote(ColumnId::Exception, 0.7, 0.65, "option_b"),
        ];
        
        let consensus = voting_system.generate_consensus(&tie_votes).unwrap();
        
        // Verify tie was broken
        assert!(consensus.winning_concept == ConceptId::new("option_a") || 
                consensus.winning_concept == ConceptId::new("option_b"));
        
        // Verify tie breaking metadata
        assert!(consensus.tie_breaking_applied);
        assert_eq!(consensus.tie_candidates.len(), 2);
        assert!(consensus.tie_candidates.contains(&ConceptId::new("option_a")));
        assert!(consensus.tie_candidates.contains(&ConceptId::new("option_b")));
        
        // Test confidence-based tie breaking
        let confidence_tie_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.9, 0.85, "high_conf"),
            create_column_vote(ColumnId::Structural, 0.6, 0.55, "low_conf"),
            create_column_vote(ColumnId::Temporal, 0.9, 0.85, "high_conf"),
            create_column_vote(ColumnId::Exception, 0.6, 0.55, "low_conf"),
        ];
        
        let confidence_consensus = voting_system.generate_consensus(&confidence_tie_votes).unwrap();
        
        // Higher confidence option should win
        assert_eq!(confidence_consensus.winning_concept, ConceptId::new("high_conf"));
    }
    
    #[test]
    fn test_consensus_quality_metrics() {
        let voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Test high quality consensus
        let high_quality_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.95, 0.9, "consensus"),
            create_column_vote(ColumnId::Structural, 0.92, 0.87, "consensus"),
            create_column_vote(ColumnId::Temporal, 0.88, 0.83, "consensus"),
            create_column_vote(ColumnId::Exception, 0.85, 0.8, "consensus"),
        ];
        
        let high_quality = voting_system.generate_consensus(&high_quality_votes).unwrap();
        
        // Verify quality metrics
        assert!(high_quality.consensus_strength > 0.9, "Should have high consensus strength");
        assert!(high_quality.agreement_level > 0.95, "Should have high agreement");
        assert!(high_quality.confidence_variance < 0.1, "Should have low variance");
        assert_eq!(high_quality.unanimity_level, 1.0); // All columns agree
        
        // Test low quality consensus
        let low_quality_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.6, 0.55, "weak_consensus"),
            create_column_vote(ColumnId::Structural, 0.4, 0.35, "other"),
            create_column_vote(ColumnId::Temporal, 0.55, 0.5, "weak_consensus"),
            create_column_vote(ColumnId::Exception, 0.45, 0.4, "another"),
        ];
        
        let low_quality = voting_system.generate_consensus(&low_quality_votes).unwrap();
        
        // Verify quality metrics
        assert!(low_quality.consensus_strength < 0.7, "Should have low consensus strength");
        assert!(low_quality.agreement_level < 0.6, "Should have low agreement");
        assert!(low_quality.confidence_variance > 0.1, "Should have high variance");
        assert!(low_quality.unanimity_level < 0.6); // Disagreement exists
    }
    
    #[test]
    fn test_voting_with_inhibited_columns() {
        let voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Create votes that simulate post-inhibition state
        let inhibited_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.9, 0.85, "winner"),        // Strong winner
            create_column_vote(ColumnId::Structural, 0.2, 0.15, "suppressed"),  // Heavily inhibited
            create_column_vote(ColumnId::Temporal, 0.1, 0.05, "suppressed"),    // Heavily inhibited
            create_column_vote(ColumnId::Exception, 0.05, 0.02, "suppressed"),  // Almost eliminated
        ];
        
        let consensus = voting_system.generate_consensus(&inhibited_votes).unwrap();
        
        // Winner should be clear
        assert_eq!(consensus.winning_concept, ConceptId::new("winner"));
        
        // Verify inhibition handling
        assert_eq!(consensus.dominant_column, Some(ColumnId::Semantic));
        assert!(consensus.inhibition_effectiveness > 0.8, "Should recognize effective inhibition");
        
        // Suppressed votes should have minimal influence
        let suppressed_total_weight: f32 = consensus.vote_breakdown.iter()
            .filter(|v| v.column_id != ColumnId::Semantic)
            .map(|v| v.effective_weight)
            .sum();
        
        let winner_weight = consensus.vote_breakdown.iter()
            .find(|v| v.column_id == ColumnId::Semantic)
            .unwrap()
            .effective_weight;
        
        assert!(winner_weight > suppressed_total_weight * 3.0, 
               "Winner should have much more influence than suppressed columns");
    }
    
    #[test]
    fn test_temporal_voting_consistency() {
        let mut voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Process same concept multiple times
        let consistent_votes = vec![
            create_column_vote(ColumnId::Semantic, 0.8, 0.75, "consistent_concept"),
            create_column_vote(ColumnId::Structural, 0.7, 0.65, "consistent_concept"),
            create_column_vote(ColumnId::Temporal, 0.6, 0.55, "consistent_concept"),
            create_column_vote(ColumnId::Exception, 0.3, 0.25, "outlier"),
        ];
        
        let mut consensus_results = Vec::new();
        
        // Process multiple rounds
        for _ in 0..10 {
            let consensus = voting_system.generate_consensus(&consistent_votes).unwrap();
            consensus_results.push(consensus);
        }
        
        // Verify consistency
        let winning_concepts: Vec<_> = consensus_results.iter()
            .map(|c| c.winning_concept.clone())
            .collect();
        
        let consistent_winner = &winning_concepts[0];
        let consistency_rate = winning_concepts.iter()
            .filter(|c| *c == consistent_winner)
            .count() as f32 / winning_concepts.len() as f32;
        
        assert!(consistency_rate > 0.9, "Voting should be consistent: {:.2}", consistency_rate);
        
        // Verify temporal adaptation
        let voting_history = voting_system.get_voting_history();
        assert_eq!(voting_history.len(), 10);
        
        let consensus_strengths: Vec<_> = consensus_results.iter()
            .map(|c| c.consensus_strength)
            .collect();
        
        // Later votes should potentially be stronger due to learning
        let early_avg = consensus_strengths[0..3].iter().sum::<f32>() / 3.0;
        let late_avg = consensus_strengths[7..10].iter().sum::<f32>() / 3.0;
        
        // Not strictly required but often the case with adaptation
        if late_avg > early_avg {
            assert!(late_avg - early_avg > 0.01, "Should show improvement over time");
        }
    }
    
    #[test]
    fn test_voting_performance_metrics() {
        let voting_system = CorticalVotingSystem::new(VotingConfig::default()).unwrap();
        
        // Process multiple voting rounds to generate metrics
        for i in 0..100 {
            let votes = create_random_votes(i);
            let _consensus = voting_system.generate_consensus(&votes).unwrap();
        }
        
        // Verify performance metrics
        let metrics = voting_system.get_performance_metrics();
        
        // Processing metrics
        assert_eq!(metrics.total_votes_processed, 100);
        assert!(metrics.average_processing_time < Duration::from_millis(2));
        assert!(metrics.consensus_agreement_rate > 0.9); // >90% should reach consensus
        
        // Quality metrics
        assert!(metrics.average_consensus_strength > 0.6);
        assert!(metrics.voting_efficiency > 0.8);
        
        // Resource metrics
        assert!(metrics.memory_usage < 2_000_000); // <2MB for voting
        assert!(metrics.cpu_utilization < 0.05); // Low CPU overhead
        
        // Timing metrics
        assert!(metrics.fastest_vote < Duration::from_micros(500));
        assert!(metrics.slowest_vote < Duration::from_millis(5));
    }
    
    // Helper functions
    fn create_column_vote(column_id: ColumnId, confidence: f32, activation: f32, concept: &str) -> ColumnVote {
        ColumnVote {
            column_id,
            confidence,
            activation,
            neural_output: vec![activation; 6],
            processing_time: Duration::from_micros(300),
        }
    }
    
    fn create_random_votes(seed: usize) -> Vec<ColumnVote> {
        let concepts = vec!["dog", "cat", "bird", "fish", "car", "tree"];
        let concept = concepts[seed % concepts.len()];
        
        let base_confidence = 0.5 + (seed as f32 * 0.01) % 0.4;
        
        vec![
            create_column_vote(ColumnId::Semantic, base_confidence + 0.1, base_confidence, concept),
            create_column_vote(ColumnId::Structural, base_confidence, base_confidence * 0.9, concept),
            create_column_vote(ColumnId::Temporal, base_confidence - 0.1, base_confidence * 0.8, concept),
            create_column_vote(ColumnId::Exception, base_confidence * 0.5, base_confidence * 0.4, 
                              if seed % 5 == 0 { "outlier" } else { concept }),
        ]
    }
}
```

## Implementation
```rust
use crate::multi_column::{ColumnVote, ColumnId};
use crate::ttfs_encoding::ConceptId;
use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

/// Cortical voting system for consensus generation from column responses
#[derive(Debug)]
pub struct CorticalVotingSystem {
    /// Voting configuration
    config: VotingConfig,
    
    /// Current voting mode
    voting_mode: VotingMode,
    
    /// Consensus threshold for decision making
    consensus_threshold: f32,
    
    /// Column expertise levels for weighted voting
    column_expertise: HashMap<ColumnId, f32>,
    
    /// Voting state tracking
    state: Arc<Mutex<VotingState>>,
    
    /// Performance monitoring
    performance_metrics: Arc<Mutex<VotingPerformanceMetrics>>,
    
    /// Adaptive threshold manager
    threshold_adapter: Arc<Mutex<ThresholdAdapter>>,
    
    /// Voting history for analysis
    voting_history: Arc<Mutex<Vec<ConsensusResult>>>,
}

/// Configuration for cortical voting behavior
#[derive(Debug, Clone)]
pub struct VotingConfig {
    /// Voting mode
    pub voting_mode: VotingMode,
    
    /// Base consensus threshold
    pub consensus_threshold: f32,
    
    /// Quorum threshold (minimum participation rate)
    pub quorum_threshold: f32,
    
    /// Minimum participating columns
    pub min_participating_columns: usize,
    
    /// Enable confidence-based weighting
    pub enable_confidence_weighting: bool,
    
    /// Enable expertise-based adjustment
    pub enable_expertise_adjustment: bool,
    
    /// Enable adaptive threshold adjustment
    pub enable_adaptive_threshold: bool,
    
    /// Adaptation sensitivity
    pub adaptation_sensitivity: f32,
    
    /// Tie-breaking strategy
    pub tie_breaking_strategy: TieBreakingStrategy,
    
    /// Quality threshold for accepting consensus
    pub quality_threshold: f32,
    
    /// Enable temporal consistency tracking
    pub enable_temporal_consistency: bool,
    
    /// History window size for adaptation
    pub history_window_size: usize,
}

/// Voting modes for consensus generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingMode {
    /// Democratic voting (equal weights)
    Democratic,
    
    /// Weighted consensus (confidence-based)
    WeightedConsensus,
    
    /// Expertise-weighted voting
    ExpertiseWeighted,
    
    /// Adaptive voting (changes based on context)
    Adaptive,
    
    /// Hierarchical voting (ordered preferences)
    Hierarchical,
}

/// Tie-breaking strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreakingStrategy {
    /// Break ties by highest confidence
    HighestConfidence,
    
    /// Break ties by expertise level
    ExpertiseLevel,
    
    /// Break ties randomly
    Random,
    
    /// Break ties by column priority
    ColumnPriority,
    
    /// Break ties by temporal precedence
    TemporalPrecedence,
}

/// Internal state of voting system
#[derive(Debug, Default)]
pub struct VotingState {
    /// Total votes cast
    pub total_votes_cast: u64,
    
    /// Consensus history
    pub consensus_history: Vec<ConsensusEvent>,
    
    /// Column expertise tracking
    pub column_expertise: HashMap<ColumnId, ExpertiseMetrics>,
    
    /// Current consensus round
    pub current_round: u64,
    
    /// Last voting timestamp
    pub last_voting_time: Option<SystemTime>,
}

/// Consensus event for history tracking
#[derive(Debug, Clone)]
pub struct ConsensusEvent {
    /// Timestamp
    pub timestamp: SystemTime,
    
    /// Participating columns
    pub participating_columns: Vec<ColumnId>,
    
    /// Winning concept
    pub winning_concept: ConceptId,
    
    /// Consensus strength
    pub consensus_strength: f32,
    
    /// Agreement level
    pub agreement_level: f32,
    
    /// Voting mode used
    pub voting_mode: VotingMode,
}

/// Expertise metrics for columns
#[derive(Debug, Clone)]
pub struct ExpertiseMetrics {
    /// Accuracy rate
    pub accuracy_rate: f32,
    
    /// Confidence correlation
    pub confidence_correlation: f32,
    
    /// Consistency score
    pub consistency_score: f32,
    
    /// Total votes cast
    pub total_votes: u64,
    
    /// Correct predictions
    pub correct_predictions: u64,
}

/// Consensus result from voting
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Winning concept
    pub winning_concept: ConceptId,
    
    /// Consensus strength (0.0-1.0)
    pub consensus_strength: f32,
    
    /// Agreement level between columns
    pub agreement_level: f32,
    
    /// Supporting columns
    pub supporting_columns: Vec<ColumnId>,
    
    /// Dissenting columns
    pub dissenting_columns: Vec<ColumnId>,
    
    /// Total votes counted
    pub total_votes: usize,
    
    /// Participating columns
    pub participating_columns: usize,
    
    /// Voting confidence
    pub voting_confidence: f32,
    
    /// Processing time
    pub processing_time: Duration,
    
    /// Vote breakdown
    pub vote_breakdown: Vec<VoteInfo>,
    
    /// Quality metrics
    pub quality_metrics: ConsensusQualityMetrics,
    
    /// Tie breaking information
    pub tie_breaking_applied: bool,
    pub tie_candidates: Vec<ConceptId>,
    
    /// Dominance information
    pub dominant_column: Option<ColumnId>,
    pub inhibition_effectiveness: f32,
    
    /// Temporal consistency
    pub consistency_with_history: f32,
    
    /// Unanimity level
    pub unanimity_level: f32,
    
    /// Confidence variance
    pub confidence_variance: f32,
}

/// Individual vote information
#[derive(Debug, Clone)]
pub struct VoteInfo {
    /// Column identifier
    pub column_id: ColumnId,
    
    /// Original confidence
    pub original_confidence: f32,
    
    /// Effective weight in voting
    pub effective_weight: f32,
    
    /// Concept voted for
    pub concept: ConceptId,
    
    /// Expertise contribution
    pub expertise_contribution: f32,
    
    /// Confidence contribution
    pub confidence_contribution: f32,
}

/// Quality metrics for consensus
#[derive(Debug, Clone)]
pub struct ConsensusQualityMetrics {
    /// Overall quality score
    pub overall_quality: f32,
    
    /// Separation quality
    pub separation_quality: f32,
    
    /// Confidence coherence
    pub confidence_coherence: f32,
    
    /// Stability measure
    pub stability_measure: f32,
    
    /// Predictive confidence
    pub predictive_confidence: f32,
}

/// Adaptive threshold management
#[derive(Debug)]
pub struct ThresholdAdapter {
    /// Adaptation statistics
    pub adaptation_stats: AdaptationStatistics,
    
    /// Consensus quality history
    pub quality_history: Vec<f32>,
    
    /// Target quality level
    pub target_quality: f32,
    
    /// Adaptation rate
    pub adaptation_rate: f32,
    
    /// Stability factor
    pub stability_factor: f32,
}

/// Adaptation statistics
#[derive(Debug, Default)]
pub struct AdaptationStatistics {
    /// Total adaptations
    pub total_adaptations: u64,
    
    /// Successful adaptations
    pub successful_adaptations: u64,
    
    /// Average improvement
    pub average_improvement: f32,
    
    /// Adaptation efficiency
    pub adaptation_efficiency: f32,
}

/// Performance metrics for voting system
#[derive(Debug, Default)]
pub struct VotingPerformanceMetrics {
    /// Total votes processed
    pub total_votes_processed: u64,
    
    /// Average processing time
    pub average_processing_time: Duration,
    
    /// Consensus agreement rate
    pub consensus_agreement_rate: f32,
    
    /// Average consensus strength
    pub average_consensus_strength: f32,
    
    /// Voting efficiency
    pub voting_efficiency: f32,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// CPU utilization
    pub cpu_utilization: f32,
    
    /// Fastest vote time
    pub fastest_vote: Duration,
    
    /// Slowest vote time
    pub slowest_vote: Duration,
    
    /// Quality distribution
    pub quality_distribution: QualityDistribution,
}

/// Quality distribution metrics
#[derive(Debug, Default)]
pub struct QualityDistribution {
    /// High quality votes (>0.8)
    pub high_quality_rate: f32,
    
    /// Medium quality votes (0.5-0.8)
    pub medium_quality_rate: f32,
    
    /// Low quality votes (<0.5)
    pub low_quality_rate: f32,
    
    /// Average quality score
    pub average_quality: f32,
}

/// Cortical voting errors
#[derive(Debug, thiserror::Error)]
pub enum VotingError {
    #[error("Invalid voting configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Insufficient quorum: required {required}, got {actual}")]
    InsufficientQuorum { required: usize, actual: usize },
    
    #[error("No valid votes provided")]
    NoValidVotes,
    
    #[error("Consensus generation failed: {0}")]
    ConsensusGenerationFailed(String),
    
    #[error("Tie breaking failed: {0}")]
    TieBreakingFailed(String),
    
    #[error("Threshold adaptation error: {0}")]
    ThresholdAdaptationError(String),
}

impl CorticalVotingSystem {
    /// Create new cortical voting system
    pub fn new(config: VotingConfig) -> Result<Self, VotingError> {
        Self::validate_config(&config)?;
        
        // Initialize column expertise with default values
        let mut column_expertise = HashMap::new();
        column_expertise.insert(ColumnId::Semantic, 0.8);
        column_expertise.insert(ColumnId::Structural, 0.7);
        column_expertise.insert(ColumnId::Temporal, 0.6);
        column_expertise.insert(ColumnId::Exception, 0.75);
        
        Ok(Self {
            voting_mode: config.voting_mode,
            consensus_threshold: config.consensus_threshold,
            column_expertise,
            config,
            state: Arc::new(Mutex::new(VotingState::default())),
            performance_metrics: Arc::new(Mutex::new(VotingPerformanceMetrics::default())),
            threshold_adapter: Arc::new(Mutex::new(ThresholdAdapter::new())),
            voting_history: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Generate consensus from column votes
    pub fn generate_consensus(&self, votes: &[ColumnVote]) -> Result<ConsensusResult, VotingError> {
        let start_time = Instant::now();
        
        // Validate input
        if votes.is_empty() {
            return Err(VotingError::NoValidVotes);
        }
        
        // Check quorum requirements
        let participating_columns = self.count_participating_columns(votes);
        if participating_columns < self.config.min_participating_columns {
            return Err(VotingError::InsufficientQuorum {
                required: self.config.min_participating_columns,
                actual: participating_columns,
            });
        }
        
        // Convert votes to concepts
        let concept_votes = self.extract_concept_votes(votes);
        
        // Generate vote breakdown with weights
        let vote_breakdown = self.calculate_vote_weights(&concept_votes)?;
        
        // Determine winning concept
        let (winning_concept, tie_info) = self.determine_winner(&vote_breakdown)?;
        
        // Calculate consensus metrics
        let consensus_metrics = self.calculate_consensus_metrics(&vote_breakdown, &winning_concept);
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(votes, &consensus_metrics);
        
        // Create consensus result
        let consensus_result = ConsensusResult {
            winning_concept: winning_concept.clone(),
            consensus_strength: consensus_metrics.consensus_strength,
            agreement_level: consensus_metrics.agreement_level,
            supporting_columns: self.get_supporting_columns(&vote_breakdown, &winning_concept),
            dissenting_columns: self.get_dissenting_columns(&vote_breakdown, &winning_concept),
            total_votes: votes.len(),
            participating_columns,
            voting_confidence: consensus_metrics.voting_confidence,
            processing_time: start_time.elapsed(),
            vote_breakdown,
            quality_metrics,
            tie_breaking_applied: tie_info.tie_occurred,
            tie_candidates: tie_info.candidates,
            dominant_column: self.find_dominant_column(votes),
            inhibition_effectiveness: self.calculate_inhibition_effectiveness(votes),
            consistency_with_history: self.calculate_historical_consistency(&winning_concept),
            unanimity_level: consensus_metrics.unanimity_level,
            confidence_variance: consensus_metrics.confidence_variance,
        };
        
        let processing_time = start_time.elapsed();
        
        // Update system state
        self.update_voting_state(&consensus_result);
        self.update_performance_metrics(processing_time, &consensus_result);
        
        // Apply adaptive threshold adjustment if enabled
        if self.config.enable_adaptive_threshold {
            self.adapt_threshold(&consensus_result)?;
        }
        
        // Add to history
        self.add_to_history(&consensus_result);
        
        Ok(consensus_result)
    }
    
    /// Extract concept votes from column votes
    fn extract_concept_votes(&self, votes: &[ColumnVote]) -> Vec<ConceptVote> {
        votes.iter()
            .filter(|v| v.confidence > 0.01) // Filter out essentially zero votes
            .map(|v| ConceptVote {
                column_id: v.column_id,
                concept: self.extract_concept_from_vote(v),
                confidence: v.confidence,
                activation: v.activation,
            })
            .collect()
    }
    
    /// Extract concept from vote (mock implementation)
    fn extract_concept_from_vote(&self, vote: &ColumnVote) -> ConceptId {
        // In a real implementation, this would extract the concept from neural output
        // For testing, we'll use a mock approach
        ConceptId::new("extracted_concept")
    }
    
    /// Calculate vote weights based on voting mode
    fn calculate_vote_weights(&self, concept_votes: &[ConceptVote]) -> Result<Vec<VoteInfo>, VotingError> {
        let mut vote_breakdown = Vec::new();
        
        for concept_vote in concept_votes {
            let base_weight = match self.voting_mode {
                VotingMode::Democratic => 1.0 / concept_votes.len() as f32,
                VotingMode::WeightedConsensus => self.calculate_confidence_weight(concept_vote),
                VotingMode::ExpertiseWeighted => self.calculate_expertise_weight(concept_vote),
                VotingMode::Adaptive => self.calculate_adaptive_weight(concept_vote),
                VotingMode::Hierarchical => self.calculate_hierarchical_weight(concept_vote),
            };
            
            // Apply additional weighting factors
            let confidence_contribution = if self.config.enable_confidence_weighting {
                concept_vote.confidence
            } else {
                1.0
            };
            
            let expertise_contribution = if self.config.enable_expertise_adjustment {
                self.column_expertise.get(&concept_vote.column_id).copied().unwrap_or(0.5)
            } else {
                1.0
            };
            
            let effective_weight = base_weight * confidence_contribution * expertise_contribution;
            
            vote_breakdown.push(VoteInfo {
                column_id: concept_vote.column_id,
                original_confidence: concept_vote.confidence,
                effective_weight,
                concept: concept_vote.concept.clone(),
                expertise_contribution,
                confidence_contribution,
            });
        }
        
        // Normalize weights to sum to 1.0
        let total_weight: f32 = vote_breakdown.iter().map(|v| v.effective_weight).sum();
        if total_weight > 0.0 {
            for vote in &mut vote_breakdown {
                vote.effective_weight /= total_weight;
            }
        }
        
        Ok(vote_breakdown)
    }
    
    /// Calculate confidence-based weight
    fn calculate_confidence_weight(&self, concept_vote: &ConceptVote) -> f32 {
        concept_vote.confidence
    }
    
    /// Calculate expertise-based weight
    fn calculate_expertise_weight(&self, concept_vote: &ConceptVote) -> f32 {
        self.column_expertise.get(&concept_vote.column_id).copied().unwrap_or(0.5)
    }
    
    /// Calculate adaptive weight
    fn calculate_adaptive_weight(&self, concept_vote: &ConceptVote) -> f32 {
        // Adaptive weighting based on historical performance
        let base_weight = concept_vote.confidence;
        let expertise_factor = self.column_expertise.get(&concept_vote.column_id).copied().unwrap_or(0.5);
        (base_weight + expertise_factor) / 2.0
    }
    
    /// Calculate hierarchical weight
    fn calculate_hierarchical_weight(&self, concept_vote: &ConceptVote) -> f32 {
        // Hierarchical weighting based on column priority
        let priority_weight = match concept_vote.column_id {
            ColumnId::Semantic => 1.0,     // Highest priority
            ColumnId::Exception => 0.9,    // High priority
            ColumnId::Structural => 0.8,   // Medium priority
            ColumnId::Temporal => 0.7,     // Lower priority
        };
        
        concept_vote.confidence * priority_weight
    }
    
    /// Determine winning concept and handle ties
    fn determine_winner(&self, vote_breakdown: &[VoteInfo]) -> Result<(ConceptId, TieInfo), VotingError> {
        // Group votes by concept
        let mut concept_weights: HashMap<ConceptId, f32> = HashMap::new();
        
        for vote in vote_breakdown {
            *concept_weights.entry(vote.concept.clone()).or_insert(0.0) += vote.effective_weight;
        }
        
        if concept_weights.is_empty() {
            return Err(VotingError::ConsensusGenerationFailed("No valid concepts found".to_string()));
        }
        
        // Find maximum weight
        let max_weight = concept_weights.values().cloned().fold(0.0f32, f32::max);
        
        // Find all concepts with maximum weight (potential ties)
        let max_concepts: Vec<_> = concept_weights.iter()
            .filter(|(_, &weight)| (weight - max_weight).abs() < 0.001)
            .map(|(concept, _)| concept.clone())
            .collect();
        
        if max_concepts.len() == 1 {
            // No tie
            Ok((max_concepts[0].clone(), TieInfo {
                tie_occurred: false,
                candidates: Vec::new(),
            }))
        } else {
            // Tie detected - apply tie breaking
            let winner = self.break_tie(&max_concepts, vote_breakdown)?;
            Ok((winner, TieInfo {
                tie_occurred: true,
                candidates: max_concepts,
            }))
        }
    }
    
    /// Break ties using configured strategy
    fn break_tie(&self, tied_concepts: &[ConceptId], vote_breakdown: &[VoteInfo]) -> Result<ConceptId, VotingError> {
        match self.config.tie_breaking_strategy {
            TieBreakingStrategy::HighestConfidence => {
                // Find concept with highest original confidence
                let mut best_concept = tied_concepts[0].clone();
                let mut highest_confidence = 0.0f32;
                
                for concept in tied_concepts {
                    let max_confidence = vote_breakdown.iter()
                        .filter(|v| v.concept == *concept)
                        .map(|v| v.original_confidence)
                        .fold(0.0f32, f32::max);
                    
                    if max_confidence > highest_confidence {
                        highest_confidence = max_confidence;
                        best_concept = concept.clone();
                    }
                }
                
                Ok(best_concept)
            }
            TieBreakingStrategy::ExpertiseLevel => {
                // Find concept supported by highest expertise column
                let mut best_concept = tied_concepts[0].clone();
                let mut highest_expertise = 0.0f32;
                
                for concept in tied_concepts {
                    let max_expertise = vote_breakdown.iter()
                        .filter(|v| v.concept == *concept)
                        .map(|v| v.expertise_contribution)
                        .fold(0.0f32, f32::max);
                    
                    if max_expertise > highest_expertise {
                        highest_expertise = max_expertise;
                        best_concept = concept.clone();
                    }
                }
                
                Ok(best_concept)
            }
            TieBreakingStrategy::ColumnPriority => {
                // Use hierarchical column priority
                let column_priorities = vec![
                    ColumnId::Semantic,
                    ColumnId::Exception, 
                    ColumnId::Structural,
                    ColumnId::Temporal
                ];
                
                for priority_column in column_priorities {
                    for concept in tied_concepts {
                        if vote_breakdown.iter().any(|v| v.concept == *concept && v.column_id == priority_column) {
                            return Ok(concept.clone());
                        }
                    }
                }
                
                Ok(tied_concepts[0].clone()) // Fallback
            }
            TieBreakingStrategy::Random => {
                // Random selection (deterministic for testing)
                let index = (tied_concepts.len() * 37) % tied_concepts.len(); // Pseudo-random
                Ok(tied_concepts[index].clone())
            }
            TieBreakingStrategy::TemporalPrecedence => {
                // Use first concept encountered (for temporal consistency)
                Ok(tied_concepts[0].clone())
            }
        }
    }
    
    /// Calculate consensus metrics
    fn calculate_consensus_metrics(&self, vote_breakdown: &[VoteInfo], winning_concept: &ConceptId) -> ConsensusMetrics {
        // Group votes by concept
        let mut concept_weights: HashMap<ConceptId, f32> = HashMap::new();
        for vote in vote_breakdown {
            *concept_weights.entry(vote.concept.clone()).or_insert(0.0) += vote.effective_weight;
        }
        
        let winner_weight = concept_weights.get(winning_concept).copied().unwrap_or(0.0);
        let total_weight: f32 = concept_weights.values().sum();
        
        // Calculate consensus strength
        let consensus_strength = if total_weight > 0.0 {
            winner_weight / total_weight
        } else {
            0.0
        };
        
        // Calculate agreement level
        let supporting_votes = vote_breakdown.iter()
            .filter(|v| v.concept == *winning_concept)
            .count();
        let agreement_level = supporting_votes as f32 / vote_breakdown.len() as f32;
        
        // Calculate voting confidence
        let avg_confidence = vote_breakdown.iter()
            .filter(|v| v.concept == *winning_concept)
            .map(|v| v.original_confidence)
            .sum::<f32>() / supporting_votes.max(1) as f32;
        
        // Calculate unanimity level
        let unanimity_level = if concept_weights.len() == 1 { 1.0 } else { agreement_level };
        
        // Calculate confidence variance
        let confidences: Vec<_> = vote_breakdown.iter().map(|v| v.original_confidence).collect();
        let mean_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter()
            .map(|c| (c - mean_confidence).powi(2))
            .sum::<f32>() / confidences.len() as f32;
        
        ConsensusMetrics {
            consensus_strength,
            agreement_level,
            voting_confidence: avg_confidence,
            unanimity_level,
            confidence_variance: variance.sqrt(),
        }
    }
    
    /// Calculate quality metrics
    fn calculate_quality_metrics(&self, votes: &[ColumnVote], consensus_metrics: &ConsensusMetrics) -> ConsensusQualityMetrics {
        // Overall quality based on multiple factors
        let separation_quality = self.calculate_separation_quality(votes);
        let confidence_coherence = 1.0 - consensus_metrics.confidence_variance;
        let stability_measure = consensus_metrics.consensus_strength;
        let predictive_confidence = consensus_metrics.voting_confidence;
        
        let overall_quality = (separation_quality * 0.3 + 
                             confidence_coherence * 0.2 + 
                             stability_measure * 0.3 + 
                             predictive_confidence * 0.2).clamp(0.0, 1.0);
        
        ConsensusQualityMetrics {
            overall_quality,
            separation_quality,
            confidence_coherence,
            stability_measure,
            predictive_confidence,
        }
    }
    
    /// Calculate separation quality
    fn calculate_separation_quality(&self, votes: &[ColumnVote]) -> f32 {
        if votes.len() < 2 {
            return 1.0;
        }
        
        let confidences: Vec<_> = votes.iter().map(|v| v.confidence).collect();
        let max_conf = confidences.iter().cloned().fold(0.0f32, f32::max);
        let min_conf = confidences.iter().cloned().fold(1.0f32, f32::min);
        
        max_conf - min_conf
    }
    
    /// Get supporting columns for a concept
    fn get_supporting_columns(&self, vote_breakdown: &[VoteInfo], concept: &ConceptId) -> Vec<ColumnId> {
        vote_breakdown.iter()
            .filter(|v| v.concept == *concept)
            .map(|v| v.column_id)
            .collect()
    }
    
    /// Get dissenting columns for a concept
    fn get_dissenting_columns(&self, vote_breakdown: &[VoteInfo], concept: &ConceptId) -> Vec<ColumnId> {
        vote_breakdown.iter()
            .filter(|v| v.concept != *concept)
            .map(|v| v.column_id)
            .collect()
    }
    
    /// Find dominant column (highest confidence)
    fn find_dominant_column(&self, votes: &[ColumnVote]) -> Option<ColumnId> {
        votes.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .map(|v| v.column_id)
    }
    
    /// Calculate inhibition effectiveness
    fn calculate_inhibition_effectiveness(&self, votes: &[ColumnVote]) -> f32 {
        if votes.is_empty() {
            return 0.0;
        }
        
        let max_conf = votes.iter().map(|v| v.confidence).fold(0.0f32, f32::max);
        let avg_others = votes.iter()
            .filter(|v| v.confidence < max_conf)
            .map(|v| v.confidence)
            .sum::<f32>() / (votes.len() - 1).max(1) as f32;
        
        if max_conf > 0.0 {
            1.0 - (avg_others / max_conf)
        } else {
            0.0
        }
    }
    
    /// Calculate historical consistency
    fn calculate_historical_consistency(&self, _concept: &ConceptId) -> f32 {
        // Mock implementation - would compare with historical decisions
        0.8
    }
    
    /// Count participating columns
    fn count_participating_columns(&self, votes: &[ColumnVote]) -> usize {
        votes.iter()
            .filter(|v| v.confidence > 0.01) // Threshold for participation
            .count()
    }
    
    /// Validate configuration
    fn validate_config(config: &VotingConfig) -> Result<(), VotingError> {
        if config.consensus_threshold < 0.0 || config.consensus_threshold > 1.0 {
            return Err(VotingError::InvalidConfiguration(
                "Consensus threshold must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if config.quorum_threshold < 0.0 || config.quorum_threshold > 1.0 {
            return Err(VotingError::InvalidConfiguration(
                "Quorum threshold must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if config.min_participating_columns == 0 {
            return Err(VotingError::InvalidConfiguration(
                "Minimum participating columns must be at least 1".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Update voting state
    fn update_voting_state(&self, consensus_result: &ConsensusResult) {
        if let Ok(mut state) = self.state.lock() {
            state.total_votes_cast += consensus_result.total_votes as u64;
            state.current_round += 1;
            state.last_voting_time = Some(SystemTime::now());
            
            // Add consensus event to history
            let event = ConsensusEvent {
                timestamp: SystemTime::now(),
                participating_columns: consensus_result.vote_breakdown.iter().map(|v| v.column_id).collect(),
                winning_concept: consensus_result.winning_concept.clone(),
                consensus_strength: consensus_result.consensus_strength,
                agreement_level: consensus_result.agreement_level,
                voting_mode: self.voting_mode,
            };
            
            state.consensus_history.push(event);
            
            // Limit history size
            if state.consensus_history.len() > self.config.history_window_size {
                state.consensus_history.drain(0..10);
            }
        }
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&self, processing_time: Duration, consensus_result: &ConsensusResult) {
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.total_votes_processed += 1;
            
            // Update timing metrics
            let total_time = metrics.average_processing_time * (metrics.total_votes_processed - 1) as u32 + processing_time;
            metrics.average_processing_time = total_time / metrics.total_votes_processed as u32;
            
            if metrics.total_votes_processed == 1 || processing_time < metrics.fastest_vote {
                metrics.fastest_vote = processing_time;
            }
            if metrics.total_votes_processed == 1 || processing_time > metrics.slowest_vote {
                metrics.slowest_vote = processing_time;
            }
            
            // Update consensus metrics
            let has_consensus = consensus_result.consensus_strength > self.consensus_threshold;
            if has_consensus {
                metrics.consensus_agreement_rate = 
                    (metrics.consensus_agreement_rate * (metrics.total_votes_processed - 1) as f32 + 1.0) 
                    / metrics.total_votes_processed as f32;
            } else {
                metrics.consensus_agreement_rate = 
                    (metrics.consensus_agreement_rate * (metrics.total_votes_processed - 1) as f32) 
                    / metrics.total_votes_processed as f32;
            }
            
            // Update quality distribution
            let quality = consensus_result.quality_metrics.overall_quality;
            if quality > 0.8 {
                metrics.quality_distribution.high_quality_rate += 1.0;
            } else if quality > 0.5 {
                metrics.quality_distribution.medium_quality_rate += 1.0;
            } else {
                metrics.quality_distribution.low_quality_rate += 1.0;
            }
            
            // Normalize quality distribution
            let total = metrics.total_votes_processed as f32;
            metrics.quality_distribution.high_quality_rate /= total;
            metrics.quality_distribution.medium_quality_rate /= total;
            metrics.quality_distribution.low_quality_rate /= total;
        }
    }
    
    /// Adapt threshold based on consensus result
    fn adapt_threshold(&self, consensus_result: &ConsensusResult) -> Result<(), VotingError> {
        if let Ok(mut adapter) = self.threshold_adapter.lock() {
            let quality = consensus_result.quality_metrics.overall_quality;
            adapter.quality_history.push(quality);
            
            // Adapt threshold if quality is consistently different from target
            if adapter.quality_history.len() >= 10 {
                let recent_avg_quality = adapter.quality_history.iter().rev().take(5).sum::<f32>() / 5.0;
                let quality_diff = recent_avg_quality - adapter.target_quality;
                
                if quality_diff.abs() > 0.1 {
                    // Adjust threshold
                    let adjustment = quality_diff * adapter.adaptation_rate;
                    // In real implementation, would update the actual threshold
                    adapter.adaptation_stats.total_adaptations += 1;
                }
                
                // Limit history size
                if adapter.quality_history.len() > 50 {
                    adapter.quality_history.drain(0..10);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add consensus result to history
    fn add_to_history(&self, consensus_result: &ConsensusResult) {
        if let Ok(mut history) = self.voting_history.lock() {
            history.push(consensus_result.clone());
            
            // Limit history size
            if history.len() > self.config.history_window_size {
                history.drain(0..10);
            }
        }
    }
    
    /// Check if system is ready
    pub fn is_ready(&self) -> bool {
        true
    }
    
    /// Get voting mode
    pub fn get_voting_mode(&self) -> VotingMode {
        self.voting_mode
    }
    
    /// Get consensus threshold
    pub fn get_consensus_threshold(&self) -> f32 {
        self.consensus_threshold
    }
    
    /// Get total number of voters (columns)
    pub fn get_total_voters(&self) -> usize {
        4 // Fixed number of columns
    }
    
    /// Get configuration
    pub fn get_configuration(&self) -> &VotingConfig {
        &self.config
    }
    
    /// Get voting state
    pub fn get_voting_state(&self) -> VotingState {
        self.state.lock().unwrap().clone()
    }
    
    /// Set column expertise
    pub fn set_column_expertise(&mut self, column_id: ColumnId, expertise: f32) {
        self.column_expertise.insert(column_id, expertise.clamp(0.0, 1.0));
    }
    
    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics {
        self.threshold_adapter.lock().unwrap().adaptation_stats.clone()
    }
    
    /// Get voting history
    pub fn get_voting_history(&self) -> Vec<ConsensusResult> {
        self.voting_history.lock().unwrap().clone()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> VotingPerformanceMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }
}

// Supporting types and implementations

#[derive(Debug, Clone)]
struct ConceptVote {
    column_id: ColumnId,
    concept: ConceptId,
    confidence: f32,
    activation: f32,
}

#[derive(Debug, Clone)]
struct TieInfo {
    tie_occurred: bool,
    candidates: Vec<ConceptId>,
}

#[derive(Debug)]
struct ConsensusMetrics {
    consensus_strength: f32,
    agreement_level: f32,
    voting_confidence: f32,
    unanimity_level: f32,
    confidence_variance: f32,
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            voting_mode: VotingMode::WeightedConsensus,
            consensus_threshold: 0.6,
            quorum_threshold: 0.5,
            min_participating_columns: 2,
            enable_confidence_weighting: true,
            enable_expertise_adjustment: true,
            enable_adaptive_threshold: false,
            adaptation_sensitivity: 0.1,
            tie_breaking_strategy: TieBreakingStrategy::HighestConfidence,
            quality_threshold: 0.7,
            enable_temporal_consistency: true,
            history_window_size: 100,
        }
    }
}

impl ThresholdAdapter {
    fn new() -> Self {
        Self {
            adaptation_stats: AdaptationStatistics::default(),
            quality_history: Vec::new(),
            target_quality: 0.8,
            adaptation_rate: 0.05,
            stability_factor: 0.9,
        }
    }
}

impl Clone for VotingState {
    fn clone(&self) -> Self {
        Self {
            total_votes_cast: self.total_votes_cast,
            consensus_history: self.consensus_history.clone(),
            column_expertise: self.column_expertise.clone(),
            current_round: self.current_round,
            last_voting_time: self.last_voting_time,
        }
    }
}

impl Default for ExpertiseMetrics {
    fn default() -> Self {
        Self {
            accuracy_rate: 0.5,
            confidence_correlation: 0.5,
            consistency_score: 0.5,
            total_votes: 0,
            correct_predictions: 0,
        }
    }
}
```

## Verification Steps
1. Implement cortical voting system with multiple voting modes (democratic, weighted, expertise-based, adaptive)
2. Add consensus generation with configurable thresholds and quorum requirements
3. Implement comprehensive tie-breaking mechanisms with multiple strategies
4. Add adaptive threshold adjustment based on consensus quality feedback
5. Implement expertise tracking and weighting for columns based on historical performance
6. Add quality metrics calculation for consensus validation and optimization
7. Implement temporal consistency tracking and historical analysis
8. Add comprehensive performance monitoring and efficiency metrics

## Success Criteria
- [ ] Voting system initializes with proper configuration in <100ms
- [ ] Consensus generation achieves >95% agreement for clear scenarios
- [ ] Processing time stays under 2ms for 4-column voting (sub-2ms target)
- [ ] Weighted voting modes show appropriate influence distribution
- [ ] Tie-breaking mechanisms resolve conflicts deterministically
- [ ] Adaptive threshold adjustment improves consensus quality by >10%
- [ ] Expertise weighting influences decisions appropriately
- [ ] Quality metrics accurately reflect consensus strength and stability
- [ ] Temporal consistency tracking maintains decision coherence
- [ ] Integration with multi-column processor and lateral inhibition successful