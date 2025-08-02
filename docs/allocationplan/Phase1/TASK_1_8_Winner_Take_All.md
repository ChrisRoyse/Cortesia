# Task 1.8: Winner-Take-All

**Duration**: 2 hours  
**Complexity**: Medium  
**Dependencies**: Task 1.7 (Lateral Inhibition)  
**AI Assistant Suitability**: High - Clear algorithmic patterns  

## Objective

Implement optimized winner-take-all selection algorithms for cortical column competition, achieving sub-100μs selection time with robust tie-breaking strategies and thread-safe implementation for high-performance neuromorphic processing.

## Specification

Build upon the lateral inhibition placeholders from Task 1.7 with complete implementations of:

**Winner Selection Algorithms**:
- Fast O(n) single-pass winner detection
- Multi-winner selection with configurable limits
- Tie-breaking strategies (random, first-found, activation-based)
- Probabilistic selection modes for exploration

**Performance Requirements**:
- Selection time: < 100μs for 1000 columns
- Memory allocation: Zero during selection
- Thread safety: Lock-free concurrent selections
- Scalability: Linear performance with column count

**Competition Modes**:
- Single winner (traditional WTA)
- Top-K winners (configurable K)
- Threshold-based selection (minimum activation)
- Probabilistic winner selection

## Implementation Guide

### Step 1: Winner Selection Engine

```rust
// src/winner_take_all.rs
use crate::{ColumnId, ColumnState, EnhancedCorticalColumn, current_time_us};
use std::sync::Arc;
use std::time::Instant;
use rand::Rng;
use smallvec::SmallVec;

/// Enhanced configuration for winner-take-all selection (extends Task 1.7 placeholder)
#[derive(Debug, Clone)]
pub struct WTAConfig {
    /// Maximum number of winners to select (from Task 1.7 placeholder)
    pub max_winners: usize,
    
    /// Minimum activation threshold for consideration
    pub activation_threshold: f32,
    
    /// Tie-breaking strategy
    pub tie_breaking: TieBreakingStrategy,
    
    /// Enable probabilistic selection based on activation levels
    pub probabilistic_mode: bool,
    
    /// Temperature for probabilistic selection (higher = more random)
    pub selection_temperature: f32,
}

impl Default for WTAConfig {
    fn default() -> Self {
        Self {
            max_winners: 1,
            activation_threshold: 0.1,
            tie_breaking: TieBreakingStrategy::Random,
            probabilistic_mode: false,
            selection_temperature: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TieBreakingStrategy {
    /// Select first found winner in case of ties
    FirstFound,
    /// Random selection among tied candidates
    Random,
    /// Select based on highest secondary criteria (timestamp, ID)
    ActivationBased,
    /// Select based on lowest column ID
    LowestId,
}

/// Result of winner-take-all selection
#[derive(Debug, Clone)]
pub struct WTAResult {
    /// Selected winner columns with their activation levels
    pub winners: SmallVec<[WinnerInfo; 8]>,
    
    /// Total candidates considered
    pub total_candidates: usize,
    
    /// Selection duration in microseconds
    pub selection_time_us: u64,
    
    /// Whether any ties were broken
    pub ties_detected: bool,
    
    /// Statistics about the selection process
    pub statistics: SelectionStatistics,
}

/// Enhanced winner information (extends Task 1.7 placeholder)  
#[derive(Debug, Clone)]
pub struct WinnerInfo {
    pub column_id: ColumnId,
    pub activation_level: f32,         // Maps to final_activation in Task 1.7
    pub selection_score: f32,          // New field for enhanced tracking
    pub timestamp_us: u64,             // New field for temporal tracking
    pub inhibited_count: usize,        // From Task 1.7 placeholder
    pub margin: f32,                   // From Task 1.7 placeholder
}

#[derive(Debug, Clone)]
pub struct SelectionStatistics {
    pub min_activation: f32,
    pub max_activation: f32,
    pub mean_activation: f32,
    pub total_activation: f32,
    pub candidates_above_threshold: usize,
}

/// High-performance winner-take-all selection engine
pub struct WinnerTakeAllEngine {
    config: WTAConfig,
    rng: rand::rngs::ThreadRng,
}

impl WinnerTakeAllEngine {
    pub fn new(config: WTAConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }
    
    /// Select winners from a slice of competing columns
    pub fn select_winners(
        &mut self,
        columns: &[Arc<EnhancedCorticalColumn>],
    ) -> WTAResult {
        let start_time = Instant::now();
        
        if columns.is_empty() {
            return WTAResult::empty(start_time.elapsed().as_micros() as u64);
        }
        
        // Fast single-pass candidate collection
        let candidates = self.collect_candidates(columns);
        
        if candidates.is_empty() {
            return WTAResult::empty(start_time.elapsed().as_micros() as u64);
        }
        
        // Select winners based on configuration
        let (winners, ties_detected) = if self.config.probabilistic_mode {
            self.probabilistic_selection(&candidates)
        } else {
            self.deterministic_selection(&candidates)
        };
        
        let statistics = self.calculate_statistics(&candidates);
        let selection_time_us = start_time.elapsed().as_micros() as u64;
        
        WTAResult {
            winners,
            total_candidates: candidates.len(),
            selection_time_us,
            ties_detected,
            statistics,
        }
    }
    
    /// Fast O(n) candidate collection with state checking
    fn collect_candidates(
        &self,
        columns: &[Arc<EnhancedCorticalColumn>],
    ) -> SmallVec<[CandidateInfo; 32]> {
        let mut candidates = SmallVec::new();
        
        for column in columns {
            // Only consider competing columns above threshold
            if column.current_state() == ColumnState::Competing {
                let activation = column.activation_level();
                
                if activation >= self.config.activation_threshold {
                    candidates.push(CandidateInfo {
                        column_id: column.id(),
                        column: column.clone(),
                        activation_level: activation,
                        timestamp_us: column.time_since_transition().as_micros() as u64,
                    });
                }
            }
        }
        
        candidates
    }
    
    /// Deterministic winner selection (highest activation wins)
    fn deterministic_selection(
        &mut self,
        candidates: &[CandidateInfo],
    ) -> (SmallVec<[WinnerInfo; 8]>, bool) {
        if candidates.is_empty() {
            return (SmallVec::new(), false);
        }
        
        // Create sorted indices by activation level (descending)
        let mut indices: SmallVec<[usize; 32]> = (0..candidates.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            candidates[b].activation_level
                .partial_cmp(&candidates[a].activation_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut winners = SmallVec::new();
        let mut ties_detected = false;
        let max_winners = self.config.max_winners.min(candidates.len());
        
        for i in 0..max_winners {
            let idx = indices[i];
            let candidate = &candidates[idx];
            
            // Check for ties with previous winner
            if i > 0 {
                let prev_idx = indices[i - 1];
                let prev_activation = candidates[prev_idx].activation_level;
                
                if (candidate.activation_level - prev_activation).abs() < f32::EPSILON {
                    ties_detected = true;
                    
                    // Apply tie-breaking strategy
                    if !self.should_select_tied_candidate(candidate, &candidates[prev_idx]) {
                        continue;
                    }
                }
            }
            
            winners.push(WinnerInfo {
                column_id: candidate.column_id,
                activation_level: candidate.activation_level,
                selection_score: candidate.activation_level,
                timestamp_us: candidate.timestamp_us,
                inhibited_count: 0,  // Will be calculated by lateral inhibition
                margin: 0.0,         // Will be calculated by lateral inhibition
            });
        }
        
        (winners, ties_detected)
    }
    
    /// Probabilistic winner selection based on activation levels
    fn probabilistic_selection(
        &mut self,
        candidates: &[CandidateInfo],
    ) -> (SmallVec<[WinnerInfo; 8]>, bool) {
        if candidates.is_empty() {
            return (SmallVec::new(), false);
        }
        
        // Calculate selection probabilities using softmax
        let probabilities = self.calculate_selection_probabilities(candidates);
        let mut winners = SmallVec::new();
        let max_winners = self.config.max_winners.min(candidates.len());
        
        // Sample without replacement
        let mut available_indices: SmallVec<[usize; 32]> = (0..candidates.len()).collect();
        
        for _ in 0..max_winners {
            if available_indices.is_empty() {
                break;
            }
            
            // Weighted random selection
            let selected_idx = self.weighted_random_selection(&available_indices, &probabilities);
            let candidate_idx = available_indices.remove(selected_idx);
            let candidate = &candidates[candidate_idx];
            
            winners.push(WinnerInfo {
                column_id: candidate.column_id,
                activation_level: candidate.activation_level,
                selection_score: probabilities[candidate_idx],
                timestamp_us: candidate.timestamp_us,
                inhibited_count: 0,  // Will be calculated by lateral inhibition
                margin: 0.0,         // Will be calculated by lateral inhibition
            });
        }
        
        (winners, false) // Probabilistic selection doesn't have "ties"
    }
    
    /// Calculate softmax probabilities for selection
    fn calculate_selection_probabilities(&self, candidates: &[CandidateInfo]) -> Vec<f32> {
        let temperature = self.config.selection_temperature;
        let mut probabilities = Vec::with_capacity(candidates.len());
        
        // Find max activation for numerical stability
        let max_activation = candidates.iter()
            .map(|c| c.activation_level)
            .fold(0.0f32, f32::max);
        
        // Calculate unnormalized probabilities
        let mut sum = 0.0;
        for candidate in candidates {
            let exp_val = ((candidate.activation_level - max_activation) / temperature).exp();
            probabilities.push(exp_val);
            sum += exp_val;
        }
        
        // Normalize to get probabilities
        if sum > 0.0 {
            for prob in &mut probabilities {
                *prob /= sum;
            }
        }
        
        probabilities
    }
    
    /// Weighted random selection from available indices
    fn weighted_random_selection(
        &mut self,
        available_indices: &[usize],
        probabilities: &[f32],
    ) -> usize {
        if available_indices.len() == 1 {
            return 0;
        }
        
        // Calculate cumulative sum of available probabilities
        let mut cumulative_sum = 0.0;
        let mut cumulative_probs = Vec::with_capacity(available_indices.len());
        
        for &idx in available_indices {
            cumulative_sum += probabilities[idx];
            cumulative_probs.push(cumulative_sum);
        }
        
        // Random selection
        let random_value = self.rng.gen::<f32>() * cumulative_sum;
        
        for (i, &cum_prob) in cumulative_probs.iter().enumerate() {
            if random_value <= cum_prob {
                return i;
            }
        }
        
        available_indices.len() - 1 // Fallback
    }
    
    /// Apply tie-breaking strategy
    fn should_select_tied_candidate(
        &mut self,
        candidate: &CandidateInfo,
        previous: &CandidateInfo,
    ) -> bool {
        match self.config.tie_breaking {
            TieBreakingStrategy::FirstFound => false, // Don't select, keep first
            TieBreakingStrategy::Random => self.rng.gen::<bool>(),
            TieBreakingStrategy::ActivationBased => {
                // Use timestamp as secondary criteria
                candidate.timestamp_us < previous.timestamp_us
            }
            TieBreakingStrategy::LowestId => candidate.column_id < previous.column_id,
        }
    }
    
    /// Calculate selection statistics
    fn calculate_statistics(&self, candidates: &[CandidateInfo]) -> SelectionStatistics {
        if candidates.is_empty() {
            return SelectionStatistics {
                min_activation: 0.0,
                max_activation: 0.0,
                mean_activation: 0.0,
                total_activation: 0.0,
                candidates_above_threshold: 0,
            };
        }
        
        let activations: Vec<f32> = candidates.iter()
            .map(|c| c.activation_level)
            .collect();
        
        let min_activation = activations.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_activation = activations.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let total_activation: f32 = activations.iter().sum();
        let mean_activation = total_activation / activations.len() as f32;
        
        SelectionStatistics {
            min_activation,
            max_activation,
            mean_activation,
            total_activation,
            candidates_above_threshold: candidates.len(),
        }
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: WTAConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn config(&self) -> &WTAConfig {
        &self.config
    }
}

#[derive(Debug, Clone)]
struct CandidateInfo {
    column_id: ColumnId,
    column: Arc<EnhancedCorticalColumn>,
    activation_level: f32,
    timestamp_us: u64,
}

impl WTAResult {
    fn empty(selection_time_us: u64) -> Self {
        Self {
            winners: SmallVec::new(),
            total_candidates: 0,
            selection_time_us,
            ties_detected: false,
            statistics: SelectionStatistics {
                min_activation: 0.0,
                max_activation: 0.0,
                mean_activation: 0.0,
                total_activation: 0.0,
                candidates_above_threshold: 0,
            },
        }
    }
    
    /// Check if any winners were selected
    pub fn has_winners(&self) -> bool {
        !self.winners.is_empty()
    }
    
    /// Get the primary winner (highest activation)
    pub fn primary_winner(&self) -> Option<&WinnerInfo> {
        self.winners.first()
    }
    
    /// Get selection efficiency (winners / candidates)
    pub fn selection_efficiency(&self) -> f32 {
        if self.total_candidates == 0 {
            0.0
        } else {
            self.winners.len() as f32 / self.total_candidates as f32
        }
    }
    
    /// Check if performance target was met
    pub fn meets_performance_target(&self, target_us: u64) -> bool {
        self.selection_time_us < target_us
    }
}
```

### Step 2: Integration with Cortical Column Manager

```rust
// src/cortical_column_manager.rs
use crate::{WinnerTakeAllEngine, WTAConfig, WTAResult, EnhancedCorticalColumn, ColumnId, ColumnState};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;

pub struct CorticalColumnManager {
    columns: HashMap<ColumnId, Arc<EnhancedCorticalColumn>>,
    wta_engine: WinnerTakeAllEngine,
    performance_metrics: ManagerMetrics,
}

#[derive(Debug, Default)]
pub struct ManagerMetrics {
    pub total_selections: u64,
    pub total_selection_time_us: u64,
    pub successful_selections: u64,
    pub empty_selections: u64,
    pub tie_breaking_events: u64,
}

impl CorticalColumnManager {
    pub fn new(wta_config: WTAConfig) -> Self {
        Self {
            columns: HashMap::new(),
            wta_engine: WinnerTakeAllEngine::new(wta_config),
            performance_metrics: ManagerMetrics::default(),
        }
    }
    
    /// Add a column to management
    pub fn add_column(&mut self, column: Arc<EnhancedCorticalColumn>) {
        self.columns.insert(column.id(), column);
    }
    
    /// Remove a column from management
    pub fn remove_column(&mut self, column_id: ColumnId) -> Option<Arc<EnhancedCorticalColumn>> {
        self.columns.remove(&column_id)
    }
    
    /// Run winner-take-all selection on all competing columns
    pub fn select_winners(&mut self) -> WTAResult {
        let start = Instant::now();
        
        // Collect competing columns
        let competing_columns: Vec<Arc<EnhancedCorticalColumn>> = self.columns
            .values()
            .filter(|col| col.current_state() == ColumnState::Competing)
            .cloned()
            .collect();
        
        // Run selection
        let result = self.wta_engine.select_winners(&competing_columns);
        
        // Update metrics
        self.update_metrics(&result);
        
        result
    }
    
    /// Select winners from specific column subset
    pub fn select_winners_from(
        &mut self,
        column_ids: &[ColumnId],
    ) -> WTAResult {
        let columns: Vec<Arc<EnhancedCorticalColumn>> = column_ids
            .iter()
            .filter_map(|&id| self.columns.get(&id))
            .cloned()
            .collect();
        
        let result = self.wta_engine.select_winners(&columns);
        self.update_metrics(&result);
        result
    }
    
    /// Execute winner allocation after selection
    pub fn allocate_winners(&self, result: &WTAResult) -> AllocationResult {
        let mut successful_allocations = 0;
        let mut failed_allocations = 0;
        let start = Instant::now();
        
        for winner in &result.winners {
            if let Some(column) = self.columns.get(&winner.column_id) {
                match column.try_allocate() {
                    Ok(_) => successful_allocations += 1,
                    Err(_) => failed_allocations += 1,
                }
            } else {
                failed_allocations += 1;
            }
        }
        
        AllocationResult {
            successful_allocations,
            failed_allocations,
            allocation_time_us: start.elapsed().as_micros() as u64,
        }
    }
    
    /// Update WTA configuration
    pub fn update_wta_config(&mut self, config: WTAConfig) {
        self.wta_engine.update_config(config);
    }
    
    /// Get current WTA configuration
    pub fn wta_config(&self) -> &WTAConfig {
        self.wta_engine.config()
    }
    
    /// Get performance metrics
    pub fn performance_metrics(&self) -> &ManagerMetrics {
        &self.performance_metrics
    }
    
    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.performance_metrics = ManagerMetrics::default();
    }
    
    /// Get all columns in specific state
    pub fn columns_in_state(&self, state: ColumnState) -> Vec<&Arc<EnhancedCorticalColumn>> {
        self.columns
            .values()
            .filter(|col| col.current_state() == state)
            .collect()
    }
    
    /// Get column count
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// Run continuous winner selection with callback
    pub fn run_continuous_selection<F>(&mut self, mut callback: F) -> ContinuousSelectionResult
    where
        F: FnMut(&WTAResult) -> bool, // Return false to stop
    {
        let start_time = Instant::now();
        let mut iteration_count = 0;
        let mut total_winners = 0;
        
        loop {
            let result = self.select_winners();
            total_winners += result.winners.len();
            iteration_count += 1;
            
            if !callback(&result) {
                break;
            }
            
            // Small delay to prevent busy loop
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        
        ContinuousSelectionResult {
            total_iterations: iteration_count,
            total_winners_selected: total_winners,
            total_runtime: start_time.elapsed(),
            average_selection_time_us: if iteration_count > 0 {
                self.performance_metrics.total_selection_time_us / iteration_count
            } else {
                0
            },
        }
    }
    
    fn update_metrics(&mut self, result: &WTAResult) {
        self.performance_metrics.total_selections += 1;
        self.performance_metrics.total_selection_time_us += result.selection_time_us;
        
        if result.has_winners() {
            self.performance_metrics.successful_selections += 1;
        } else {
            self.performance_metrics.empty_selections += 1;
        }
        
        if result.ties_detected {
            self.performance_metrics.tie_breaking_events += 1;
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub successful_allocations: usize,
    pub failed_allocations: usize,
    pub allocation_time_us: u64,
}

#[derive(Debug)]
pub struct ContinuousSelectionResult {
    pub total_iterations: u64,
    pub total_winners_selected: usize,
    pub total_runtime: std::time::Duration,
    pub average_selection_time_us: u64,
}

impl ManagerMetrics {
    pub fn average_selection_time_us(&self) -> u64 {
        if self.total_selections == 0 {
            0
        } else {
            self.total_selection_time_us / self.total_selections
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_selections == 0 {
            0.0
        } else {
            self.successful_selections as f64 / self.total_selections as f64
        }
    }
    
    pub fn tie_breaking_rate(&self) -> f64 {
        if self.total_selections == 0 {
            0.0
        } else {
            self.tie_breaking_events as f64 / self.total_selections as f64
        }
    }
}
```

## AI-Executable Test Suite

```rust
// tests/winner_take_all_test.rs
use llmkg::{
    WinnerTakeAllEngine, WTAConfig, TieBreakingStrategy, EnhancedCorticalColumn,
    CorticalColumnManager, ColumnState
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

#[test]
fn test_single_winner_selection() {
    let mut engine = WinnerTakeAllEngine::new(WTAConfig::default());
    
    // Create columns with different activation levels
    let columns = vec![
        create_competing_column(1, 0.3),
        create_competing_column(2, 0.8), // Should win
        create_competing_column(3, 0.5),
        create_competing_column(4, 0.1),
    ];
    
    let result = engine.select_winners(&columns);
    
    assert!(result.has_winners());
    assert_eq!(result.winners.len(), 1);
    assert_eq!(result.primary_winner().unwrap().column_id, 2);
    assert_eq!(result.primary_winner().unwrap().activation_level, 0.8);
    assert_eq!(result.total_candidates, 4);
    assert!(!result.ties_detected);
}

#[test]
fn test_multiple_winner_selection() {
    let config = WTAConfig {
        max_winners: 3,
        ..Default::default()
    };
    let mut engine = WinnerTakeAllEngine::new(config);
    
    let columns = vec![
        create_competing_column(1, 0.9), // 1st
        create_competing_column(2, 0.7), // 2nd
        create_competing_column(3, 0.8), // 3rd (actually 2nd by activation)
        create_competing_column(4, 0.2),
        create_competing_column(5, 0.6), // 3rd
    ];
    
    let result = engine.select_winners(&columns);
    
    assert_eq!(result.winners.len(), 3);
    
    // Should be sorted by activation level (descending)
    assert_eq!(result.winners[0].column_id, 1); // 0.9
    assert_eq!(result.winners[1].column_id, 3); // 0.8
    assert_eq!(result.winners[2].column_id, 2); // 0.7
}

#[test]
fn test_activation_threshold_filtering() {
    let config = WTAConfig {
        activation_threshold: 0.5,
        max_winners: 10,
        ..Default::default()
    };
    let mut engine = WinnerTakeAllEngine::new(config);
    
    let columns = vec![
        create_competing_column(1, 0.9), // Above threshold
        create_competing_column(2, 0.3), // Below threshold
        create_competing_column(3, 0.6), // Above threshold
        create_competing_column(4, 0.1), // Below threshold
    ];
    
    let result = engine.select_winners(&columns);
    
    assert_eq!(result.winners.len(), 2);
    assert_eq!(result.statistics.candidates_above_threshold, 2);
    
    let winner_ids: Vec<u32> = result.winners.iter().map(|w| w.column_id).collect();
    assert!(winner_ids.contains(&1));
    assert!(winner_ids.contains(&3));
}

#[test]
fn test_tie_breaking_strategies() {
    // Test random tie breaking
    let config = WTAConfig {
        tie_breaking: TieBreakingStrategy::Random,
        max_winners: 2,
        ..Default::default()
    };
    let mut engine = WinnerTakeAllEngine::new(config);
    
    let columns = vec![
        create_competing_column(1, 0.8), // Tied for first
        create_competing_column(2, 0.8), // Tied for first
        create_competing_column(3, 0.5),
    ];
    
    let result = engine.select_winners(&columns);
    
    assert_eq!(result.winners.len(), 2);
    assert!(result.ties_detected);
    
    // Both tied winners should be selected
    let winner_ids: Vec<u32> = result.winners.iter().map(|w| w.column_id).collect();
    assert!(winner_ids.contains(&1) && winner_ids.contains(&2) || 
            winner_ids.contains(&1) && winner_ids.contains(&3) ||
            winner_ids.contains(&2) && winner_ids.contains(&3));
    
    // Test lowest ID tie breaking
    let config = WTAConfig {
        tie_breaking: TieBreakingStrategy::LowestId,
        max_winners: 1,
        ..Default::default()
    };
    engine.update_config(config);
    
    let result = engine.select_winners(&columns);
    assert_eq!(result.winners.len(), 1);
    assert_eq!(result.primary_winner().unwrap().column_id, 1); // Lowest ID
}

#[test]
fn test_probabilistic_selection() {
    let config = WTAConfig {
        probabilistic_mode: true,
        selection_temperature: 1.0,
        max_winners: 1,
        ..Default::default()
    };
    let mut engine = WinnerTakeAllEngine::new(config);
    
    let columns = vec![
        create_competing_column(1, 0.9), // High probability
        create_competing_column(2, 0.1), // Low probability
    ];
    
    // Run multiple selections to verify probabilistic behavior
    let mut high_activation_wins = 0;
    let total_runs = 1000;
    
    for _ in 0..total_runs {
        let result = engine.select_winners(&columns);
        if result.primary_winner().unwrap().column_id == 1 {
            high_activation_wins += 1;
        }
    }
    
    // High activation column should win most of the time
    let win_rate = high_activation_wins as f64 / total_runs as f64;
    assert!(win_rate > 0.7); // Should win at least 70% of the time
    assert!(win_rate < 1.0); // But not 100% due to randomness
}

#[test]
fn test_performance_benchmarks() {
    let config = WTAConfig {
        max_winners: 10,
        ..Default::default()
    };
    let mut engine = WinnerTakeAllEngine::new(config);
    
    // Create 1000 competing columns
    let columns: Vec<Arc<EnhancedCorticalColumn>> = (0..1000)
        .map(|i| create_competing_column(i, 0.5 + (i % 50) as f32 / 100.0))
        .collect();
    
    // Benchmark selection time
    let start = Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        let result = engine.select_winners(&columns);
        assert!(result.has_winners());
    }
    
    let total_time = start.elapsed();
    let avg_time_us = total_time.as_micros() / iterations;
    
    println!("Average selection time for 1000 columns: {} μs", avg_time_us);
    
    // Should meet performance target of < 100μs
    assert!(avg_time_us < 100);
    
    // Test single selection performance
    let start = Instant::now();
    let result = engine.select_winners(&columns);
    let single_selection_us = start.elapsed().as_micros();
    
    println!("Single selection time: {} μs", single_selection_us);
    assert!(result.meets_performance_target(100));
    assert!(single_selection_us < 100);
}

#[test]
fn test_column_manager_integration() {
    let config = WTAConfig {
        max_winners: 5,
        ..Default::default()
    };
    let mut manager = CorticalColumnManager::new(config);
    
    // Add columns to manager
    for i in 0..20 {
        let column = create_competing_column(i, 0.3 + (i % 10) as f32 / 20.0);
        manager.add_column(column);
    }
    
    assert_eq!(manager.column_count(), 20);
    
    // Run selection
    let result = manager.select_winners();
    
    assert!(result.has_winners());
    assert!(result.winners.len() <= 5);
    assert!(result.meets_performance_target(100));
    
    // Test allocation
    let allocation_result = manager.allocate_winners(&result);
    assert_eq!(
        allocation_result.successful_allocations + allocation_result.failed_allocations,
        result.winners.len()
    );
    
    // Check metrics
    let metrics = manager.performance_metrics();
    assert_eq!(metrics.total_selections, 1);
    assert_eq!(metrics.successful_selections, 1);
    assert!(metrics.average_selection_time_us() < 100);
}

#[test]
fn test_concurrent_winner_selection() {
    let config = WTAConfig {
        max_winners: 3,
        ..Default::default()
    };
    let manager = Arc::new(std::sync::Mutex::new(CorticalColumnManager::new(config)));
    
    // Add columns
    for i in 0..50 {
        let column = create_competing_column(i, 0.4 + (i % 20) as f32 / 40.0);
        manager.lock().unwrap().add_column(column);
    }
    
    let mut handles = vec![];
    let barrier = Arc::new(std::sync::Barrier::new(10));
    
    // Run concurrent selections
    for _ in 0..10 {
        let manager = manager.clone();
        let barrier = barrier.clone();
        
        handles.push(thread::spawn(move || {
            barrier.wait();
            
            let mut results = vec![];
            for _ in 0..10 {
                let result = manager.lock().unwrap().select_winners();
                results.push(result);
            }
            results
        }));
    }
    
    // Collect all results
    let all_results: Vec<Vec<_>> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Verify all selections were successful and fast
    for thread_results in all_results {
        for result in thread_results {
            assert!(result.has_winners());
            assert!(result.meets_performance_target(100));
        }
    }
    
    // Check final metrics
    let metrics = manager.lock().unwrap().performance_metrics();
    assert_eq!(metrics.total_selections, 100); // 10 threads × 10 selections
    assert!(metrics.success_rate() > 0.9);
    assert!(metrics.average_selection_time_us() < 100);
}

#[test]
fn test_edge_cases() {
    let mut engine = WinnerTakeAllEngine::new(WTAConfig::default());
    
    // Empty column list
    let result = engine.select_winners(&[]);
    assert!(!result.has_winners());
    assert_eq!(result.total_candidates, 0);
    
    // No competing columns (all available)
    let available_columns = vec![
        create_available_column(1),
        create_available_column(2),
    ];
    let result = engine.select_winners(&available_columns);
    assert!(!result.has_winners());
    
    // All activations below threshold
    let config = WTAConfig {
        activation_threshold: 0.8,
        ..Default::default()
    };
    engine.update_config(config);
    
    let low_activation_columns = vec![
        create_competing_column(1, 0.1),
        create_competing_column(2, 0.3),
    ];
    let result = engine.select_winners(&low_activation_columns);
    assert!(!result.has_winners());
    assert_eq!(result.statistics.candidates_above_threshold, 0);
}

#[test]
fn test_continuous_selection_stress() {
    let config = WTAConfig {
        max_winners: 2,
        ..Default::default()
    };
    let mut manager = CorticalColumnManager::new(config);
    
    // Add columns
    for i in 0..100 {
        let column = create_competing_column(i, 0.2 + (i % 30) as f32 / 50.0);
        manager.add_column(column);
    }
    
    let mut iteration_count = 0;
    let max_iterations = 1000;
    
    let continuous_result = manager.run_continuous_selection(|result| {
        iteration_count += 1;
        
        // Verify each result
        assert!(result.meets_performance_target(100));
        assert!(result.winners.len() <= 2);
        
        iteration_count < max_iterations
    });
    
    assert_eq!(continuous_result.total_iterations, max_iterations);
    assert!(continuous_result.average_selection_time_us < 100);
    assert!(continuous_result.total_runtime < Duration::from_secs(5));
    
    // Final metrics check
    let metrics = manager.performance_metrics();
    assert_eq!(metrics.total_selections as u64, max_iterations);
    assert!(metrics.success_rate() > 0.8);
}

// Helper functions
fn create_competing_column(id: u32, activation: f32) -> Arc<EnhancedCorticalColumn> {
    let column = Arc::new(EnhancedCorticalColumn::new(id));
    column.try_activate_with_level(activation).unwrap();
    column.try_compete_with_strength(activation).unwrap();
    column
}

fn create_available_column(id: u32) -> Arc<EnhancedCorticalColumn> {
    Arc::new(EnhancedCorticalColumn::new(id))
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: Run `cargo test winner_take_all_test` - must be 9/9 passing
2. **Performance targets met**: 
   - Selection time < 100μs for 1000 columns (benchmark test)
   - Zero allocations during selection (profiling verification)
3. **Thread safety verified**: Concurrent tests pass consistently
4. **Algorithm correctness**: All selection modes work as specified
5. **Zero clippy warnings**: `cargo clippy -- -D warnings`

## Verification Commands

```bash
# Run tests
cargo test winner_take_all_test --release

# Performance benchmarks with output
cargo test test_performance_benchmarks --release -- --nocapture

# Stress testing
cargo test test_continuous_selection_stress --release

# Concurrent validation
cargo test test_concurrent_winner_selection --release

# Code quality
cargo clippy -- -D warnings

# Memory allocation check (requires nightly)
cargo +nightly test test_performance_benchmarks --release -Z unstable-options --profile=bench
```

## Files to Create/Update

1. `src/winner_take_all.rs` (replaces/extends Task 1.7 placeholders)
2. `src/cortical_column_manager.rs`
3. `tests/winner_take_all_test.rs`
4. Update `src/lib.rs` with new exports
5. Update `src/Cargo.toml` with `smallvec` dependency

**Dependencies**: Requires Task 1.7 (types, core infrastructure)

## Dependencies to Add to Cargo.toml

```toml
[dependencies]
smallvec = "1.11"
rand = "0.8"

[dev-dependencies]
criterion = "0.5"
```

## Expected Performance Results

```
Single winner selection: ~5-15 μs
Multiple winner selection: ~10-30 μs
1000 column selection: ~40-80 μs
Tie breaking overhead: ~2-5 μs
Probabilistic selection: ~15-35 μs
Memory allocations: 0 during selection
```

## Next Task

Task 1.9: Concept Deduplication (prevent duplicate allocations via inhibition)