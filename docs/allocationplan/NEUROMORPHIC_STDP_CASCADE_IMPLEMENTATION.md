# Neuromorphic STDP Learning Rules and Cascade Correlation Implementation

**Core Innovation**: Complete implementation of Spike-Timing-Dependent Plasticity (STDP) learning rules and Cascade Correlation Networks using ruv-FANN architectures for biological neural adaptation.

## Overview

This document provides the complete implementation of the two most critical missing neuromorphic components:

1. **STDP Learning Rules**: Biological synaptic adaptation based on spike timing
2. **Cascade Correlation Networks**: Dynamic neural network growth and adaptation

Both systems integrate with the ruv-FANN ecosystem and support the 4-column parallel processing architecture.

## STDP Learning Rules Implementation

### Core STDP Mathematical Model

The STDP learning rule follows the biological principle that synaptic strength changes based on the relative timing of pre- and post-synaptic spikes:

```
Δw = η × f(Δt)

where:
Δt = t_post - t_pre (spike timing difference)
η = learning rate
f(Δt) = A_+ × exp(-Δt/τ_+) if Δt > 0 (potentiation)
f(Δt) = -A_- × exp(Δt/τ_-) if Δt < 0 (depression)
```

### Rust Implementation

```rust
// src/snn_processing/stdp_learning.rs
use crate::ttfs_encoding::TTFSSpikePattern;
use crate::multi_column::ColumnVote;
use std::time::Duration;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct STDPLearningRule {
    // Biological time constants (milliseconds)
    pub tau_plus: f32,      // Potentiation time constant (typically 20ms)
    pub tau_minus: f32,     // Depression time constant (typically 20ms)
    
    // Learning amplitudes
    pub a_plus: f32,        // Potentiation amplitude (typically 0.005)
    pub a_minus: f32,       // Depression amplitude (typically 0.00525)
    
    // Learning rate and bounds
    pub learning_rate: f32, // Global learning rate (typically 0.01)
    pub min_weight: f32,    // Minimum synaptic weight (0.0)
    pub max_weight: f32,    // Maximum synaptic weight (1.0)
    
    // Biological realism parameters
    pub spike_trace_decay: f32,  // Exponential decay of spike traces
    pub homeostatic_factor: f32, // Weight normalization factor
}

impl Default for STDPLearningRule {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,    // 20ms potentiation window
            tau_minus: 20.0,   // 20ms depression window
            a_plus: 0.005,     // Moderate potentiation
            a_minus: 0.00525,  // Slightly stronger depression (realistic)
            learning_rate: 0.01,
            min_weight: 0.0,
            max_weight: 1.0,
            spike_trace_decay: 0.95,
            homeostatic_factor: 1.0,
        }
    }
}

impl STDPLearningRule {
    pub fn new_biological() -> Self {
        Self::default()
    }
    
    pub fn new_accelerated() -> Self {
        // Faster learning for real-time applications
        Self {
            tau_plus: 5.0,
            tau_minus: 5.0,
            a_plus: 0.02,
            a_minus: 0.021,
            learning_rate: 0.05,
            ..Default::default()
        }
    }
    
    /// Core STDP weight update function
    pub fn update_synaptic_weight(
        &self,
        current_weight: f32,
        pre_spike_time: Duration,
        post_spike_time: Duration,
    ) -> f32 {
        let delta_t = (post_spike_time.as_nanos() as f32 - pre_spike_time.as_nanos() as f32) / 1_000_000.0; // Convert to ms
        
        let weight_change = if delta_t > 0.0 {
            // Post-synaptic spike after pre-synaptic (Potentiation: LTP)
            self.a_plus * (-delta_t / self.tau_plus).exp()
        } else if delta_t < 0.0 {
            // Pre-synaptic spike after post-synaptic (Depression: LTD)
            -self.a_minus * (delta_t / self.tau_minus).exp()
        } else {
            0.0 // Simultaneous spikes - no change
        };
        
        // Apply learning rate and clamp to bounds
        let new_weight = current_weight + (self.learning_rate * weight_change);
        new_weight.clamp(self.min_weight, self.max_weight)
    }
    
    /// Update lateral inhibition strengths between cortical columns
    pub fn update_lateral_inhibition_strength(
        &self,
        current_strength: f32,
        winning_column_spike_time: Duration,
        losing_column_spike_time: Duration,
    ) -> f32 {
        // Strengthen inhibition if winning column spiked first
        self.update_synaptic_weight(current_strength, losing_column_spike_time, winning_column_spike_time)
    }
    
    /// Batch update all synaptic weights in a column
    pub fn update_column_weights(
        &self,
        column_weights: &mut HashMap<(NeuronId, NeuronId), f32>,
        spike_pattern: &TTFSSpikePattern,
        column_response_time: Duration,
    ) -> STDPUpdateResult {
        let mut updates_applied = 0;
        let mut total_weight_change = 0.0;
        
        for ((pre_neuron, post_neuron), current_weight) in column_weights.iter_mut() {
            // Find spike times for pre and post neurons
            if let (Some(pre_spike), Some(post_spike)) = (
                self.find_neuron_spike_time(*pre_neuron, spike_pattern),
                Some(column_response_time), // Post-synaptic spike is column response
            ) {
                let old_weight = *current_weight;
                *current_weight = self.update_synaptic_weight(old_weight, pre_spike, post_spike);
                
                total_weight_change += (*current_weight - old_weight).abs();
                updates_applied += 1;
            }
        }
        
        STDPUpdateResult {
            updates_applied,
            total_weight_change,
            average_weight: column_weights.values().sum::<f32>() / column_weights.len() as f32,
            convergence_indicator: if total_weight_change < 0.001 { true } else { false },
        }
    }
    
    fn find_neuron_spike_time(&self, neuron_id: NeuronId, spike_pattern: &TTFSSpikePattern) -> Option<Duration> {
        spike_pattern.spike_sequence
            .iter()
            .find(|spike| spike.neuron_id == neuron_id)
            .map(|spike| spike.timing)
    }
}

#[derive(Debug, Clone)]
pub struct STDPUpdateResult {
    pub updates_applied: usize,
    pub total_weight_change: f32,
    pub average_weight: f32,
    pub convergence_indicator: bool,
}

// Integration with Multi-Column Processor
impl crate::multi_column::MultiColumnProcessor {
    pub fn apply_stdp_learning(&mut self, spike_pattern: &TTFSSpikePattern, winning_column: &ColumnVote) -> Result<(), NeuromorphicError> {
        let stdp_rule = STDPLearningRule::new_biological();
        
        // Update winning column's internal weights
        self.update_column_internal_weights(&stdp_rule, spike_pattern, winning_column)?;
        
        // Update lateral inhibition connections
        self.update_lateral_connections(&stdp_rule, winning_column)?;
        
        // Apply homeostatic scaling to prevent runaway potentiation
        self.apply_homeostatic_scaling()?;
        
        Ok(())
    }
    
    fn update_column_internal_weights(
        &mut self,
        stdp_rule: &STDPLearningRule,
        spike_pattern: &TTFSSpikePattern,
        winning_column: &ColumnVote,
    ) -> Result<(), NeuromorphicError> {
        match winning_column.column_id {
            ColumnId::Semantic => {
                let result = stdp_rule.update_column_weights(
                    &mut self.semantic_column.internal_weights,
                    spike_pattern,
                    winning_column.processing_time,
                );
                self.semantic_column.last_stdp_update = Some(result);
            }
            ColumnId::Structural => {
                let result = stdp_rule.update_column_weights(
                    &mut self.structural_column.internal_weights,
                    spike_pattern,
                    winning_column.processing_time,
                );
                self.structural_column.last_stdp_update = Some(result);
            }
            ColumnId::Temporal => {
                let result = stdp_rule.update_column_weights(
                    &mut self.temporal_column.internal_weights,
                    spike_pattern,
                    winning_column.processing_time,
                );
                self.temporal_column.last_stdp_update = Some(result);
            }
            ColumnId::Exception => {
                let result = stdp_rule.update_column_weights(
                    &mut self.exception_column.internal_weights,
                    spike_pattern,
                    winning_column.processing_time,
                );
                self.exception_column.last_stdp_update = Some(result);
            }
        }
        
        Ok(())
    }
    
    fn update_lateral_connections(&mut self, stdp_rule: &STDPLearningRule, winning_column: &ColumnVote) -> Result<(), NeuromorphicError> {
        // Strengthen inhibition from winning column to other columns
        let winning_spike_time = Duration::from_nanos(winning_column.processing_time.as_nanos());
        
        for (column_pair, current_strength) in self.lateral_inhibition.connection_strengths.iter_mut() {
            if column_pair.0 == winning_column.column_id {
                // This connection originates from the winning column
                let losing_spike_time = self.get_column_response_time(column_pair.1);
                *current_strength = stdp_rule.update_lateral_inhibition_strength(
                    *current_strength,
                    winning_spike_time,
                    losing_spike_time,
                );
            }
        }
        
        Ok(())
    }
    
    fn apply_homeostatic_scaling(&mut self) -> Result<(), NeuromorphicError> {
        // Implement synaptic scaling to maintain network stability
        let target_average_weight = 0.5;
        
        // Apply to each column
        self.semantic_column.apply_homeostatic_scaling(target_average_weight)?;
        self.structural_column.apply_homeostatic_scaling(target_average_weight)?;
        self.temporal_column.apply_homeostatic_scaling(target_average_weight)?;
        self.exception_column.apply_homeostatic_scaling(target_average_weight)?;
        
        Ok(())
    }
}
```

## Cascade Correlation Networks Implementation

Cascade Correlation allows the neural network to grow new neurons dynamically when existing ones can't learn the required patterns.

### Rust Implementation

```rust
// src/snn_processing/cascade_correlation.rs
use crate::ruv_fann_integration::NetworkSelector;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct CascadeCorrelationNetwork {
    base_network: Box<dyn ruv_fann::NeuralNetwork>,
    candidate_pool: Vec<CandidateNeuron>,
    hidden_neurons: Vec<HiddenNeuron>,
    
    // Learning parameters
    correlation_threshold: f32,    // Minimum correlation to add neuron (0.4)
    max_hidden_neurons: usize,     // Maximum network size (1000)
    candidate_pool_size: usize,    // Number of candidates to test (8)
    max_epochs: usize,             // Training epochs per candidate (1000)
    
    // Performance tracking
    error_history: VecDeque<f32>,
    growth_history: Vec<NetworkGrowthEvent>,
    
    // Integration with ruv-FANN
    network_selector: NetworkSelector,
    ephemeral_networks: HashMap<ConceptId, Box<dyn ruv_fann::NeuralNetwork>>,
}

impl CascadeCorrelationNetwork {
    pub fn new_with_base_architecture(base_arch_id: usize) -> Result<Self, NeuromorphicError> {
        let base_network = ruv_fann::load_architecture(base_arch_id)?;
        
        Ok(Self {
            base_network,
            candidate_pool: Vec::new(),
            hidden_neurons: Vec::new(),
            correlation_threshold: 0.4,
            max_hidden_neurons: 1000,
            candidate_pool_size: 8,
            max_epochs: 1000,
            error_history: VecDeque::with_capacity(1000),
            growth_history: Vec::new(),
            network_selector: NetworkSelector::with_29_architectures(),
            ephemeral_networks: HashMap::new(),
        })
    }
    
    /// Core cascade correlation adaptation function
    pub async fn adapt_to_new_pattern(
        &mut self,
        input_pattern: &TTFSSpikePattern,
        desired_output: &ColumnVote,
    ) -> Result<NetworkGrowth, NeuromorphicError> {
        // Phase 1: Try learning with existing network
        let current_error = self.calculate_prediction_error(input_pattern, desired_output).await?;
        self.error_history.push_back(current_error);
        
        if current_error < self.error_threshold() {
            return Ok(NetworkGrowth::NoGrowthNeeded {
                current_error,
                threshold: self.error_threshold(),
            });
        }
        
        // Phase 2: Generate and train candidate neurons
        let candidates = self.generate_candidate_neurons(input_pattern).await?;
        let best_candidate = self.train_candidates_for_correlation(&candidates, current_error).await?;
        
        // Phase 3: Evaluate if candidate improves network performance
        if best_candidate.correlation_score > self.correlation_threshold {
            let neuron_id = self.add_neuron_to_network(best_candidate).await?;
            self.freeze_previous_weights(); // Cascade correlation principle
            
            // Record growth event
            self.growth_history.push(NetworkGrowthEvent {
                timestamp: std::time::Instant::now(),
                neuron_id,
                correlation_improvement: best_candidate.correlation_score,
                error_before: current_error,
                error_after: self.calculate_prediction_error(input_pattern, desired_output).await?,
                trigger_pattern: input_pattern.clone(),
            });
            
            Ok(NetworkGrowth::NeuronAdded {
                neuron_id,
                correlation_improvement: best_candidate.correlation_score,
                new_network_size: self.hidden_neurons.len(),
                performance_gain: current_error - self.error_history.back().unwrap_or(&0.0),
            })
        } else {
            // Phase 4: Try ephemeral network creation if correlation fails
            self.try_ephemeral_network_creation(input_pattern, desired_output).await
        }
    }
    
    async fn generate_candidate_neurons(&self, input_pattern: &TTFSSpikePattern) -> Result<Vec<CandidateNeuron>, NeuromorphicError> {
        let mut candidates = Vec::with_capacity(self.candidate_pool_size);
        
        for i in 0..self.candidate_pool_size {
            // Create candidate with random weights connected to all inputs and hidden neurons
            let mut weights = Vec::new();
            
            // Weights from input neurons
            for _ in 0..input_pattern.neural_features.len() {
                weights.push(self.random_weight());
            }
            
            // Weights from existing hidden neurons
            for _ in 0..self.hidden_neurons.len() {
                weights.push(self.random_weight());
            }
            
            // Select appropriate activation function based on ruv-FANN analysis
            let activation_function = self.network_selector.suggest_activation_function(input_pattern)?;
            
            candidates.push(CandidateNeuron {
                id: CandidateId(i),
                weights,
                activation_function,
                correlation_score: 0.0,
                training_error: f32::INFINITY,
                ruv_fann_compatibility: true,
            });
        }
        
        Ok(candidates)
    }
    
    async fn train_candidates_for_correlation(
        &mut self,
        candidates: &[CandidateNeuron],
        target_error: f32,
    ) -> Result<CandidateNeuron, NeuromorphicError> {
        let mut best_candidate = None;
        let mut best_correlation = 0.0;
        
        // Train each candidate to maximize correlation with residual error
        for candidate in candidates {
            let mut trained_candidate = candidate.clone();
            
            // Training loop using ruv-FANN optimization
            for epoch in 0..self.max_epochs {
                let correlation = self.calculate_error_correlation(&trained_candidate, target_error).await?;
                
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_candidate = Some(trained_candidate.clone());
                }
                
                // Update weights using gradient descent on correlation
                self.update_candidate_weights(&mut trained_candidate, correlation).await?;
                
                // Early stopping if correlation is high enough
                if correlation > self.correlation_threshold * 1.2 {
                    break;
                }
            }
        }
        
        best_candidate.ok_or(NeuromorphicError::NoCandidateFound)
    }
    
    async fn calculate_error_correlation(&self, candidate: &CandidateNeuron, target_error: f32) -> Result<f32, NeuromorphicError> {
        // Calculate correlation between candidate output and network residual error
        let mut correlation_sum = 0.0;
        let mut candidate_outputs = Vec::new();
        let mut residual_errors = Vec::new();
        
        // Use recent training examples for correlation calculation
        for spike_pattern in self.get_recent_training_patterns() {
            let candidate_output = self.calculate_candidate_output(candidate, spike_pattern)?;
            let network_output = self.base_network.forward(&spike_pattern.neural_features)?;
            let residual_error = target_error - network_output[0]; // Assuming single output
            
            candidate_outputs.push(candidate_output);
            residual_errors.push(residual_error);
        }
        
        // Calculate Pearson correlation coefficient
        if candidate_outputs.len() >= 2 {
            let correlation = self.pearson_correlation(&candidate_outputs, &residual_errors);
            Ok(correlation)
        } else {
            Ok(0.0)
        }
    }
    
    async fn add_neuron_to_network(&mut self, candidate: CandidateNeuron) -> Result<NeuronId, NeuromorphicError> {
        let neuron_id = NeuronId(self.hidden_neurons.len());
        
        let hidden_neuron = HiddenNeuron {
            id: neuron_id,
            weights: candidate.weights,
            activation_function: candidate.activation_function,
            frozen: false, // Will be frozen after output weights are trained
            correlation_score: candidate.correlation_score,
            creation_timestamp: std::time::Instant::now(),
        };
        
        self.hidden_neurons.push(hidden_neuron);
        
        // Train output weights for the new neuron
        self.train_output_weights().await?;
        
        Ok(neuron_id)
    }
    
    fn freeze_previous_weights(&mut self) {
        // Cascade correlation principle: freeze all weights except output weights
        for neuron in &mut self.hidden_neurons {
            neuron.frozen = true;
        }
    }
    
    async fn try_ephemeral_network_creation(
        &mut self,
        input_pattern: &TTFSSpikePattern,
        desired_output: &ColumnVote,
    ) -> Result<NetworkGrowth, NeuromorphicError> {
        // Create a specialized ephemeral network for this specific pattern type
        let optimal_architecture = self.network_selector.select_optimal_architecture(input_pattern);
        let ephemeral_network = ruv_fann::create_ephemeral_network(optimal_architecture.architecture_id)?;
        
        // Train the ephemeral network specifically for this pattern
        let training_result = ephemeral_network.train_for_pattern(input_pattern, desired_output).await?;
        
        if training_result.final_error < self.error_threshold() {
            self.ephemeral_networks.insert(input_pattern.concept_id, ephemeral_network);
            
            Ok(NetworkGrowth::EphemeralNetworkCreated {
                network_id: input_pattern.concept_id,
                architecture_used: optimal_architecture.architecture_id,
                final_error: training_result.final_error,
                specialization: format!("Pattern-specific {} network", optimal_architecture.name),
            })
        } else {
            Ok(NetworkGrowth::AdaptationFailed {
                attempted_methods: vec!["cascade_correlation", "ephemeral_network"],
                final_error: training_result.final_error,
                threshold: self.error_threshold(),
            })
        }
    }
    
    // Helper functions
    fn random_weight(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(-0.5..0.5)
    }
    
    fn error_threshold(&self) -> f32 {
        0.01 // 1% error threshold
    }
    
    fn pearson_correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;
        
        let numerator: f32 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f32 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f32 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandidateNeuron {
    pub id: CandidateId,
    pub weights: Vec<f32>,
    pub activation_function: ActivationFunction,
    pub correlation_score: f32,
    pub training_error: f32,
    pub ruv_fann_compatibility: bool,
}

#[derive(Debug, Clone)]
pub struct HiddenNeuron {
    pub id: NeuronId,
    pub weights: Vec<f32>,
    pub activation_function: ActivationFunction,
    pub frozen: bool,
    pub correlation_score: f32,
    pub creation_timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
pub enum NetworkGrowth {
    NoGrowthNeeded {
        current_error: f32,
        threshold: f32,
    },
    NeuronAdded {
        neuron_id: NeuronId,
        correlation_improvement: f32,
        new_network_size: usize,
        performance_gain: f32,
    },
    EphemeralNetworkCreated {
        network_id: ConceptId,
        architecture_used: usize,
        final_error: f32,
        specialization: String,
    },
    AdaptationFailed {
        attempted_methods: Vec<&'static str>,
        final_error: f32,
        threshold: f32,
    },
}

#[derive(Debug, Clone)]
pub struct NetworkGrowthEvent {
    pub timestamp: std::time::Instant,
    pub neuron_id: NeuronId,
    pub correlation_improvement: f32,
    pub error_before: f32,
    pub error_after: f32,
    pub trigger_pattern: TTFSSpikePattern,
}
```

## Integration with ruv-FANN Architectures

### Architecture Selection for Different Tasks

```rust
// src/ruv_fann_integration/network_selector.rs
impl NetworkSelector {
    pub fn select_for_stdp_learning(&self, spike_pattern: &TTFSSpikePattern) -> ruv_fann::Architecture {
        match spike_pattern.pattern_type() {
            PatternType::Semantic => ruv_fann::Architecture::MultilayerPerceptron, // #1
            PatternType::Structural => ruv_fann::Architecture::GraphNeuralNetwork, // #15
            PatternType::Temporal => ruv_fann::Architecture::LSTM, // #6
            PatternType::Exception => ruv_fann::Architecture::SparselyConnected, // #28
        }
    }
    
    pub fn select_for_cascade_correlation(&self, error_pattern: &ErrorPattern) -> ruv_fann::Architecture {
        if error_pattern.requires_memory() {
            ruv_fann::Architecture::GatedRecurrentUnit // #7
        } else if error_pattern.is_highly_nonlinear() {
            ruv_fann::Architecture::DeepBeliefNetwork // #11
        } else {
            ruv_fann::Architecture::CascadeCorrelationNetwork // #29
        }
    }
}
```

## Performance Monitoring and Metrics

```rust
// src/monitoring/neuromorphic_metrics.rs
pub struct NeuromorphicMetrics {
    pub stdp_updates_per_second: f32,
    pub cascade_growth_events: usize,
    pub synaptic_weight_distribution: HashMap<String, f32>,
    pub network_complexity_growth: Vec<(Instant, usize)>,
    pub learning_convergence_rate: f32,
    pub biological_realism_score: f32,
}

impl NeuromorphicMetrics {
    pub fn collect_stdp_metrics(&self, processor: &MultiColumnProcessor) -> STDPMetrics {
        STDPMetrics {
            average_weight_change: processor.get_average_weight_change(),
            potentiation_events: processor.count_potentiation_events(),
            depression_events: processor.count_depression_events(),
            homeostatic_scaling_applied: processor.homeostatic_scaling_count(),
            lateral_inhibition_strength: processor.lateral_inhibition.average_strength(),
        }
    }
    
    pub fn collect_cascade_metrics(&self, network: &CascadeCorrelationNetwork) -> CascadeMetrics {
        CascadeMetrics {
            total_neurons_added: network.hidden_neurons.len(),
            network_growth_rate: network.calculate_growth_rate(),
            correlation_threshold_hits: network.correlation_successes(),
            ephemeral_networks_created: network.ephemeral_networks.len(),
            adaptation_success_rate: network.calculate_adaptation_success_rate(),
        }
    }
}
```

## Testing and Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdp_biological_timing() {
        let stdp = STDPLearningRule::new_biological();
        
        // Test potentiation (post after pre)
        let pre_time = Duration::from_millis(0);
        let post_time = Duration::from_millis(10);
        let weight_change = stdp.update_synaptic_weight(0.5, pre_time, post_time);
        assert!(weight_change > 0.5); // Should be potentiated
        
        // Test depression (pre after post)
        let pre_time = Duration::from_millis(10);
        let post_time = Duration::from_millis(0);
        let weight_change = stdp.update_synaptic_weight(0.5, pre_time, post_time);
        assert!(weight_change < 0.5); // Should be depressed
    }
    
    #[tokio::test]
    async fn test_cascade_correlation_growth() {
        let mut network = CascadeCorrelationNetwork::new_with_base_architecture(1).unwrap();
        
        let difficult_pattern = create_xor_spike_pattern();
        let desired_output = create_xor_target_output();
        
        let growth_result = network.adapt_to_new_pattern(&difficult_pattern, &desired_output).await.unwrap();
        
        match growth_result {
            NetworkGrowth::NeuronAdded { correlation_improvement, .. } => {
                assert!(correlation_improvement > 0.4);
            }
            _ => panic!("Expected neuron addition for XOR problem"),
        }
    }
    
    #[test]
    fn test_ruv_fann_integration() {
        let selector = NetworkSelector::with_29_architectures();
        
        let semantic_pattern = create_semantic_spike_pattern();
        let architecture = selector.select_for_stdp_learning(&semantic_pattern);
        
        assert_eq!(architecture.id(), 1); // Should select MLP
        assert!(architecture.supports_stdp());
    }
}
```

This comprehensive implementation provides the missing STDP learning rules and cascade correlation networks that integrate seamlessly with the ruv-FANN ecosystem and multi-column parallel processing architecture.