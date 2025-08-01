use std::collections::HashMap;
use crate::core::types::EntityKey;
use serde::{Serialize, Deserialize};

/// Configuration for activation propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub decay_rate: f32,
    pub decay_factor: f32,
    pub inhibition_strength: f32,
    pub default_threshold: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.001,
            decay_rate: 0.1,
            decay_factor: 0.95,
            inhibition_strength: 2.0,
            default_threshold: 0.5,
        }
    }
}

/// Result of activation propagation
#[derive(Debug, Clone)]
pub struct PropagationResult {
    pub final_activations: HashMap<EntityKey, f32>,
    pub iterations_completed: usize,
    pub converged: bool,
    pub activation_trace: Vec<crate::core::brain_types::ActivationStep>,
    pub total_energy: f32,
}

/// Statistics about the activation network
#[derive(Debug, Clone)]
pub struct ActivationStatistics {
    pub total_entities: usize,
    pub total_gates: usize,
    pub total_relationships: usize,
    pub active_entities: usize,
    pub inhibitory_connections: usize,
    pub average_activation: f32,
}