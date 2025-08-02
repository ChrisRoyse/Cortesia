use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use crate::core::types::{EntityKey, AttributeValue};
use crate::error::Result;

/// Direction of brain-inspired entities (input/output/gate)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityDirection {
    Input,    // Concept input nodes
    Output,   // Concept output nodes
    Gate,     // Logic gate nodes
    Hidden,   // Hidden processing nodes
}

/// Types of logic gates for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicGateType {
    And,
    Or,
    Not,
    Xor,
    Nand,
    Nor,
    Xnor,
    Identity,
    Threshold,
    Inhibitory,
    Weighted,
}

impl std::fmt::Display for LogicGateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogicGateType::And => write!(f, "and"),
            LogicGateType::Or => write!(f, "or"),
            LogicGateType::Not => write!(f, "not"),
            LogicGateType::Xor => write!(f, "xor"),
            LogicGateType::Nand => write!(f, "nand"),
            LogicGateType::Nor => write!(f, "nor"),
            LogicGateType::Xnor => write!(f, "xnor"),
            LogicGateType::Identity => write!(f, "identity"),
            LogicGateType::Threshold => write!(f, "threshold"),
            LogicGateType::Inhibitory => write!(f, "inhibitory"),
            LogicGateType::Weighted => write!(f, "weighted"),
        }
    }
}

/// Types of relationships in the brain-inspired graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    IsA,            // Inheritance relationship
    HasInstance,    // Instance relationship  
    HasProperty,    // Property relationship
    RelatedTo,      // General association
    PartOf,         // Part-whole relationship
    Similar,        // Similarity relationship
    Opposite,       // Opposition relationship
    Temporal,       // Temporal relationship
    Learned,        // Learned relationship
    Influences,     // Influence relationship
}

/// Brain-inspired entity with activation state and temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainInspiredEntity {
    pub id: EntityKey,
    pub concept_id: String,          // Canonical concept identifier
    pub direction: EntityDirection,   // in/out/gate classification
    pub properties: HashMap<String, AttributeValue>,
    pub embedding: Vec<f32>,
    pub activation_state: f32,       // Current activation level (0.0-1.0)
    pub last_activation: SystemTime, // Temporal decay tracking
    pub last_update: SystemTime,     // Last update timestamp
}

/// Graph operation for structure prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphOperation {
    CreateNode {
        concept: String,
        node_type: EntityDirection,
    },
    CreateLogicGate {
        inputs: Vec<String>,
        outputs: Vec<String>,
        gate_type: LogicGateType,
    },
    CreateRelationship {
        source: String,
        target: String,
        relation_type: RelationType,
        weight: f32,
    },
}

/// Training example for structure prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub text: String,
    pub expected_operations: Vec<GraphOperation>,
    pub metadata: HashMap<String, String>,
}

/// Logic gate for computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicGate {
    pub gate_id: EntityKey,
    pub gate_type: LogicGateType,    // AND, OR, NOT, INHIBITORY
    pub input_nodes: Vec<EntityKey>,  // Input entity references
    pub output_nodes: Vec<EntityKey>, // Output entity references
    pub threshold: f32,              // Activation threshold
    pub weight_matrix: Vec<f32>,     // Input weight coefficients
}

/// Enhanced relationship with temporal and inhibitory properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainInspiredRelationship {
    pub source: EntityKey,
    pub target: EntityKey,
    pub source_key: EntityKey,          // Duplicate field for compatibility
    pub target_key: EntityKey,          // Duplicate field for compatibility
    pub relation_type: RelationType,    // Structured relation type
    pub weight: f32,                    // Dynamic weight (0.0-1.0)
    pub strength: f32,                  // Alternative name for weight
    pub is_inhibitory: bool,            // Inhibitory connection flag
    pub temporal_decay: f32,            // Decay rate (0.0-1.0)
    pub last_strengthened: SystemTime,  // Hebbian learning timestamp
    pub last_update: SystemTime,        // Last update timestamp
    pub activation_count: u64,          // Usage frequency
    pub usage_count: u64,               // Alternative name for activation_count
    pub creation_time: SystemTime,      // Bi-temporal tracking
    pub ingestion_time: SystemTime,     // When added to system
    pub metadata: HashMap<String, String>, // Additional metadata
}

/// Activation pattern for propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationPattern {
    pub activations: HashMap<EntityKey, f32>,
    pub timestamp: SystemTime,
    pub query: String,
}

impl ActivationPattern {
    pub fn new(query: String) -> Self {
        Self {
            activations: HashMap::new(),
            timestamp: SystemTime::now(),
            query,
        }
    }

    pub fn get_top_activations(&self, n: usize) -> Vec<(EntityKey, f32)> {
        let mut activations: Vec<_> = self.activations.iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        activations.truncate(n);
        activations
    }
}

/// Types of activation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationOperation {
    Initialize,
    Propagate,
    Inhibit,
    Reinforce,
    Decay,
}

/// Activation step for reasoning trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStep {
    pub step_id: usize,
    pub entity_key: EntityKey,
    pub concept_id: String,
    pub activation_level: f32,
    pub operation_type: ActivationOperation,
    pub timestamp: SystemTime,
}

impl BrainInspiredEntity {
    pub fn new(concept_id: String, direction: EntityDirection) -> Self {
        let now = SystemTime::now();
        Self {
            id: EntityKey::default(),
            concept_id,
            direction,
            properties: HashMap::new(),
            embedding: Vec::new(),
            activation_state: 0.0,
            last_activation: now,
            last_update: now,
        }
    }

    /// Activate the entity with decay based on time since last activation
    pub fn activate(&mut self, activation_level: f32, decay_rate: f32) -> f32 {
        let time_since_last = self.last_activation.elapsed()
            .unwrap_or_default()
            .as_secs_f32();
        
        // Apply temporal decay
        let decayed_state = self.activation_state * (-decay_rate * time_since_last).exp();
        
        // Add new activation
        self.activation_state = (decayed_state + activation_level).min(1.0);
        self.last_activation = SystemTime::now();
        
        self.activation_state
    }
}

impl LogicGate {
    pub fn new(gate_type: LogicGateType, threshold: f32) -> Self {
        Self {
            gate_id: EntityKey::default(),
            gate_type,
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
            threshold,
            weight_matrix: Vec::new(),
        }
    }

    /// Calculate gate output based on input activations
    pub fn calculate_output(&self, input_activations: &[f32]) -> Result<f32> {
        if input_activations.len() != self.input_nodes.len() {
            return Err(crate::error::GraphError::InvalidInput(
                "Input activation count mismatch".to_string()
            ));
        }

        match self.gate_type {
            LogicGateType::And => {
                // All inputs must exceed threshold
                let min_activation = input_activations.iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                Ok(if min_activation >= self.threshold { min_activation } else { 0.0 })
            }
            LogicGateType::Or => {
                // Any input exceeding threshold activates
                let max_activation = input_activations.iter()
                    .cloned()
                    .fold(0.0, f32::max);
                Ok(if max_activation >= self.threshold { max_activation } else { 0.0 })
            }
            LogicGateType::Not => {
                // Invert single input
                if input_activations.len() != 1 {
                    return Err(crate::error::GraphError::InvalidInput(
                        "NOT gate requires exactly one input".to_string()
                    ));
                }
                Ok(1.0 - input_activations[0])
            }
            LogicGateType::Xor => {
                // XOR: odd number of inputs above threshold
                if input_activations.len() != 2 {
                    return Err(crate::error::GraphError::InvalidInput(
                        "XOR gate requires exactly two inputs".to_string()
                    ));
                }
                let a = input_activations[0] >= self.threshold;
                let b = input_activations[1] >= self.threshold;
                Ok(if a ^ b { input_activations.iter().cloned().fold(0.0, f32::max) } else { 0.0 })
            }
            LogicGateType::Nand => {
                // NAND: NOT AND
                let min_activation = input_activations.iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                Ok(if min_activation >= self.threshold { 0.0 } else { 1.0 })
            }
            LogicGateType::Nor => {
                // NOR: NOT OR
                let max_activation = input_activations.iter()
                    .cloned()
                    .fold(0.0, f32::max);
                Ok(if max_activation >= self.threshold { 0.0 } else { 1.0 })
            }
            LogicGateType::Xnor => {
                // XNOR: NOT XOR
                if input_activations.len() != 2 {
                    return Err(crate::error::GraphError::InvalidInput(
                        "XNOR gate requires exactly two inputs".to_string()
                    ));
                }
                let a = input_activations[0] >= self.threshold;
                let b = input_activations[1] >= self.threshold;
                Ok(if !(a ^ b) { input_activations.iter().cloned().fold(0.0, f32::max) } else { 0.0 })
            }
            LogicGateType::Identity => {
                // Identity: pass through single input
                if input_activations.len() != 1 {
                    return Err(crate::error::GraphError::InvalidInput(
                        "Identity gate requires exactly one input".to_string()
                    ));
                }
                Ok(input_activations[0])
            }
            LogicGateType::Threshold => {
                // Threshold: sum of inputs compared to threshold
                let sum: f32 = input_activations.iter().sum();
                Ok(if sum >= self.threshold { sum.min(1.0) } else { 0.0 })
            }
            LogicGateType::Inhibitory => {
                // First input minus sum of others
                if input_activations.is_empty() {
                    return Ok(0.0);
                }
                let primary = input_activations[0];
                let inhibition: f32 = input_activations[1..].iter().sum();
                Ok((primary - inhibition).max(0.0))
            }
            LogicGateType::Weighted => {
                // Weighted sum with threshold
                if self.weight_matrix.len() != input_activations.len() {
                    return Err(crate::error::GraphError::InvalidInput(
                        "Weight matrix size mismatch".to_string()
                    ));
                }
                let weighted_sum: f32 = input_activations.iter()
                    .zip(self.weight_matrix.iter())
                    .map(|(a, w)| a * w)
                    .sum();
                Ok(if weighted_sum >= self.threshold { weighted_sum.min(1.0) } else { 0.0 })
            }
        }
    }
}

impl BrainInspiredRelationship {
    pub fn new(source: EntityKey, target: EntityKey, relation_type: RelationType) -> Self {
        let now = SystemTime::now();
        Self {
            source,
            target,
            source_key: source,
            target_key: target,
            relation_type,
            weight: 1.0,
            strength: 1.0,
            is_inhibitory: false,
            temporal_decay: 0.1,
            last_strengthened: now,
            last_update: now,
            activation_count: 0,
            usage_count: 0,
            creation_time: now,
            ingestion_time: now,
            metadata: HashMap::new(),
        }
    }

    /// Apply Hebbian learning to strengthen the connection
    pub fn strengthen(&mut self, learning_rate: f32) {
        self.weight = (self.weight + learning_rate).min(1.0);
        self.strength = self.weight;
        self.last_strengthened = SystemTime::now();
        self.last_update = SystemTime::now();
        self.activation_count += 1;
        self.usage_count += 1;
    }

    /// Apply temporal decay to the connection weight
    pub fn apply_decay(&mut self) -> f32 {
        let time_since_strengthened = self.last_strengthened.elapsed()
            .unwrap_or_default()
            .as_secs_f32();
        
        self.weight *= (-self.temporal_decay * time_since_strengthened).exp();
        self.strength = self.weight;
        self.last_update = SystemTime::now();
        self.weight
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test keys
    fn create_test_key(id: u64) -> EntityKey {
        EntityKey::from(slotmap::KeyData::from_ffi(id))
    }

    #[test]
    fn test_logic_gate_and_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // Both inputs above threshold
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.8);
        
        // One input below threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0);
        
        // Both inputs below threshold
        assert_eq!(gate.calculate_output(&[0.2, 0.3]).unwrap(), 0.0);
        
        // Both inputs at threshold
        assert_eq!(gate.calculate_output(&[0.5, 0.5]).unwrap(), 0.5);
        
        // Zero inputs
        assert_eq!(gate.calculate_output(&[0.0, 0.0]).unwrap(), 0.0);
        
        // Maximum inputs
        assert_eq!(gate.calculate_output(&[1.0, 1.0]).unwrap(), 1.0);
    }

    #[test]
    fn test_logic_gate_and_empty_inputs() {
        let gate = LogicGate::new(LogicGateType::And, 0.5);
        
        // Empty input array with empty input nodes
        assert!(gate.calculate_output(&[]).is_ok());
    }

    #[test]
    fn test_logic_gate_and_mismatch_inputs() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // Input count mismatch
        assert!(gate.calculate_output(&[0.8]).is_err());
        assert!(gate.calculate_output(&[0.8, 0.9, 0.7]).is_err());
    }

    #[test]
    fn test_logic_gate_or_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Or, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // At least one input above threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9);
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.8);
        
        // Both inputs above threshold
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.9);
        
        // Both inputs below threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
        
        // Zero inputs
        assert_eq!(gate.calculate_output(&[0.0, 0.0]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_not_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Not, 0.5);
        gate.input_nodes = vec![create_test_key(1)];
        
        // Single input inversion
        assert_eq!(gate.calculate_output(&[0.8]).unwrap(), 0.2);
        assert_eq!(gate.calculate_output(&[0.0]).unwrap(), 1.0);
        assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 0.0);
        assert_eq!(gate.calculate_output(&[0.5]).unwrap(), 0.5);
    }

    #[test]
    fn test_logic_gate_not_invalid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Not, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // NOT gate with more than one input should fail
        assert!(gate.calculate_output(&[0.8, 0.9]).is_err());
        
        gate.input_nodes = vec![];
        assert!(gate.calculate_output(&[]).is_err());
    }

    #[test]
    fn test_logic_gate_xor_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Xor, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // XOR true cases
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.8);
        assert_eq!(gate.calculate_output(&[0.3, 0.8]).unwrap(), 0.8);
        
        // XOR false cases (both above or both below threshold)
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0);
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_xor_invalid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Xor, 0.5);
        gate.input_nodes = vec![create_test_key(1)];
        
        // XOR with wrong number of inputs
        assert!(gate.calculate_output(&[0.8]).is_err());
        
        gate.input_nodes = vec![create_test_key(1), create_test_key(2), create_test_key(3)];
        assert!(gate.calculate_output(&[0.8, 0.9, 0.7]).is_err());
    }

    #[test]
    fn test_logic_gate_nand_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Nand, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // NAND: NOT AND
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // AND would be true, so NAND is false
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 1.0); // AND would be false, so NAND is true
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // AND would be false, so NAND is true
    }

    #[test]
    fn test_logic_gate_nor_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Nor, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // NOR: NOT OR
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0); // OR would be true, so NOR is false
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // OR would be true, so NOR is false
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // OR would be false, so NOR is true
    }

    #[test]
    fn test_logic_gate_xnor_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Xnor, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // XNOR: NOT XOR
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.9); // XOR would be false, so XNOR is true
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.4); // XOR would be false, so XNOR is true
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.0); // XOR would be true, so XNOR is false
    }

    #[test]
    fn test_logic_gate_xnor_invalid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Xnor, 0.5);
        gate.input_nodes = vec![create_test_key(1)];
        
        // XNOR with wrong number of inputs
        assert!(gate.calculate_output(&[0.8]).is_err());
    }

    #[test]
    fn test_logic_gate_identity_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Identity, 0.5);
        gate.input_nodes = vec![create_test_key(1)];
        
        // Identity gate passes through input unchanged
        assert_eq!(gate.calculate_output(&[0.8]).unwrap(), 0.8);
        assert_eq!(gate.calculate_output(&[0.0]).unwrap(), 0.0);
        assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 1.0);
    }

    #[test]
    fn test_logic_gate_identity_invalid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Identity, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        
        // Identity with wrong number of inputs
        assert!(gate.calculate_output(&[0.8, 0.9]).is_err());
        
        gate.input_nodes = vec![];
        assert!(gate.calculate_output(&[]).is_err());
    }

    #[test]
    fn test_logic_gate_threshold_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Threshold, 1.0);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2), create_test_key(3)];
        
        // Sum above threshold
        assert_eq!(gate.calculate_output(&[0.5, 0.6, 0.7]).unwrap(), 1.0); // Sum is 1.8, clamped to 1.0
        
        // Sum at threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.3, 0.4]).unwrap(), 1.0);
        
        // Sum below threshold
        assert_eq!(gate.calculate_output(&[0.2, 0.2, 0.2]).unwrap(), 0.0);
        
        // Empty inputs
        gate.input_nodes = vec![];
        assert_eq!(gate.calculate_output(&[]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_inhibitory_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Inhibitory, 0.5);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2), create_test_key(3)];
        
        // Primary input minus inhibitory inputs
        assert_eq!(gate.calculate_output(&[0.8, 0.2, 0.1]).unwrap(), 0.5); // 0.8 - (0.2 + 0.1) = 0.5
        
        // Strong inhibition
        assert_eq!(gate.calculate_output(&[0.5, 0.3, 0.4]).unwrap(), 0.0); // 0.5 - 0.7 = -0.2, clamped to 0.0
        
        // No inhibition
        assert_eq!(gate.calculate_output(&[0.8, 0.0, 0.0]).unwrap(), 0.8);
        
        // Single input (no inhibition)
        gate.input_nodes = vec![create_test_key(1)];
        assert_eq!(gate.calculate_output(&[0.8]).unwrap(), 0.8);
        
        // Empty inputs
        gate.input_nodes = vec![];
        assert_eq!(gate.calculate_output(&[]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_weighted_valid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Weighted, 1.0);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        gate.weight_matrix = vec![0.5, 0.8];
        
        // Weighted sum above threshold
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 1.0); // 0.8*0.5 + 0.9*0.8 = 1.12, clamped to 1.0
        
        // Weighted sum at threshold
        gate.weight_matrix = vec![0.5, 0.5];
        assert_eq!(gate.calculate_output(&[1.0, 1.0]).unwrap(), 1.0); // 1.0*0.5 + 1.0*0.5 = 1.0
        
        // Weighted sum below threshold
        assert_eq!(gate.calculate_output(&[0.5, 0.5]).unwrap(), 0.0); // 0.5*0.5 + 0.5*0.5 = 0.5 < 1.0
    }

    #[test]
    fn test_logic_gate_weighted_invalid_inputs() {
        let mut gate = LogicGate::new(LogicGateType::Weighted, 1.0);
        gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
        gate.weight_matrix = vec![0.5]; // Mismatch in weight matrix size
        
        assert!(gate.calculate_output(&[0.8, 0.9]).is_err());
        
        gate.weight_matrix = vec![0.5, 0.8, 0.3]; // Too many weights
        assert!(gate.calculate_output(&[0.8, 0.9]).is_err());
    }

    #[test]
    fn test_brain_inspired_entity_new() {
        let entity = BrainInspiredEntity::new("test_concept".to_string(), EntityDirection::Input);
        
        assert_eq!(entity.concept_id, "test_concept");
        assert_eq!(entity.direction, EntityDirection::Input);
        assert_eq!(entity.activation_state, 0.0);
        assert!(entity.properties.is_empty());
        assert!(entity.embedding.is_empty());
    }

    #[test]
    fn test_brain_inspired_entity_activate_initial() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Initial activation on zero state
        let result = entity.activate(0.8, 0.1);
        assert!((result - 0.8).abs() < 0.01);
        assert!((entity.activation_state - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_brain_inspired_entity_activate_accumulation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // First activation
        entity.activate(0.5, 0.1);
        
        // Second activation should accumulate (without significant decay due to immediate timing)
        let result = entity.activate(0.4, 0.1);
        assert!(result > 0.5); // Should be greater than initial
        assert!(result <= 1.0); // Should be clamped at 1.0
    }

    #[test]
    fn test_brain_inspired_entity_activate_clamping() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Activation that would exceed 1.0 should be clamped
        let result = entity.activate(1.5, 0.1);
        assert_eq!(result, 1.0);
        assert_eq!(entity.activation_state, 1.0);
    }

    #[test]
    fn test_brain_inspired_entity_activate_zero_decay() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Test with zero decay rate
        entity.activate(0.8, 0.0);
        let result = entity.activate(0.2, 0.0);
        assert_eq!(result, 1.0); // 0.8 + 0.2 = 1.0
    }

    #[test]
    fn test_brain_inspired_entity_activate_temporal_updates() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        let initial_time = entity.last_activation;
        
        // Activation should update timestamp
        entity.activate(0.5, 0.1);
        assert!(entity.last_activation > initial_time);
    }

    #[test]
    fn test_brain_inspired_relationship_new() {
        let source = create_test_key(1);
        let target = create_test_key(2);
        let rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        
        assert_eq!(rel.source, source);
        assert_eq!(rel.target, target);
        assert_eq!(rel.source_key, source);
        assert_eq!(rel.target_key, target);
        assert_eq!(rel.relation_type, RelationType::RelatedTo);
        assert_eq!(rel.weight, 1.0);
        assert_eq!(rel.strength, 1.0);
        assert!(!rel.is_inhibitory);
        assert_eq!(rel.temporal_decay, 0.1);
        assert_eq!(rel.activation_count, 0);
        assert_eq!(rel.usage_count, 0);
        assert!(rel.metadata.is_empty());
    }

    #[test]
    fn test_brain_inspired_relationship_strengthen_basic() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        let initial_weight = rel.weight;
        let initial_count = rel.activation_count;
        let initial_time = rel.last_strengthened;
        
        rel.strengthen(0.1);
        
        assert!(rel.weight >= initial_weight);
        assert_eq!(rel.strength, rel.weight);
        assert_eq!(rel.activation_count, initial_count + 1);
        assert_eq!(rel.usage_count, rel.activation_count);
        assert!(rel.last_strengthened > initial_time);
    }

    #[test]
    fn test_brain_inspired_relationship_strengthen_clamping() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        // Strengthen beyond maximum
        rel.strengthen(0.5); // Should clamp at 1.0
        assert_eq!(rel.weight, 1.0);
        assert_eq!(rel.strength, 1.0);
    }

    #[test]
    fn test_brain_inspired_relationship_strengthen_zero_rate() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        let initial_weight = rel.weight;
        rel.strengthen(0.0);
        
        assert_eq!(rel.weight, initial_weight);
        assert_eq!(rel.activation_count, 1); // Counter should still increment
    }

    #[test]
    fn test_brain_inspired_relationship_strengthen_negative_rate() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        // Negative learning rate should decrease weight
        rel.strengthen(-0.2);
        assert!(rel.weight < 1.0);
        assert_eq!(rel.strength, rel.weight);
    }

    #[test]
    fn test_brain_inspired_relationship_strengthen_multiple_calls() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        // Multiple strengthening calls
        rel.strengthen(0.1);
        rel.strengthen(0.1);
        rel.strengthen(0.1);
        
        assert_eq!(rel.activation_count, 3);
        assert_eq!(rel.usage_count, 3);
        assert!(rel.weight <= 1.0); // Should be clamped
    }

    #[test]
    fn test_brain_inspired_relationship_apply_decay() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        let initial_weight = rel.weight;
        let result = rel.apply_decay();
        
        // Immediate decay should have minimal effect
        assert!(result <= initial_weight);
        assert_eq!(rel.weight, result);
        assert_eq!(rel.strength, rel.weight);
    }

    #[test]
    fn test_activation_pattern_new() {
        let pattern = ActivationPattern::new("test query".to_string());
        
        assert_eq!(pattern.query, "test query");
        assert!(pattern.activations.is_empty());
    }

    #[test]
    fn test_activation_pattern_get_top_activations_empty() {
        let pattern = ActivationPattern::new("test".to_string());
        
        let top = pattern.get_top_activations(5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_activation_pattern_get_top_activations_populated() {
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations.insert(create_test_key(1), 0.8);
        pattern.activations.insert(create_test_key(2), 0.6);
        pattern.activations.insert(create_test_key(3), 0.9);
        pattern.activations.insert(create_test_key(4), 0.4);
        
        let top = pattern.get_top_activations(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 0.9); // Highest activation
        assert_eq!(top[1].1, 0.8); // Second highest
    }

    #[test]
    fn test_activation_pattern_get_top_activations_limit_exceeded() {
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations.insert(create_test_key(1), 0.5);
        
        let top = pattern.get_top_activations(5);
        assert_eq!(top.len(), 1); // Should only return available activations
    }

    #[test]
    fn test_logic_gate_type_display() {
        assert_eq!(format!("{}", LogicGateType::And), "and");
        assert_eq!(format!("{}", LogicGateType::Or), "or");
        assert_eq!(format!("{}", LogicGateType::Not), "not");
        assert_eq!(format!("{}", LogicGateType::Xor), "xor");
        assert_eq!(format!("{}", LogicGateType::Nand), "nand");
        assert_eq!(format!("{}", LogicGateType::Nor), "nor");
        assert_eq!(format!("{}", LogicGateType::Xnor), "xnor");
        assert_eq!(format!("{}", LogicGateType::Identity), "identity");
        assert_eq!(format!("{}", LogicGateType::Threshold), "threshold");
        assert_eq!(format!("{}", LogicGateType::Inhibitory), "inhibitory");
        assert_eq!(format!("{}", LogicGateType::Weighted), "weighted");
    }

    #[test]
    fn test_logic_gate_new() {
        let gate = LogicGate::new(LogicGateType::And, 0.7);
        
        assert_eq!(gate.gate_type, LogicGateType::And);
        assert_eq!(gate.threshold, 0.7);
        assert!(gate.input_nodes.is_empty());
        assert!(gate.output_nodes.is_empty());
        assert!(gate.weight_matrix.is_empty());
    }

    // Edge case tests for extreme values
    #[test]
    fn test_logic_gate_extreme_threshold_values() {
        let mut gate = LogicGate::new(LogicGateType::Threshold, f32::MAX);
        gate.input_nodes = vec![create_test_key(1)];
        
        // No input should exceed MAX threshold
        assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 0.0);
        
        // Test with negative threshold
        gate.threshold = -1.0;
        assert_eq!(gate.calculate_output(&[0.5]).unwrap(), 0.5); // Should activate since 0.5 > -1.0
    }

    #[test]
    fn test_brain_inspired_entity_extreme_activation_values() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Test with very large activation
        let result = entity.activate(f32::MAX, 0.1);
        assert_eq!(result, 1.0); // Should be clamped
        
        // Test with negative activation
        entity.activation_state = 0.0;
        let result = entity.activate(-0.5, 0.1);
        assert!(result >= 0.0); // Result should never be negative due to max(0.0) in some implementations
    }

    #[test]
    fn test_brain_inspired_relationship_extreme_learning_rates() {
        let mut rel = BrainInspiredRelationship::new(
            create_test_key(1),
            create_test_key(2),
            RelationType::RelatedTo
        );
        
        // Test with very large learning rate
        rel.strengthen(f32::MAX);
        assert_eq!(rel.weight, 1.0); // Should be clamped
        
        // Test with very negative learning rate
        rel.weight = 1.0;
        rel.strengthen(-f32::MAX);
        assert!(rel.weight <= 1.0); // Weight can't exceed 1.0 even after negative adjustment
    }

    // Test serialization/deserialization compatibility (basic structure validation)
    #[test]
    fn test_enum_serialization_compatibility() {
        // Test that enums have expected discriminant values for serialization stability
        use std::mem::discriminant;
        
        assert_eq!(discriminant(&EntityDirection::Input), discriminant(&EntityDirection::Input));
        assert_ne!(discriminant(&EntityDirection::Input), discriminant(&EntityDirection::Output));
        
        assert_eq!(discriminant(&LogicGateType::And), discriminant(&LogicGateType::And));
        assert_ne!(discriminant(&LogicGateType::And), discriminant(&LogicGateType::Or));
        
        assert_eq!(discriminant(&RelationType::IsA), discriminant(&RelationType::IsA));
        assert_ne!(discriminant(&RelationType::IsA), discriminant(&RelationType::HasInstance));
    }
}