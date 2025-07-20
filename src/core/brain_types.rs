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

/// Types of logic gates for neural processing
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

/// Graph operation for neural structure prediction
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

/// Logic gate for neural computation
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

/// Activation pattern for neural propagation
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

    #[test]
    fn test_logic_gate_and() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![EntityKey::from(slotmap::KeyData::from_ffi(1)), EntityKey::from(slotmap::KeyData::from_ffi(2))];
        
        // Both inputs high
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.8);
        
        // One input low
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_or() {
        let mut gate = LogicGate::new(LogicGateType::Or, 0.5);
        gate.input_nodes = vec![EntityKey::from(slotmap::KeyData::from_ffi(1)), EntityKey::from(slotmap::KeyData::from_ffi(2))];
        
        // At least one input high
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9);
        
        // Both inputs low
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
    }

    #[test]
    fn test_entity_activation_decay() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Initial activation
        entity.activate(0.8, 0.1);
        assert!((entity.activation_state - 0.8).abs() < 0.01);
        
        // Activation adds up
        entity.activate(0.3, 0.1);
        assert!(entity.activation_state > 0.8);
        assert!(entity.activation_state <= 1.0);
    }

    #[test]
    fn test_hebbian_learning() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_weight = relationship.weight;
        relationship.strengthen(0.1);
        assert!(relationship.weight >= initial_weight, "Weight should increase or stay same: {} >= {}", relationship.weight, initial_weight);
        assert_eq!(relationship.activation_count, 1);
    }
}