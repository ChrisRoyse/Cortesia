# Task 04e: Create Neural Connection Relationship

**Estimated Time**: 8 minutes  
**Dependencies**: 04d_create_temporal_relationship.md  
**Next Task**: 04f_create_relationship_trait.md  

## Objective
Create the NeuralConnectionRelationship for neural pathway metadata.

## Single Action
Add NeuralConnectionRelationship struct to relationship_types.rs.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralConnectionRelationship {
    pub id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub connection_type: ConnectionType,
    pub connection_strength: f32,
    pub activation_pattern: Vec<f32>,
    pub ttfs_encoding: Option<f32>,
    pub synaptic_weight: f32,
    pub learning_rate: f32,
    pub usage_count: i32,
    pub last_activated: DateTime<Utc>,
    pub established_at: DateTime<Utc>,
    pub plasticity_enabled: bool,
    pub inhibitory_strength: f32,
    pub delay_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    Excitatory,       // Increases activation
    Inhibitory,       // Decreases activation
    Modulatory,       // Modifies other connections
    Gating,          // Controls information flow
    Feedback,        // Feedback connection
    Feedforward,     // Feedforward connection
}

impl NeuralConnectionRelationship {
    pub fn new(
        source_node_id: String,
        target_node_id: String,
        connection_type: ConnectionType,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            target_node_id,
            connection_type,
            connection_strength: 0.5,
            activation_pattern: Vec::new(),
            ttfs_encoding: None,
            synaptic_weight: 1.0,
            learning_rate: 0.01,
            usage_count: 0,
            last_activated: Utc::now(),
            established_at: Utc::now(),
            plasticity_enabled: true,
            inhibitory_strength: 0.0,
            delay_ms: 1.0,
        }
    }
    
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.connection_strength = strength.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_ttfs_encoding(mut self, encoding: f32) -> Self {
        self.ttfs_encoding = Some(encoding);
        self
    }
    
    pub fn set_inhibitory(mut self, strength: f32) -> Self {
        self.connection_type = ConnectionType::Inhibitory;
        self.inhibitory_strength = strength.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_delay(mut self, delay_ms: f32) -> Self {
        self.delay_ms = delay_ms.max(0.0);
        self
    }
    
    pub fn activate(&mut self, input_pattern: Vec<f32>) -> Vec<f32> {
        self.activation_pattern = input_pattern.clone();
        self.usage_count += 1;
        self.last_activated = Utc::now();
        
        // Apply synaptic transformation
        let mut output = input_pattern;
        for value in &mut output {
            *value *= self.synaptic_weight * self.connection_strength;
            
            // Apply inhibition if inhibitory connection
            if matches!(self.connection_type, ConnectionType::Inhibitory) {
                *value *= -self.inhibitory_strength;
            }
        }
        
        // Apply Hebbian learning if plasticity enabled
        if self.plasticity_enabled {
            self.apply_hebbian_learning();
        }
        
        output
    }
    
    pub fn apply_hebbian_learning(&mut self) {
        // Simple Hebbian learning: strengthen with use
        let learning_factor = self.learning_rate * 0.1;
        self.synaptic_weight += learning_factor;
        self.synaptic_weight = self.synaptic_weight.clamp(0.1, 2.0);
    }
    
    pub fn decay(&mut self, decay_rate: f32) {
        // Synaptic decay over time
        self.synaptic_weight *= 1.0 - decay_rate;
        self.synaptic_weight = self.synaptic_weight.max(0.1);
    }
    
    pub fn is_active(&self) -> bool {
        self.connection_strength > 0.1 && self.synaptic_weight > 0.1
    }
    
    pub fn get_effective_strength(&self) -> f32 {
        self.connection_strength * self.synaptic_weight
    }
    
    pub fn validate(&self) -> bool {
        !self.source_node_id.is_empty() &&
        !self.target_node_id.is_empty() &&
        self.source_node_id != self.target_node_id &&
        self.connection_strength >= 0.0 &&
        self.connection_strength <= 1.0 &&
        self.synaptic_weight >= 0.0 &&
        self.learning_rate >= 0.0 &&
        self.delay_ms >= 0.0
    }
}

#[cfg(test)]
mod neural_connection_tests {
    use super::*;
    
    #[test]
    fn test_neural_connection_creation() {
        let conn = NeuralConnectionRelationship::new(
            "neuron_a".to_string(),
            "neuron_b".to_string(),
            ConnectionType::Excitatory,
        );
        
        assert_eq!(conn.source_node_id, "neuron_a");
        assert_eq!(conn.target_node_id, "neuron_b");
        assert_eq!(conn.connection_type, ConnectionType::Excitatory);
        assert_eq!(conn.connection_strength, 0.5);
        assert!(conn.plasticity_enabled);
        assert!(conn.validate());
    }
    
    #[test]
    fn test_neural_activation() {
        let mut conn = NeuralConnectionRelationship::new(
            "input".to_string(),
            "output".to_string(),
            ConnectionType::Excitatory,
        ).with_strength(0.8);
        
        let input_pattern = vec![1.0, 0.8, 0.6];
        let output_pattern = conn.activate(input_pattern.clone());
        
        assert_eq!(conn.usage_count, 1);
        assert_eq!(conn.activation_pattern, input_pattern);
        
        // Output should be scaled by connection strength and synaptic weight
        let expected_scale = conn.connection_strength * conn.synaptic_weight;
        assert!((output_pattern[0] - expected_scale).abs() < 0.01);
    }
    
    #[test]
    fn test_inhibitory_connection() {
        let mut conn = NeuralConnectionRelationship::new(
            "inhibitor".to_string(),
            "target".to_string(),
            ConnectionType::Excitatory,
        ).set_inhibitory(0.7);
        
        assert_eq!(conn.connection_type, ConnectionType::Inhibitory);
        assert_eq!(conn.inhibitory_strength, 0.7);
        
        let input = vec![1.0];
        let output = conn.activate(input);
        
        // Should be negative due to inhibition
        assert!(output[0] < 0.0);
    }
    
    #[test]
    fn test_hebbian_learning() {
        let mut conn = NeuralConnectionRelationship::new(
            "learner_a".to_string(),
            "learner_b".to_string(),
            ConnectionType::Excitatory,
        );
        
        let initial_weight = conn.synaptic_weight;
        
        // Activate multiple times to trigger learning
        for _ in 0..5 {
            conn.activate(vec![1.0]);
        }
        
        assert!(conn.synaptic_weight > initial_weight);
        assert_eq!(conn.usage_count, 5);
    }
    
    #[test]
    fn test_synaptic_decay() {
        let mut conn = NeuralConnectionRelationship::new(
            "decaying".to_string(),
            "target".to_string(),
            ConnectionType::Excitatory,
        );
        
        let initial_weight = conn.synaptic_weight;
        conn.decay(0.1);
        
        assert!(conn.synaptic_weight < initial_weight);
        assert!(conn.synaptic_weight >= 0.1); // Should not decay below minimum
    }
    
    #[test]
    fn test_ttfs_encoding() {
        let conn = NeuralConnectionRelationship::new(
            "ttfs_source".to_string(),
            "ttfs_target".to_string(),
            ConnectionType::Modulatory,
        ).with_ttfs_encoding(0.85)
         .with_delay(2.5);
        
        assert_eq!(conn.ttfs_encoding, Some(0.85));
        assert_eq!(conn.delay_ms, 2.5);
        assert_eq!(conn.connection_type, ConnectionType::Modulatory);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run neural connection tests
cargo test neural_connection_tests
```

## Acceptance Criteria
- [ ] NeuralConnectionRelationship struct compiles
- [ ] Activation and learning mechanisms work
- [ ] Inhibitory connections function correctly
- [ ] Hebbian learning simulation works
- [ ] Tests pass

## Duration
6-8 minutes for neural connection implementation.