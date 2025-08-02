# Task 03f: Create Neural Pathway Struct

**Estimated Time**: 9 minutes  
**Dependencies**: 03e_create_version_node_struct.md  
**Next Task**: 03g_create_node_trait_interface.md  

## Objective
Create the NeuralPathwayNode data structure for neural network integration.

## Single Action
Add NeuralPathwayNode struct and neural types to node_types.rs.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralPathwayNode {
    pub id: String,
    pub pathway_type: PathwayType,
    pub source_node: String,
    pub target_node: String,
    pub activation_pattern: Vec<f32>,
    pub connection_strength: f32,
    pub ttfs_encoding: Option<f32>,
    pub learning_rate: f32,
    pub usage_count: i32,
    pub last_activated: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub is_inhibitory: bool,
    pub temporal_dynamics: TemporalDynamics,
    pub plasticity_params: PlasticityParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PathwayType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Feedback,
    Feedforward,
    Lateral,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalDynamics {
    pub decay_rate: f32,
    pub refractory_period: f32,
    pub spike_threshold: f32,
    pub reset_potential: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlasticityParameters {
    pub hebbian_rate: f32,
    pub anti_hebbian_rate: f32,
    pub homeostatic_scaling: f32,
    pub spike_timing_dependency: bool,
}

impl NeuralPathwayNode {
    pub fn new(
        pathway_type: PathwayType,
        source_node: String,
        target_node: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            pathway_type,
            source_node,
            target_node,
            activation_pattern: Vec::new(),
            connection_strength: 0.5,
            ttfs_encoding: None,
            learning_rate: 0.01,
            usage_count: 0,
            last_activated: Utc::now(),
            created_at: Utc::now(),
            is_inhibitory: false,
            temporal_dynamics: TemporalDynamics::default(),
            plasticity_params: PlasticityParameters::default(),
        }
    }
    
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.connection_strength = strength.clamp(0.0, 1.0);
        self
    }
    
    pub fn set_inhibitory(mut self, inhibitory: bool) -> Self {
        self.is_inhibitory = inhibitory;
        if inhibitory {
            self.pathway_type = PathwayType::Inhibitory;
        }
        self
    }
    
    pub fn activate(&mut self, spike_pattern: Vec<f32>) {
        self.activation_pattern = spike_pattern;
        self.usage_count += 1;
        self.last_activated = Utc::now();
        
        // Apply Hebbian learning
        if self.plasticity_params.hebbian_rate > 0.0 {
            self.connection_strength += self.plasticity_params.hebbian_rate * self.learning_rate;
            self.connection_strength = self.connection_strength.clamp(0.0, 1.0);
        }
    }
    
    pub fn decay(&mut self) {
        let decay_factor = self.temporal_dynamics.decay_rate;
        self.connection_strength *= 1.0 - decay_factor;
    }
}

impl Default for TemporalDynamics {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,
            refractory_period: 2.0,
            spike_threshold: 0.7,
            reset_potential: 0.0,
        }
    }
}

impl Default for PlasticityParameters {
    fn default() -> Self {
        Self {
            hebbian_rate: 0.01,
            anti_hebbian_rate: 0.005,
            homeostatic_scaling: 0.001,
            spike_timing_dependency: true,
        }
    }
}

#[cfg(test)]
mod neural_tests {
    use super::*;
    
    #[test]
    fn test_neural_pathway_creation() {
        let pathway = NeuralPathwayNode::new(
            PathwayType::Excitatory,
            "source_1".to_string(),
            "target_1".to_string(),
        );
        
        assert_eq!(pathway.pathway_type, PathwayType::Excitatory);
        assert_eq!(pathway.source_node, "source_1");
        assert_eq!(pathway.target_node, "target_1");
        assert_eq!(pathway.connection_strength, 0.5);
        assert!(!pathway.is_inhibitory);
    }
    
    #[test]
    fn test_pathway_activation_learning() {
        let mut pathway = NeuralPathwayNode::new(
            PathwayType::Excitatory,
            "a".to_string(),
            "b".to_string(),
        );
        
        let initial_strength = pathway.connection_strength;
        pathway.activate(vec![0.8, 0.9, 0.7]);
        
        assert!(pathway.connection_strength >= initial_strength);
        assert_eq!(pathway.usage_count, 1);
        assert_eq!(pathway.activation_pattern, vec![0.8, 0.9, 0.7]);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run neural tests
cargo test neural_tests
```

## Acceptance Criteria
- [ ] NeuralPathwayNode struct compiles without errors
- [ ] Temporal dynamics and plasticity implemented
- [ ] Activation and learning mechanisms work
- [ ] Hebbian learning simulation functions
- [ ] Tests pass

## Duration
7-9 minutes for neural pathway implementation and testing.