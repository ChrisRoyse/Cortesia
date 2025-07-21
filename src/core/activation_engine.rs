use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ahash::AHashMap;
use slotmap::SlotMap;

use crate::core::brain_types::{
    BrainInspiredEntity, LogicGate, BrainInspiredRelationship, 
    ActivationPattern
};
use crate::core::types::EntityKey;
// Re-export for external use
pub use crate::core::activation_config::{ActivationConfig, PropagationResult, ActivationStatistics};
use crate::core::activation_processors::ActivationProcessors;
use crate::error::Result;

/// Neural activation propagation engine
pub struct ActivationPropagationEngine {
    entities: Arc<RwLock<AHashMap<EntityKey, BrainInspiredEntity>>>,
    logic_gates: Arc<RwLock<AHashMap<EntityKey, LogicGate>>>,
    relationships: Arc<RwLock<AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>>>,
    entity_arena: Arc<RwLock<SlotMap<EntityKey, ()>>>,  // For allocating EntityKeys
    config: ActivationConfig,
    processors: ActivationProcessors,
}

impl std::fmt::Debug for ActivationPropagationEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationPropagationEngine")
            .field("entities", &"Arc<RwLock<AHashMap>>")
            .field("logic_gates", &"Arc<RwLock<AHashMap>>")
            .field("relationships", &"Arc<RwLock<AHashMap>>")
            .field("entity_arena", &"Arc<RwLock<SlotMap>>")
            .field("config", &self.config)
            .finish()
    }
}

impl ActivationPropagationEngine {
    pub fn new(config: ActivationConfig) -> Self {
        let processors = ActivationProcessors::new(config.clone());
        Self {
            entities: Arc::new(RwLock::new(AHashMap::new())),
            logic_gates: Arc::new(RwLock::new(AHashMap::new())),
            relationships: Arc::new(RwLock::new(AHashMap::new())),
            entity_arena: Arc::new(RwLock::new(SlotMap::with_key())),
            config,
            processors,
        }
    }

    /// Add an entity to the propagation network
    pub async fn add_entity(&self, mut entity: BrainInspiredEntity) -> Result<EntityKey> {
        // Allocate a proper EntityKey
        let mut arena = self.entity_arena.write().await;
        let key = arena.insert(());
        drop(arena);
        
        // Update entity with the allocated key
        entity.id = key;
        
        let mut entities = self.entities.write().await;
        entities.insert(key, entity);
        Ok(key)
    }

    /// Add a logic gate to the propagation network
    pub async fn add_logic_gate(&self, mut gate: LogicGate) -> Result<EntityKey> {
        // Allocate a proper EntityKey for the gate
        let mut arena = self.entity_arena.write().await;
        let key = arena.insert(());
        drop(arena);
        
        // Update gate with the allocated key
        gate.gate_id = key;
        
        let mut gates = self.logic_gates.write().await;
        gates.insert(key, gate);
        Ok(key)
    }

    /// Add a relationship to the propagation network
    pub async fn add_relationship(&self, relationship: BrainInspiredRelationship) -> Result<()> {
        let mut relationships = self.relationships.write().await;
        relationships.insert((relationship.source, relationship.target), relationship);
        Ok(())
    }

    /// Propagate activation through the network
    pub async fn propagate_activation(
        &self,
        initial_pattern: &ActivationPattern,
    ) -> Result<PropagationResult> {
        // Validate initial activations for NaN/Infinity
        for (key, &activation) in &initial_pattern.activations {
            if !activation.is_finite() {
                return Err(crate::error::GraphError::InvalidInput(
                    format!("Invalid activation value for entity {:?}: {}", key, activation)
                ));
            }
        }
        
        let mut current_activations = initial_pattern.activations.clone();
        let mut trace = Vec::new();
        let mut converged = false;
        
        // Memory limit for activation trace (max 10000 entries)
        const MAX_TRACE_ENTRIES: usize = 10000;

        // Get immutable references to data structures
        let entities = self.entities.read().await;
        let gates = self.logic_gates.read().await;
        let relationships = self.relationships.read().await;

        for iteration in 0..self.config.max_iterations {
            let previous_activations = current_activations.clone();
            
            // Step 1: Update entity activations based on incoming connections
            self.processors.update_entity_activations(
                &mut current_activations,
                &*entities,
                &*relationships,
                &mut trace,
                iteration,
            ).await?;
            
            // Check trace memory limit
            if trace.len() > MAX_TRACE_ENTRIES {
                // Keep only the most recent entries
                trace.drain(0..trace.len() - MAX_TRACE_ENTRIES);
            }

            // Step 2: Process logic gates
            self.processors.process_logic_gates(
                &mut current_activations,
                &*entities,
                &*gates,
                &mut trace,
                iteration,
            ).await?;

            // Step 3: Apply inhibitory connections
            self.processors.apply_inhibitory_connections(
                &mut current_activations,
                &*relationships,
                &mut trace,
                iteration,
            ).await?;
            
            // Validate all activations are still finite after processing
            for (key, activation) in &mut current_activations {
                if !activation.is_finite() {
                    // Clamp to valid range if NaN or Infinity
                    *activation = activation.clamp(0.0, 1.0);
                    if !activation.is_finite() {
                        *activation = 0.0; // Reset if still invalid
                    }
                }
            }

            // Step 4: Apply temporal decay
            self.processors.apply_temporal_decay(&mut current_activations, &*entities, &*relationships).await?;

            // Check for convergence
            if self.processors.has_converged(&previous_activations, &current_activations) {
                converged = true;
                break;
            }
        }

        let total_energy = current_activations.values().map(|&v| v * v).sum();

        Ok(PropagationResult {
            final_activations: current_activations,
            iterations_completed: if converged { 
                // At least 1 iteration was completed if we entered the loop
                (trace.len() / 4).max(1)
            } else { 
                self.config.max_iterations 
            },
            converged,
            activation_trace: trace,
            total_energy,
        })
    }


    /// Get current state of all entities
    pub async fn get_current_state(&self) -> Result<HashMap<EntityKey, f32>> {
        let entities = self.entities.read().await;
        let activations = entities.iter()
            .map(|(key, entity)| (*key, entity.activation_state))
            .collect();
        Ok(activations)
    }

    /// Reset all activations to zero
    pub async fn reset_activations(&self) -> Result<()> {
        let mut entities = self.entities.write().await;
        for (_, entity) in entities.iter_mut() {
            entity.activation_state = 0.0;
        }
        Ok(())
    }

    /// Get activation statistics
    pub async fn get_activation_statistics(&self) -> Result<ActivationStatistics> {
        let entities = self.entities.read().await;
        let gates = self.logic_gates.read().await;
        let relationships = self.relationships.read().await;

        let total_entities = entities.len();
        let total_gates = gates.len();
        let total_relationships = relationships.len();
        
        let active_entities = entities.values()
            .filter(|e| e.activation_state > 0.1)
            .count();

        let inhibitory_connections = relationships.values()
            .filter(|r| r.is_inhibitory)
            .count();

        let average_activation = if !entities.is_empty() {
            entities.values().map(|e| e.activation_state).sum::<f32>() / entities.len() as f32
        } else {
            0.0
        };

        Ok(ActivationStatistics {
            total_entities,
            total_gates,
            total_relationships,
            active_entities,
            inhibitory_connections,
            average_activation,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::{EntityDirection, LogicGateType, RelationType, ActivationStep, ActivationOperation};
    use std::time::SystemTime;

    fn create_test_config() -> ActivationConfig {
        ActivationConfig {
            max_iterations: 10,
            convergence_threshold: 0.001,
            decay_rate: 0.1,
            inhibition_strength: 2.0,
            default_threshold: 0.5,
        }
    }

    #[tokio::test]
    async fn test_activation_engine_creation() {
        let config = ActivationConfig::default();
        let _engine = ActivationPropagationEngine::new(config);
    }

    #[tokio::test]
    async fn test_simple_activation_propagation() {
        let config = create_test_config();
        let engine = ActivationPropagationEngine::new(config);

        // Create a simple network: Input -> Output
        let input_entity = BrainInspiredEntity::new("input".to_string(), EntityDirection::Input);
        let output_entity = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);

        let input_key = engine.add_entity(input_entity).await.unwrap();
        let output_key = engine.add_entity(output_entity).await.unwrap();

        // Create relationships to connect input -> output
        let relationship = BrainInspiredRelationship::new(
            input_key, 
            output_key, 
            RelationType::RelatedTo
        );
        engine.add_relationship(relationship).await.unwrap();

        // Create initial activation pattern
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations.insert(input_key, 0.8);

        // Propagate activation
        let result = engine.propagate_activation(&pattern).await.unwrap();

        assert!(!result.final_activations.is_empty());
        assert!(result.converged || result.iterations_completed > 0);
        assert!(result.final_activations.len() >= 1);
    }

    #[tokio::test]
    async fn test_inhibitory_connections() {
        let config = create_test_config();
        let engine = ActivationPropagationEngine::new(config);

        // Create entities
        let excitatory = BrainInspiredEntity::new("excitatory".to_string(), EntityDirection::Input);
        let inhibitory = BrainInspiredEntity::new("inhibitory".to_string(), EntityDirection::Input);
        let target = BrainInspiredEntity::new("target".to_string(), EntityDirection::Output);

        let exc_key = engine.add_entity(excitatory).await.unwrap();
        let inh_key = engine.add_entity(inhibitory).await.unwrap();
        let target_key = engine.add_entity(target).await.unwrap();

        // Create relationships
        let excitatory_rel = BrainInspiredRelationship::new(exc_key, target_key, RelationType::RelatedTo);
        let mut inhibitory_rel = BrainInspiredRelationship::new(inh_key, target_key, RelationType::RelatedTo);
        inhibitory_rel.is_inhibitory = true;

        engine.add_relationship(excitatory_rel).await.unwrap();
        engine.add_relationship(inhibitory_rel).await.unwrap();

        // Test with only excitatory activation
        let mut pattern = ActivationPattern::new("test_excitatory".to_string());
        pattern.activations.insert(exc_key, 0.8);

        let result = engine.propagate_activation(&pattern).await.unwrap();
        let target_activation_excitatory = result.final_activations.get(&target_key).copied().unwrap_or(0.0);

        // Test with both excitatory and inhibitory activation
        let mut pattern = ActivationPattern::new("test_inhibitory".to_string());
        pattern.activations.insert(exc_key, 0.8);
        pattern.activations.insert(inh_key, 0.6);

        let result = engine.propagate_activation(&pattern).await.unwrap();
        let target_activation_inhibited = result.final_activations.get(&target_key).copied().unwrap_or(0.0);

        // Target activation should be lower when inhibitory input is present
        assert!(target_activation_inhibited < target_activation_excitatory);
    }

    /// Tests for ActivationProcessors private methods
    mod processor_tests {
        use super::*;

        #[tokio::test]
        async fn test_update_entity_activations_happy_path() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            // Create test entities
            let mut entities = AHashMap::new();
            let mut relationships = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            // Create source and target entities
            let source_entity = BrainInspiredEntity::new("source".to_string(), EntityDirection::Input);
            let target_entity = BrainInspiredEntity::new("target".to_string(), EntityDirection::Output);
            
            let source_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let target_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            entities.insert(source_key, source_entity);
            entities.insert(target_key, target_entity);

            // Create relationship
            let relationship = BrainInspiredRelationship::new(source_key, target_key, RelationType::RelatedTo);
            relationships.insert((source_key, target_key), relationship);

            // Set initial activation
            activations.insert(source_key, 0.8);

            // Update entity activations
            processors.update_entity_activations(
                &mut activations,
                &entities,
                &relationships,
                &mut trace,
                0
            ).await.unwrap();

            // Target should have received activation
            assert!(activations.get(&target_key).is_some());
            let target_activation = activations.get(&target_key).copied().unwrap_or(0.0);
            assert!(target_activation > 0.0);
            assert!(target_activation <= 1.0);

            // Source activation should remain unchanged (input entity)
            assert_eq!(activations.get(&source_key), Some(&0.8));

            // Trace should contain activation steps
            assert!(!trace.is_empty());
        }

        #[tokio::test]
        async fn test_update_entity_activations_empty_patterns() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let entities = AHashMap::new();
            let relationships = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            // Test with empty data structures
            let result = processors.update_entity_activations(
                &mut activations,
                &entities,
                &relationships,
                &mut trace,
                0
            ).await;

            assert!(result.is_ok());
            assert!(activations.is_empty());
            assert!(trace.is_empty());
        }

        #[tokio::test]
        async fn test_update_entity_activations_disconnected_network() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut entities = AHashMap::new();
            let relationships = AHashMap::new(); // No relationships
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            // Create isolated entities
            let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Output);
            let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Output);
            
            let key1 = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let key2 = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            entities.insert(key1, entity1);
            entities.insert(key2, entity2);

            activations.insert(key1, 0.5);
            activations.insert(key2, 0.3);

            processors.update_entity_activations(
                &mut activations,
                &entities,
                &relationships,
                &mut trace,
                0
            ).await.unwrap();

            // Activations should remain unchanged for disconnected entities
            assert_eq!(activations.get(&key1), Some(&0.5));
            assert_eq!(activations.get(&key2), Some(&0.3));
        }

        #[tokio::test]
        async fn test_apply_inhibitory_connections_happy_path() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut relationships = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            let source_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let target_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Create inhibitory relationship
            let mut relationship = BrainInspiredRelationship::new(source_key, target_key, RelationType::RelatedTo);
            relationship.is_inhibitory = true;
            relationship.weight = 0.5;
            relationships.insert((source_key, target_key), relationship);

            // Set initial activations
            activations.insert(source_key, 0.8);
            activations.insert(target_key, 0.6);

            let initial_target_activation = activations[&target_key];

            processors.apply_inhibitory_connections(
                &mut activations,
                &relationships,
                &mut trace,
                0
            ).await.unwrap();

            // Target activation should be reduced due to inhibition
            let final_target_activation = activations[&target_key];
            assert!(final_target_activation < initial_target_activation);
            assert!(final_target_activation >= 0.0);
        }

        #[tokio::test]
        async fn test_apply_inhibitory_connections_zero_activation() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut relationships = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            let source_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let target_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Create inhibitory relationship
            let mut relationship = BrainInspiredRelationship::new(source_key, target_key, RelationType::RelatedTo);
            relationship.is_inhibitory = true;
            relationships.insert((source_key, target_key), relationship);

            // Set zero activation for target
            activations.insert(source_key, 0.5);
            activations.insert(target_key, 0.0);

            processors.apply_inhibitory_connections(
                &mut activations,
                &relationships,
                &mut trace,
                0
            ).await.unwrap();

            // Zero activation should remain zero
            assert_eq!(activations[&target_key], 0.0);
        }

        #[tokio::test]
        async fn test_apply_inhibitory_connections_negative_activation() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut relationships = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            let source_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let target_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Create inhibitory relationship
            let mut relationship = BrainInspiredRelationship::new(source_key, target_key, RelationType::RelatedTo);
            relationship.is_inhibitory = true;
            relationships.insert((source_key, target_key), relationship);

            // Set negative activation (edge case)
            activations.insert(source_key, 0.5);
            activations.insert(target_key, -0.1);

            processors.apply_inhibitory_connections(
                &mut activations,
                &relationships,
                &mut trace,
                0
            ).await.unwrap();

            // Negative activation should not be affected by inhibition logic
            assert!(activations[&target_key] <= 0.0);
        }

        #[tokio::test]
        async fn test_process_logic_gates_happy_path() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut entities = AHashMap::new();
            let mut gates = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            // Create input entities
            let input1 = BrainInspiredEntity::new("input1".to_string(), EntityDirection::Input);
            let input2 = BrainInspiredEntity::new("input2".to_string(), EntityDirection::Input);
            let output = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);

            let input1_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let input2_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));
            let output_key = slotmap::Key::from(slotmap::KeyData::from_ffi(3));
            let gate_key = slotmap::Key::from(slotmap::KeyData::from_ffi(4));

            entities.insert(input1_key, input1);
            entities.insert(input2_key, input2);
            entities.insert(output_key, output);

            // Create AND gate
            let gate = LogicGate {
                gate_id: gate_key,
                gate_type: LogicGateType::And,
                input_nodes: vec![input1_key, input2_key],
                output_nodes: vec![output_key],
                threshold: 0.5,
                weight_matrix: vec![1.0, 1.0],
            };
            gates.insert(gate_key, gate);

            // Set input activations
            activations.insert(input1_key, 0.8);
            activations.insert(input2_key, 0.7);

            processors.process_logic_gates(
                &mut activations,
                &entities,
                &gates,
                &mut trace,
                0
            ).await.unwrap();

            // Gate should have calculated output
            assert!(activations.contains_key(&gate_key));
            // Output entity should have received propagated activation
            assert!(activations.contains_key(&output_key));
            
            let gate_output = activations[&gate_key];
            assert!(gate_output >= 0.0 && gate_output <= 1.0);
        }

        #[tokio::test]
        async fn test_process_logic_gates_empty_inputs() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let entities = AHashMap::new();
            let mut gates = AHashMap::new();
            let mut activations = HashMap::new();
            let mut trace = Vec::new();

            let gate_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));

            // Create gate with no inputs
            let gate = LogicGate {
                gate_id: gate_key,
                gate_type: LogicGateType::And,
                input_nodes: vec![], // Empty inputs
                output_nodes: vec![],
                threshold: 0.5,
                weight_matrix: vec![],
            };
            gates.insert(gate_key, gate);

            processors.process_logic_gates(
                &mut activations,
                &entities,
                &gates,
                &mut trace,
                0
            ).await.unwrap();

            // Gate with no inputs should not be processed
            assert!(!activations.contains_key(&gate_key));
        }

        #[tokio::test]
        async fn test_apply_temporal_decay_happy_path() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut entities = AHashMap::new();
            let mut relationships = AHashMap::new();
            let mut activations = HashMap::new();

            let entity_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let other_key = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Create connected entities (not input entities)
            let mut entity = BrainInspiredEntity::new("entity".to_string(), EntityDirection::Output);
            // Set last_activation to past time to trigger decay
            entity.last_activation = SystemTime::now() - std::time::Duration::from_secs(1);
            entities.insert(entity_key, entity);

            // Create relationship to ensure entity is connected
            let relationship = BrainInspiredRelationship::new(entity_key, other_key, RelationType::RelatedTo);
            relationships.insert((entity_key, other_key), relationship);

            // Set initial activation
            activations.insert(entity_key, 0.8);

            let initial_activation = activations[&entity_key];

            processors.apply_temporal_decay(
                &mut activations,
                &entities,
                &relationships
            ).await.unwrap();

            // Activation should have decayed
            let final_activation = activations[&entity_key];
            assert!(final_activation < initial_activation);
            assert!(final_activation >= 0.0);
        }

        #[tokio::test]
        async fn test_apply_temporal_decay_input_entities_unchanged() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut entities = AHashMap::new();
            let relationships = AHashMap::new();
            let mut activations = HashMap::new();

            let entity_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));

            // Create input entity
            let entity = BrainInspiredEntity::new("input".to_string(), EntityDirection::Input);
            entities.insert(entity_key, entity);

            // Set initial activation
            activations.insert(entity_key, 0.8);

            let initial_activation = activations[&entity_key];

            processors.apply_temporal_decay(
                &mut activations,
                &entities,
                &relationships
            ).await.unwrap();

            // Input entity activation should remain unchanged
            let final_activation = activations[&entity_key];
            assert_eq!(final_activation, initial_activation);
        }

        #[tokio::test]
        async fn test_apply_temporal_decay_disconnected_entities() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut entities = AHashMap::new();
            let relationships = AHashMap::new(); // No relationships
            let mut activations = HashMap::new();

            let entity_key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));

            // Create disconnected entity
            let entity = BrainInspiredEntity::new("disconnected".to_string(), EntityDirection::Output);
            entities.insert(entity_key, entity);

            // Set initial activation
            activations.insert(entity_key, 0.8);

            let initial_activation = activations[&entity_key];

            processors.apply_temporal_decay(
                &mut activations,
                &entities,
                &relationships
            ).await.unwrap();

            // Disconnected entity activation should remain unchanged
            let final_activation = activations[&entity_key];
            assert_eq!(final_activation, initial_activation);
        }

        #[tokio::test]
        async fn test_has_converged_true() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut previous = HashMap::new();
            let mut current = HashMap::new();

            let key1 = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let key2 = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Very small changes (below threshold)
            previous.insert(key1, 0.5);
            previous.insert(key2, 0.3);
            
            current.insert(key1, 0.5005); // 0.0005 change
            current.insert(key2, 0.3003); // 0.0003 change

            assert!(processors.has_converged(&previous, &current));
        }

        #[tokio::test]
        async fn test_has_converged_false() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let mut previous = HashMap::new();
            let mut current = HashMap::new();

            let key1 = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            let key2 = slotmap::Key::from(slotmap::KeyData::from_ffi(2));

            // Large changes (above threshold)
            previous.insert(key1, 0.5);
            previous.insert(key2, 0.3);
            
            current.insert(key1, 0.52); // 0.02 change
            current.insert(key2, 0.31); // 0.01 change

            assert!(!processors.has_converged(&previous, &current));
        }

        #[tokio::test]
        async fn test_has_converged_new_activations() {
            let config = create_test_config();
            let processors = ActivationProcessors::new(config);

            let previous = HashMap::new();
            let mut current = HashMap::new();

            let key1 = slotmap::Key::from(slotmap::KeyData::from_ffi(1));

            // New activation appears
            current.insert(key1, 0.5);

            assert!(!processors.has_converged(&previous, &current));
        }
    }

    /// Tests for propagation edge cases and error handling
    mod propagation_edge_cases {
        use super::*;

        #[tokio::test]
        async fn test_propagate_activation_invalid_input() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            let mut pattern = ActivationPattern::new("invalid_test".to_string());
            let key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            
            // Test with NaN activation
            pattern.activations.insert(key, f32::NAN);
            
            let result = engine.propagate_activation(&pattern).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_propagate_activation_infinity_input() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            let mut pattern = ActivationPattern::new("infinity_test".to_string());
            let key = slotmap::Key::from(slotmap::KeyData::from_ffi(1));
            
            // Test with infinity activation
            pattern.activations.insert(key, f32::INFINITY);
            
            let result = engine.propagate_activation(&pattern).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_propagate_activation_max_iterations() {
            let mut config = create_test_config();
            config.max_iterations = 1; // Force early termination
            config.convergence_threshold = 0.0; // Never converge
            
            let engine = ActivationPropagationEngine::new(config);

            // Create a simple network that won't converge quickly
            let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Input);
            let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Output);

            let key1 = engine.add_entity(entity1).await.unwrap();
            let key2 = engine.add_entity(entity2).await.unwrap();

            let relationship = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
            engine.add_relationship(relationship).await.unwrap();

            let mut pattern = ActivationPattern::new("max_iterations_test".to_string());
            pattern.activations.insert(key1, 0.8);

            let result = engine.propagate_activation(&pattern).await.unwrap();

            // Should not converge and hit max iterations
            assert!(!result.converged);
            assert_eq!(result.iterations_completed, 1);
        }

        #[tokio::test]
        async fn test_propagate_activation_cycle_detection() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            // Create a cycle: A -> B -> C -> A
            let entity_a = BrainInspiredEntity::new("a".to_string(), EntityDirection::Input);
            let entity_b = BrainInspiredEntity::new("b".to_string(), EntityDirection::Hidden);
            let entity_c = BrainInspiredEntity::new("c".to_string(), EntityDirection::Output);

            let key_a = engine.add_entity(entity_a).await.unwrap();
            let key_b = engine.add_entity(entity_b).await.unwrap();
            let key_c = engine.add_entity(entity_c).await.unwrap();

            // Create cyclic relationships
            let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
            let rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
            let rel_ca = BrainInspiredRelationship::new(key_c, key_a, RelationType::RelatedTo);

            engine.add_relationship(rel_ab).await.unwrap();
            engine.add_relationship(rel_bc).await.unwrap();
            engine.add_relationship(rel_ca).await.unwrap();

            let mut pattern = ActivationPattern::new("cycle_test".to_string());
            pattern.activations.insert(key_a, 0.8);

            let result = engine.propagate_activation(&pattern).await.unwrap();

            // Cycle should not cause infinite loop or errors
            assert!(result.iterations_completed <= config.max_iterations);
            assert!(!result.final_activations.is_empty());
        }

        #[tokio::test]
        async fn test_propagate_activation_memory_limit() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            // Create a large network to test memory limits
            let mut entity_keys = Vec::new();
            for i in 0..50 {
                let entity = BrainInspiredEntity::new(
                    format!("entity_{}", i), 
                    if i == 0 { EntityDirection::Input } else { EntityDirection::Output }
                );
                let key = engine.add_entity(entity).await.unwrap();
                entity_keys.push(key);
            }

            // Connect all entities to create many activation steps
            for i in 0..entity_keys.len()-1 {
                let rel = BrainInspiredRelationship::new(
                    entity_keys[i], 
                    entity_keys[i+1], 
                    RelationType::RelatedTo
                );
                engine.add_relationship(rel).await.unwrap();
            }

            let mut pattern = ActivationPattern::new("memory_test".to_string());
            pattern.activations.insert(entity_keys[0], 0.8);

            let result = engine.propagate_activation(&pattern).await.unwrap();

            // Should complete without memory issues
            assert!(result.activation_trace.len() <= 10000); // MAX_TRACE_ENTRIES
        }
    }

    /// Integration tests for complete activation scenarios
    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_complex_network_integration() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            // Create a complex network with inputs, gates, and outputs
            let input1 = BrainInspiredEntity::new("input1".to_string(), EntityDirection::Input);
            let input2 = BrainInspiredEntity::new("input2".to_string(), EntityDirection::Input);
            let hidden = BrainInspiredEntity::new("hidden".to_string(), EntityDirection::Hidden);
            let output = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);

            let input1_key = engine.add_entity(input1).await.unwrap();
            let input2_key = engine.add_entity(input2).await.unwrap();
            let hidden_key = engine.add_entity(hidden).await.unwrap();
            let output_key = engine.add_entity(output).await.unwrap();

            // Create logic gate
            let gate = LogicGate {
                gate_id: slotmap::Key::from(slotmap::KeyData::from_ffi(999)),
                gate_type: LogicGateType::And,
                input_nodes: vec![input1_key, input2_key],
                output_nodes: vec![hidden_key],
                threshold: 0.5,
                weight_matrix: vec![1.0, 1.0],
            };
            let gate_key = engine.add_logic_gate(gate).await.unwrap();

            // Create relationships
            let rel1 = BrainInspiredRelationship::new(input1_key, gate_key, RelationType::RelatedTo);
            let rel2 = BrainInspiredRelationship::new(input2_key, gate_key, RelationType::RelatedTo);
            let rel3 = BrainInspiredRelationship::new(hidden_key, output_key, RelationType::RelatedTo);

            engine.add_relationship(rel1).await.unwrap();
            engine.add_relationship(rel2).await.unwrap();
            engine.add_relationship(rel3).await.unwrap();

            // Add inhibitory connection
            let mut inhibitory_rel = BrainInspiredRelationship::new(
                input1_key, 
                output_key, 
                RelationType::RelatedTo
            );
            inhibitory_rel.is_inhibitory = true;
            inhibitory_rel.weight = 0.3;
            engine.add_relationship(inhibitory_rel).await.unwrap();

            // Test propagation
            let mut pattern = ActivationPattern::new("complex_test".to_string());
            pattern.activations.insert(input1_key, 0.9);
            pattern.activations.insert(input2_key, 0.7);

            let result = engine.propagate_activation(&pattern).await.unwrap();

            // Verify complex behavior
            assert!(result.final_activations.contains_key(&output_key));
            assert!(result.total_energy > 0.0);
            assert!(!result.activation_trace.is_empty());

            // Test that inhibition affected the output
            let output_activation = result.final_activations[&output_key];
            assert!(output_activation < 1.0); // Should be reduced by inhibition
        }

        #[tokio::test]
        async fn test_statistics_collection() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            // Add various entities and relationships
            let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Input);
            let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Output);
            
            let key1 = engine.add_entity(entity1).await.unwrap();
            let key2 = engine.add_entity(entity2).await.unwrap();

            let gate = LogicGate::new(LogicGateType::And, 0.5);
            let _gate_key = engine.add_logic_gate(gate).await.unwrap();

            let mut rel = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
            rel.is_inhibitory = true;
            engine.add_relationship(rel).await.unwrap();

            let stats = engine.get_activation_statistics().await.unwrap();

            assert_eq!(stats.total_entities, 2);
            assert_eq!(stats.total_gates, 1);
            assert_eq!(stats.total_relationships, 1);
            assert_eq!(stats.inhibitory_connections, 1);
            assert!(stats.average_activation >= 0.0);
        }

        #[tokio::test]
        async fn test_state_management() {
            let config = create_test_config();
            let engine = ActivationPropagationEngine::new(config);

            let entity = BrainInspiredEntity::new("test_entity".to_string(), EntityDirection::Input);
            let key = engine.add_entity(entity).await.unwrap();

            // Set activation through propagation
            let mut pattern = ActivationPattern::new("state_test".to_string());
            pattern.activations.insert(key, 0.75);
            
            let _result = engine.propagate_activation(&pattern).await.unwrap();

            // Get current state
            let state = engine.get_current_state().await.unwrap();
            assert!(state.contains_key(&key));

            // Reset activations
            engine.reset_activations().await.unwrap();
            
            let reset_state = engine.get_current_state().await.unwrap();
            assert_eq!(reset_state[&key], 0.0);
        }
    }
}