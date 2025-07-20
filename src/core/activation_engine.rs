use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ahash::AHashMap;

use crate::core::brain_types::{
    BrainInspiredEntity, LogicGate, LogicGateType, BrainInspiredRelationship, 
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
    config: ActivationConfig,
    processors: ActivationProcessors,
}

impl std::fmt::Debug for ActivationPropagationEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationPropagationEngine")
            .field("entities", &"Arc<RwLock<AHashMap>>")
            .field("logic_gates", &"Arc<RwLock<AHashMap>>")
            .field("relationships", &"Arc<RwLock<AHashMap>>")
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
            config,
            processors,
        }
    }

    /// Add an entity to the propagation network
    pub async fn add_entity(&self, entity: BrainInspiredEntity) -> Result<()> {
        let mut entities = self.entities.write().await;
        entities.insert(entity.id, entity);
        Ok(())
    }

    /// Add a logic gate to the propagation network
    pub async fn add_logic_gate(&self, gate: LogicGate) -> Result<()> {
        let mut gates = self.logic_gates.write().await;
        gates.insert(gate.gate_id, gate);
        Ok(())
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
        let mut current_activations = initial_pattern.activations.clone();
        let mut trace = Vec::new();
        let mut converged = false;

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
                trace.len() / 4 // Approximate iterations (4 steps per iteration)
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
    use crate::core::brain_types::EntityDirection;

    #[tokio::test]
    async fn test_activation_engine_creation() {
        let config = ActivationConfig::default();
        let _engine = ActivationPropagationEngine::new(config);
    }

    #[tokio::test]
    async fn test_simple_activation_propagation() {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);

        // Create a simple network: Input -> Gate -> Output
        let input_entity = BrainInspiredEntity::new("input".to_string(), EntityDirection::Input);
        let output_entity = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);
        let gate = LogicGate::new(LogicGateType::And, 0.5);

        let input_key = input_entity.id;
        let output_key = output_entity.id;
        let gate_key = gate.gate_id;

        engine.add_entity(input_entity).await.unwrap();
        engine.add_entity(output_entity).await.unwrap();
        engine.add_logic_gate(gate).await.unwrap();

        // Create initial activation pattern
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations.insert(input_key, 0.8);

        // Propagate activation
        let result = engine.propagate_activation(&pattern).await.unwrap();

        assert!(!result.final_activations.is_empty());
        assert!(result.iterations_completed > 0);
    }

    #[tokio::test]
    async fn test_inhibitory_connections() {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);

        // Create entities
        let excitatory = BrainInspiredEntity::new("excitatory".to_string(), EntityDirection::Input);
        let inhibitory = BrainInspiredEntity::new("inhibitory".to_string(), EntityDirection::Input);
        let target = BrainInspiredEntity::new("target".to_string(), EntityDirection::Output);

        let exc_key = excitatory.id;
        let inh_key = inhibitory.id;
        let target_key = target.id;

        engine.add_entity(excitatory).await.unwrap();
        engine.add_entity(inhibitory).await.unwrap();
        engine.add_entity(target).await.unwrap();

        // Create relationships
        let excitatory_rel = BrainInspiredRelationship::new(exc_key, target_key, crate::core::brain_types::RelationType::RelatedTo);
        let mut inhibitory_rel = BrainInspiredRelationship::new(inh_key, target_key, crate::core::brain_types::RelationType::RelatedTo);
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
}