use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use ahash::AHashMap;

use crate::core::brain_types::{
    BrainInspiredEntity, LogicGate, LogicGateType, BrainInspiredRelationship, 
    ActivationPattern, ActivationStep, EntityDirection, ActivationOperation
};
use crate::core::types::EntityKey;
use crate::error::Result;

/// Configuration for activation propagation
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub decay_rate: f32,
    pub inhibition_strength: f32,
    pub default_threshold: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.001,
            decay_rate: 0.1,
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
    pub activation_trace: Vec<ActivationStep>,
    pub total_energy: f32,
}

/// Neural activation propagation engine
pub struct ActivationPropagationEngine {
    entities: Arc<RwLock<AHashMap<EntityKey, BrainInspiredEntity>>>,
    logic_gates: Arc<RwLock<AHashMap<EntityKey, LogicGate>>>,
    relationships: Arc<RwLock<AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>>>,
    config: ActivationConfig,
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
        Self {
            entities: Arc::new(RwLock::new(AHashMap::new())),
            logic_gates: Arc::new(RwLock::new(AHashMap::new())),
            relationships: Arc::new(RwLock::new(AHashMap::new())),
            config,
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
            self.update_entity_activations(
                &mut current_activations,
                &*entities,
                &*relationships,
                &mut trace,
                iteration,
            ).await?;

            // Step 2: Process logic gates
            self.process_logic_gates(
                &mut current_activations,
                &*entities,
                &*gates,
                &mut trace,
                iteration,
            ).await?;

            // Step 3: Apply inhibitory connections
            self.apply_inhibitory_connections(
                &mut current_activations,
                &*relationships,
                &mut trace,
                iteration,
            ).await?;

            // Step 4: Apply temporal decay
            self.apply_temporal_decay(&mut current_activations, &*entities).await?;

            // Check for convergence
            if self.has_converged(&previous_activations, &current_activations) {
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

    /// Update entity activations based on incoming connections
    async fn update_entity_activations(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        entities: &AHashMap<EntityKey, BrainInspiredEntity>,
        relationships: &AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>,
        trace: &mut Vec<ActivationStep>,
        _iteration: usize,
    ) -> Result<()> {
        let mut updates = HashMap::new();

        for (entity_key, entity) in entities.iter() {
            if matches!(entity.direction, EntityDirection::Gate) {
                continue; // Skip gates in this step
            }

            let mut incoming_activation = 0.0;
            let mut connection_count = 0;

            // Sum incoming activations from connected entities
            for ((source, target), relationship) in relationships.iter() {
                if *target == *entity_key && !relationship.is_inhibitory {
                    if let Some(&source_activation) = activations.get(source) {
                        incoming_activation += source_activation * relationship.weight;
                        connection_count += 1;
                    }
                }
            }

            // Calculate new activation
            let current_activation = activations.get(entity_key).copied().unwrap_or(0.0);
            let new_activation = if connection_count > 0 {
                (current_activation + incoming_activation / connection_count as f32).min(1.0)
            } else {
                current_activation
            };

            if (new_activation - current_activation).abs() > 0.001 {
                updates.insert(*entity_key, new_activation);
                
                trace.push(ActivationStep {
                    step_id: trace.len(),
                    entity_key: *entity_key,
                    concept_id: format!("entity_{:?}", entity_key),
                    activation_level: new_activation,
                    operation_type: ActivationOperation::Propagate,
                    timestamp: std::time::SystemTime::now(),
                });
            }
        }

        // Apply updates
        for (key, value) in updates {
            activations.insert(key, value);
        }

        Ok(())
    }

    /// Process logic gates
    async fn process_logic_gates(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        entities: &AHashMap<EntityKey, BrainInspiredEntity>,
        gates: &AHashMap<EntityKey, LogicGate>,
        trace: &mut Vec<ActivationStep>,
        _iteration: usize,
    ) -> Result<()> {
        for (gate_key, gate) in gates.iter() {
            // Collect input activations
            let input_activations: Vec<f32> = gate.input_nodes.iter()
                .map(|node_key| activations.get(node_key).copied().unwrap_or(0.0))
                .collect();

            if input_activations.is_empty() {
                continue;
            }

            // Calculate gate output
            let gate_output = gate.calculate_output(&input_activations)?;

            // Update gate activation
            activations.insert(*gate_key, gate_output);

            trace.push(ActivationStep {
                step_id: trace.len(),
                entity_key: *gate_key,
                concept_id: format!("gate_{:?}", gate_key),
                activation_level: gate_output,
                operation_type: ActivationOperation::Propagate,
                timestamp: std::time::SystemTime::now(),
            });

            // Propagate to output nodes
            for output_node in &gate.output_nodes {
                if let Some(entity) = entities.get(output_node) {
                    if matches!(entity.direction, EntityDirection::Output) {
                        let current = activations.get(output_node).copied().unwrap_or(0.0);
                        let new_activation = (current + gate_output * 0.8).min(1.0); // 0.8 is propagation strength
                        
                        activations.insert(*output_node, new_activation);

                        trace.push(ActivationStep {
                            step_id: trace.len(),
                            entity_key: *output_node,
                            concept_id: format!("output_{:?}", output_node),
                            activation_level: new_activation,
                            operation_type: ActivationOperation::Propagate,
                            timestamp: std::time::SystemTime::now(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply inhibitory connections
    async fn apply_inhibitory_connections(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        relationships: &AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>,
        trace: &mut Vec<ActivationStep>,
        _iteration: usize,
    ) -> Result<()> {
        let mut inhibition_updates = HashMap::new();

        for ((source, target), relationship) in relationships.iter() {
            if relationship.is_inhibitory {
                if let (Some(&source_activation), Some(&target_activation)) = 
                    (activations.get(source), activations.get(target)) {
                    
                    // Apply inhibition: stronger source activation reduces target activation
                    let inhibition_effect = source_activation * relationship.weight * self.config.inhibition_strength;
                    let new_target_activation = (target_activation - inhibition_effect).max(0.0);
                    
                    if (new_target_activation - target_activation).abs() > 0.001 {
                        inhibition_updates.insert(*target, new_target_activation);
                        
                        trace.push(ActivationStep {
                            step_id: trace.len(),
                            entity_key: *target,
                            concept_id: format!("inhibited_{:?}", target),
                            activation_level: new_target_activation,
                            operation_type: ActivationOperation::Inhibit,
                            timestamp: std::time::SystemTime::now(),
                        });
                    }
                }
            }
        }

        // Apply inhibition updates
        for (key, value) in inhibition_updates {
            activations.insert(key, value);
        }

        Ok(())
    }

    /// Apply temporal decay to all activations
    async fn apply_temporal_decay(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        entities: &AHashMap<EntityKey, BrainInspiredEntity>,
    ) -> Result<()> {
        for (entity_key, activation) in activations.iter_mut() {
            if let Some(entity) = entities.get(entity_key) {
                let time_since_last = entity.last_activation.elapsed()
                    .unwrap_or_default()
                    .as_secs_f32();
                
                let decay_factor = (-self.config.decay_rate * time_since_last).exp();
                *activation *= decay_factor;
            }
        }

        Ok(())
    }

    /// Check if the activation pattern has converged
    fn has_converged(
        &self,
        previous: &HashMap<EntityKey, f32>,
        current: &HashMap<EntityKey, f32>,
    ) -> bool {
        let mut max_change: f32 = 0.0;

        for (key, &current_value) in current.iter() {
            let previous_value = previous.get(key).copied().unwrap_or(0.0);
            let change = (current_value - previous_value).abs();
            max_change = max_change.max(change);
        }

        max_change < self.config.convergence_threshold
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