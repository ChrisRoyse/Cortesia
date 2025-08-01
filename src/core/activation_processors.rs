use std::collections::HashMap;
use ahash::AHashMap;
use crate::core::brain_types::{
    BrainInspiredEntity, LogicGate, BrainInspiredRelationship, 
    ActivationStep, EntityDirection, ActivationOperation
};
use crate::core::types::EntityKey;
use crate::core::activation_config::ActivationConfig;
use crate::error::Result;

pub struct ActivationProcessors {
    config: ActivationConfig,
}

impl ActivationProcessors {
    pub fn new(config: ActivationConfig) -> Self {
        Self { config }
    }

    /// Update entity activations based on incoming connections
    pub async fn update_entity_activations(
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
            
            // Input entities maintain their activation (external inputs)
            let new_activation = if matches!(entity.direction, EntityDirection::Input) {
                current_activation
            } else if connection_count > 0 {
                // Use the sum of weighted activations directly
                incoming_activation.min(1.0)
            } else {
                current_activation
            };

            // Always update if entity isn't in activations map yet or if there's a significant change
            if !activations.contains_key(entity_key) || (new_activation - current_activation).abs() > 0.001 {
                updates.insert(*entity_key, new_activation);
                
                trace.push(ActivationStep {
                    step_id: trace.len(),
                    entity_key: *entity_key,
                    concept_id: format!("entity_{entity_key:?}"),
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
    pub async fn process_logic_gates(
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
                concept_id: format!("gate_{gate_key:?}"),
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
                            concept_id: format!("output_{output_node:?}"),
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
    pub async fn apply_inhibitory_connections(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        relationships: &AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>,
        trace: &mut Vec<ActivationStep>,
        _iteration: usize,
    ) -> Result<()> {
        // Apply inhibition in multiple passes to allow chain effects
        // This ensures that if A inhibits B and B inhibits C,
        // B's reduced activation will properly affect its inhibition of C
        for pass in 0..2 {  // Two passes to propagate chain effects
            // Collect all inhibition effects for this pass
            let mut inhibition_effects: HashMap<EntityKey, f32> = HashMap::new();

            for ((source, target), relationship) in relationships.iter() {
                if relationship.is_inhibitory {
                    if let Some(&source_activation) = activations.get(source) {
                        // Calculate inhibition effect based on current activation
                        let effect = source_activation * relationship.weight * self.config.inhibition_strength;
                        
                        // Accumulate inhibition effects for each target
                        *inhibition_effects.entry(*target).or_insert(0.0) += effect;
                    }
                }
            }

            // Apply all inhibition effects for this pass
            let mut inhibition_updates = HashMap::new();
            
            for (target, total_inhibition) in inhibition_effects.iter() {
                // Get current activation (default to 0.0 if not present)
                let target_activation = activations.get(target).copied().unwrap_or(0.0);
                
                if target_activation > 0.0 {
                    // Apply inhibition with a divisive inhibition model
                    // This prevents complete suppression and allows competition
                    let new_target_activation = target_activation / (1.0 + total_inhibition);
                    
                    if (new_target_activation - target_activation).abs() > 0.001 {
                        inhibition_updates.insert(*target, new_target_activation);
                        
                        // Only add trace entries in the first pass to avoid duplication
                        if pass == 0 {
                            trace.push(ActivationStep {
                                step_id: trace.len(),
                                entity_key: *target,
                                concept_id: format!("inhibited_{target:?}"),
                                activation_level: new_target_activation,
                                operation_type: ActivationOperation::Inhibit,
                                timestamp: std::time::SystemTime::now(),
                            });
                        }
                    }
                }
            }

            // Apply inhibition updates immediately for this pass
            for (key, value) in inhibition_updates {
                activations.insert(key, value);
            }
        }

        Ok(())
    }

    /// Apply temporal decay to all activations
    pub async fn apply_temporal_decay(
        &self,
        activations: &mut HashMap<EntityKey, f32>,
        entities: &AHashMap<EntityKey, BrainInspiredEntity>,
        relationships: &AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>,
    ) -> Result<()> {
        for (entity_key, activation) in activations.iter_mut() {
            if let Some(entity) = entities.get(entity_key) {
                // Skip decay for input entities (they maintain external activation)
                if matches!(entity.direction, EntityDirection::Input) {
                    continue;
                }
                
                // Check if entity has any connections (incoming or outgoing)
                let has_connections = relationships.iter().any(|((source, target), _)| {
                    *source == *entity_key || *target == *entity_key
                });
                
                // Only apply decay to connected entities
                if has_connections {
                    let time_since_last = entity.last_activation.elapsed()
                        .unwrap_or_default()
                        .as_secs_f32();
                    
                    let decay_factor = (-self.config.decay_rate * time_since_last).exp();
                    *activation *= decay_factor;
                }
                // Disconnected entities maintain their activation
            }
        }

        Ok(())
    }

    /// Check if the activation pattern has converged
    pub fn has_converged(
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
}