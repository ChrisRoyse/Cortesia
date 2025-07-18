//! Inhibition matrix operations and management

use crate::cognitive::inhibitory::InhibitionMatrix;
use crate::core::types::EntityKey;

pub trait InhibitionMatrixOps {
    fn get_inhibition_strength(&self, source: EntityKey, target: EntityKey, inhibition_type: InhibitionType) -> f32;
    fn set_inhibition_strength(&mut self, source: EntityKey, target: EntityKey, strength: f32, inhibition_type: InhibitionType);
    fn update_inhibition_strength(&mut self, source: EntityKey, target: EntityKey, delta: f32, inhibition_type: InhibitionType);
    fn get_total_inhibition(&self, source: EntityKey, target: EntityKey) -> f32;
}

#[derive(Debug, Clone, Copy)]
pub enum InhibitionType {
    Lateral,
    Hierarchical,
    Contextual,
    Temporal,
}

impl InhibitionMatrixOps for InhibitionMatrix {
    fn get_inhibition_strength(&self, source: EntityKey, target: EntityKey, inhibition_type: InhibitionType) -> f32 {
        let map = match inhibition_type {
            InhibitionType::Lateral => &self.lateral_inhibition,
            InhibitionType::Hierarchical => &self.hierarchical_inhibition,
            InhibitionType::Contextual => &self.contextual_inhibition,
            InhibitionType::Temporal => &self.temporal_inhibition,
        };
        
        map.get(&(source, target)).copied().unwrap_or(0.0)
    }
    
    fn set_inhibition_strength(&mut self, source: EntityKey, target: EntityKey, strength: f32, inhibition_type: InhibitionType) {
        let map = match inhibition_type {
            InhibitionType::Lateral => &mut self.lateral_inhibition,
            InhibitionType::Hierarchical => &mut self.hierarchical_inhibition,
            InhibitionType::Contextual => &mut self.contextual_inhibition,
            InhibitionType::Temporal => &mut self.temporal_inhibition,
        };
        
        if strength > 0.0 {
            map.insert((source, target), strength.clamp(0.0, 1.0));
        } else {
            map.remove(&(source, target));
        }
    }
    
    fn update_inhibition_strength(&mut self, source: EntityKey, target: EntityKey, delta: f32, inhibition_type: InhibitionType) {
        let current = self.get_inhibition_strength(source, target, inhibition_type);
        self.set_inhibition_strength(source, target, current + delta, inhibition_type);
    }
    
    fn get_total_inhibition(&self, source: EntityKey, target: EntityKey) -> f32 {
        let lateral = self.lateral_inhibition.get(&(source, target)).copied().unwrap_or(0.0);
        let hierarchical = self.hierarchical_inhibition.get(&(source, target)).copied().unwrap_or(0.0);
        let contextual = self.contextual_inhibition.get(&(source, target)).copied().unwrap_or(0.0);
        let temporal = self.temporal_inhibition.get(&(source, target)).copied().unwrap_or(0.0);
        
        // Combined inhibition with different weights
        (lateral * 0.4 + hierarchical * 0.3 + contextual * 0.2 + temporal * 0.1).min(1.0)
    }
}