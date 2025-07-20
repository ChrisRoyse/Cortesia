/// Test traits for the attention manager module
/// These traits expose internal functionality for testing purposes
use crate::core::types::EntityKey;
use crate::cognitive::working_memory::MemoryItem;
use crate::cognitive::attention_manager::AttentionType;
use ahash::AHashMap;
use crate::error::Result;
use async_trait::async_trait;

/// Trait exposing internal calculation methods for testing
#[async_trait]
pub trait AttentionCalculator {
    /// Calculate attention weights for given targets
    async fn calculate_attention_weights(
        &self,
        targets: &[EntityKey],
        available_capacity: f32,
        attention_type: &AttentionType,
    ) -> Result<AHashMap<EntityKey, f32>>;
    
    /// Calculate memory load from working memory items
    fn calculate_memory_load(&self, memory_items: &[MemoryItem]) -> f32;
}

/// Trait for testing attention state management
pub trait AttentionStateManager {
    /// Get the current internal attention state
    fn get_internal_state(&self) -> &AttentionState;
    
    /// Set the internal attention state (for testing)
    fn set_internal_state(&mut self, state: AttentionState);
}

// Re-export AttentionState for trait usage
pub use crate::cognitive::attention_manager::AttentionState;