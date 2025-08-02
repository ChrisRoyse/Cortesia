//! Model Registry
//! 
//! Manages registration and discovery of available models with their metadata.
//! Provides intelligent model selection based on task complexity and resource constraints.

use std::collections::HashMap;
use crate::enhanced_knowledge_storage::types::*;
// use crate::models::{smollm, minilm}; // Removed unused imports

/// Registry of available models with their metadata and capabilities
#[derive(Debug)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    /// Create a registry pre-populated with default small language models
    pub fn with_default_models() -> Self {
        let mut registry = Self::new();
        registry.register_default_models();
        registry
    }
    
    /// Register a model with its metadata
    pub fn register_model(&mut self, model_id: &str, metadata: ModelMetadata) {
        self.models.insert(model_id.to_string(), metadata);
    }
    
    /// Get metadata for a specific model
    pub fn get_model_metadata(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }
    
    /// Suggest the optimal model for a given task complexity
    pub fn suggest_optimal_model(&self, complexity: TaskComplexity) -> Option<&ModelMetadata> {
        let target_complexity = match complexity {
            TaskComplexity::Low => ComplexityLevel::Low,
            TaskComplexity::Medium => ComplexityLevel::Medium,
            TaskComplexity::High => ComplexityLevel::High,
        };
        
        // Find models matching the target complexity level
        let matching_models: Vec<_> = self.models
            .values()
            .filter(|m| m.complexity_level == target_complexity)
            .collect();
        
        if !matching_models.is_empty() {
            // Return the smallest model (by parameters) that matches complexity
            matching_models.iter()
                .min_by_key(|m| m.parameters)
                .copied()
        } else {
            // Fallback: find the most suitable model across all complexities
            self.find_best_fallback_model(target_complexity)
        }
    }
    
    /// List all models that match a specific complexity level
    pub fn list_models_by_complexity(&self, complexity: TaskComplexity) -> Vec<&ModelMetadata> {
        let target_complexity = match complexity {
            TaskComplexity::Low => ComplexityLevel::Low,
            TaskComplexity::Medium => ComplexityLevel::Medium,
            TaskComplexity::High => ComplexityLevel::High,
        };
        
        self.models
            .values()
            .filter(|m| m.complexity_level == target_complexity)
            .collect()
    }
    
    /// Get models that fit within a memory constraint
    pub fn get_models_within_memory_limit(&self, max_memory: u64) -> Vec<&ModelMetadata> {
        self.models
            .values()
            .filter(|m| m.memory_footprint <= max_memory)
            .collect()
    }
    
    /// Get all available model IDs
    pub fn get_model_ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
    
    /// Check if a model is registered
    pub fn has_model(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }
    
    /// Remove a model from the registry
    pub fn unregister_model(&mut self, model_id: &str) -> Option<ModelMetadata> {
        self.models.remove(model_id)
    }
    
    /// Get total number of registered models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
    
    /// Register default small language models
    fn register_default_models(&mut self) {
        // Register SmolLM variants
        self.register_smollm_models();
        self.register_minilm_models();
    }
    
    /// Register SmolLM model variants
    fn register_smollm_models(&mut self) {
        // SmolLM2-135M - Smallest, for low complexity tasks
        self.register_model("smollm2_135m", ModelMetadata {
            name: "SmolLM2-135M".to_string(),
            parameters: 135_000_000,
            memory_footprint: 270_000_000, // ~270MB
            complexity_level: ComplexityLevel::Low,
            model_type: "Language Model".to_string(),
            huggingface_id: "HuggingFaceTB/SmolLM2-135M".to_string(),
            supported_tasks: vec![
                "text_generation".to_string(),
                "entity_extraction".to_string(),
                "simple_classification".to_string(),
            ],
        });
        
        // SmolLM2-360M - Medium complexity tasks
        self.register_model("smollm2_360m", ModelMetadata {
            name: "SmolLM2-360M".to_string(),
            parameters: 360_000_000,
            memory_footprint: 720_000_000, // ~720MB
            complexity_level: ComplexityLevel::Medium,
            model_type: "Language Model".to_string(),
            huggingface_id: "HuggingFaceTB/SmolLM2-360M".to_string(),
            supported_tasks: vec![
                "text_generation".to_string(),
                "entity_extraction".to_string(),
                "relationship_extraction".to_string(),
                "semantic_analysis".to_string(),
                "classification".to_string(),
            ],
        });
        
        // For high complexity tasks, register a variant that maps to the NER model
        // The 1.7B model has been removed as it's too large
        self.register_model("smollm2_high", ModelMetadata {
            name: "SmolLM2-High".to_string(),
            parameters: 340_000_000, // Maps to BERT large
            memory_footprint: 1_360_000_000, // ~1.36GB
            complexity_level: ComplexityLevel::High,
            model_type: "Language Model".to_string(),
            huggingface_id: "dbmdz/bert-large-cased-finetuned-conll03-english".to_string(),
            supported_tasks: vec![
                "text_generation".to_string(),
                "complex_reasoning".to_string(),
                "multi_step_analysis".to_string(),
                "entity_extraction".to_string(),
                "relationship_extraction".to_string(),
                "semantic_analysis".to_string(),
            ],
        });
    }
    
    /// Register MiniLM model variants for specialized tasks
    fn register_minilm_models(&mut self) {
        // MiniLM-L6-v2 - Sentence embeddings and similarity
        self.register_model("minilm_l6_v2", ModelMetadata {
            name: "all-MiniLM-L6-v2".to_string(),
            parameters: 22_000_000,
            memory_footprint: 90_000_000, // ~90MB
            complexity_level: ComplexityLevel::Low,
            model_type: "Sentence Transformer".to_string(),
            huggingface_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            supported_tasks: vec![
                "sentence_embeddings".to_string(),
                "semantic_similarity".to_string(),
                "text_classification".to_string(),
            ],
        });
        
        // MiniLM-L12-v2 - Higher quality embeddings
        self.register_model("minilm_l12_v2", ModelMetadata {
            name: "all-MiniLM-L12-v2".to_string(),
            parameters: 33_000_000,
            memory_footprint: 130_000_000, // ~130MB
            complexity_level: ComplexityLevel::Medium,
            model_type: "Sentence Transformer".to_string(),
            huggingface_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            supported_tasks: vec![
                "sentence_embeddings".to_string(),
                "semantic_similarity".to_string(),
                "text_classification".to_string(),
                "semantic_search".to_string(),
            ],
        });
    }
    
    /// Find the best fallback model when no exact complexity match exists
    fn find_best_fallback_model(&self, target_complexity: ComplexityLevel) -> Option<&ModelMetadata> {
        let target_params = match target_complexity {
            ComplexityLevel::Low => 200_000_000,    // ~200M parameters
            ComplexityLevel::Medium => 500_000_000, // ~500M parameters  
            ComplexityLevel::High => 1_500_000_000, // ~1.5B parameters
        };
        
        // Find the model with parameters closest to target
        self.models
            .values()
            .min_by_key(|m| {
                (m.parameters as i64 - target_params as i64).abs()
            })
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::with_default_models()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);
        
        let default_registry = ModelRegistry::with_default_models();
        assert!(default_registry.model_count() > 0);
    }
    
    #[test]
    fn test_model_registration_and_retrieval() {
        let mut registry = ModelRegistry::new();
        
        let metadata = ModelMetadata {
            name: "Test Model".to_string(),
            parameters: 100_000_000,
            memory_footprint: 200_000_000,
            complexity_level: ComplexityLevel::Low,
            model_type: "Test".to_string(),
            huggingface_id: "test/model".to_string(),
            supported_tasks: vec!["test".to_string()],
        };
        
        registry.register_model("test_model", metadata.clone());
        
        let retrieved = registry.get_model_metadata("test_model").unwrap();
        assert_eq!(retrieved.name, "Test Model");
        assert_eq!(retrieved.parameters, 100_000_000);
    }
    
    #[test]
    fn test_optimal_model_suggestion() {
        let registry = ModelRegistry::with_default_models();
        
        let low_model = registry.suggest_optimal_model(TaskComplexity::Low).unwrap();
        let medium_model = registry.suggest_optimal_model(TaskComplexity::Medium).unwrap();
        let high_model = registry.suggest_optimal_model(TaskComplexity::High).unwrap();
        
        assert!(low_model.parameters < high_model.parameters);
        assert_eq!(low_model.complexity_level, ComplexityLevel::Low);
        assert_eq!(medium_model.complexity_level, ComplexityLevel::Medium);
        assert_eq!(high_model.complexity_level, ComplexityLevel::High);
        
        // Verify each complexity level selects a different model
        assert_ne!(low_model.name, medium_model.name);
        assert_ne!(medium_model.name, high_model.name);
        assert_ne!(low_model.name, high_model.name);
    }
    
    #[test]
    fn test_memory_constraint_filtering() {
        let registry = ModelRegistry::with_default_models();
        
        let small_memory_models = registry.get_models_within_memory_limit(500_000_000);
        let large_memory_models = registry.get_models_within_memory_limit(5_000_000_000);
        
        assert!(!small_memory_models.is_empty());
        assert!(large_memory_models.len() >= small_memory_models.len());
        
        for model in &small_memory_models {
            assert!(model.memory_footprint <= 500_000_000);
        }
    }
    
    #[test]
    fn test_complexity_level_filtering() {
        let registry = ModelRegistry::with_default_models();
        
        let low_models = registry.list_models_by_complexity(TaskComplexity::Low);
        let medium_models = registry.list_models_by_complexity(TaskComplexity::Medium);
        let high_models = registry.list_models_by_complexity(TaskComplexity::High);
        
        for model in &low_models {
            assert_eq!(model.complexity_level, ComplexityLevel::Low);
        }
        
        for model in &medium_models {
            assert_eq!(model.complexity_level, ComplexityLevel::Medium);
        }
        
        for model in &high_models {
            assert_eq!(model.complexity_level, ComplexityLevel::High);
        }
    }
}