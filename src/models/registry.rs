//! Model registry for managing and discovering available models

use super::{ModelMetadata, ModelSize};
use super::smollm::{self, SmolLMVariant};
use super::tinyllama::{self, TinyLlamaVariant};
use super::openelm::{self, OpenELMVariant};
use super::minilm::{self, MiniLMVariant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Model registry for discovering and managing available models
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.populate_default_models();
        registry
    }

    /// Populate registry with default models
    fn populate_default_models(&mut self) {
        // Add SmolLM models
        for variant in smollm::available_variants() {
            if let Ok(model) = match variant {
                SmolLMVariant::SmolLM135M => smollm::smollm_135m().build(),
                SmolLMVariant::SmolLM360M => smollm::smollm_360m().build(),
                SmolLMVariant::SmolLM1_7B => smollm::smollm_1_7b().build(),
                SmolLMVariant::SmolLM135MInstruct => smollm::smollm_135m_instruct().build(),
                SmolLMVariant::SmolLM360MInstruct => smollm::smollm_360m_instruct().build(),
                SmolLMVariant::SmolLM1_7BInstruct => smollm::smollm_1_7b_instruct().build(),
                SmolLMVariant::SmolLM2_135M => smollm::smollm2_135m().build(),
                SmolLMVariant::SmolLM2_360M => smollm::smollm2_360m().build(),
                SmolLMVariant::SmolLM2_1_7B => smollm::smollm2_1_7b().build(),
            } {
                self.models.insert(model.metadata.huggingface_id.clone(), model.metadata);
            }
        }

        // Add TinyLlama models
        for variant in tinyllama::available_variants() {
            if let Ok(model) = match variant {
                TinyLlamaVariant::TinyLlama1_1B => tinyllama::tinyllama_1_1b().build(),
                TinyLlamaVariant::TinyLlama1_1BChat => tinyllama::tinyllama_1_1b_chat().build(),
                TinyLlamaVariant::TinyLlama1_1BChatV0_1 => tinyllama::tinyllama_1_1b_chat_v0_1().build(),
                TinyLlamaVariant::TinyLlama1_1BChatV0_3 => tinyllama::tinyllama_1_1b_chat_v0_3().build(), 
                TinyLlamaVariant::TinyLlama1_1BChatV0_6 => tinyllama::tinyllama_1_1b_chat_v0_6().build(),
                TinyLlamaVariant::TinyLlama1_1BChatV1_0 => tinyllama::tinyllama_1_1b_chat_v1_0().build(),
                TinyLlamaVariant::TinyLlama1_1BIntermediate => tinyllama::tinyllama_1_1b_intermediate().build(),
            } {
                self.models.insert(model.metadata.huggingface_id.clone(), model.metadata);
            }
        }

        // Add OpenELM models
        for variant in openelm::available_variants() {
            if let Ok(model) = match variant {
                OpenELMVariant::OpenELM270M => openelm::openelm_270m().build(),
                OpenELMVariant::OpenELM450M => openelm::openelm_450m().build(),
                OpenELMVariant::OpenELM1_1B => openelm::openelm_1_1b().build(),
                OpenELMVariant::OpenELM3B => openelm::openelm_3b().build(),
                OpenELMVariant::OpenELM270MInstruct => openelm::openelm_270m_instruct().build(),
                OpenELMVariant::OpenELM450MInstruct => openelm::openelm_450m_instruct().build(),
                OpenELMVariant::OpenELM1_1BInstruct => openelm::openelm_1_1b_instruct().build(),
                OpenELMVariant::OpenELM3BInstruct => openelm::openelm_3b_instruct().build(),
            } {
                self.models.insert(model.metadata.huggingface_id.clone(), model.metadata);
            }
        }

        // Add MiniLM models
        for variant in minilm::available_variants() {
            if let Ok(model) = match variant {
                MiniLMVariant::MiniLML12H384 => minilm::minilm_l12_h384().build(),
                MiniLMVariant::MiniLMMultilingualL12H384 => minilm::minilm_multilingual_l12_h384().build(),
                MiniLMVariant::AllMiniLML6V2 => minilm::all_minilm_l6_v2().build(),
                MiniLMVariant::AllMiniLML12V2 => minilm::all_minilm_l12_v2().build(),
                MiniLMVariant::MsMarcoMiniLML6V2 => minilm::ms_marco_minilm_l6_v2().build(),
                MiniLMVariant::MsMarcoMiniLML12V2 => minilm::ms_marco_minilm_l12_v2().build(),
            } {
                self.models.insert(model.metadata.huggingface_id.clone(), model.metadata);
            }
        }
    }

    /// Get all registered models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// Get models by size category
    pub fn list_models_by_size(&self, size: ModelSize) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| metadata.size_category == size)
            .collect()
    }

    /// Get models by family
    pub fn list_models_by_family(&self, family: &str) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| metadata.family == family)
            .collect()
    }

    /// Get models with specific capabilities
    pub fn list_models_with_capability(&self, check: impl Fn(&super::ModelCapabilities) -> bool) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| check(&metadata.capabilities))
            .collect()
    }

    /// Get model by HuggingFace ID
    pub fn get_model(&self, huggingface_id: &str) -> Option<&ModelMetadata> {
        self.models.get(huggingface_id)
    }

    /// Search models by name or description
    pub fn search_models(&self, query: &str) -> Vec<&ModelMetadata> {
        let query = query.to_lowercase();
        self.models.values()
            .filter(|metadata| {
                metadata.name.to_lowercase().contains(&query) ||
                metadata.description.to_lowercase().contains(&query) ||
                metadata.family.to_lowercase().contains(&query)
            })
            .collect()
    }

    /// Get models in parameter range
    pub fn list_models_in_parameter_range(&self, min_params: u64, max_params: u64) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| metadata.parameters >= min_params && metadata.parameters <= max_params)
            .collect()
    }

    /// Get the smallest model by parameters
    pub fn get_smallest_model(&self) -> Option<&ModelMetadata> {
        self.models.values().min_by_key(|metadata| metadata.parameters)
    }

    /// Get the largest model by parameters (within 500M limit)
    pub fn get_largest_model(&self) -> Option<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| metadata.parameters <= 500_000_000)
            .max_by_key(|metadata| metadata.parameters)
    }

    /// Get recommended models for different use cases
    pub fn get_recommended_models(&self) -> RecommendedModels {
        let mut recommended = RecommendedModels::default();

        // Best overall small models
        if let Some(model) = self.get_model("HuggingFaceTB/SmolLM-360M") {
            recommended.best_overall = Some(model.clone());
        }

        // Best for chat
        if let Some(model) = self.get_model("HuggingFaceTB/SmolLM-360M-Instruct") {
            recommended.best_chat = Some(model.clone());
        }

        // Most efficient (smallest with good performance)
        if let Some(model) = self.get_model("HuggingFaceTB/SmolLM-135M") {
            recommended.most_efficient = Some(model.clone());
        }

        // Best for embeddings
        if let Some(model) = self.get_model("sentence-transformers/all-MiniLM-L6-v2") {
            recommended.best_embeddings = Some(model.clone());
        }

        // Best multilingual
        if let Some(model) = self.get_model("microsoft/Multilingual-MiniLM-L12-H384") {
            recommended.best_multilingual = Some(model.clone());
        }

        // Latest and greatest
        if let Some(model) = self.get_model("HuggingFaceTB/SmolLM2-360M") {
            recommended.latest = Some(model.clone());
        }

        recommended
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        let total_models = self.models.len();
        let families: std::collections::HashSet<_> = self.models.values().map(|m| &m.family).collect();
        
        let mut size_distribution = HashMap::new();
        for metadata in self.models.values() {
            *size_distribution.entry(metadata.size_category).or_insert(0) += 1;
        }

        let mut capability_stats = CapabilityStatistics::default();
        for metadata in self.models.values() {
            if metadata.capabilities.text_generation { capability_stats.text_generation += 1; }
            if metadata.capabilities.instruction_following { capability_stats.instruction_following += 1; }
            if metadata.capabilities.chat { capability_stats.chat += 1; }
            if metadata.capabilities.code_generation { capability_stats.code_generation += 1; }
            if metadata.capabilities.reasoning { capability_stats.reasoning += 1; }
            if metadata.capabilities.multilingual { capability_stats.multilingual += 1; }
        }

        let total_parameters: u64 = self.models.values().map(|m| m.parameters).sum();
        let avg_parameters = if total_models > 0 { total_parameters / total_models as u64 } else { 0 };

        RegistryStatistics {
            total_models,
            total_families: families.len(),
            size_distribution,
            capability_stats,
            total_parameters,
            average_parameters: avg_parameters,
        }
    }

    /// Register a custom model
    pub fn register_model(&mut self, metadata: ModelMetadata) {
        self.models.insert(metadata.huggingface_id.clone(), metadata);
    }

    /// Unregister a model
    pub fn unregister_model(&mut self, huggingface_id: &str) -> Option<ModelMetadata> {
        self.models.remove(huggingface_id)
    }

    /// Check if a model is registered
    pub fn is_registered(&self, huggingface_id: &str) -> bool {
        self.models.contains_key(huggingface_id)
    }

    /// Get model count
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommended models for different use cases
#[derive(Debug, Clone, Default)]
pub struct RecommendedModels {
    pub best_overall: Option<ModelMetadata>,
    pub best_chat: Option<ModelMetadata>,
    pub most_efficient: Option<ModelMetadata>,
    pub best_embeddings: Option<ModelMetadata>,
    pub best_multilingual: Option<ModelMetadata>,
    pub latest: Option<ModelMetadata>,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    pub total_models: usize,
    pub total_families: usize,
    pub size_distribution: HashMap<ModelSize, usize>,
    pub capability_stats: CapabilityStatistics,
    pub total_parameters: u64,
    pub average_parameters: u64,
}

/// Capability statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityStatistics {
    pub text_generation: usize,
    pub instruction_following: usize,
    pub chat: usize,
    pub code_generation: usize,
    pub reasoning: usize,
    pub multilingual: usize,
}

/// Model filter for searching
#[derive(Debug, Clone, Default)]
pub struct ModelFilter {
    pub families: Option<Vec<String>>,
    pub size_categories: Option<Vec<ModelSize>>,
    pub min_parameters: Option<u64>,
    pub max_parameters: Option<u64>,
    pub capabilities: Option<Vec<String>>,
    pub licenses: Option<Vec<String>>,
}

impl ModelFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_families(mut self, families: Vec<String>) -> Self {
        self.families = Some(families);
        self
    }

    pub fn with_size_categories(mut self, sizes: Vec<ModelSize>) -> Self {
        self.size_categories = Some(sizes);
        self
    }

    pub fn with_parameter_range(mut self, min: u64, max: u64) -> Self {
        self.min_parameters = Some(min);
        self.max_parameters = Some(max);
        self
    }

    pub fn matches(&self, metadata: &ModelMetadata) -> bool {
        if let Some(ref families) = self.families {
            if !families.contains(&metadata.family) {
                return false;
            }
        }

        if let Some(ref sizes) = self.size_categories {
            if !sizes.contains(&metadata.size_category) {
                return false;
            }
        }

        if let Some(min) = self.min_parameters {
            if metadata.parameters < min {
                return false;
            }
        }

        if let Some(max) = self.max_parameters {
            if metadata.parameters > max {
                return false;
            }
        }

        true
    }
}

impl ModelRegistry {
    /// Filter models using a filter
    pub fn filter_models(&self, filter: &ModelFilter) -> Vec<&ModelMetadata> {
        self.models.values()
            .filter(|metadata| filter.matches(metadata))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(registry.model_count() > 0);
        
        let stats = registry.get_statistics();
        assert!(stats.total_models > 0);
        assert!(stats.total_families > 0);
    }

    #[test]
    fn test_model_filtering() {
        let registry = ModelRegistry::new();
        
        // Test filtering by family
        let smollm_models = registry.list_models_by_family("SmolLM");
        assert!(!smollm_models.is_empty());
        
        // Test filtering by size
        let small_models = registry.list_models_by_size(ModelSize::Small);
        assert!(!small_models.is_empty());
        
        // Test parameter range filtering
        let small_param_models = registry.list_models_in_parameter_range(100_000_000, 200_000_000);
        assert!(!small_param_models.is_empty());
    }

    #[test]
    fn test_model_search() {
        let registry = ModelRegistry::new();
        
        let smol_results = registry.search_models("SmolLM");
        assert!(!smol_results.is_empty());
        
        let instruct_results = registry.search_models("instruct");
        assert!(!instruct_results.is_empty());
    }

    #[test]
    fn test_recommendations() {
        let registry = ModelRegistry::new();
        let recommendations = registry.get_recommended_models();
        
        assert!(recommendations.best_overall.is_some());
        assert!(recommendations.most_efficient.is_some());
    }

    #[test]
    fn test_model_filter() {
        let registry = ModelRegistry::new();
        
        let filter = ModelFilter::new()
            .with_families(vec!["SmolLM".to_string()])
            .with_parameter_range(100_000_000, 400_000_000);
        
        let filtered_models = registry.filter_models(&filter);
        assert!(!filtered_models.is_empty());
        
        // Verify all results match the filter
        for model in &filtered_models {
            assert_eq!(model.family, "SmolLM");
            assert!(model.parameters >= 100_000_000);
            assert!(model.parameters <= 400_000_000);
        }
    }

    #[test]
    fn test_capability_filtering() {
        let registry = ModelRegistry::new();
        
        let chat_models = registry.list_models_with_capability(|caps| caps.chat);
        assert!(!chat_models.is_empty());
        
        let multilingual_models = registry.list_models_with_capability(|caps| caps.multilingual);
        assert!(!multilingual_models.is_empty());
    }
}