//! Model loader utilities for downloading and loading models from HuggingFace

use super::{Model, ModelConfig, LoadingConfig};
use crate::error::{GraphError, Result};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Model loading state
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingState {
    NotLoaded,
    Downloading(f32), // Progress percentage
    Loading,
    Loaded,
    Failed(String),
}

/// Model loader for downloading and loading models from HuggingFace Hub
#[derive(Debug)]
pub struct ModelLoader {
    cache_dir: PathBuf,
    loading_config: LoadingConfig,
    loaded_models: HashMap<String, Model>,
    loading_states: HashMap<String, LoadingState>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(cache_dir: Option<PathBuf>, loading_config: LoadingConfig) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("huggingface")
                .join("transformers")
        });

        Self {
            cache_dir,
            loading_config,
            loaded_models: HashMap::new(),
            loading_states: HashMap::new(),
        }
    }

    /// Create a default model loader
    pub fn default() -> Self {
        Self::new(None, LoadingConfig::default())
    }

    /// Set cache directory
    pub fn with_cache_dir<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = cache_dir.as_ref().to_path_buf();
        self
    }

    /// Set loading configuration
    pub fn with_loading_config(mut self, config: LoadingConfig) -> Self {
        self.loading_config = config;
        self
    }

    /// Load a model from HuggingFace Hub
    pub async fn load_model(&mut self, model_id: &str, config: Option<ModelConfig>) -> Result<&mut Model> {
        if self.loaded_models.contains_key(model_id) {
            return Ok(self.loaded_models.get_mut(model_id).unwrap());
        }

        self.loading_states.insert(model_id.to_string(), LoadingState::Downloading(0.0));

        // Check if model exists in cache
        let model_cache_path = self.get_model_cache_path(model_id);
        if !model_cache_path.exists() {
            self.download_model(model_id).await?;
        }

        self.loading_states.insert(model_id.to_string(), LoadingState::Loading);

        // Load the model
        let model_config = config.unwrap_or_default();
        let model = self.load_model_from_cache(model_id, model_config)?;

        self.loaded_models.insert(model_id.to_string(), model);
        self.loading_states.insert(model_id.to_string(), LoadingState::Loaded);

        Ok(self.loaded_models.get_mut(model_id).unwrap())
    }

    /// Download model from HuggingFace Hub
    async fn download_model(&mut self, model_id: &str) -> Result<()> {
        // Create cache directory if it doesn't exist
        if !self.cache_dir.exists() {
            std::fs::create_dir_all(&self.cache_dir)
                .map_err(|e| GraphError::StorageError(format!("Failed to create cache directory: {e}")))?;
        }

        // Simulate download process (in a real implementation, this would use huggingface-hub crate)
        let model_cache_path = self.get_model_cache_path(model_id);
        std::fs::create_dir_all(&model_cache_path)
            .map_err(|e| GraphError::StorageError(format!("Failed to create model cache directory: {e}")))?;

        // Create placeholder files to simulate download
        let config_path = model_cache_path.join("config.json");
        let model_path = model_cache_path.join("pytorch_model.bin");
        let tokenizer_path = model_cache_path.join("tokenizer.json");

        std::fs::write(&config_path, r#"{"model_type": "transformer"}"#)
            .map_err(|e| GraphError::StorageError(format!("Failed to write config: {e}")))?;
        
        std::fs::write(&model_path, b"placeholder model data")
            .map_err(|e| GraphError::StorageError(format!("Failed to write model: {e}")))?;
        
        std::fs::write(&tokenizer_path, r#"{"tokenizer": "placeholder"}"#)
            .map_err(|e| GraphError::StorageError(format!("Failed to write tokenizer: {e}")))?;

        // Update progress
        for i in 1..=10 {
            self.loading_states.insert(model_id.to_string(), LoadingState::Downloading(i as f32 * 10.0));
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        Ok(())
    }

    /// Load model from cache
    fn load_model_from_cache(&self, model_id: &str, config: ModelConfig) -> Result<Model> {
        let model_cache_path = self.get_model_cache_path(model_id);
        
        if !model_cache_path.exists() {
            return Err(GraphError::StorageError(format!("Model {model_id} not found in cache")));
        }

        // Create a placeholder model (in a real implementation, this would load the actual model)
        let metadata = super::ModelMetadata {
            name: model_id.to_string(),
            family: "Unknown".to_string(),
            parameters: 0,
            size_category: super::ModelSize::Small,
            huggingface_id: model_id.to_string(),
            architecture: "Transformer".to_string(),
            capabilities: super::ModelCapabilities {
                text_generation: true,
                instruction_following: false,
                chat: false,
                code_generation: false,
                reasoning: false,
                multilingual: false,
            },
            context_length: 2048,
            vocab_size: 50000,
            training_tokens: None,
            release_date: "Unknown".to_string(),
            license: "Unknown".to_string(),
            description: format!("Loaded model: {model_id}"),
        };

        let mut model = Model::new(metadata, config);
        model.load()?;

        Ok(model)
    }

    /// Get model cache path
    fn get_model_cache_path(&self, model_id: &str) -> PathBuf {
        // Replace '/' with '--' for filesystem safety
        let safe_model_id = model_id.replace('/', "--");
        self.cache_dir.join(&safe_model_id)
    }

    /// Unload a model
    pub fn unload_model(&mut self, model_id: &str) -> Result<()> {
        if let Some(mut model) = self.loaded_models.remove(model_id) {
            model.unload()?;
            self.loading_states.insert(model_id.to_string(), LoadingState::NotLoaded);
        }
        Ok(())
    }

    /// Get loading state of a model
    pub fn get_loading_state(&self, model_id: &str) -> LoadingState {
        self.loading_states.get(model_id).cloned().unwrap_or(LoadingState::NotLoaded)
    }

    /// Get all loaded models
    pub fn get_loaded_models(&self) -> Vec<&str> {
        self.loaded_models.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.loaded_models.contains_key(model_id)
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Clear cache for a specific model
    pub fn clear_model_cache(&self, model_id: &str) -> Result<()> {
        let model_cache_path = self.get_model_cache_path(model_id);
        if model_cache_path.exists() {
            std::fs::remove_dir_all(&model_cache_path)
                .map_err(|e| GraphError::StorageError(format!("Failed to clear cache: {e}")))?;
        }
        Ok(())
    }

    /// Clear all cache
    pub fn clear_all_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| GraphError::StorageError(format!("Failed to clear all cache: {e}")))?;
        }
        Ok(())
    }

    /// Get cache size in bytes
    pub fn get_cache_size(&self) -> Result<u64> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let mut total_size = 0u64;
        for entry in walkdir::WalkDir::new(&self.cache_dir) {
            let entry = entry.map_err(|e| GraphError::StorageError(format!("Failed to read cache directory: {e}")))?;
            if entry.file_type().is_file() {
                let metadata = entry.metadata().map_err(|e| GraphError::StorageError(format!("Failed to read file metadata: {e}")))?;
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }

    /// List cached models
    pub fn list_cached_models(&self) -> Result<Vec<String>> {
        if !self.cache_dir.exists() {
            return Ok(vec![]);
        }

        let mut cached_models = Vec::new();
        for entry in std::fs::read_dir(&self.cache_dir)
            .map_err(|e| GraphError::StorageError(format!("Failed to read cache directory: {e}")))? {
            let entry = entry.map_err(|e| GraphError::StorageError(format!("Failed to read directory entry: {e}")))?;
            if entry.file_type().map_err(|e| GraphError::StorageError(format!("Failed to read file type: {e}")))?.is_dir() {
                let model_id = entry.file_name().to_string_lossy().replace("--", "/");
                cached_models.push(model_id);
            }
        }

        Ok(cached_models)
    }
}

/// Download progress callback
pub type ProgressCallback = Box<dyn Fn(f32) + Send + Sync>;

/// Advanced model loader with progress callbacks
pub struct AdvancedModelLoader {
    loader: ModelLoader,
    progress_callbacks: HashMap<String, ProgressCallback>,
}

impl AdvancedModelLoader {
    pub fn new(loader: ModelLoader) -> Self {
        Self {
            loader,
            progress_callbacks: HashMap::new(),
        }
    }

    pub fn set_progress_callback<F>(&mut self, model_id: &str, callback: F)
    where
        F: Fn(f32) + Send + Sync + 'static,
    {
        self.progress_callbacks.insert(model_id.to_string(), Box::new(callback));
    }

    pub async fn load_model(&mut self, model_id: &str, config: Option<ModelConfig>) -> Result<&mut Model> {
        // Set up progress monitoring
        if let Some(callback) = self.progress_callbacks.get(model_id) {
            // In a real implementation, this would wire up the callback to the download progress
            callback(100.0);
        }

        self.loader.load_model(model_id, config).await
    }
}

/// Model loading statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingStatistics {
    pub total_models_loaded: usize,
    pub total_cache_size: u64,
    pub average_load_time: f64,
    pub most_used_models: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_loader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(Some(temp_dir.path().to_path_buf()), LoadingConfig::default());
        
        assert_eq!(loader.cache_dir(), temp_dir.path());
        assert_eq!(loader.get_loaded_models().len(), 0);
        assert!(!loader.is_model_loaded("test-model"));
    }

    #[test]
    fn test_model_cache_path() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(Some(temp_dir.path().to_path_buf()), LoadingConfig::default());
        
        let cache_path = loader.get_model_cache_path("microsoft/DialoGPT-medium");
        let expected_path = temp_dir.path().join("microsoft--DialoGPT-medium");
        
        assert_eq!(cache_path, expected_path);
    }

    #[tokio::test]
    async fn test_model_loading() {
        let temp_dir = TempDir::new().unwrap();
        let mut loader = ModelLoader::new(Some(temp_dir.path().to_path_buf()), LoadingConfig::default());
        
        let result = loader.load_model("test-model", None).await;
        assert!(result.is_ok());
        
        assert!(loader.is_model_loaded("test-model"));
        assert_eq!(loader.get_loaded_models().len(), 1);
        assert_eq!(loader.get_loading_state("test-model"), LoadingState::Loaded);
    }

    #[test]
    fn test_cache_operations() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(Some(temp_dir.path().to_path_buf()), LoadingConfig::default());
        
        assert_eq!(loader.get_cache_size().unwrap(), 0);
        assert_eq!(loader.list_cached_models().unwrap().len(), 0);
    }
}