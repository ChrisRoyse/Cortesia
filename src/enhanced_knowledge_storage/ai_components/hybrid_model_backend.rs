//! Hybrid Model Backend
//! 
//! Intelligent backend that uses local models when available,
//! falls back to downloading from HuggingFace Hub

use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, instrument};

use super::types::*;
use super::ai_model_backend::{AIModelBackend, AIBackendConfig};
use super::local_model_backend::{LocalModelBackend, LocalModelConfig};
use crate::enhanced_knowledge_storage::types::*;
use crate::enhanced_knowledge_storage::model_management::ModelBackend;

/// Configuration for hybrid model backend
#[derive(Debug, Clone)]
pub struct HybridModelConfig {
    pub prefer_local: bool,
    pub local_model_dir: PathBuf,
    pub enable_download: bool,
    pub cache_downloaded: bool,
}

impl Default for HybridModelConfig {
    fn default() -> Self {
        Self {
            prefer_local: true,
            local_model_dir: PathBuf::from("model_weights"),
            enable_download: true,
            cache_downloaded: true,
        }
    }
}

/// Hybrid model backend that combines local and remote models
pub struct HybridModelBackend {
    config: HybridModelConfig,
    local_backend: Option<LocalModelBackend>,
    remote_backend: Option<AIModelBackend>,
    model_availability: Arc<RwLock<HashMap<String, ModelAvailability>>>,
}

#[derive(Debug, Clone)]
enum ModelAvailability {
    Local,
    Remote,
    Both,
    None,
}

impl HybridModelBackend {
    /// Create a new hybrid model backend
    pub async fn new(config: HybridModelConfig) -> Result<Self> {
        info!("Initializing hybrid model backend");
        
        // Try to create local backend
        let local_backend = match LocalModelBackend::new(LocalModelConfig {
            model_weights_dir: config.local_model_dir.clone(),
            ..Default::default()
        }) {
            Ok(backend) => {
                info!("Local model backend initialized successfully");
                Some(backend)
            }
            Err(e) => {
                warn!("Failed to initialize local backend: {}", e);
                None
            }
        };
        
        // Create remote backend if downloads are enabled
        let remote_backend = if config.enable_download {
            match AIModelBackend::new(AIBackendConfig::default()).await {
                Ok(backend) => {
                    info!("Remote model backend initialized successfully");
                    Some(backend)
                }
                Err(e) => {
                    warn!("Failed to initialize remote backend: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        if local_backend.is_none() && remote_backend.is_none() {
            return Err(EnhancedStorageError::ConfigurationError(
                "No model backends available".to_string()
            ));
        }
        
        let mut backend = Self {
            config,
            local_backend,
            remote_backend,
            model_availability: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Check model availability
        backend.update_model_availability().await?;
        
        Ok(backend)
    }
    
    /// Update model availability information
    async fn update_model_availability(&mut self) -> Result<()> {
        let mut availability = self.model_availability.write().await;
        availability.clear();
        
        // List of models we care about
        let target_models = vec![
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "dbmdz/bert-large-cased-finetuned-conll03-english",
        ];
        
        for model_id in target_models {
            let local_available = self.local_backend.as_ref()
                .map(|b| b.is_model_available(model_id))
                .unwrap_or(false);
            
            let remote_available = self.remote_backend.is_some();
            
            let availability_status = match (local_available, remote_available) {
                (true, true) => ModelAvailability::Both,
                (true, false) => ModelAvailability::Local,
                (false, true) => ModelAvailability::Remote,
                (false, false) => ModelAvailability::None,
            };
            
            availability.insert(model_id.to_string(), availability_status);
        }
        
        Ok(())
    }
    
    /// Get the best backend for a model
    async fn get_backend_for_model(&self, model_id: &str) -> Result<ModelBackendChoice> {
        let availability = self.model_availability.read().await;
        let model_avail = availability.get(model_id)
            .unwrap_or(&ModelAvailability::None);
        
        match (model_avail, self.config.prefer_local) {
            (ModelAvailability::Both, true) | (ModelAvailability::Local, _) => {
                if let Some(backend) = &self.local_backend {
                    Ok(ModelBackendChoice::Local(backend))
                } else {
                    Err(EnhancedStorageError::ModelNotFound(
                        format!("Local backend not available for {}", model_id)
                    ))
                }
            }
            (ModelAvailability::Both, false) | (ModelAvailability::Remote, _) => {
                if let Some(backend) = &self.remote_backend {
                    Ok(ModelBackendChoice::Remote(backend))
                } else {
                    Err(EnhancedStorageError::ModelNotFound(
                        format!("Remote backend not available for {}", model_id)
                    ))
                }
            }
            (ModelAvailability::None, _) => {
                Err(EnhancedStorageError::ModelNotFound(
                    format!("Model {} not available in any backend", model_id)
                ))
            }
        }
    }
}

enum ModelBackendChoice<'a> {
    Local(&'a LocalModelBackend),
    Remote(&'a AIModelBackend),
}

#[async_trait::async_trait]
impl ModelBackend for HybridModelBackend {
    async fn load_model(&self, model_id: &str) -> Result<ModelHandle> {
        info!("Loading model {} through hybrid backend", model_id);
        
        // Apply model translation
        let backend_model_id = crate::enhanced_knowledge_storage::model_management::get_backend_model_id(model_id);
        
        match self.get_backend_for_model(&backend_model_id).await? {
            ModelBackendChoice::Local(backend) => {
                debug!("Using local backend for {}", backend_model_id);
                
                // Load from local backend and convert to ModelHandle
                let local_model = backend.load_model(&backend_model_id).await?;
                
                // Get model metadata
                let metadata = ModelMetadata {
                    name: backend_model_id.to_string(),
                    parameters: match backend_model_id {
                        "bert-base-uncased" => 110_000_000,
                        "sentence-transformers/all-MiniLM-L6-v2" => 22_000_000,
                        "dbmdz/bert-large-cased-finetuned-conll03-english" => 340_000_000,
                        _ => 100_000_000,
                    },
                    memory_footprint: match backend_model_id {
                        "bert-base-uncased" => 440_000_000,
                        "sentence-transformers/all-MiniLM-L6-v2" => 90_000_000,
                        "dbmdz/bert-large-cased-finetuned-conll03-english" => 1_360_000_000,
                        _ => 400_000_000,
                    },
                    complexity_level: match backend_model_id {
                        "sentence-transformers/all-MiniLM-L6-v2" => ComplexityLevel::Low,
                        "bert-base-uncased" => ComplexityLevel::Medium,
                        "dbmdz/bert-large-cased-finetuned-conll03-english" => ComplexityLevel::High,
                        _ => ComplexityLevel::Medium,
                    },
                    capabilities: ModelCapabilities {
                        embeddings: true,
                        ner: backend_model_id.contains("ner"),
                        classification: true,
                        generation: false,
                    },
                    huggingface_id: backend_model_id.to_string(),
                };
                
                Ok(ModelHandle {
                    model_id: backend_model_id.to_string(),
                    backend_type: BackendType::Local,
                    metadata,
                    memory_usage: metadata.memory_footprint,
                })
            }
            ModelBackendChoice::Remote(backend) => {
                debug!("Using remote backend for {}", backend_model_id);
                backend.load_model(&backend_model_id).await
            }
        }
    }
    
    async fn unload_model(&self, handle: &ModelHandle) -> Result<()> {
        match handle.backend_type {
            BackendType::Local => {
                // Local models are cached, just report success
                Ok(())
            }
            _ => {
                if let Some(backend) = &self.remote_backend {
                    backend.unload_model(handle).await
                } else {
                    Ok(())
                }
            }
        }
    }
    
    async fn generate_text(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        match handle.backend_type {
            BackendType::Local => {
                // For now, return a placeholder for local models
                // In production, implement actual generation
                Ok(format!("Local generation for: {}", prompt))
            }
            _ => {
                if let Some(backend) = &self.remote_backend {
                    backend.generate_text(handle, prompt, max_tokens).await
                } else {
                    Err(EnhancedStorageError::ModelError(
                        "Remote backend not available".to_string()
                    ))
                }
            }
        }
    }
    
    async fn get_memory_usage(&self, handle: &ModelHandle) -> Result<u64> {
        Ok(handle.memory_usage)
    }
    
    async fn health_check(&self) -> Result<()> {
        let mut any_healthy = false;
        
        if let Some(local) = &self.local_backend {
            any_healthy = true;
            debug!("Local backend is healthy");
        }
        
        if let Some(remote) = &self.remote_backend {
            if remote.health_check().await.is_ok() {
                any_healthy = true;
                debug!("Remote backend is healthy");
            }
        }
        
        if any_healthy {
            Ok(())
        } else {
            Err(EnhancedStorageError::ServiceUnavailable(
                "No healthy backends available".to_string()
            ))
        }
    }
}

use std::collections::HashMap;
use crate::enhanced_knowledge_storage::model_management::{ModelHandle, BackendType};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_config() {
        let config = HybridModelConfig::default();
        assert!(config.prefer_local);
        assert!(config.enable_download);
    }
}