//! Test Data Registry System
//! 
//! Manages test data generation, caching, and lifecycle for all simulation components.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;
use crate::infrastructure::{TestConfig, DataGenerator, GenerationParams, DataSize};

/// Properties describing a test dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProperties {
    pub size: DataSize,
    pub entity_count: usize,
    pub relationship_count: usize,
    pub embedding_dimension: usize,
    pub checksum: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Descriptor for a test dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDescriptor {
    pub id: String,
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub properties: DataProperties,
    pub tags: Vec<String>,
}

/// Test data registry for managing all test datasets
pub struct TestDataRegistry {
    config: TestConfig,
    data_dir: PathBuf,
    datasets: HashMap<String, DatasetDescriptor>,
    generator: DataGenerator,
}

impl TestDataRegistry {
    /// Create a new test data registry
    pub async fn new(config: &TestConfig) -> Result<Self> {
        let data_dir = config.data_directory.clone().unwrap_or_else(|| {
            std::env::temp_dir().join("llmkg_test_data")
        });

        // Ensure data directory exists
        fs::create_dir_all(&data_dir).await?;

        let generator = DataGenerator::new()?;
        let mut registry = Self {
            config: config.clone(),
            data_dir,
            datasets: HashMap::new(),
            generator,
        };

        // Load existing datasets
        registry.load_existing_datasets().await?;

        Ok(registry)
    }

    /// Generate test data with specified parameters
    pub async fn generate_data(&mut self, params: &GenerationParams) -> Result<String> {
        let dataset_id = Uuid::new_v4().to_string();
        let file_path = self.data_dir.join(format!("{}.json", dataset_id));

        // Generate the data
        let data = self.generator.generate(params).await?;
        
        // Save to file
        let serialized = serde_json::to_string_pretty(&data)?;
        fs::write(&file_path, serialized).await?;

        // Calculate checksum
        let checksum = self.calculate_checksum(&file_path).await?;

        // Create descriptor
        let properties = DataProperties {
            size: params.size.clone(),
            entity_count: params.entity_count,
            relationship_count: params.relationship_count,
            embedding_dimension: params.embedding_dimension,
            checksum,
            generated_at: chrono::Utc::now(),
        };

        let descriptor = DatasetDescriptor {
            id: dataset_id.clone(),
            name: format!("Generated Dataset {}", dataset_id[..8].to_uppercase()),
            description: format!("Auto-generated test data: {} entities, {} relationships", 
                                params.entity_count, params.relationship_count),
            file_path,
            properties,
            tags: params.tags.clone(),
        };

        // Register the dataset
        self.datasets.insert(dataset_id.clone(), descriptor);
        self.save_registry().await?;

        Ok(dataset_id)
    }

    /// Get dataset by ID
    pub fn get_dataset(&self, dataset_id: &str) -> Option<&DatasetDescriptor> {
        self.datasets.get(dataset_id)
    }

    /// List all datasets matching tags
    pub fn list_datasets(&self, tags: Option<&[String]>) -> Vec<&DatasetDescriptor> {
        self.datasets.values()
            .filter(|dataset| {
                if let Some(filter_tags) = tags {
                    filter_tags.iter().any(|tag| dataset.tags.contains(tag))
                } else {
                    true
                }
            })
            .collect()
    }

    /// Load dataset data
    pub async fn load_dataset_data(&self, dataset_id: &str) -> Result<serde_json::Value> {
        let descriptor = self.get_dataset(dataset_id)
            .ok_or_else(|| anyhow!("Dataset not found: {}", dataset_id))?;

        let content = fs::read_to_string(&descriptor.file_path).await?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        Ok(data)
    }

    /// Cleanup old or unused datasets
    pub async fn cleanup(&self) -> Result<()> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(7);
        
        for (id, descriptor) in &self.datasets {
            if descriptor.properties.generated_at < cutoff_time {
                if let Err(e) = fs::remove_file(&descriptor.file_path).await {
                    log::warn!("Failed to cleanup dataset {}: {}", id, e);
                }
            }
        }

        Ok(())
    }

    /// Load existing datasets from disk
    async fn load_existing_datasets(&mut self) -> Result<()> {
        let registry_file = self.data_dir.join("registry.json");
        
        if registry_file.exists() {
            let content = fs::read_to_string(&registry_file).await?;
            let datasets: HashMap<String, DatasetDescriptor> = serde_json::from_str(&content)
                .unwrap_or_default();
            
            // Validate datasets still exist
            for (id, descriptor) in datasets {
                if descriptor.file_path.exists() {
                    self.datasets.insert(id, descriptor);
                }
            }
        }

        Ok(())
    }

    /// Save registry to disk
    async fn save_registry(&self) -> Result<()> {
        let registry_file = self.data_dir.join("registry.json");
        let content = serde_json::to_string_pretty(&self.datasets)?;
        fs::write(&registry_file, content).await?;
        Ok(())
    }

    /// Calculate file checksum
    async fn calculate_checksum(&self, file_path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};
        
        let content = fs::read(file_path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let result = hasher.finalize();
        Ok(hex::encode(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::TestConfig;

    #[tokio::test]
    async fn test_data_registry_creation() {
        let config = TestConfig::default();
        let registry = TestDataRegistry::new(&config).await;
        assert!(registry.is_ok());
    }

    #[tokio::test]
    async fn test_data_generation() {
        let config = TestConfig::default();
        let mut registry = TestDataRegistry::new(&config).await.unwrap();
        
        let params = GenerationParams {
            size: DataSize::Small,
            entity_count: 100,
            relationship_count: 200,
            embedding_dimension: 128,
            tags: vec!["test".to_string()],
        };

        let dataset_id = registry.generate_data(&params).await.unwrap();
        assert!(!dataset_id.is_empty());
        
        let descriptor = registry.get_dataset(&dataset_id);
        assert!(descriptor.is_some());
    }
}