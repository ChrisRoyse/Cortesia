// Enhanced memory-mapped storage with file persistence and zero-copy operations

use crate::core::types::{EntityKey, EntityData};
use crate::embedding::quantizer::ProductQuantizer;
use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

// Memory-mapped file header with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMapHeader {
    pub magic: [u8; 8],           // "LLMKGDB\0"
    pub version: u32,             // File format version
    pub entity_count: u32,        // Number of entities
    pub embedding_dim: u32,       // Embedding dimension
    pub quantizer_subvectors: u32, // Number of subvectors in quantizer
    pub total_file_size: u64,     // Total file size
    pub entity_section_offset: u64,
    pub entity_section_size: u64,
    pub embedding_section_offset: u64,
    pub embedding_section_size: u64,
    pub quantizer_section_offset: u64,
    pub quantizer_section_size: u64,
    pub index_section_offset: u64,
    pub index_section_size: u64,
    pub checksum: u64,            // Data integrity checksum
}

impl Default for MMapHeader {
    fn default() -> Self {
        Self {
            magic: *b"LLMKGDB\0",
            version: 1,
            entity_count: 0,
            embedding_dim: 0,
            quantizer_subvectors: 0,
            total_file_size: 0,
            entity_section_offset: 0,
            entity_section_size: 0,
            embedding_section_offset: 0,
            embedding_section_size: 0,
            quantizer_section_offset: 0,
            quantizer_section_size: 0,
            index_section_offset: 0,
            index_section_size: 0,
            checksum: 0,
        }
    }
}

// Compact entity representation for memory-mapped storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMapEntity {
    pub entity_key: u64,          // EntityKey hash
    pub embedding_offset: u32,    // Offset in quantized embedding section
    pub property_size: u16,       // Size of property data
    pub relationship_count: u16,  // Number of relationships
    pub flags: u32,               // Entity flags (type, etc.)
}

impl Default for MMapEntity {
    fn default() -> Self {
        Self {
            entity_key: 0,
            embedding_offset: 0,
            property_size: 0,
            relationship_count: 0,
            flags: 0,
        }
    }
}

/// Persistent memory-mapped storage with Product Quantization integration
pub struct PersistentMMapStorage {
    file_path: PathBuf,
    file: Option<File>,
    
    // Memory-mapped regions
    header: MMapHeader,
    entities: Vec<MMapEntity>,
    quantized_embeddings: Vec<u8>,
    
    // Integrated quantizer
    quantizer: Arc<RwLock<ProductQuantizer>>,
    
    // Fast lookup structures
    entity_index: RwLock<HashMap<EntityKey, u32>>, // EntityKey -> index in entities vec
    
    // Statistics
    memory_usage: AtomicU64,
    file_size: AtomicU64,
    read_count: AtomicU64,
    write_count: AtomicU64,
    
    // Configuration
    auto_sync: bool,
    compression_enabled: bool,
}

impl PersistentMMapStorage {
    /// Create new persistent storage with optional file path
    pub fn new<P: AsRef<Path>>(file_path: Option<P>, embedding_dim: usize) -> Result<Self> {
        let file_path = file_path
            .map(|p| p.as_ref().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("llmkg_storage.db"));
        
        let quantizer_subvectors = if embedding_dim >= 256 { 8 } else { 4 };
        let quantizer = Arc::new(RwLock::new(
            ProductQuantizer::new(embedding_dim, quantizer_subvectors)?
        ));
        
        let mut storage = Self {
            file_path,
            file: None,
            header: MMapHeader::default(),
            entities: Vec::new(),
            quantized_embeddings: Vec::new(),
            quantizer,
            entity_index: RwLock::new(HashMap::new()),
            memory_usage: AtomicU64::new(0),
            file_size: AtomicU64::new(0),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            auto_sync: true,
            compression_enabled: true,
        };
        
        storage.header.embedding_dim = embedding_dim as u32;
        storage.header.quantizer_subvectors = quantizer_subvectors as u32;
        
        Ok(storage)
    }
    
    /// Load existing storage from file
    pub fn load<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        
        if !file_path.exists() {
            return Err(GraphError::IndexCorruption);
        }
        
        let mut file = File::open(&file_path)?;
        
        // Read and validate header  
        let header: MMapHeader = bincode::deserialize_from(&mut file)
            .map_err(|_| GraphError::IndexCorruption)?;
        
        if &header.magic != b"LLMKGDB\0" {
            return Err(GraphError::IndexCorruption);
        }
        
        if header.version != 1 {
            return Err(GraphError::IndexCorruption);
        }
        
        // Create quantizer with loaded parameters
        let quantizer = Arc::new(RwLock::new(
            ProductQuantizer::new(header.embedding_dim as usize, header.quantizer_subvectors as usize)?
        ));
        
        let file_size = header.total_file_size;
        
        let mut storage = Self {
            file_path,
            file: Some(file),
            header,
            entities: Vec::new(),
            quantized_embeddings: Vec::new(),
            quantizer,
            entity_index: RwLock::new(HashMap::new()),
            memory_usage: AtomicU64::new(0),
            file_size: AtomicU64::new(file_size),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            auto_sync: true,
            compression_enabled: true,
        };
        
        // Load sections from file
        storage.load_sections()?;
        
        Ok(storage)
    }
    
    /// Load all data sections from file
    fn load_sections(&mut self) -> Result<()> {
        let file = self.file.as_mut().ok_or(GraphError::IndexCorruption)?;
        
        // Load entities - they're right after the header
        if self.header.entity_count > 0 {
            
            for _i in 0..self.header.entity_count {
                let entity: MMapEntity = bincode::deserialize_from(&mut *file)
                    .map_err(|_| GraphError::IndexCorruption)?;
                self.entities.push(entity);
            }
            
            // Build entity index
            let mut index = self.entity_index.write();
            for (i, entity) in self.entities.iter().enumerate() {
                index.insert(EntityKey::from_raw(entity.entity_key), i as u32);
            }
        }
        
        // Load quantized embeddings - they're right after the entities
        if self.header.embedding_section_size > 0 {
            self.quantized_embeddings = vec![0u8; self.header.embedding_section_size as usize];
            file.read_exact(&mut self.quantized_embeddings)?;
        }
        
        // Load quantizer data
        if self.header.quantizer_section_size > 0 {
            file.seek(SeekFrom::Start(self.header.quantizer_section_offset))?;
            let mut quantizer_data = vec![0u8; self.header.quantizer_section_size as usize];
            file.read_exact(&mut quantizer_data)?;
            
            // Deserialize quantizer (simplified - in real implementation use proper serialization)
            // For now, we'll retrain the quantizer if needed
        }
        
        self.memory_usage.store(
            (self.entities.len() * std::mem::size_of::<MMapEntity>() + self.quantized_embeddings.len()) as u64,
            Ordering::Relaxed
        );
        
        Ok(())
    }
    
    /// Add entity with automatic quantization
    pub fn add_entity(&mut self, entity_key: EntityKey, _data: &EntityData, embedding: &[f32]) -> Result<()> {
        // Quantize embedding
        let quantized = {
            let quantizer = self.quantizer.read();
            if !quantizer.is_trained() {
                drop(quantizer);
                // Auto-train quantizer if not trained yet
                let mut quantizer = self.quantizer.write();
                if !quantizer.is_trained() {
                    quantizer.train_adaptive(&[embedding.to_vec()])?;
                }
                quantizer.encode(embedding)?
            } else {
                quantizer.encode(embedding)?
            }
        };
        
        // Store quantized embedding
        let embedding_offset = self.quantized_embeddings.len() as u32;
        self.quantized_embeddings.extend_from_slice(&quantized);
        
        // Create compact entity
        let entity = MMapEntity {
            entity_key: entity_key.as_raw(),
            embedding_offset,
            property_size: 0, // TODO: serialize properties
            relationship_count: 0, // relationships stored separately
            flags: 0,
        };
        
        // Add to storage
        let entity_index = self.entities.len() as u32;
        self.entities.push(entity);
        
        // Update index
        {
            let mut index = self.entity_index.write();
            index.insert(entity_key, entity_index);
        }
        
        // Update memory usage
        self.memory_usage.fetch_add(
            (std::mem::size_of::<MMapEntity>() + quantized.len()) as u64,
            Ordering::Relaxed
        );
        
        self.write_count.fetch_add(1, Ordering::Relaxed);
        
        // Auto-sync if enabled
        if self.auto_sync && self.entities.len() % 100 == 0 {
            self.sync_to_disk()?;
        }
        
        Ok(())
    }
    
    /// Batch add entities for better performance
    pub fn batch_add_entities(&mut self, entities_data: &[(EntityKey, EntityData, Vec<f32>)]) -> Result<()> {
        if entities_data.is_empty() {
            return Ok(());
        }
        
        // Train quantizer if needed with all embeddings
        {
            let quantizer = self.quantizer.read();
            if !quantizer.is_trained() {
                drop(quantizer);
                let embeddings: Vec<Vec<f32>> = entities_data.iter()
                    .map(|(_, _, emb)| emb.clone())
                    .collect();
                
                let mut quantizer = self.quantizer.write();
                if !quantizer.is_trained() {
                    quantizer.train_adaptive(&embeddings)?;
                }
            }
        }
        
        // Batch quantize embeddings
        let embeddings: Vec<Vec<f32>> = entities_data.iter()
            .map(|(_, _, emb)| emb.clone())
            .collect();
        
        let quantized_batch = {
            let quantizer = self.quantizer.read();
            quantizer.batch_encode(&embeddings)?
        };
        
        // Add all entities
        let mut index = self.entity_index.write();
        let mut total_memory_added = 0;
        
        for ((entity_key, _data, _embedding), quantized) in entities_data.iter().zip(quantized_batch.iter()) {
            let embedding_offset = self.quantized_embeddings.len() as u32;
            self.quantized_embeddings.extend_from_slice(quantized);
            
            let entity = MMapEntity {
                entity_key: entity_key.as_raw(),
                embedding_offset,
                property_size: 0,
                relationship_count: 0, // relationships stored separately
                flags: 0,
            };
            
            let entity_index = self.entities.len() as u32;
            self.entities.push(entity);
            index.insert(*entity_key, entity_index);
            
            total_memory_added += std::mem::size_of::<MMapEntity>() + quantized.len();
        }
        
        self.memory_usage.fetch_add(total_memory_added as u64, Ordering::Relaxed);
        self.write_count.fetch_add(entities_data.len() as u64, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get entity by key
    pub fn get_entity(&self, entity_key: EntityKey) -> Option<&MMapEntity> {
        let index = self.entity_index.read();
        let entity_index = *index.get(&entity_key)?;
        drop(index);
        
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.entities.get(entity_index as usize)
    }
    
    /// Get quantized embedding for entity
    pub fn get_quantized_embedding(&self, entity_key: EntityKey) -> Option<&[u8]> {
        let entity = self.get_entity(entity_key)?;
        let start = entity.embedding_offset as usize;
        let quantizer = self.quantizer.read();
        let size = quantizer.num_subspaces();
        drop(quantizer);
        
        if start + size <= self.quantized_embeddings.len() {
            Some(&self.quantized_embeddings[start..start + size])
        } else {
            None
        }
    }
    
    /// Reconstruct full embedding from quantized data
    pub fn get_reconstructed_embedding(&self, entity_key: EntityKey) -> Result<Option<Vec<f32>>> {
        if let Some(quantized) = self.get_quantized_embedding(entity_key) {
            let quantizer = self.quantizer.read();
            Ok(Some(quantizer.decode(quantized)?))
        } else {
            Ok(None)
        }
    }
    
    /// Similarity search using quantized embeddings
    pub fn similarity_search(&self, query: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let quantizer = self.quantizer.read();
        let mut results = Vec::new();
        
        // Use asymmetric distance for efficiency
        for entity in &self.entities {
            let entity_key = EntityKey::from_raw(entity.entity_key);
            if let Some(quantized) = self.get_quantized_embedding(entity_key) {
                if let Ok(distance) = quantizer.asymmetric_distance(query, quantized) {
                    let similarity = 1.0 / (1.0 + distance);
                    results.push((entity_key, similarity));
                }
            }
        }
        
        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Sync data to disk
    pub fn sync_to_disk(&mut self) -> Result<()> {
        self.header.entity_count = self.entities.len() as u32;
        self.header.embedding_section_size = self.quantized_embeddings.len() as u64;
        
        // Open/create file for writing
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;
        
        // Write header
        bincode::serialize_into(&mut file, &self.header)
            .map_err(|_| GraphError::IndexCorruption)?;
        
        // Write entities
        for entity in &self.entities {
            bincode::serialize_into(&mut file, entity)
                .map_err(|_| GraphError::IndexCorruption)?;
        }
        
        // Write quantized embeddings
        if !self.quantized_embeddings.is_empty() {
            file.write_all(&self.quantized_embeddings)?;
        }
        
        let file_size = file.seek(SeekFrom::Current(0))?;
        file.sync_all()?;
        
        self.file = Some(file);
        self.file_size.store(file_size, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Get comprehensive storage statistics
    pub fn storage_stats(&self) -> StorageStats {
        let quantizer = self.quantizer.read();
        let compression_stats = quantizer.compression_stats(self.header.embedding_dim as usize);
        
        StorageStats {
            entity_count: self.entities.len(),
            memory_usage_bytes: self.memory_usage.load(Ordering::Relaxed),
            file_size_bytes: self.file_size.load(Ordering::Relaxed),
            quantized_embedding_bytes: self.quantized_embeddings.len(),
            compression_ratio: compression_stats.compression_ratio,
            avg_bytes_per_entity: if self.entities.len() > 0 {
                self.memory_usage.load(Ordering::Relaxed) / self.entities.len() as u64
            } else {
                0
            },
            read_operations: self.read_count.load(Ordering::Relaxed),
            write_operations: self.write_count.load(Ordering::Relaxed),
            quantizer_trained: quantizer.is_trained(),
            quantizer_quality: quantizer.training_quality(),
        }
    }
    
    /// Enable/disable auto-sync
    pub fn set_auto_sync(&mut self, enabled: bool) {
        self.auto_sync = enabled;
    }
    
    /// Get quantizer for external operations
    pub fn get_quantizer(&self) -> Arc<RwLock<ProductQuantizer>> {
        self.quantizer.clone()
    }
    
    /// Compact storage to remove unused space
    pub fn compact(&mut self) -> Result<()> {
        // Rebuild embeddings array to remove fragmentation
        let mut new_embeddings = Vec::new();
        let quantizer = self.quantizer.read();
        let codes_size = quantizer.num_subspaces();
        drop(quantizer);
        
        for entity in &mut self.entities {
            let old_offset = entity.embedding_offset as usize;
            let new_offset = new_embeddings.len() as u32;
            
            if old_offset + codes_size <= self.quantized_embeddings.len() {
                new_embeddings.extend_from_slice(&self.quantized_embeddings[old_offset..old_offset + codes_size]);
                entity.embedding_offset = new_offset;
            }
        }
        
        self.quantized_embeddings = new_embeddings;
        
        if self.auto_sync {
            self.sync_to_disk()?;
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub entity_count: usize,
    pub memory_usage_bytes: u64,
    pub file_size_bytes: u64,
    pub quantized_embedding_bytes: usize,
    pub compression_ratio: f32,
    pub avg_bytes_per_entity: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub quantizer_trained: bool,
    pub quantizer_quality: f32,
}

impl std::fmt::Display for StorageStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Storage Stats:\n")?;
        write!(f, "  Entities: {}\n", self.entity_count)?;
        write!(f, "  Memory: {} KB\n", self.memory_usage_bytes / 1024)?;
        write!(f, "  File size: {} KB\n", self.file_size_bytes / 1024)?;
        write!(f, "  Compression: {:.1}:1\n", self.compression_ratio)?;
        write!(f, "  Avg bytes/entity: {}\n", self.avg_bytes_per_entity)?;
        write!(f, "  Operations: {} reads, {} writes", self.read_operations, self.write_operations)
    }
}

impl From<io::Error> for GraphError {
    fn from(_err: io::Error) -> Self {
        GraphError::IndexCorruption // Simplified error mapping
    }
}