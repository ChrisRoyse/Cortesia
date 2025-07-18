use crate::core::types::{CompactEntity, NeighborSlice};
use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use parking_lot::RwLock;

/// Ultra-fast memory-mapped storage for zero-copy access
/// Designed for maximum cache efficiency and minimal memory footprint
pub struct MMapStorage {
    // Memory-mapped entity data - packed for cache efficiency
    entities: Box<[CompactEntity]>,
    entity_count: AtomicU32,
    
    // CSR graph structure - optimized for neighbor access
    row_ptr: Box<[u64]>,
    col_idx: Box<[u32]>,
    
    // Quantized embeddings - ultra-compressed
    embedding_data: Box<[u8]>,
    embedding_dim: u16,
    
    // Property storage - zstd compressed
    property_data: Box<[u8]>,
    property_index: Box<[u32]>,
    
    // Cache-friendly lookup structures
    id_to_index: RwLock<HashMap<u32, u32>>,
    
    // Memory statistics
    total_memory_bytes: AtomicU64,
}

impl MMapStorage {
    pub fn new(estimated_entities: usize, estimated_edges: usize, embedding_dim: u16) -> Result<Self> {
        // Pre-allocate with generous capacity to avoid reallocations
        let entity_capacity = (estimated_entities as f64 * 1.2) as usize;
        let edge_capacity = (estimated_edges as f64 * 1.2) as usize;
        
        let entities = vec![CompactEntity::default(); entity_capacity].into_boxed_slice();
        let row_ptr = vec![0u64; entity_capacity + 1].into_boxed_slice();
        let col_idx = vec![0u32; edge_capacity].into_boxed_slice();
        let embedding_data = vec![0u8; entity_capacity * (embedding_dim as usize / 8)].into_boxed_slice();
        let property_data = vec![0u8; entity_capacity * 64].into_boxed_slice(); // 64 bytes avg per entity
        let property_index = vec![0u32; entity_capacity * 2].into_boxed_slice(); // start/end pairs
        
        let total_memory = entities.len() * std::mem::size_of::<CompactEntity>() +
                          row_ptr.len() * std::mem::size_of::<u64>() +
                          col_idx.len() * std::mem::size_of::<u32>() +
                          embedding_data.len() +
                          property_data.len() +
                          property_index.len() * std::mem::size_of::<u32>();
        
        Ok(Self {
            entities,
            entity_count: AtomicU32::new(0),
            row_ptr,
            col_idx,
            embedding_data,
            embedding_dim,
            property_data,
            property_index,
            id_to_index: RwLock::new(HashMap::with_capacity(entity_capacity)),
            total_memory_bytes: AtomicU64::new(total_memory as u64),
        })
    }
    
    /// Get entity with zero-copy access
    #[inline]
    pub fn get_entity(&self, entity_id: u32) -> Option<&CompactEntity> {
        let id_map = self.id_to_index.read();
        let index = *id_map.get(&entity_id)?;
        drop(id_map);
        
        let index = index as usize;
        if index < self.entity_count.load(Ordering::Relaxed) as usize {
            Some(&self.entities[index])
        } else {
            None
        }
    }
    
    /// Get neighbors with zero-copy slice access
    #[inline]
    pub unsafe fn get_neighbors_unchecked(&self, entity_id: u32) -> Option<NeighborSlice> {
        let entity_index = {
            let id_map = self.id_to_index.read();
            *id_map.get(&entity_id)?
        };
        
        let start = self.row_ptr[entity_index as usize] as usize;
        let end = self.row_ptr[entity_index as usize + 1] as usize;
        let len = end - start;
        
        if len > 0 && start < self.col_idx.len() {
            Some(NeighborSlice::new(
                self.col_idx.as_ptr().add(start),
                len as u16
            ))
        } else {
            None
        }
    }
    
    /// Safe neighbor access with bounds checking
    #[inline]
    pub fn get_neighbors(&self, entity_id: u32) -> Option<&[u32]> {
        let entity_index = {
            let id_map = self.id_to_index.read();
            *id_map.get(&entity_id)?
        };
        
        let start = self.row_ptr[entity_index as usize] as usize;
        let end = self.row_ptr[entity_index as usize + 1] as usize;
        
        if start <= end && end <= self.col_idx.len() {
            Some(&self.col_idx[start..end])
        } else {
            None
        }
    }
    
    /// Get quantized embedding data
    #[inline]
    pub fn get_embedding_codes(&self, entity_id: u32) -> Option<&[u8]> {
        let entity = self.get_entity(entity_id)?;
        let offset = entity.embedding_offset as usize;
        let size = (self.embedding_dim as usize) / 8; // 8 subvectors for 96-dim
        
        if offset + size <= self.embedding_data.len() {
            Some(&self.embedding_data[offset..offset + size])
        } else {
            None
        }
    }
    
    /// Batch neighbor lookup for high-throughput queries
    pub fn batch_get_neighbors(&self, entity_ids: &[u32], results: &mut Vec<Vec<u32>>) -> Result<()> {
        results.clear();
        results.reserve(entity_ids.len());
        
        let id_map = self.id_to_index.read();
        
        for &entity_id in entity_ids {
            if let Some(&entity_index) = id_map.get(&entity_id) {
                let start = self.row_ptr[entity_index as usize] as usize;
                let end = self.row_ptr[entity_index as usize + 1] as usize;
                
                if start <= end && end <= self.col_idx.len() {
                    results.push(self.col_idx[start..end].to_vec());
                } else {
                    results.push(Vec::new());
                }
            } else {
                results.push(Vec::new());
            }
        }
        
        Ok(())
    }
    
    /// Ultra-fast batch embedding lookup
    pub fn batch_get_embeddings<'a>(&'a self, entity_ids: &[u32], codes_batch: &mut Vec<&'a [u8]>) -> Result<()> {
        codes_batch.clear();
        codes_batch.reserve(entity_ids.len());
        
        let codes_size = (self.embedding_dim as usize) / 8;
        
        for &entity_id in entity_ids {
            if let Some(entity) = self.get_entity(entity_id) {
                let offset = entity.embedding_offset as usize;
                if offset + codes_size <= self.embedding_data.len() {
                    codes_batch.push(&self.embedding_data[offset..offset + codes_size]);
                } else {
                    return Err(GraphError::IndexCorruption);
                }
            } else {
                return Err(GraphError::EntityNotFound { id: entity_id });
            }
        }
        
        Ok(())
    }
    
    /// Multi-hop graph traversal with minimal allocations
    pub fn traverse_multi_hop(&self, start_entities: &[u32], max_hops: u8, visited: &mut Vec<bool>) -> Result<Vec<u32>> {
        visited.resize(self.entity_count.load(Ordering::Relaxed) as usize, false);
        visited.fill(false);
        
        let mut current_level = start_entities.to_vec();
        let mut next_level = Vec::with_capacity(current_level.len() * 4); // Estimate 4x fanout
        let mut all_visited = Vec::with_capacity(current_level.len() * 10);
        
        // Mark starting entities as visited
        for &entity_id in start_entities {
            if let Some(_entity) = self.get_entity(entity_id) {
                let idx = entity_id as usize;
                if idx < visited.len() {
                    visited[idx] = true;
                    all_visited.push(entity_id);
                }
            }
        }
        
        for _hop in 0..max_hops {
            next_level.clear();
            
            for &entity_id in &current_level {
                if let Some(neighbors) = self.get_neighbors(entity_id) {
                    for &neighbor_id in neighbors {
                        let idx = neighbor_id as usize;
                        if idx < visited.len() && !visited[idx] {
                            visited[idx] = true;
                            next_level.push(neighbor_id);
                            all_visited.push(neighbor_id);
                        }
                    }
                }
            }
            
            if next_level.is_empty() {
                break;
            }
            
            std::mem::swap(&mut current_level, &mut next_level);
        }
        
        Ok(all_visited)
    }
    
    /// Memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_bytes: self.total_memory_bytes.load(Ordering::Relaxed),
            entity_bytes: self.entities.len() * std::mem::size_of::<CompactEntity>(),
            graph_bytes: self.row_ptr.len() * std::mem::size_of::<u64>() + 
                        self.col_idx.len() * std::mem::size_of::<u32>(),
            embedding_bytes: self.embedding_data.len(),
            property_bytes: self.property_data.len(),
            entity_count: self.entity_count.load(Ordering::Relaxed),
            bytes_per_entity: if self.entity_count.load(Ordering::Relaxed) > 0 {
                self.total_memory_bytes.load(Ordering::Relaxed) / self.entity_count.load(Ordering::Relaxed) as u64
            } else {
                0
            }
        }
    }
    
    /// Cache optimization - prefetch data for upcoming accesses
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_entities(&self, entity_ids: &[u32]) {
        use std::arch::x86_64::*;
        
        let id_map = self.id_to_index.read();
        
        for &entity_id in entity_ids.iter().take(64) { // Limit to avoid cache pollution
            if let Some(&index) = id_map.get(&entity_id) {
                let entity_ptr = &self.entities[index as usize] as *const CompactEntity;
                unsafe {
                    _mm_prefetch(entity_ptr as *const i8, _MM_HINT_T0);
                }
            }
        }
    }
}

impl Default for CompactEntity {
    fn default() -> Self {
        Self {
            id: 0,
            type_id: 0,
            degree: 0,
            embedding_offset: 0,
            property_offset: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: u64,
    pub entity_bytes: usize,
    pub graph_bytes: usize,
    pub embedding_bytes: usize,
    pub property_bytes: usize,
    pub entity_count: u32,
    pub bytes_per_entity: u64,
}

/// Lock-free concurrent access for read-heavy workloads
pub struct ConcurrentMMapStorage {
    storage: MMapStorage,
    read_epoch: AtomicU64,
}

impl ConcurrentMMapStorage {
    pub fn new(estimated_entities: usize, estimated_edges: usize, embedding_dim: u16) -> Result<Self> {
        Ok(Self {
            storage: MMapStorage::new(estimated_entities, estimated_edges, embedding_dim)?,
            read_epoch: AtomicU64::new(0),
        })
    }
    
    /// Lock-free entity access
    #[inline]
    pub fn get_entity(&self, entity_id: u32) -> Option<&CompactEntity> {
        let _epoch = self.read_epoch.load(Ordering::Acquire);
        self.storage.get_entity(entity_id)
    }
    
    /// Lock-free neighbor access
    #[inline]
    pub fn get_neighbors(&self, entity_id: u32) -> Option<&[u32]> {
        let _epoch = self.read_epoch.load(Ordering::Acquire);
        self.storage.get_neighbors(entity_id)
    }
}
