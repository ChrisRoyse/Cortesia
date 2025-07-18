//! Hybrid graph storage combining immutable CSR with mutable overlay
//! 
//! This provides the best of both worlds:
//! - Fast reads via CSR format
//! - Efficient updates via delta storage
//! - Periodic compaction to maintain performance

use crate::core::types::Relationship;
use crate::error::Result;
use crate::storage::csr::CSRGraph;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;

/// Hybrid graph with immutable base and mutable delta
pub struct HybridGraph {
    /// Immutable CSR for bulk of data
    base_csr: Arc<CSRGraph>,
    
    /// Recent additions not yet in CSR
    delta_additions: Arc<RwLock<HashMap<u32, Vec<(u32, u8, f32)>>>>,
    
    /// Recent deletions to filter from CSR
    delta_deletions: Arc<RwLock<HashSet<(u32, u32, u8)>>>,
    
    /// Compaction settings
    compaction_threshold: usize,
    last_compaction: Arc<RwLock<Instant>>,
    
    /// Statistics
    stats: Arc<RwLock<GraphStats>>,
}

#[derive(Default, Clone)]
pub struct GraphStats {
    total_nodes: u32,
    total_edges: u64,
    delta_size: usize,
    compactions: u64,
}

impl HybridGraph {
    pub fn new(initial_csr: CSRGraph) -> Self {
        Self {
            base_csr: Arc::new(initial_csr),
            delta_additions: Arc::new(RwLock::new(HashMap::new())),
            delta_deletions: Arc::new(RwLock::new(HashSet::new())),
            compaction_threshold: 10_000, // Compact after 10k changes
            last_compaction: Arc::new(RwLock::new(Instant::now())),
            stats: Arc::new(RwLock::new(GraphStats::default())),
        }
    }
    
    /// Add an edge with automatic compaction
    pub async fn add_edge(&self, from: u32, to: u32, rel_type: u8, weight: f32) -> Result<()> {
        {
            let mut additions = self.delta_additions.write().await;
            additions.entry(from)
                .or_insert_with(Vec::new)
                .push((to, rel_type, weight));
            
            let mut stats = self.stats.write().await;
            stats.delta_size += 1;
        }
        
        // Check if compaction needed
        if self.should_compact().await {
            self.compact().await?;
        }
        
        Ok(())
    }
    
    /// Remove an edge
    pub async fn remove_edge(&self, from: u32, to: u32, rel_type: u8) -> Result<()> {
        let mut deletions = self.delta_deletions.write().await;
        deletions.insert((from, to, rel_type));
        
        let mut stats = self.stats.write().await;
        stats.delta_size += 1;
        
        Ok(())
    }
    
    /// Get neighbors combining CSR and delta
    pub async fn get_neighbors(&self, node_id: u32) -> Vec<u32> {
        let mut neighbors = HashSet::new();
        
        // Get from CSR
        let csr_neighbors = self.base_csr.get_neighbors(node_id);
        neighbors.extend(csr_neighbors.iter().copied());
        
        // Add from delta
        let additions = self.delta_additions.read().await;
        if let Some(delta_edges) = additions.get(&node_id) {
            for (to, _, _) in delta_edges {
                neighbors.insert(*to);
            }
        }
        
        // Filter deletions
        let deletions = self.delta_deletions.read().await;
        neighbors.retain(|&to| !deletions.iter().any(|(from, t, _)| *from == node_id && *t == to));
        
        neighbors.into_iter().collect()
    }
    
    /// Check if compaction is needed
    async fn should_compact(&self) -> bool {
        let stats = self.stats.read().await;
        stats.delta_size >= self.compaction_threshold
    }
    
    /// Compact delta into new CSR
    pub async fn compact(&self) -> Result<()> {
        let _start = Instant::now();
        
        // Collect all edges
        let mut all_edges = Vec::new();
        
        // Add base CSR edges
        for node_id in 0..self.base_csr.node_count() {
            for (to, rel_type, weight) in self.base_csr.get_edges(node_id) {
                all_edges.push(Relationship {
                    from: node_id,
                    to,
                    rel_type,
                    weight,
                });
            }
        }
        
        // Apply deletions
        let deletions = self.delta_deletions.read().await;
        all_edges.retain(|e| !deletions.contains(&(e.from, e.to, e.rel_type)));
        
        // Add additions
        let additions = self.delta_additions.read().await;
        for (&from, edges) in additions.iter() {
            for &(to, rel_type, weight) in edges {
                all_edges.push(Relationship {
                    from,
                    to,
                    rel_type,
                    weight,
                });
            }
        }
        
        // Build new CSR
        let node_count = all_edges.iter()
            .flat_map(|e| vec![e.from, e.to])
            .max()
            .unwrap_or(0) + 1;
            
        let _new_csr = CSRGraph::from_edges(all_edges, node_count)?;
        
        // Atomic swap
        // Note: In real implementation, would use Arc::swap or similar
        // This is simplified for illustration
        
        // Clear deltas
        self.delta_additions.write().await.clear();
        self.delta_deletions.write().await.clear();
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.delta_size = 0;
        stats.compactions += 1;
        
        *self.last_compaction.write().await = Instant::now();
        
        Ok(())
    }
    
    /// Get statistics
    pub async fn stats(&self) -> GraphStats {
        self.stats.read().await.clone()
    }
}

