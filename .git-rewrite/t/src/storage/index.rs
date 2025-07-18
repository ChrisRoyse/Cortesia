use crate::error::Result;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

#[derive(Debug, Clone)]
pub struct HNSWNode {
    pub id: u32,
    pub level: usize,
    pub connections: Vec<Vec<u32>>, // Connections at each level
}

pub struct HNSWIndex {
    nodes: Vec<HNSWNode>,
    entry_point: Option<u32>,
    max_connections: usize,
    max_connections_top: usize,
    level_multiplier: f64,
    rng_seed: u64,
}

impl HNSWIndex {
    pub fn new(max_connections: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_connections,
            max_connections_top: max_connections,
            level_multiplier: 1.0 / (2.0_f64.ln()),
            rng_seed: 42,
        }
    }
    
    pub fn insert(&mut self, id: u32, embedding: &[f32]) -> Result<()> {
        let level = self.random_level();
        
        let mut node = HNSWNode {
            id,
            level,
            connections: vec![Vec::new(); level + 1],
        };
        
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.nodes.push(node);
            return Ok(());
        }
        
        let entry_point = self.entry_point.unwrap();
        let mut current_closest = vec![entry_point];
        
        // Search from top level down to level 1
        for lev in (1..=self.get_node_level(entry_point)).rev() {
            current_closest = self.search_level(embedding, &current_closest, 1, lev)?;
        }
        
        // Search level 0 and connect
        for lev in (0..=level).rev() {
            let candidates = self.search_level(
                embedding,
                &current_closest,
                if lev == 0 { self.max_connections } else { self.max_connections_top },
                lev,
            )?;
            
            let max_conn = if lev == 0 { self.max_connections } else { self.max_connections_top };
            let selected = self.select_neighbors(&candidates, max_conn, embedding)?;
            
            for &neighbor in &selected {
                node.connections[lev].push(neighbor);
                
                // Add bidirectional connection
                // Simplified version without pruning for now
                if let Some(neighbor_node) = self.get_node_mut(neighbor) {
                    if neighbor_node.connections.len() > lev {
                        neighbor_node.connections[lev].push(id);
                    }
                }
            }
            
            current_closest = selected;
        }
        
        // Update entry point if necessary
        if level > self.get_node_level(entry_point) {
            self.entry_point = Some(id);
        }
        
        self.nodes.push(node);
        Ok(())
    }
    
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        if let Some(entry_point) = self.entry_point {
            let mut current_closest = vec![entry_point];
            
            // Navigate from top level to level 1
            for lev in (1..=self.get_node_level(entry_point)).rev() {
                current_closest = self.search_level(query, &current_closest, 1, lev)?;
            }
            
            // Search level 0 for k nearest neighbors
            let candidates = self.search_level(query, &current_closest, k * 2, 0)?;
            
            // Calculate distances and sort
            let mut results: Vec<(u32, f32)> = candidates
                .into_iter()
                .map(|id| {
                    let dist = self.distance(query, &self.get_embedding(id).unwrap_or_default());
                    (id, dist)
                })
                .collect();
            
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
            
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }
    
    fn search_level(&self, query: &[f32], entry_points: &[u32], num_closest: usize, level: usize) -> Result<Vec<u32>> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        
        for &ep in entry_points {
            let dist = self.distance(query, &self.get_embedding(ep)?);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            w.push((OrderedFloat(dist), ep));
            visited.insert(ep);
        }
        
        while let Some(Reverse((curr_dist, curr))) = candidates.pop() {
            if !w.is_empty() && curr_dist.0 > w.peek().unwrap().0.0 {
                break;
            }
            
            if let Some(node) = self.get_node(curr) {
                if node.connections.len() > level {
                    for &neighbor in &node.connections[level] {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            
                            let dist = self.distance(query, &self.get_embedding(neighbor)?);
                            
                            if w.len() < num_closest || dist < w.peek().unwrap().0.0 {
                                candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                                w.push((OrderedFloat(dist), neighbor));
                                
                                if w.len() > num_closest {
                                    w.pop();
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(w.into_iter().map(|(_, id)| id).collect())
    }
    
    fn select_neighbors(&self, candidates: &[u32], max_count: usize, query: &[f32]) -> Result<Vec<u32>> {
        if candidates.len() <= max_count {
            return Ok(candidates.to_vec());
        }
        
        // Simple greedy selection - in production, use more sophisticated heuristics
        let mut distances: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, &self.get_embedding(id).unwrap_or_default());
                (id, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(max_count);
        
        Ok(distances.into_iter().map(|(id, _)| id).collect())
    }
    
    fn random_level(&mut self) -> usize {
        // Simple PRNG for level selection
        self.rng_seed = self.rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let uniform = (self.rng_seed as f64) / (u64::MAX as f64);
        (-uniform.ln() * self.level_multiplier).floor() as usize
    }
    
    fn get_node(&self, id: u32) -> Option<&HNSWNode> {
        self.nodes.iter().find(|node| node.id == id)
    }
    
    fn get_node_mut(&mut self, id: u32) -> Option<&mut HNSWNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }
    
    fn get_node_level(&self, id: u32) -> usize {
        self.get_node(id).map(|node| node.level).unwrap_or(0)
    }
    
    fn get_embedding(&self, _id: u32) -> Result<Vec<f32>> {
        // Placeholder - in real implementation, this would fetch from embedding store
        Ok(vec![0.0; 96])
    }
    
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn memory_usage(&self) -> usize {
        self.nodes.iter()
            .map(|node| {
                std::mem::size_of::<HNSWNode>() + 
                node.connections.iter()
                    .map(|level| level.capacity() * std::mem::size_of::<u32>())
                    .sum::<usize>()
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}