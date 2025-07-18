//! Pattern analysis for optimization opportunities

use super::types::*;
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

impl PatternDetector {
    /// Create new pattern detector
    pub fn new() -> Self {
        Self {
            detection_threshold: 0.5,
            analysis_config: PatternAnalysisConfig::default(),
            last_analysis: None,
        }
    }

    /// Identify optimization opportunities
    pub async fn identify_optimization_opportunities(
        &mut self,
        graph: &BrainEnhancedKnowledgeGraph,
        cache: &mut PatternCache,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Check cache first
        if let Some(cached_results) = self.check_cache_for_opportunities(cache) {
            return Ok(cached_results);
        }
        
        // Analyze for different optimization types
        opportunities.extend(self.identify_attribute_bubbling_opportunities(graph).await?);
        opportunities.extend(self.identify_hierarchy_consolidation_opportunities(graph).await?);
        opportunities.extend(self.identify_subgraph_factorization_opportunities(graph).await?);
        opportunities.extend(self.identify_connection_pruning_opportunities(graph).await?);
        
        // Cache results
        self.cache_opportunities(&opportunities, cache);
        
        // Update last analysis time
        self.last_analysis = Some(Instant::now());
        
        Ok(opportunities)
    }

    /// Identify attribute bubbling opportunities
    pub async fn identify_attribute_bubbling_opportunities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        for entity_key in entity_keys {
            if let Some(opportunity) = self.analyze_attribute_bubbling(graph, entity_key).await? {
                opportunities.push(opportunity);
            }
        }
        
        Ok(opportunities)
    }

    /// Analyze attribute bubbling for a specific entity
    async fn analyze_attribute_bubbling(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        entity_key: EntityKey,
    ) -> Result<Option<OptimizationOpportunity>> {
        // Get entity data
        let entity_data = graph.get_entity_data(entity_key);
        if entity_data.is_none() {
            return Ok(None);
        }
        
        let data = entity_data.unwrap();
        
        // Get children and analyze their attributes
        let children = graph.get_child_entities(entity_key).await;
        if children.len() < 2 {
            return Ok(None);
        }
        
        // Check for common attributes that can be bubbled up
        let mut common_attributes = HashMap::new();
        let mut first_child = true;
        let children_count = children.len(); // Store length before consuming children
        
        for (child_key, _) in children {
            if let Some(child_data) = graph.get_entity_data(child_key) {
                if first_child {
                    // Initialize with first child's properties as a single attribute
                    common_attributes.insert("properties".to_string(), child_data.properties.clone());
                    first_child = false;
                } else {
                    // Keep only if properties match
                    if let Some(existing_props) = common_attributes.get("properties") {
                        if existing_props != &child_data.properties {
                            common_attributes.remove("properties");
                        }
                    }
                }
            }
        }
        
        // If we have common attributes, this is an opportunity
        if !common_attributes.is_empty() {
            let improvement = self.calculate_bubbling_improvement(&common_attributes, children_count);
            
            if improvement > self.detection_threshold {
                return Ok(Some(OptimizationOpportunity {
                    opportunity_id: format!("bubbling_{:?}", entity_key),
                    optimization_type: OptimizationType::AttributeBubbling,
                    affected_entities: vec![entity_key],
                    estimated_improvement: improvement,
                    implementation_cost: 0.1,
                    risk_level: RiskLevel::Low,
                    prerequisites: vec![],
                }));
            }
        }
        
        Ok(None)
    }

    /// Calculate improvement from attribute bubbling
    fn calculate_bubbling_improvement(&self, common_attributes: &HashMap<String, String>, child_count: usize) -> f32 {
        let attribute_count = common_attributes.len();
        let storage_saved = (attribute_count * (child_count - 1)) as f32;
        
        // Normalize based on typical attribute counts
        (storage_saved / (child_count as f32 * 10.0)).min(1.0)
    }

    /// Identify hierarchy consolidation opportunities
    async fn identify_hierarchy_consolidation_opportunities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Look for shallow hierarchies that can be consolidated
        for entity_key in entity_keys {
            let children = graph.get_child_entities(entity_key).await;
            
            // Check if this entity has exactly one child (potential consolidation)
            if children.len() == 1 {
                let (child_key, _) = children[0];
                let grandchildren = graph.get_child_entities(child_key).await;
                
                // If the child also has few children, consider consolidation
                if grandchildren.len() <= 2 {
                    let improvement = self.calculate_consolidation_improvement(1, grandchildren.len());
                    
                    if improvement > self.detection_threshold {
                        opportunities.push(OptimizationOpportunity {
                            opportunity_id: format!("consolidation_{:?}_{:?}", entity_key, child_key),
                            optimization_type: OptimizationType::HierarchyConsolidation,
                            affected_entities: vec![entity_key, child_key],
                            estimated_improvement: improvement,
                            implementation_cost: 0.15,
                            risk_level: RiskLevel::Medium,
                            prerequisites: vec!["data_integrity_check".to_string()],
                        });
                    }
                }
            }
        }
        
        Ok(opportunities)
    }

    /// Calculate improvement from hierarchy consolidation
    fn calculate_consolidation_improvement(&self, levels_removed: usize, entities_affected: usize) -> f32 {
        let traversal_savings = levels_removed as f32 * 0.1;
        let memory_savings = (entities_affected as f32 * 0.05).min(0.3);
        
        traversal_savings + memory_savings
    }

    /// Identify subgraph factorization opportunities
    async fn identify_subgraph_factorization_opportunities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Look for repeated subgraph patterns
        let mut subgraph_patterns = HashMap::new();
        
        for entity_key in entity_keys {
            let neighbors = graph.get_neighbors(entity_key);
            
            if neighbors.len() >= 3 {
                // Create a pattern signature for this subgraph
                let mut pattern_signature = neighbors.iter().map(|n| format!("{:?}", n)).collect::<Vec<_>>();
                pattern_signature.sort();
                let signature = format!("{:?}", pattern_signature);
                
                subgraph_patterns.entry(signature).or_insert_with(Vec::new).push(entity_key);
            }
        }
        
        // Find patterns that appear multiple times
        for (pattern, entities) in subgraph_patterns {
            if entities.len() >= 2 {
                let improvement = self.calculate_factorization_improvement(entities.len());
                
                if improvement > self.detection_threshold {
                    opportunities.push(OptimizationOpportunity {
                        opportunity_id: format!("factorization_{}", pattern.len()),
                        optimization_type: OptimizationType::SubgraphFactorization,
                        affected_entities: entities,
                        estimated_improvement: improvement,
                        implementation_cost: 0.25,
                        risk_level: RiskLevel::High,
                        prerequisites: vec!["pattern_validation".to_string(), "impact_analysis".to_string()],
                    });
                }
            }
        }
        
        Ok(opportunities)
    }

    /// Calculate improvement from subgraph factorization
    fn calculate_factorization_improvement(&self, pattern_count: usize) -> f32 {
        let duplication_savings = ((pattern_count - 1) as f32 * 0.2).min(0.8);
        let query_optimization = (pattern_count as f32 * 0.05).min(0.3);
        
        duplication_savings + query_optimization
    }

    /// Identify connection pruning opportunities
    async fn identify_connection_pruning_opportunities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        for entity_key in entity_keys {
            let neighbors = graph.get_neighbors_with_weights(entity_key).await;
            
            // Look for weak connections that can be pruned
            let weak_connections: Vec<_> = neighbors.iter()
                .filter(|(_, weight)| *weight < 0.1)
                .collect();
            
            if weak_connections.len() > 0 {
                let improvement = self.calculate_pruning_improvement(weak_connections.len(), neighbors.len());
                
                if improvement > self.detection_threshold {
                    opportunities.push(OptimizationOpportunity {
                        opportunity_id: format!("pruning_{:?}", entity_key),
                        optimization_type: OptimizationType::ConnectionPruning,
                        affected_entities: vec![entity_key],
                        estimated_improvement: improvement,
                        implementation_cost: 0.05,
                        risk_level: RiskLevel::Low,
                        prerequisites: vec!["connection_analysis".to_string()],
                    });
                }
            }
        }
        
        Ok(opportunities)
    }

    /// Calculate improvement from connection pruning
    fn calculate_pruning_improvement(&self, connections_pruned: usize, total_connections: usize) -> f32 {
        let ratio = connections_pruned as f32 / total_connections as f32;
        let memory_savings = ratio * 0.3;
        let query_speedup = ratio * 0.2;
        
        memory_savings + query_speedup
    }

    /// Check cache for optimization opportunities
    fn check_cache_for_opportunities(&self, cache: &PatternCache) -> Option<Vec<OptimizationOpportunity>> {
        // Check if we have recent cached results
        if let Some(last_analysis) = self.last_analysis {
            if last_analysis.elapsed() < self.analysis_config.cache_ttl {
                // Return cached opportunities if available
                // This is a simplified implementation
                return None;
            }
        }
        None
    }

    /// Cache optimization opportunities
    fn cache_opportunities(&self, opportunities: &[OptimizationOpportunity], cache: &mut PatternCache) {
        for opportunity in opportunities {
            let cached_pattern = CachedPattern {
                pattern_id: opportunity.opportunity_id.clone(),
                pattern_type: opportunity.optimization_type.clone(),
                detection_time: Instant::now(),
                expiry_time: Instant::now() + self.analysis_config.cache_ttl,
                hit_count: 0,
                efficiency_score: opportunity.estimated_improvement,
            };
            
            cache.cached_patterns.insert(opportunity.opportunity_id.clone(), cached_pattern);
        }
    }
}

impl PatternCache {
    /// Create new pattern cache
    pub fn new() -> Self {
        Self {
            cached_patterns: HashMap::new(),
            cache_capacity: 1000,
            hit_count: 0,
            miss_count: 0,
            last_cleanup: Instant::now(),
        }
    }

    /// Get cached pattern
    pub fn get_pattern(&mut self, pattern_id: &str) -> Option<CachedPattern> {
        if let Some(pattern) = self.cached_patterns.get(pattern_id).cloned() {
            // Check if pattern is still valid
            if pattern.expiry_time > Instant::now() {
                self.hit_count += 1;
                return Some(pattern);
            } else {
                // Pattern expired, remove it
                self.cached_patterns.remove(pattern_id);
            }
        }
        
        self.miss_count += 1;
        None
    }

    /// Put pattern in cache
    pub fn put_pattern(&mut self, pattern: CachedPattern) {
        // Check capacity
        if self.cached_patterns.len() >= self.cache_capacity {
            self.cleanup_expired_patterns();
        }
        
        self.cached_patterns.insert(pattern.pattern_id.clone(), pattern);
    }

    /// Clean up expired patterns
    pub fn cleanup_expired_patterns(&mut self) {
        let now = Instant::now();
        self.cached_patterns.retain(|_, pattern| pattern.expiry_time > now);
        self.last_cleanup = now;
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize, f32) {
        let total_requests = self.hit_count + self.miss_count;
        let hit_rate = if total_requests > 0 {
            self.hit_count as f32 / total_requests as f32
        } else {
            0.0
        };
        
        (self.hit_count, self.miss_count, hit_rate)
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cached_patterns.clear();
        self.hit_count = 0;
        self.miss_count = 0;
        self.last_cleanup = Instant::now();
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PatternCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for pattern analysis
impl PatternDetector {
    /// Calculate similarity between two entity sets
    pub fn calculate_set_similarity(&self, set1: &HashSet<EntityKey>, set2: &HashSet<EntityKey>) -> f32 {
        let intersection = set1.intersection(set2).count();
        let union = set1.union(set2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Check if pattern meets minimum size requirements
    pub fn meets_size_requirements(&self, pattern_size: usize) -> bool {
        pattern_size >= self.analysis_config.min_pattern_size && 
        pattern_size <= self.analysis_config.max_pattern_size
    }

    /// Calculate pattern quality score
    pub fn calculate_pattern_quality(&self, pattern: &[EntityKey], graph: &BrainEnhancedKnowledgeGraph) -> f32 {
        let mut quality_score = 0.0;
        let mut connection_count = 0;
        
        // Check connectivity within pattern
        for i in 0..pattern.len() {
            for j in i + 1..pattern.len() {
                if graph.core_graph.has_relationship(pattern[i], pattern[j]) {
                    connection_count += 1;
                }
            }
        }
        
        // Calculate density
        let max_connections = pattern.len() * (pattern.len() - 1) / 2;
        if max_connections > 0 {
            quality_score = connection_count as f32 / max_connections as f32;
        }
        
        quality_score
    }

    /// Validate pattern against analysis scope
    pub fn validate_pattern_scope(&self, pattern: &[EntityKey]) -> bool {
        match &self.analysis_config.analysis_scope {
            AnalysisScope::LocalEntities(entities) => {
                pattern.iter().all(|entity| entities.contains(entity))
            }
            AnalysisScope::GlobalGraph => true,
            AnalysisScope::SubgraphRegion(region) => {
                pattern.iter().all(|entity| region.contains(entity))
            }
            AnalysisScope::RecentActivity(_) => {
                // Would need to check activity timestamps
                true
            }
        }
    }
}

/// Pattern analysis utilities
pub struct PatternAnalysisUtils;

impl PatternAnalysisUtils {
    /// Find common neighbors between entities
    pub fn find_common_neighbors(
        entity1: EntityKey,
        entity2: EntityKey,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> Vec<EntityKey> {
        let neighbors1: HashSet<_> = graph.get_neighbors(entity1).into_iter().map(|(k, _)| k).collect();
        let neighbors2: HashSet<_> = graph.get_neighbors(entity2).into_iter().map(|(k, _)| k).collect();
        
        neighbors1.intersection(&neighbors2).cloned().collect()
    }

    /// Calculate structural similarity between entities
    pub fn calculate_structural_similarity(
        entity1: EntityKey,
        entity2: EntityKey,
        graph: &BrainEnhancedKnowledgeGraph,
    ) -> f32 {
        let neighbors1: HashSet<_> = graph.get_neighbors(entity1).into_iter().map(|(k, _)| k).collect();
        let neighbors2: HashSet<_> = graph.get_neighbors(entity2).into_iter().map(|(k, _)| k).collect();
        
        let intersection = neighbors1.intersection(&neighbors2).count();
        let union = neighbors1.union(&neighbors2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Find entity clusters based on connectivity
    pub fn find_entity_clusters(
        entities: &[EntityKey],
        graph: &BrainEnhancedKnowledgeGraph,
        similarity_threshold: f32,
    ) -> Vec<Vec<EntityKey>> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();
        
        for &entity in entities {
            if visited.contains(&entity) {
                continue;
            }
            
            let mut cluster = Vec::new();
            let mut to_visit = vec![entity];
            
            while let Some(current) = to_visit.pop() {
                if visited.contains(&current) {
                    continue;
                }
                
                visited.insert(current);
                cluster.push(current);
                
                // Find similar entities
                for &other in entities {
                    if !visited.contains(&other) {
                        let similarity = Self::calculate_structural_similarity(current, other, graph);
                        if similarity >= similarity_threshold {
                            to_visit.push(other);
                        }
                    }
                }
            }
            
            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }
        
        clusters
    }
}