//! Database branching and versioning for LLMKG
//! Provides git-like branching for knowledge graphs

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::versioning::{MultiDatabaseVersionManager, VersionId};
use crate::federation::DatabaseId;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Branch metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub branch_name: String,
    pub database_id: DatabaseId,
    pub created_at: DateTime<Utc>,
    pub created_from: DatabaseId,
    pub parent_version: Option<VersionId>,
    pub description: Option<String>,
    pub is_active: bool,
}

/// Database branching manager
pub struct DatabaseBranchManager {
    /// Map of database IDs to knowledge engines
    databases: Arc<RwLock<HashMap<DatabaseId, Arc<RwLock<KnowledgeEngine>>>>>,
    /// Version manager for tracking changes
    version_manager: Arc<MultiDatabaseVersionManager>,
    /// Branch metadata
    branches: Arc<RwLock<HashMap<String, BranchInfo>>>,
}

impl DatabaseBranchManager {
    pub fn new(version_manager: Arc<MultiDatabaseVersionManager>) -> Self {
        Self {
            databases: Arc::new(RwLock::new(HashMap::new())),
            version_manager,
            branches: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new branch by copying an existing database
    pub async fn create_branch(
        &self,
        source_db_id: &DatabaseId,
        branch_name: String,
        description: Option<String>,
    ) -> Result<DatabaseId> {
        // Generate new database ID for the branch
        let branch_db_id = DatabaseId::new(format!("{}_{}", source_db_id.as_str(), branch_name));
        
        // Get source database
        let databases = self.databases.read().await;
        let source_engine = databases.get(source_db_id)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Source database not found: {}", source_db_id.as_str())
            ))?;
        
        // Create snapshot of source database
        let _snapshot_id = self.version_manager.create_snapshot(
            source_db_id,
            format!("Branch snapshot for {branch_name}"),
            Some(format!("Creating branch '{branch_name}'")),
        ).await?;
        
        // Create new database engine for branch
        let source_lock = source_engine.read().await;
        let new_engine = self.copy_knowledge_engine(&source_lock).await?;
        drop(source_lock);
        
        // Register the new database
        self.version_manager.register_database(branch_db_id.clone()).await?;
        
        // Store branch info
        let branch_info = BranchInfo {
            branch_name: branch_name.clone(),
            database_id: branch_db_id.clone(),
            created_at: Utc::now(),
            created_from: source_db_id.clone(),
            parent_version: None, // Could track specific version
            description,
            is_active: true,
        };
        
        // Add to databases and branches
        drop(databases);
        let mut databases = self.databases.write().await;
        databases.insert(branch_db_id.clone(), Arc::new(RwLock::new(new_engine)));
        
        let mut branches = self.branches.write().await;
        branches.insert(branch_name, branch_info);
        
        Ok(branch_db_id)
    }
    
    /// Copy a knowledge engine's contents
    async fn copy_knowledge_engine(&self, source: &KnowledgeEngine) -> Result<KnowledgeEngine> {
        // Create new engine with same parameters
        let _stats = source.get_memory_stats();
        let new_engine = KnowledgeEngine::new(384, 1_000_000)?; // TODO: Get actual params
        
        // Copy all triples
        let all_triples = source.query_triples(TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: usize::MAX,
            min_confidence: 0.0,
            include_chunks: true,
        })?;
        
        // Store in new engine
        for triple in all_triples.triples {
            new_engine.store_triple(triple, None)?;
        }
        
        // Copy chunks if any
        for node in all_triples.nodes {
            if matches!(&node.node_type, crate::core::triple::NodeType::Chunk) {
                // For chunk nodes, we need to extract the content
                match &node.content {
                    crate::core::triple::NodeContent::Chunk { text, .. } => {
                        new_engine.store_chunk(text.clone(), None)?;
                    }
                    _ => {
                        // Skip non-chunk content in chunk nodes
                        continue;
                    }
                }
            }
        }
        
        Ok(new_engine)
    }
    
    /// List all branches
    pub async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        let branches = self.branches.read().await;
        Ok(branches.values().cloned().collect())
    }
    
    /// Switch to a different branch
    pub async fn switch_branch(&self, branch_name: &str) -> Result<DatabaseId> {
        let branches = self.branches.read().await;
        let branch = branches.get(branch_name)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Branch not found: {branch_name}")
            ))?;
        
        Ok(branch.database_id.clone())
    }
    
    /// Compare two branches
    pub async fn compare_branches(
        &self,
        branch1: &str,
        branch2: &str,
    ) -> Result<BranchComparison> {
        let branches = self.branches.read().await;
        
        let info1 = branches.get(branch1)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Branch not found: {branch1}")
            ))?;
            
        let info2 = branches.get(branch2)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Branch not found: {branch2}")
            ))?;
        
        // Get databases
        let databases = self.databases.read().await;
        let db1 = databases.get(&info1.database_id).unwrap();
        let db2 = databases.get(&info2.database_id).unwrap();
        
        // Compare triple counts
        let stats1 = db1.read().await.get_memory_stats();
        let stats2 = db2.read().await.get_memory_stats();
        
        // Find differences (simplified)
        let differences = self.find_differences(db1, db2).await?;
        
        Ok(BranchComparison {
            branch1: branch1.to_string(),
            branch2: branch2.to_string(),
            stats1: BranchStats {
                total_triples: stats1.total_triples,
                total_nodes: stats1.total_nodes,
                total_bytes: stats1.total_bytes,
            },
            stats2: BranchStats {
                total_triples: stats2.total_triples,
                total_nodes: stats2.total_nodes,
                total_bytes: stats2.total_bytes,
            },
            differences,
        })
    }
    
    /// Find differences between two databases
    async fn find_differences(
        &self,
        db1: &Arc<RwLock<KnowledgeEngine>>,
        db2: &Arc<RwLock<KnowledgeEngine>>,
    ) -> Result<DifferenceSet> {
        let engine1 = db1.read().await;
        let engine2 = db2.read().await;
        
        // Get all triples from both
        let triples1 = engine1.query_triples(TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: 10000, // Limit for performance
            min_confidence: 0.0,
            include_chunks: false,
        })?;
        
        let triples2 = engine2.query_triples(TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: 10000,
            min_confidence: 0.0,
            include_chunks: false,
        })?;
        
        // Convert to sets for comparison
        let set1: std::collections::HashSet<_> = triples1.triples.iter()
            .map(|t| format!("{}-{}-{}", t.subject, t.predicate, t.object))
            .collect();
            
        let set2: std::collections::HashSet<_> = triples2.triples.iter()
            .map(|t| format!("{}-{}-{}", t.subject, t.predicate, t.object))
            .collect();
        
        // Find differences
        let only_in_1: Vec<String> = set1.difference(&set2).cloned().collect();
        let only_in_2: Vec<String> = set2.difference(&set1).cloned().collect();
        
        Ok(DifferenceSet {
            only_in_first: only_in_1.len(),
            only_in_second: only_in_2.len(),
            common: set1.intersection(&set2).count(),
            sample_differences: only_in_1.into_iter().take(10)
                .chain(only_in_2.into_iter().take(10))
                .collect(),
        })
    }
    
    /// Merge changes from one branch into another
    pub async fn merge_branches(
        &self,
        source_branch: &str,
        target_branch: &str,
        merge_strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        // Get branch info
        let branches = self.branches.read().await;
        let source_info = branches.get(source_branch)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Source branch not found: {source_branch}")
            ))?;
        let target_info = branches.get(target_branch)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Target branch not found: {target_branch}")
            ))?;
        
        // Get differences
        let databases = self.databases.read().await;
        let source_db = databases.get(&source_info.database_id).unwrap();
        let target_db = databases.get(&target_info.database_id).unwrap();
        
        let _differences = self.find_differences(source_db, target_db).await?;
        
        // Apply merge strategy
        match merge_strategy {
            MergeStrategy::AcceptSource => {
                // Copy all from source to target
                let source_engine = source_db.read().await;
                let target_engine = target_db.write().await;
                
                // This is simplified - real implementation would handle conflicts
                let source_triples = source_engine.query_triples(TripleQuery {
                    subject: None,
                    predicate: None,
                    object: None,
                    limit: usize::MAX,
                    min_confidence: 0.0,
                    include_chunks: false,
                })?;
                
                let mut added = 0;
                for triple in source_triples.triples {
                    if target_engine.store_triple(triple, None).is_ok() {
                        added += 1;
                    }
                }
                
                Ok(MergeResult {
                    success: true,
                    triples_added: added,
                    triples_removed: 0,
                    conflicts_resolved: 0,
                    message: format!("Merged {added} triples from {source_branch} to {target_branch}"),
                })
            }
            MergeStrategy::AcceptTarget => {
                // No changes needed
                Ok(MergeResult {
                    success: true,
                    triples_added: 0,
                    triples_removed: 0,
                    conflicts_resolved: 0,
                    message: "No changes - keeping target branch as is".to_string(),
                })
            }
            MergeStrategy::Manual => {
                // Would need UI for manual resolution
                Err(crate::error::GraphError::InvalidInput(
                    "Manual merge not implemented - use AcceptSource or AcceptTarget".to_string()
                ))
            }
        }
    }
    
    /// Delete a branch
    pub async fn delete_branch(&self, branch_name: &str) -> Result<()> {
        let mut branches = self.branches.write().await;
        let branch_info = branches.remove(branch_name)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(
                format!("Branch not found: {branch_name}")
            ))?;
        
        // Remove database
        let mut databases = self.databases.write().await;
        databases.remove(&branch_info.database_id);
        
        Ok(())
    }
}

/// Branch comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchComparison {
    pub branch1: String,
    pub branch2: String,
    pub stats1: BranchStats,
    pub stats2: BranchStats,
    pub differences: DifferenceSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchStats {
    pub total_triples: usize,
    pub total_nodes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferenceSet {
    pub only_in_first: usize,
    pub only_in_second: usize,
    pub common: usize,
    pub sample_differences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    AcceptSource,
    AcceptTarget,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub success: bool,
    pub triples_added: usize,
    pub triples_removed: usize,
    pub conflicts_resolved: usize,
    pub message: String,
}

lazy_static::lazy_static! {
    static ref BRANCH_MANAGER: Arc<RwLock<Option<DatabaseBranchManager>>> = 
        Arc::new(RwLock::new(None));
}

/// Initialize the branch manager
pub async fn initialize_branch_manager(version_manager: Arc<MultiDatabaseVersionManager>) {
    let mut manager = BRANCH_MANAGER.write().await;
    *manager = Some(DatabaseBranchManager::new(version_manager));
}

/// Get the branch manager
pub async fn get_branch_manager() -> Result<Arc<RwLock<Option<DatabaseBranchManager>>>> {
    Ok(BRANCH_MANAGER.clone())
}