// Multi-database federation module
// Provides federated knowledge graph functionality with cross-database operations

pub mod registry;
pub mod router;
pub mod merger;
pub mod coordinator;
pub mod types;
pub mod database_connection;
pub mod transaction_log;
pub mod two_phase_commit;

pub use registry::{DatabaseRegistry, DatabaseDescriptor};
pub use router::{QueryRouter, QueryPlan};
pub use merger::ResultMerger;
pub use coordinator::{FederationCoordinator, CrossDatabaseTransaction};
pub use types::*;

use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub database_id: String,
    pub similarity_score: f32,
    pub entity_data: HashMap<String, String>,
    pub confidence_score: Option<f32>,
}

/// Main federation manager that orchestrates multi-database operations
pub struct FederationManager {
    registry: Arc<DatabaseRegistry>,
    router: QueryRouter,
    merger: ResultMerger,
    coordinator: FederationCoordinator,
}

impl FederationManager {
    pub async fn new() -> Result<Self> {
        let registry = Arc::new(DatabaseRegistry::new()?);
        let router = QueryRouter::new(registry.clone())?;
        let merger = ResultMerger::new();
        let coordinator = FederationCoordinator::new(registry.clone()).await?;

        Ok(Self {
            registry,
            router,
            merger,
            coordinator,
        })
    }

    /// Register a new database in the federation
    pub async fn register_database(&mut self, descriptor: DatabaseDescriptor) -> Result<()> {
        self.registry.register(descriptor).await
    }

    /// Execute a federated query across multiple databases
    pub async fn execute_federated_query(&self, query: FederatedQuery) -> Result<FederatedQueryResult> {
        let plan = self.router.plan_query(&query).await?;
        let raw_results = self.router.execute_plan(plan).await?;
        let merged_result = self.merger.merge_results(raw_results, query.merge_strategy()).await?;
        Ok(merged_result)
    }

    /// Get the list of all registered databases
    pub async fn list_databases(&self) -> Vec<DatabaseDescriptor> {
        self.registry.list_databases().await
    }

    /// Check the health of all registered databases
    pub async fn health_check(&self) -> Vec<DatabaseHealth> {
        self.registry.health_check_all().await
    }
    
    /// Execute similarity search across all federated databases
    pub async fn execute_similarity_search(&self, query: &str, threshold: f32, max_results: usize) -> Result<Vec<SimilarityResult>> {
        use crate::federation::types::{FederatedQuery, QueryType, SimilarityQuery};
        
        let similarity_query = SimilarityQuery {
            query_text: query.to_string(),
            threshold,
            max_results,
            embedding_model: "default".to_string(),
        };
        
        let federated_query = FederatedQuery {
            query_id: format!("sim_{}", uuid::Uuid::new_v4()),
            query_type: QueryType::SimilaritySearch(similarity_query),
            target_databases: self.registry.list_databases().await.iter().map(|db| db.id.clone()).collect(),
            merge_strategy: crate::federation::types::MergeStrategy::SimilarityMerge,
            timeout_ms: 30000,
        };
        
        let result = self.execute_federated_query(federated_query).await?;
        
        // Extract similarity results
        match result.results {
            crate::federation::types::QueryResultData::SimilarityResults(similarities) => {
                Ok(similarities.into_iter().map(|sim| {
                    let mut entity_data = std::collections::HashMap::new();
                    if let Some(obj) = sim.metadata.as_object() {
                        for (k, v) in obj {
                            entity_data.insert(k.clone(), v.to_string());
                        }
                    }
                    SimilarityResult {
                        database_id: "default".to_string(),  // Extract from metadata if available
                        similarity_score: sim.similarity_score,
                        entity_data,
                        confidence_score: Some(sim.similarity_score),
                    }
                }).collect())
            }
            _ => Ok(vec![]),
        }
    }
}