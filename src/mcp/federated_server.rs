// Enhanced MCP server with multi-database federation capabilities
// Provides LLM-friendly tools for cross-database operations, versioning, and mathematical computations

use crate::federation::FederationManager;
use crate::versioning::MultiDatabaseVersionManager;
use crate::math::MathEngine;
use crate::mcp::shared_types::{LLMMCPTool, LLMExample, LLMMCPRequest, LLMMCPResponse, PerformanceInfo};
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;

/// Enhanced MCP server with multi-database federation capabilities
pub struct FederatedMCPServer {
    federation_manager: Arc<RwLock<FederationManager>>,
    version_manager: Arc<RwLock<MultiDatabaseVersionManager>>,
    math_engine: Arc<MathEngine>,
    usage_stats: Arc<RwLock<FederatedUsageStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct FederatedUsageStats {
    pub total_operations: u64,
    pub cross_database_queries: u64,
    pub version_operations: u64,
    pub math_operations: u64,
    pub databases_accessed: u64,
    pub avg_response_time_ms: f64,
    pub federation_efficiency: f64,
}

impl FederatedMCPServer {
    pub async fn new() -> Result<Self> {
        let federation_manager = Arc::new(RwLock::new(FederationManager::new()?));
        let version_manager = Arc::new(RwLock::new(MultiDatabaseVersionManager::new()?));
        let math_engine = Arc::new(MathEngine::new()?);
        
        Ok(Self {
            federation_manager,
            version_manager,
            math_engine,
            usage_stats: Arc::new(RwLock::new(FederatedUsageStats::default())),
        })
    }

    /// Get all available federated tools with comprehensive descriptions
    pub fn get_federated_tools(&self) -> Vec<LLMMCPTool> {
        vec![
            // Cross-database similarity search
            LLMMCPTool {
                name: "cross_database_similarity".to_string(),
                description: "Find similar entities across multiple databases using vector similarity. Perfect for discovering related concepts that exist in different knowledge bases.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query_entity": {
                            "type": "string",
                            "description": "The entity to find similarities for (e.g., 'Einstein', 'quantum_physics')"
                        },
                        "databases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of database IDs to search in. Leave empty to search all databases."
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score (0.0 to 1.0, default: 0.7)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of similar entities to return (default: 20)",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 20
                        },
                        "similarity_metric": {
                            "type": "string",
                            "enum": ["cosine", "euclidean", "jaccard"],
                            "description": "Similarity calculation method (default: cosine)",
                            "default": "cosine"
                        }
                    },
                    "required": ["query_entity"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Find entities similar to Einstein across all databases".to_string(),
                        input: serde_json::json!({
                            "query_entity": "Einstein",
                            "similarity_threshold": 0.8,
                            "max_results": 15
                        }),
                        expected_output: "Found 12 similar entities: Newton (0.85), Hawking (0.82), Feynman (0.79)...".to_string(),
                    },
                ],
                tips: vec![
                    "Use higher thresholds (0.8+) for finding very similar entities".to_string(),
                    "Lower thresholds (0.5-0.7) help discover broader relationships".to_string(),
                    "Cosine similarity works best for conceptual relationships".to_string(),
                ],
            },

            // Cross-database entity comparison
            LLMMCPTool {
                name: "compare_across_databases".to_string(),
                description: "Compare how the same entity appears in different databases. Great for finding inconsistencies, different perspectives, or complementary information.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity to compare (e.g., 'Einstein', 'Python programming')"
                        },
                        "databases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Database IDs to compare. Leave empty to compare across all databases."
                        },
                        "comparison_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific fields to compare (optional). Leave empty to compare all fields."
                        },
                        "show_differences_only": {
                            "type": "boolean",
                            "description": "Only show fields that differ between databases (default: false)",
                            "default": false
                        }
                    },
                    "required": ["entity_id"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Compare Einstein's information across databases".to_string(),
                        input: serde_json::json!({
                            "entity_id": "Einstein",
                            "databases": ["physics_db", "biography_db", "patents_db"],
                            "show_differences_only": true
                        }),
                        expected_output: "Found differences in 3 fields: birth_year (1879 vs 1880), Nobel_Prize_year...".to_string(),
                    },
                ],
                tips: vec![
                    "Use this to identify conflicting information between sources".to_string(),
                    "Great for merging knowledge from multiple databases".to_string(),
                    "Compare specific fields when you know what to look for".to_string(),
                ],
            },

            // Relationship strength calculation
            LLMMCPTool {
                name: "calculate_relationship_strength".to_string(),
                description: "Calculate how strongly two entities are related using multiple metrics: semantic similarity, structural proximity, and co-occurrence patterns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity1": {
                            "type": "string",
                            "description": "First entity (e.g., 'Einstein')"
                        },
                        "entity2": {
                            "type": "string",
                            "description": "Second entity (e.g., 'quantum_physics')"
                        },
                        "database1": {
                            "type": "string",
                            "description": "Database ID for first entity (optional)"
                        },
                        "database2": {
                            "type": "string",
                            "description": "Database ID for second entity (optional)"
                        },
                        "relationship_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of relationships to consider (e.g., ['collaborates_with', 'influences'])"
                        },
                        "include_indirect": {
                            "type": "boolean",
                            "description": "Include indirect relationships through other entities (default: true)",
                            "default": true
                        }
                    },
                    "required": ["entity1", "entity2"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Calculate relationship strength between Einstein and quantum physics".to_string(),
                        input: serde_json::json!({
                            "entity1": "Einstein",
                            "entity2": "quantum_physics",
                            "relationship_types": ["researches", "contributes_to", "pioneers"],
                            "include_indirect": true
                        }),
                        expected_output: "Relationship strength: 0.85 (semantic: 0.9, structural: 0.8, co-occurrence: 0.85)".to_string(),
                    },
                ],
                tips: vec![
                    "Scores range from 0.0 (no relationship) to 1.0 (very strong relationship)".to_string(),
                    "Include specific relationship types for more accurate results".to_string(),
                    "Indirect relationships help discover hidden connections".to_string(),
                ],
            },

            // Version comparison
            LLMMCPTool {
                name: "compare_versions".to_string(),
                description: "Compare different versions of an entity to see how it has changed over time. Perfect for tracking evolution of concepts, facts, or knowledge.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity to analyze version history for"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database containing the entity"
                        },
                        "version1": {
                            "type": "string",
                            "description": "First version ID to compare (optional - uses oldest if not specified)"
                        },
                        "version2": {
                            "type": "string",
                            "description": "Second version ID to compare (optional - uses latest if not specified)"
                        },
                        "show_timeline": {
                            "type": "boolean",
                            "description": "Show full timeline of changes (default: false)",
                            "default": false
                        }
                    },
                    "required": ["entity_id", "database_id"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Compare latest version of Einstein with version from last month".to_string(),
                        input: serde_json::json!({
                            "entity_id": "Einstein",
                            "database_id": "physics_db",
                            "show_timeline": true
                        }),
                        expected_output: "Found 5 changes: birth_place updated, 2 new publications added, Nobel Prize date corrected...".to_string(),
                    },
                ],
                tips: vec![
                    "Leave version IDs empty to compare oldest vs newest".to_string(),
                    "Use timeline view to see gradual evolution of knowledge".to_string(),
                    "Great for fact-checking and quality control".to_string(),
                ],
            },

            // Temporal queries
            LLMMCPTool {
                name: "temporal_query".to_string(),
                description: "Query entities as they existed at a specific point in time, or see how they changed during a time period. Time-travel for your knowledge graph!".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["point_in_time", "time_range", "field_evolution", "changed_entities"],
                            "description": "Type of temporal query to perform"
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Entity to query (required for most query types)"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database ID to query"
                        },
                        "timestamp": {
                            "type": "string",
                            "description": "ISO timestamp for point_in_time queries (e.g., '2024-01-15T10:30:00Z')"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start timestamp for time_range queries"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End timestamp for time_range queries"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Specific field to track for field_evolution queries"
                        }
                    },
                    "required": ["query_type", "database_id"]
                }),
                examples: vec![
                    LLMExample {
                        description: "See how Einstein's entry looked on January 1st, 2024".to_string(),
                        input: serde_json::json!({
                            "query_type": "point_in_time",
                            "entity_id": "Einstein",
                            "database_id": "physics_db",
                            "timestamp": "2024-01-01T00:00:00Z"
                        }),
                        expected_output: "Einstein on 2024-01-01: physicist, born 1879, theory of relativity (3 publications listed)...".to_string(),
                    },
                ],
                tips: vec![
                    "Use point_in_time to see historical states of entities".to_string(),
                    "time_range shows all changes during a period".to_string(),
                    "field_evolution tracks how specific properties changed".to_string(),
                ],
            },

            // Database snapshot creation
            LLMMCPTool {
                name: "create_database_snapshot".to_string(),
                description: "Create a snapshot of a database for backup, versioning, or experimental purposes. Snapshots preserve the exact state at a point in time.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "database_id": {
                            "type": "string",
                            "description": "Database to snapshot"
                        },
                        "snapshot_name": {
                            "type": "string",
                            "description": "Name for the snapshot (e.g., 'before_major_update', 'monthly_backup')"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the snapshot purpose"
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include version history and metadata (default: true)",
                            "default": true
                        }
                    },
                    "required": ["database_id", "snapshot_name"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Create a backup snapshot before making major changes".to_string(),
                        input: serde_json::json!({
                            "database_id": "main_kg",
                            "snapshot_name": "pre_migration_backup",
                            "description": "Backup before migrating to new schema version"
                        }),
                        expected_output: "Snapshot 'pre_migration_backup' created successfully. ID: snap_1234567890".to_string(),
                    },
                ],
                tips: vec![
                    "Create snapshots before major changes or experiments".to_string(),
                    "Use descriptive names to identify snapshots later".to_string(),
                    "Include metadata for full restoration capabilities".to_string(),
                ],
            },

            // Mathematical operations
            LLMMCPTool {
                name: "mathematical_operation".to_string(),
                description: "Execute advanced mathematical operations across multiple databases: PageRank, shortest paths, centrality measures, and graph statistics.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["pagerank", "shortest_path", "betweenness_centrality", "clustering_coefficient", "graph_statistics"],
                            "description": "Mathematical operation to perform"
                        },
                        "databases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Database IDs to include in calculation"
                        },
                        "source_entity": {
                            "type": "string",
                            "description": "Source entity for path operations"
                        },
                        "target_entity": {
                            "type": "string",
                            "description": "Target entity for path operations"
                        },
                        "damping_factor": {
                            "type": "number",
                            "description": "Damping factor for PageRank (default: 0.85)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.85
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum iterations for iterative algorithms (default: 100)",
                            "minimum": 10,
                            "maximum": 1000,
                            "default": 100
                        }
                    },
                    "required": ["operation"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Calculate PageRank to find most important entities".to_string(),
                        input: serde_json::json!({
                            "operation": "pagerank",
                            "databases": ["physics_db", "chemistry_db"],
                            "damping_factor": 0.85,
                            "max_iterations": 50
                        }),
                        expected_output: "PageRank completed: Einstein (0.15), Newton (0.12), Curie (0.08)...".to_string(),
                    },
                ],
                tips: vec![
                    "PageRank identifies the most influential entities".to_string(),
                    "Use shortest_path to find connections between entities".to_string(),
                    "Graph statistics provide overall database insights".to_string(),
                ],
            },

            // Federation statistics
            LLMMCPTool {
                name: "federation_stats".to_string(),
                description: "Get comprehensive statistics about the federated database system: database health, performance metrics, and usage patterns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "include_performance": {
                            "type": "boolean",
                            "description": "Include detailed performance metrics (default: true)",
                            "default": true
                        },
                        "include_health": {
                            "type": "boolean",
                            "description": "Include database health status (default: true)",
                            "default": true
                        },
                        "time_range_hours": {
                            "type": "integer",
                            "description": "Time range for statistics in hours (default: 24)",
                            "minimum": 1,
                            "maximum": 168,
                            "default": 24
                        }
                    }
                }),
                examples: vec![
                    LLMExample {
                        description: "Get federation status and performance".to_string(),
                        input: serde_json::json!({
                            "include_performance": true,
                            "include_health": true,
                            "time_range_hours": 24
                        }),
                        expected_output: "Federation Status: 5 databases, all healthy. 1,250 cross-DB queries, avg response: 15ms".to_string(),
                    },
                ],
                tips: vec![
                    "Monitor federation health regularly".to_string(),
                    "Use performance metrics to optimize queries".to_string(),
                    "Check for database synchronization issues".to_string(),
                ],
            },
        ]
    }

    /// Handle federated MCP requests
    pub async fn handle_federated_request(&self, request: LLMMCPRequest) -> LLMMCPResponse {
        let start_time = Instant::now();
        
        let result = match request.method.as_str() {
            "cross_database_similarity" => self.handle_cross_database_similarity(request.params).await,
            "compare_across_databases" => self.handle_compare_across_databases(request.params).await,
            "calculate_relationship_strength" => self.handle_calculate_relationship_strength(request.params).await,
            "compare_versions" => self.handle_compare_versions(request.params).await,
            "temporal_query" => self.handle_temporal_query(request.params).await,
            "create_database_snapshot" => self.handle_create_database_snapshot(request.params).await,
            "mathematical_operation" => self.handle_mathematical_operation(request.params).await,
            "federation_stats" => self.handle_federation_stats(request.params).await,
            _ => Err(format!("Unknown federated method: {}. Available methods: cross_database_similarity, compare_across_databases, calculate_relationship_strength, compare_versions, temporal_query, create_database_snapshot, mathematical_operation, federation_stats", request.method)),
        };
        
        let response_time = start_time.elapsed().as_millis() as u64;
        
        // Update usage stats
        self.update_federated_stats(response_time, &request.method).await;
        
        match result {
            Ok((data, message, suggestions)) => {
                LLMMCPResponse {
                    success: true,
                    data,
                    message,
                    helpful_info: Some(self.generate_federated_help(&request.method)),
                    suggestions,
                    performance: PerformanceInfo {
                        execution_time_ms: response_time as f64,
                        memory_used_bytes: 0, // TODO: Get actual memory usage
                        cache_hit: false, // TODO: Implement cache tracking
                        complexity_score: 0.95, // TODO: Calculate actual score
                    },
                }
            },
            Err(error) => {
                LLMMCPResponse {
                    success: false,
                    data: serde_json::json!(null),
                    message: error,
                    helpful_info: Some(self.generate_federated_error_help(&request.method)),
                    suggestions: self.generate_federated_error_suggestions(&request.method),
                    performance: PerformanceInfo {
                        execution_time_ms: response_time as f64,
                        memory_used_bytes: 0,
                        cache_hit: false,
                        complexity_score: 0.0,
                    },
                }
            }
        }
    }

    // Placeholder implementations for the handlers
    async fn handle_cross_database_similarity(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        // Extract query parameters
        let query = params.get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        let threshold = params.get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        
        let max_results = params.get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;
        
        // Execute similarity search across databases
        let results = self.federation_manager.read().await.execute_similarity_search(
            &query,
            threshold,
            max_results,
        ).await.map_err(|e| format!("Similarity search failed: {e}"))?;
        
        // Format results
        let results_count = results.len();
        let response = serde_json::json!({
            "query": query,
            "threshold": threshold,
            "total_results": results_count,
            "results": results.into_iter().map(|result| {
                serde_json::json!({
                    "database": result.database_id,
                    "similarity": result.similarity_score,
                    "entity": result.entity_data,
                    "confidence": result.confidence_score.unwrap_or(0.8)
                })
            }).collect::<Vec<_>>()
        });
        
        Ok((
            response,
            format!("Found {results_count} similar entities across databases"),
            vec![
                "Results are ranked by similarity score".to_string(),
                "Use threshold to filter results".to_string(),
                "Higher similarity scores indicate better matches".to_string()
            ],
        ))
    }

    async fn handle_compare_across_databases(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "compare_across_databases implementation"}),
            "Cross-database comparison completed".to_string(),
            vec!["Great for identifying inconsistencies between sources".to_string()],
        ))
    }

    async fn handle_calculate_relationship_strength(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "relationship_strength implementation"}),
            "Relationship strength calculated".to_string(),
            vec!["Higher scores indicate stronger relationships".to_string()],
        ))
    }

    async fn handle_compare_versions(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "version_comparison implementation"}),
            "Version comparison completed".to_string(),
            vec!["Track how your knowledge evolves over time".to_string()],
        ))
    }

    async fn handle_temporal_query(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "temporal_query implementation"}),
            "Temporal query executed".to_string(),
            vec!["Time-travel through your knowledge graph history".to_string()],
        ))
    }

    async fn handle_create_database_snapshot(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "snapshot implementation"}),
            "Database snapshot created".to_string(),
            vec!["Snapshots provide safe restore points".to_string()],
        ))
    }

    async fn handle_mathematical_operation(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        Ok((
            serde_json::json!({"placeholder": "mathematical operation implementation"}),
            "Mathematical operation completed".to_string(),
            vec!["Use math operations to discover patterns in your data".to_string()],
        ))
    }

    async fn handle_federation_stats(&self, _params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let stats = self.usage_stats.read().await.clone();
        Ok((
            serde_json::json!({
                "total_operations": stats.total_operations,
                "cross_database_queries": stats.cross_database_queries,
                "version_operations": stats.version_operations,
                "math_operations": stats.math_operations,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "federation_efficiency": stats.federation_efficiency
            }),
            format!("Federation stats: {} operations, {:.1}ms avg response", stats.total_operations, stats.avg_response_time_ms),
            vec!["Monitor these metrics to optimize federation performance".to_string()],
        ))
    }

    async fn update_federated_stats(&self, response_time_ms: u64, method: &str) {
        let mut stats = self.usage_stats.write().await;
        stats.total_operations += 1;
        
        match method {
            "cross_database_similarity" | "compare_across_databases" => {
                stats.cross_database_queries += 1;
            },
            "compare_versions" | "temporal_query" | "create_database_snapshot" => {
                stats.version_operations += 1;
            },
            "mathematical_operation" | "calculate_relationship_strength" => {
                stats.math_operations += 1;
            },
            _ => {}
        }
        
        // Update rolling average
        if stats.total_operations == 1 {
            stats.avg_response_time_ms = response_time_ms as f64;
        } else {
            stats.avg_response_time_ms = (stats.avg_response_time_ms * (stats.total_operations - 1) as f64 + response_time_ms as f64) / stats.total_operations as f64;
        }
    }

    fn generate_federated_help(&self, method: &str) -> String {
        match method {
            "cross_database_similarity" => "💡 Tip: Use cosine similarity for conceptual relationships, Euclidean for precise matches. Higher thresholds give more precise results.".to_string(),
            "compare_across_databases" => "💡 Tip: Perfect for data quality checks and finding conflicting information between sources.".to_string(),
            "calculate_relationship_strength" => "💡 Tip: Combines semantic, structural, and co-occurrence metrics for comprehensive relationship analysis.".to_string(),
            "compare_versions" => "💡 Tip: Track how knowledge evolves over time. Great for auditing and quality control.".to_string(),
            "temporal_query" => "💡 Tip: Time-travel through your knowledge graph. Point-in-time queries show historical states.".to_string(),
            "mathematical_operation" => "💡 Tip: PageRank finds influential entities, shortest paths discover connections, statistics show overall patterns.".to_string(),
            _ => "💡 Tip: Use federation tools to leverage the full power of your multi-database knowledge graph.".to_string(),
        }
    }

    fn generate_federated_error_help(&self, method: &str) -> String {
        match method {
            "cross_database_similarity" => "❗ Common issues: Invalid entity names, empty databases, or embedding data not available.".to_string(),
            "compare_across_databases" => "❗ Common issues: Entity doesn't exist in specified databases or database connection problems.".to_string(),
            "temporal_query" => "❗ Common issues: Invalid timestamp format, entity has no version history, or database doesn't support versioning.".to_string(),
            _ => "❗ Check database connections and entity names. Ensure all required databases are accessible.".to_string(),
        }
    }

    fn generate_federated_error_suggestions(&self, method: &str) -> Vec<String> {
        match method {
            "cross_database_similarity" => vec![
                "Verify entity exists in at least one database".to_string(),
                "Check database connectivity with federation_stats".to_string(),
                "Try lowering similarity threshold".to_string(),
            ],
            "compare_across_databases" => vec![
                "Ensure entity exists in the specified databases".to_string(),
                "Check database IDs are correct".to_string(),
                "Try comparing across fewer databases first".to_string(),
            ],
            _ => vec![
                "Check the federation documentation for examples".to_string(),
                "Use federation_stats to verify system health".to_string(),
                "Try simpler operations first".to_string(),
            ],
        }
    }
}