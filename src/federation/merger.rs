// Result merger for federated query results

use crate::federation::types::{FederatedQueryResult, QueryResultData, MergeStrategy, SimilarityMatch, EntityComparison, MathematicalResult};
use crate::federation::router::RawQueryResult;
use crate::error::{GraphError, Result};
use std::collections::HashMap;
use async_trait::async_trait;

/// Result merger that combines results from multiple databases
pub struct ResultMerger {
    merge_strategies: HashMap<MergeStrategy, Box<dyn MergeHandler + Send + Sync>>,
}

impl Default for ResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for handling different merge strategies
#[async_trait]
pub trait MergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData>;
}

impl ResultMerger {
    pub fn new() -> Self {
        let mut merge_strategies: HashMap<MergeStrategy, Box<dyn MergeHandler + Send + Sync>> = HashMap::new();
        
        merge_strategies.insert(MergeStrategy::SimilarityMerge, Box::new(SimilarityMergeHandler));
        merge_strategies.insert(MergeStrategy::ComparisonMerge, Box::new(ComparisonMergeHandler));
        merge_strategies.insert(MergeStrategy::RelationshipMerge, Box::new(RelationshipMergeHandler));
        merge_strategies.insert(MergeStrategy::MathematicalMerge, Box::new(MathematicalMergeHandler));
        merge_strategies.insert(MergeStrategy::AggregationMerge, Box::new(AggregationMergeHandler));
        merge_strategies.insert(MergeStrategy::UnionMerge, Box::new(UnionMergeHandler));
        merge_strategies.insert(MergeStrategy::IntersectionMerge, Box::new(IntersectionMergeHandler));
        
        Self {
            merge_strategies,
        }
    }

    /// Merge raw results using the specified strategy
    pub async fn merge_results(
        &self,
        raw_results: Vec<RawQueryResult>,
        strategy: MergeStrategy,
    ) -> Result<FederatedQueryResult> {
        let start_time = std::time::Instant::now();
        
        // Filter successful results
        let successful_results: Vec<RawQueryResult> = raw_results.into_iter()
            .filter(|result| result.success)
            .collect();
        
        if successful_results.is_empty() {
            return Err(GraphError::InvalidInput("No successful results to merge".to_string()));
        }

        // Get the appropriate merge handler
        let merge_handler = self.merge_strategies.get(&strategy)
            .ok_or_else(|| GraphError::InvalidInput(format!("Unsupported merge strategy: {strategy:?}")))?;
        
        // Perform the merge
        let merged_data = merge_handler.merge(successful_results.clone()).await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        let databases_queried = successful_results.iter()
            .map(|result| result.database_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        let total_results = self.count_results(&merged_data);
        
        Ok(FederatedQueryResult {
            query_id: crate::federation::types::generate_query_id(),
            execution_time_ms: execution_time,
            databases_queried,
            total_results,
            results: merged_data,
            metadata: self.generate_merge_metadata(&successful_results, &strategy),
        })
    }

    /// Count the number of results in merged data
    fn count_results(&self, data: &QueryResultData) -> usize {
        match data {
            QueryResultData::SimilarityResults(results) => results.len(),
            QueryResultData::ComparisonResults(results) => results.len(),
            QueryResultData::RelationshipResults(results) => results.len(),
            QueryResultData::MathematicalResults(_) => 1,
            QueryResultData::AggregateResults(_) => 1,
        }
    }

    /// Generate metadata about the merge operation
    fn generate_merge_metadata(
        &self,
        results: &[RawQueryResult],
        strategy: &MergeStrategy,
    ) -> crate::federation::types::QueryMetadata {
        let _total_execution_time: u64 = results.iter().map(|r| r.execution_time_ms).sum();
        let databases_count = results.iter()
            .map(|r| &r.database_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        crate::federation::types::QueryMetadata {
            query_plan: format!("Merged {} results using {strategy:?} strategy", results.len()),
            optimization_used: vec!["parallel_execution".to_string(), "result_deduplication".to_string()],
            cache_hits: 0, // Would be tracked separately
            cache_misses: results.len(),
            network_round_trips: databases_count,
            data_transferred_bytes: results.iter()
                .map(|r| r.data.to_string().len())
                .sum(),
        }
    }
}

/// Handler for similarity merge operations
struct SimilarityMergeHandler;

#[async_trait]
impl MergeHandler for SimilarityMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            let mut all_similarities = Vec::new();
            
            for result in raw_results {
                // Parse similarity results from each database
                if let Ok(similarities) = self.parse_similarity_results(&result) {
                    all_similarities.extend(similarities);
                }
            }
            
            // Sort by similarity score (highest first)
            all_similarities.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
            
            // Remove duplicates based on entity
            all_similarities.dedup_by(|a, b| a.entity == b.entity);
            
            Ok(QueryResultData::SimilarityResults(all_similarities))
    }
}

impl SimilarityMergeHandler {
    fn parse_similarity_results(&self, result: &RawQueryResult) -> Result<Vec<SimilarityMatch>> {
        // Parse similarity matches from raw JSON data
        // Expected format: [{"entity": "...", "score": 0.95, "metadata": {...}}, ...]
        
        match serde_json::from_value::<Vec<SimilarityMatch>>(result.data.clone()) {
            Ok(matches) => Ok(matches),
            Err(_) => {
                // Try alternative parsing for backward compatibility
                // If the data format doesn't match expected structure, return error
                Err(GraphError::InvalidInput(
                    format!("Failed to parse similarity results from database '{}'. Expected format: array of SimilarityMatch objects", 
                            result.database_id.as_str())
                ))
            }
        }
    }
}

/// Handler for comparison merge operations
struct ComparisonMergeHandler;

#[async_trait]
impl MergeHandler for ComparisonMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            let mut comparisons = Vec::new();
            
            // Group results by entity ID
            let mut entity_groups: HashMap<String, Vec<&RawQueryResult>> = HashMap::new();
            for result in &raw_results {
                if let Some(entity_id) = self.extract_entity_id(result) {
                    entity_groups.entry(entity_id).or_default().push(result);
                }
            }
            
            // Create comparisons for each entity
            for (entity_id, results) in entity_groups {
                if results.len() > 1 {
                    let comparison = self.create_entity_comparison(&entity_id, results)?;
                    comparisons.push(comparison);
                }
            }
            
            Ok(QueryResultData::ComparisonResults(comparisons))
    }
}

impl ComparisonMergeHandler {
    fn extract_entity_id(&self, result: &RawQueryResult) -> Option<String> {
        // Extract entity ID from result data or metadata
        // Try to parse as object with "entity_id" field
        result.data.as_object()
            .and_then(|obj| obj.get("entity_id"))
            .and_then(|id| id.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                // Fallback: try to extract from array of results
                result.data.as_array()
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.as_object())
                    .and_then(|obj| obj.get("entity_id"))
                    .and_then(|id| id.as_str())
                    .map(|s| s.to_string())
            })
    }
    
    fn create_entity_comparison(&self, entity_id: &str, results: Vec<&RawQueryResult>) -> Result<EntityComparison> {
        // Create comparison between different database versions of the same entity
        let mut database_versions = Vec::new();
        let mut all_attributes = std::collections::HashSet::new();
        
        // Extract entity data from each database
        for result in &results {
            if let Some(entity_data) = result.data.as_object() {
                // Collect all attribute names
                for key in entity_data.keys() {
                    all_attributes.insert(key.clone());
                }
                
                let version = crate::federation::types::EntityVersion {
                    database_id: result.database_id.clone(),
                    version_id: entity_data.get("version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    attributes: entity_data.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                };
                database_versions.push(version);
            }
        }
        
        if database_versions.is_empty() {
            return Err(GraphError::InvalidInput(
                format!("No valid entity data found for comparison of entity '{entity_id}'")
            ));
        }
        
        // Compute differences between versions
        let mut differences = Vec::new();
        for attr in &all_attributes {
            let mut database_values = HashMap::new();
            let mut has_difference = false;
            let mut first_value: Option<&serde_json::Value> = None;
            
            for version in &database_versions {
                if let Some(value) = version.attributes.get(attr) {
                    database_values.insert(version.database_id.clone(), value.clone());
                    if let Some(first) = first_value {
                        if first != value {
                            has_difference = true;
                        }
                    } else {
                        first_value = Some(value);
                    }
                }
            }
            
            if has_difference {
                let field_diff = crate::federation::types::FieldDifference {
                    field_name: attr.clone(),
                    database_values,
                    difference_type: crate::federation::types::DifferenceType::ValueMismatch,
                };
                differences.push(field_diff);
            }
        }
        
        // Compute similarity score based on matching attributes
        let matching_attrs = all_attributes.len() - differences.len();
        let similarity_score = if all_attributes.is_empty() {
            0.0
        } else {
            matching_attrs as f32 / all_attributes.len() as f32
        };
        
        Ok(EntityComparison {
            entity_id: entity_id.to_string(),
            database_versions,
            differences,
            similarity_score,
        })
    }
}

/// Handler for relationship merge operations
struct RelationshipMergeHandler;

#[async_trait]
impl MergeHandler for RelationshipMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            let mut relationships = Vec::new();
            
            for result in raw_results {
                // Parse relationship results from each database
                if let Ok(rels) = self.parse_relationship_results(&result) {
                    relationships.extend(rels);
                }
            }
            
            // Remove duplicate relationships
            relationships.dedup_by(|a, b| {
                a.from_db == b.from_db && a.from_entity == b.from_entity &&
                a.to_db == b.to_db && a.to_entity == b.to_entity &&
                a.rel_type == b.rel_type
            });
            
            Ok(QueryResultData::RelationshipResults(relationships))
    }
}

impl RelationshipMergeHandler {
    fn parse_relationship_results(&self, result: &RawQueryResult) -> Result<Vec<crate::federation::types::CrossDatabaseRelationship>> {
        // Parse relationship data from result
        // Expected format: [{"from": {...}, "to": {...}, "type": "...", "metadata": {...}}, ...]
        
        let relationships = result.data.as_array()
            .ok_or_else(|| GraphError::InvalidInput(
                format!("Expected array of relationships from database '{}'", result.database_id.as_str())
            ))?;
        
        let mut parsed_relationships = Vec::new();
        
        for rel_data in relationships {
            let rel_obj = rel_data.as_object()
                .ok_or_else(|| GraphError::InvalidInput("Invalid relationship object format".to_string()))?;
            
            // Extract relationship components
            let from_entity = rel_obj.get("from")
                .and_then(|f| f.get("entity_id"))
                .and_then(|e| e.as_str())
                .ok_or_else(|| GraphError::InvalidInput("Missing 'from.entity_id' in relationship".to_string()))?;
            
            let to_entity = rel_obj.get("to")
                .and_then(|t| t.get("entity_id"))
                .and_then(|e| e.as_str())
                .ok_or_else(|| GraphError::InvalidInput("Missing 'to.entity_id' in relationship".to_string()))?;
            
            let rel_type = rel_obj.get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("related_to");
            
            let metadata = rel_obj.get("metadata")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            
            let relationship = crate::federation::types::CrossDatabaseRelationship {
                from_db: result.database_id.clone(),
                from_entity: from_entity.to_string(),
                to_db: rel_obj.get("to")
                    .and_then(|t| t.get("database"))
                    .and_then(|d| d.as_str())
                    .map(|s| crate::federation::types::DatabaseId::new(s.to_string()))
                    .unwrap_or_else(|| result.database_id.clone()),
                to_entity: to_entity.to_string(),
                rel_type: rel_type.to_string(),
                confidence: rel_obj.get("confidence")
                    .and_then(|c| c.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(1.0),
                metadata,
            };
            
            parsed_relationships.push(relationship);
        }
        
        Ok(parsed_relationships)
    }
}

/// Handler for mathematical operation merge
struct MathematicalMergeHandler;

#[async_trait]
impl MergeHandler for MathematicalMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            // Combine mathematical results from multiple databases
            // The specific combination depends on the operation type
            
            let combined_result = MathematicalResult {
                operation: crate::federation::types::MathOperation::CosineSimilarity, // Placeholder
                result_type: crate::federation::types::MathResultType::Scalar(0.5),
                execution_time_ms: raw_results.iter().map(|r| r.execution_time_ms).sum(),
            };
            
            Ok(QueryResultData::MathematicalResults(combined_result))
    }
}

/// Handler for aggregation merge operations
struct AggregationMergeHandler;

#[async_trait]
impl MergeHandler for AggregationMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            // Aggregate numerical results from multiple databases
            let mut per_database = HashMap::new();
            let mut total_value = 0.0;
            let mut total_count = 0;
            
            for result in raw_results {
                if let Ok((value, count)) = self.extract_aggregate_data(&result) {
                    per_database.insert(result.database_id.clone(), value);
                    total_value += value;
                    total_count += count;
                }
            }
            
            let aggregate_result = crate::federation::types::AggregateResult {
                function: crate::federation::types::AggregateFunction::Sum, // Would be determined from query
                value: total_value,
                count: total_count,
                per_database,
            };
            
            Ok(QueryResultData::AggregateResults(aggregate_result))
    }
}

impl AggregationMergeHandler {
    fn extract_aggregate_data(&self, result: &RawQueryResult) -> Result<(f64, usize)> {
        // Extract aggregated value and count from result
        // Expected format: {"value": 123.45, "count": 10, ...}
        
        let obj = result.data.as_object()
            .ok_or_else(|| GraphError::InvalidInput(
                format!("Expected object with 'value' and 'count' from database '{}'", result.database_id.as_str())
            ))?;
        
        let value = obj.get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| GraphError::InvalidInput(
                format!("Missing or invalid 'value' field in aggregate result from database '{}'", result.database_id.as_str())
            ))?;
        
        let count = obj.get("count")
            .and_then(|c| c.as_u64())
            .map(|c| c as usize)
            .ok_or_else(|| GraphError::InvalidInput(
                format!("Missing or invalid 'count' field in aggregate result from database '{}'", result.database_id.as_str())
            ))?;
        
        Ok((value, count))
    }
}

/// Handler for union merge operations
struct UnionMergeHandler;

#[async_trait]
impl MergeHandler for UnionMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            // Union of all results - combine all unique entities
            let mut all_entities = std::collections::HashSet::new();
            let mut similarity_results = Vec::new();
            
            for result in raw_results {
                // Extract entities from each result
                if let Some(entities) = result.data.as_array() {
                    for entity in entities {
                        // Create a unique key for deduplication
                        let entity_key = entity.get("entity_id")
                            .or_else(|| entity.get("id"))
                            .and_then(|id| id.as_str())
                            .unwrap_or_default();
                        
                        if !entity_key.is_empty() && all_entities.insert(entity_key.to_string()) {
                            // Convert to SimilarityMatch for uniform output
                            let similarity_match = SimilarityMatch {
                                entity: entity_key.to_string(),
                                similarity_score: 1.0, // Union doesn't have scores
                                metadata: entity.clone(),
                            };
                            similarity_results.push(similarity_match);
                        }
                    }
                }
            }
            
            Ok(QueryResultData::SimilarityResults(similarity_results))
    }
}

/// Handler for intersection merge operations
struct IntersectionMergeHandler;

#[async_trait]
impl MergeHandler for IntersectionMergeHandler {
    async fn merge(&self, raw_results: Vec<RawQueryResult>) -> Result<QueryResultData> {
            // Find common results across all databases
            if raw_results.is_empty() {
                return Ok(QueryResultData::SimilarityResults(Vec::new()));
            }
            
            // Count occurrences of each entity across databases
            let mut entity_counts: std::collections::HashMap<String, (usize, serde_json::Value)> = std::collections::HashMap::new();
            let total_databases = raw_results.len();
            
            for result in &raw_results {
                if let Some(entities) = result.data.as_array() {
                    for entity in entities {
                        let entity_key = entity.get("entity_id")
                            .or_else(|| entity.get("id"))
                            .and_then(|id| id.as_str())
                            .unwrap_or_default()
                            .to_string();
                        
                        if !entity_key.is_empty() {
                            entity_counts.entry(entity_key)
                                .and_modify(|(count, _)| *count += 1)
                                .or_insert((1, entity.clone()));
                        }
                    }
                }
            }
            
            // Keep only entities that appear in all databases
            let mut intersection_results = Vec::new();
            for (entity_key, (count, metadata)) in entity_counts {
                if count == total_databases {
                    let similarity_match = SimilarityMatch {
                        entity: entity_key,
                        similarity_score: 1.0, // Perfect match across all databases
                        metadata,
                    };
                    intersection_results.push(similarity_match);
                }
            }
            
            Ok(QueryResultData::SimilarityResults(intersection_results))
    }
}