use crate::error::{GraphError, Result};
use crate::federation::DatabaseId;
use crate::versioning::types::*;
use crate::versioning::version_store::VersionStore;
use std::collections::HashMap;
use std::time::SystemTime;
use serde_json::Value;

/// Version merger for handling conflicts and merging versions
pub struct VersionMerger {
    /// Merge strategies and their weights
    strategy_weights: HashMap<ConflictResolution, f32>,
    /// Statistics about merge operations
    merge_stats: MergeOperationStats,
}

impl Default for VersionMerger {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionMerger {
    pub fn new() -> Self {
        let mut strategy_weights = HashMap::new();
        strategy_weights.insert(ConflictResolution::TakeVersion1, 0.3);
        strategy_weights.insert(ConflictResolution::TakeVersion2, 0.3);
        strategy_weights.insert(ConflictResolution::TakeNewest, 0.8);
        strategy_weights.insert(ConflictResolution::TakeHighestConfidence, 0.9);
        strategy_weights.insert(ConflictResolution::SmartMerge, 1.0);
        strategy_weights.insert(ConflictResolution::ManualResolution, 0.1);

        Self {
            strategy_weights,
            merge_stats: MergeOperationStats {
                total_merges: 0,
                successful_merges: 0,
                conflicts_resolved: 0,
                manual_interventions: 0,
                average_merge_time_ms: 0.0,
            },
        }
    }

    /// Perform a three-way merge between versions
    pub async fn three_way_merge(
        &self,
        store: &VersionStore,
        entity_id: &str,
        base_version: &VersionId,
        version1: &VersionId,
        version2: &VersionId,
        resolution_strategy: ConflictResolution,
    ) -> Result<MergeResult> {
        let start_time = SystemTime::now();
        
        // Get the three versions
        let base = store.get_version(entity_id, base_version).await?;
        let v1 = store.get_version(entity_id, version1).await?;
        let v2 = store.get_version(entity_id, version2).await?;

        // Reconstruct full states for each version
        let base_state = self.reconstruct_full_state(&base)?;
        let v1_state = self.reconstruct_full_state(&v1)?;
        let v2_state = self.reconstruct_full_state(&v2)?;

        // Detect conflicts
        let conflicts = self.detect_merge_conflicts(&base_state, &v1_state, &v2_state)?;

        // Apply resolution strategy
        let (merged_changes, resolved_conflicts) = self.resolve_conflicts(
            &conflicts,
            &base_state,
            &v1_state,
            &v2_state,
            &v1,
            &v2,
            resolution_strategy.clone(),
        )?;

        // Create merged version if successful
        let merged_version = if resolved_conflicts.iter().all(|c| c.suggested_resolution.is_some()) {
            // Create new version with merged changes
            let new_version_id = VersionId::generate(&v1.database_id, entity_id);
            Some(new_version_id)
        } else {
            None
        };

        let merge_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        let merge_result = MergeResult {
            success: merged_version.is_some(),
            merged_version,
            conflicts: resolved_conflicts,
            resolution_applied: resolution_strategy,
            merge_statistics: MergeStatistics {
                total_fields: base_state.len() + v1_state.len() + v2_state.len(),
                fields_merged_automatically: merged_changes.len(),
                fields_requiring_resolution: conflicts.len(),
                merge_time_ms: merge_time,
            },
        };

        Ok(merge_result)
    }

    /// Compare versions across different databases
    pub async fn compare_across_databases(
        &self,
        entity_id: &str,
        database_versions: Vec<(DatabaseId, VersionEntry)>,
    ) -> Result<CrossDatabaseComparison> {
        if database_versions.is_empty() {
            return Err(GraphError::InvalidInput("No database versions provided".to_string()));
        }

        // Reconstruct states for all database versions
        let mut states = HashMap::new();
        for (db_id, version) in &database_versions {
            let state = self.reconstruct_full_state(version)?;
            states.insert(db_id.clone(), state);
        }

        // Find consensus and conflicting fields
        let (consensus_fields, conflicting_fields) = self.analyze_cross_database_consensus(&states);

        // Calculate similarity matrix
        let similarity_matrix = self.calculate_similarity_matrix(&database_versions)?;

        // Generate recommendations
        let recommendations = self.generate_merge_recommendations(&consensus_fields, &conflicting_fields);

        Ok(CrossDatabaseComparison {
            entity_id: entity_id.to_string(),
            database_versions,
            consensus_fields,
            conflicting_fields,
            similarity_matrix,
            recommendations,
        })
    }

    /// Merge multiple versions using consensus
    pub async fn consensus_merge(
        &self,
        entity_id: &str,
        versions: Vec<VersionEntry>,
        consensus_threshold: f32,
    ) -> Result<MergeResult> {
        let start_time = SystemTime::now();
        
        if versions.is_empty() {
            return Err(GraphError::InvalidInput("No versions to merge".to_string()));
        }

        // Reconstruct states for all versions
        let mut states = Vec::new();
        for version in &versions {
            let state = self.reconstruct_full_state(version)?;
            states.push(state);
        }

        // Find consensus fields
        let consensus_fields = self.find_consensus_fields(&states, consensus_threshold);

        // Create merged changes
        let merged_changes: Vec<FieldChange> = consensus_fields
            .into_iter()
            .map(|(field_name, value)| FieldChange {
                field_name,
                old_value: None,
                new_value: value,
                change_type: ChangeType::Modified,
            })
            .collect();

        let merge_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        let merge_result = MergeResult {
            success: true,
            merged_version: Some(VersionId::generate(&versions[0].database_id, entity_id)),
            conflicts: Vec::new(),
            resolution_applied: ConflictResolution::SmartMerge,
            merge_statistics: MergeStatistics {
                total_fields: states.iter().map(|s| s.len()).sum(),
                fields_merged_automatically: merged_changes.len(),
                fields_requiring_resolution: 0,
                merge_time_ms: merge_time,
            },
        };

        Ok(merge_result)
    }

    // Helper methods

    fn reconstruct_full_state(&self, version: &VersionEntry) -> Result<HashMap<String, Value>> {
        let mut state = HashMap::new();
        
        // Apply all changes to reconstruct the full state
        for change in &version.changes {
            match change.change_type {
                ChangeType::Added | ChangeType::Modified => {
                    state.insert(change.field_name.clone(), change.new_value.clone());
                }
                ChangeType::Removed => {
                    state.remove(&change.field_name);
                }
                ChangeType::Renamed => {
                    // For renamed fields, we need more context
                    // For now, treat as modified
                    state.insert(change.field_name.clone(), change.new_value.clone());
                }
            }
        }
        
        Ok(state)
    }

    fn detect_merge_conflicts(
        &self,
        base_state: &HashMap<String, Value>,
        v1_state: &HashMap<String, Value>,
        v2_state: &HashMap<String, Value>,
    ) -> Result<Vec<MergeConflict>> {
        let mut conflicts = Vec::new();
        
        // Find all unique field names across all states
        let mut all_fields = std::collections::HashSet::new();
        all_fields.extend(base_state.keys());
        all_fields.extend(v1_state.keys());
        all_fields.extend(v2_state.keys());

        for field_name in all_fields {
            let base_value = base_state.get(field_name);
            let v1_value = v1_state.get(field_name);
            let v2_value = v2_state.get(field_name);

            let conflict_type = self.determine_conflict_type(base_value, v1_value, v2_value);
            
            if let Some(conflict_type) = conflict_type {
                let suggested_resolution = self.suggest_conflict_resolution(
                    base_value, v1_value, v2_value, &conflict_type
                );

                conflicts.push(MergeConflict {
                    field_name: field_name.clone(),
                    base_value: base_value.cloned(),
                    version1_value: v1_value.cloned(),
                    version2_value: v2_value.cloned(),
                    conflict_type,
                    suggested_resolution,
                });
            }
        }

        Ok(conflicts)
    }

    fn determine_conflict_type(
        &self,
        base_value: Option<&Value>,
        v1_value: Option<&Value>,
        v2_value: Option<&Value>,
    ) -> Option<ConflictType> {
        match (base_value, v1_value, v2_value) {
            // No conflict if values are the same
            (_, Some(v1), Some(v2)) if v1 == v2 => None,
            
            // Value conflict: both versions changed the field differently
            (Some(_), Some(v1), Some(v2)) if v1 != v2 => Some(ConflictType::ValueConflict),
            
            // Structural conflict: one added, one removed
            (None, Some(_), None) | (None, None, Some(_)) => Some(ConflictType::StructuralConflict),
            (Some(_), None, Some(_)) | (Some(_), Some(_), None) => Some(ConflictType::StructuralConflict),
            
            // Type conflict: values have different types
            (_, Some(v1), Some(v2)) if self.are_different_types(v1, v2) => Some(ConflictType::TypeConflict),
            
            _ => None,
        }
    }

    fn are_different_types(&self, v1: &Value, v2: &Value) -> bool {
        std::mem::discriminant(v1) != std::mem::discriminant(v2)
    }

    fn suggest_conflict_resolution(
        &self,
        _base_value: Option<&Value>,
        v1_value: Option<&Value>,
        v2_value: Option<&Value>,
        conflict_type: &ConflictType,
    ) -> Option<Value> {
        match conflict_type {
            ConflictType::ValueConflict => {
                // For value conflicts, prefer non-null values
                match (v1_value, v2_value) {
                    (Some(v1), Some(v2)) => {
                        // Could implement more sophisticated logic here
                        // For now, prefer strings over numbers, objects over primitives, etc.
                        if v1.is_string() && !v2.is_string() {
                            Some(v1.clone())
                        } else if v2.is_string() && !v1.is_string() {
                            Some(v2.clone())
                        } else {
                            Some(v1.clone()) // Default to first version
                        }
                    }
                    (Some(v), None) | (None, Some(v)) => Some(v.clone()),
                    _ => None,
                }
            }
            ConflictType::StructuralConflict => {
                // For structural conflicts, prefer additions over removals
                v1_value.or(v2_value).cloned()
            }
            ConflictType::TypeConflict => {
                // For type conflicts, prefer more complex types
                match (v1_value, v2_value) {
                    (Some(v1), Some(v2)) => {
                        if v1.is_object() && !v2.is_object() {
                            Some(v1.clone())
                        } else if v2.is_object() && !v1.is_object() {
                            Some(v2.clone())
                        } else if v1.is_array() && !v2.is_array() {
                            Some(v1.clone())
                        } else if v2.is_array() && !v1.is_array() {
                            Some(v2.clone())
                        } else {
                            Some(v1.clone())
                        }
                    }
                    _ => None,
                }
            }
            ConflictType::RenameConflict => {
                // For rename conflicts, manual resolution is usually needed
                None
            }
        }
    }

    fn resolve_conflicts(
        &self,
        conflicts: &[MergeConflict],
        _base_state: &HashMap<String, Value>,
        _v1_state: &HashMap<String, Value>,
        _v2_state: &HashMap<String, Value>,
        v1: &VersionEntry,
        v2: &VersionEntry,
        strategy: ConflictResolution,
    ) -> Result<(Vec<FieldChange>, Vec<MergeConflict>)> {
        let mut merged_changes = Vec::new();
        let mut resolved_conflicts = Vec::new();

        for conflict in conflicts {
            let resolved_value = match strategy {
                ConflictResolution::TakeVersion1 => conflict.version1_value.clone(),
                ConflictResolution::TakeVersion2 => conflict.version2_value.clone(),
                ConflictResolution::TakeNewest => {
                    if v1.timestamp > v2.timestamp {
                        conflict.version1_value.clone()
                    } else {
                        conflict.version2_value.clone()
                    }
                }
                ConflictResolution::TakeHighestConfidence => {
                    if v1.metadata.confidence_score > v2.metadata.confidence_score {
                        conflict.version1_value.clone()
                    } else {
                        conflict.version2_value.clone()
                    }
                }
                ConflictResolution::SmartMerge => conflict.suggested_resolution.clone(),
                ConflictResolution::ManualResolution => None,
            };

            let mut resolved_conflict = conflict.clone();
            resolved_conflict.suggested_resolution = resolved_value.clone();
            resolved_conflicts.push(resolved_conflict);

            if let Some(value) = resolved_value {
                merged_changes.push(FieldChange {
                    field_name: conflict.field_name.clone(),
                    old_value: conflict.base_value.clone(),
                    new_value: value,
                    change_type: ChangeType::Modified,
                });
            }
        }

        Ok((merged_changes, resolved_conflicts))
    }

    fn analyze_cross_database_consensus(
        &self,
        states: &HashMap<DatabaseId, HashMap<String, Value>>,
    ) -> (HashMap<String, Value>, HashMap<String, Vec<(DatabaseId, Value)>>) {
        let mut field_values: HashMap<String, Vec<(DatabaseId, Value)>> = HashMap::new();
        
        // Collect all field values across databases
        for (db_id, state) in states {
            for (field_name, value) in state {
                field_values
                    .entry(field_name.clone())
                    .or_default()
                    .push((db_id.clone(), value.clone()));
            }
        }

        let mut consensus_fields = HashMap::new();
        let mut conflicting_fields = HashMap::new();

        for (field_name, values) in field_values {
            // Check if all values are the same
            if values.len() > 1 {
                let first_value = &values[0].1;
                let all_same = values.iter().all(|(_, v)| v == first_value);
                
                if all_same {
                    consensus_fields.insert(field_name, first_value.clone());
                } else {
                    conflicting_fields.insert(field_name, values);
                }
            } else if values.len() == 1 {
                consensus_fields.insert(field_name, values[0].1.clone());
            }
        }

        (consensus_fields, conflicting_fields)
    }

    fn calculate_similarity_matrix(&self, versions: &[(DatabaseId, VersionEntry)]) -> Result<Vec<Vec<f32>>> {
        let n = versions.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    let state1 = self.reconstruct_full_state(&versions[i].1)?;
                    let state2 = self.reconstruct_full_state(&versions[j].1)?;
                    matrix[i][j] = self.calculate_state_similarity(&state1, &state2);
                }
            }
        }

        Ok(matrix)
    }

    fn calculate_state_similarity(&self, state1: &HashMap<String, Value>, state2: &HashMap<String, Value>) -> f32 {
        if state1.is_empty() && state2.is_empty() {
            return 1.0;
        }

        let all_fields: std::collections::HashSet<&String> = state1.keys().chain(state2.keys()).collect();
        let total_fields = all_fields.len() as f32;
        
        if total_fields == 0.0 {
            return 1.0;
        }

        let matching_fields = all_fields
            .into_iter()
            .filter(|field| {
                match (state1.get(field.as_str()), state2.get(field.as_str())) {
                    (Some(v1), Some(v2)) => v1 == v2,
                    (None, None) => true,
                    _ => false,
                }
            })
            .count() as f32;

        matching_fields / total_fields
    }

    fn generate_merge_recommendations(
        &self,
        consensus_fields: &HashMap<String, Value>,
        conflicting_fields: &HashMap<String, Vec<(DatabaseId, Value)>>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !consensus_fields.is_empty() {
            recommendations.push(format!(
                "Found {} fields with consensus across databases",
                consensus_fields.len()
            ));
        }

        if !conflicting_fields.is_empty() {
            recommendations.push(format!(
                "Found {} conflicting fields requiring resolution",
                conflicting_fields.len()
            ));

            for (field_name, values) in conflicting_fields {
                if values.len() == 2 {
                    recommendations.push(format!(
                        "Field '{}' conflicts between {} databases - consider manual review",
                        field_name, values.len()
                    ));
                } else {
                    recommendations.push(format!(
                        "Field '{}' has {} different values across databases - complex merge needed",
                        field_name, values.len()
                    ));
                }
            }
        }

        if recommendations.is_empty() {
            recommendations.push("All databases are in sync for this entity".to_string());
        }

        recommendations
    }

    fn find_consensus_fields(
        &self,
        states: &[HashMap<String, Value>],
        threshold: f32,
    ) -> HashMap<String, Value> {
        let mut field_consensus = HashMap::new();
        
        // Find all unique field names
        let all_fields: std::collections::HashSet<_> = states
            .iter()
            .flat_map(|state| state.keys())
            .collect();

        for field_name in all_fields {
            let mut value_counts: HashMap<String, (Value, usize)> = HashMap::new();
            let mut total_count = 0;

            for state in states {
                if let Some(value) = state.get(field_name) {
                    let value_key = serde_json::to_string(value).unwrap_or_default();
                    let entry = value_counts.entry(value_key).or_insert((value.clone(), 0));
                    entry.1 += 1;
                    total_count += 1;
                }
            }

            // Find the most common value
            if let Some((_, (consensus_value, count))) = value_counts
                .iter()
                .max_by_key(|(_, (_, count))| *count)
            {
                let consensus_ratio = *count as f32 / total_count as f32;
                if consensus_ratio >= threshold {
                    field_consensus.insert(field_name.clone(), consensus_value.clone());
                }
            }
        }

        field_consensus
    }

    /// Get merge operation statistics
    pub fn get_statistics(&self) -> &MergeOperationStats {
        &self.merge_stats
    }
}

/// Statistics about merge operations
#[derive(Debug, Clone)]
pub struct MergeOperationStats {
    pub total_merges: u64,
    pub successful_merges: u64,
    pub conflicts_resolved: u64,
    pub manual_interventions: u64,
    pub average_merge_time_ms: f64,
}