use crate::error::{GraphError, Result};
use crate::federation::DatabaseId;
use crate::versioning::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use serde_json;

/// Version store that manages version data for a specific database
pub struct VersionStore {
    database_id: DatabaseId,
    /// Entity ID -> List of versions
    entity_versions: Arc<RwLock<HashMap<String, Vec<VersionEntry>>>>,
    /// Version ID -> Version data (for quick lookup)
    version_index: Arc<RwLock<HashMap<VersionId, VersionEntry>>>,
    /// Snapshots for this database
    snapshots: Arc<RwLock<HashMap<SnapshotId, SnapshotData>>>,
    /// Statistics
    stats: Arc<RwLock<DatabaseVersionStats>>,
}

impl VersionStore {
    pub fn new(database_id: DatabaseId) -> Result<Self> {
        Ok(Self {
            database_id,
            entity_versions: Arc::new(RwLock::new(HashMap::new())),
            version_index: Arc::new(RwLock::new(HashMap::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DatabaseVersionStats {
                total_versions: 0,
                total_entities: 0,
                anchor_versions: 0,
                delta_versions: 0,
                total_storage_bytes: 0,
                compression_ratio: 1.0,
                average_version_size: 0.0,
            })),
        })
    }

    /// Create a new version for an entity
    pub async fn create_version(
        &self,
        entity_id: &str,
        changes: Vec<FieldChange>,
        author: Option<String>,
        message: Option<String>,
    ) -> Result<VersionId> {
        let version_id = VersionId::generate(&self.database_id, entity_id);
        let timestamp = SystemTime::now();

        // Determine if this should be an anchor version
        let entity_versions = self.entity_versions.read().await;
        let existing_versions = entity_versions.get(entity_id).map(|v| v.len()).unwrap_or(0);
        let is_anchor = existing_versions % 10 == 0; // Every 10th version is an anchor
        drop(entity_versions);

        // Get parent versions
        let parent_versions = self.get_parent_versions(entity_id).await?;

        // Create version entry
        let version_entry = VersionEntry {
            version_id: version_id.clone(),
            entity_id: entity_id.to_string(),
            database_id: self.database_id.clone(),
            timestamp,
            author,
            message,
            parent_versions,
            changes: changes.clone(),
            is_anchor,
            compressed_size: self.estimate_compressed_size(&changes),
            metadata: VersionMetadata {
                branch_name: None,
                tags: Vec::new(),
                confidence_score: 1.0,
                validation_status: ValidationStatus::Valid,
                source: VersionSource::UserEdit,
                checksum: self.calculate_checksum(&changes),
            },
        };

        // Store the version
        {
            let mut entity_versions = self.entity_versions.write().await;
            entity_versions.entry(entity_id.to_string()).or_insert_with(Vec::new).push(version_entry.clone());
        }

        {
            let mut version_index = self.version_index.write().await;
            version_index.insert(version_id.clone(), version_entry);
        }

        // Update statistics
        self.update_stats(is_anchor).await;

        Ok(version_id)
    }

    /// Get version history for an entity
    pub async fn get_version_history(&self, entity_id: &str) -> Result<Vec<VersionEntry>> {
        let entity_versions = self.entity_versions.read().await;
        Ok(entity_versions.get(entity_id).cloned().unwrap_or_default())
    }

    /// Get a specific version
    pub async fn get_version(&self, _entity_id: &str, version_id: &VersionId) -> Result<VersionEntry> {
        let version_index = self.version_index.read().await;
        version_index.get(version_id)
            .cloned()
            .ok_or_else(|| GraphError::InvalidInput(format!("Version not found: {}", version_id.as_str())))
    }

    /// Get the latest version of an entity
    pub async fn get_latest_version(&self, entity_id: &str) -> Result<VersionEntry> {
        let entity_versions = self.entity_versions.read().await;
        let versions = entity_versions.get(entity_id)
            .ok_or(GraphError::EntityNotFound { id: 0 })?;

        versions.last()
            .cloned()
            .ok_or_else(|| GraphError::InvalidInput("No versions found for entity".to_string()))
    }

    /// Compare two versions
    pub async fn compare_versions(
        &self,
        entity_id: &str,
        version1: &VersionId,
        version2: &VersionId,
    ) -> Result<VersionComparison> {
        let v1 = self.get_version(entity_id, version1).await?;
        let v2 = self.get_version(entity_id, version2).await?;

        let differences = self.calculate_differences(&v1, &v2);
        let similarity_score = self.calculate_similarity(&differences);
        let change_summary = self.create_change_summary(&differences);

        Ok(VersionComparison {
            entity_id: entity_id.to_string(),
            version1: version1.clone(),
            version2: version2.clone(),
            differences,
            similarity_score,
            change_summary,
        })
    }

    /// Create a snapshot
    pub async fn create_snapshot(
        &self,
        name: String,
        description: Option<String>,
    ) -> Result<SnapshotId> {
        let snapshot_id = SnapshotId::new(format!("{}:{}", self.database_id.as_str(), uuid::Uuid::new_v4()));
        
        // Capture current state of all entities
        let entity_versions = self.entity_versions.read().await;
        let mut entity_states = HashMap::new();
        
        for (entity_id, versions) in entity_versions.iter() {
            if let Some(latest_version) = versions.last() {
                entity_states.insert(entity_id.clone(), latest_version.version_id.clone());
            }
        }

        let snapshot = SnapshotData {
            id: snapshot_id.clone(),
            name,
            description,
            created_at: SystemTime::now(),
            entity_states,
        };

        let mut snapshots = self.snapshots.write().await;
        snapshots.insert(snapshot_id.clone(), snapshot);

        Ok(snapshot_id)
    }

    /// Restore from a snapshot
    pub async fn restore_snapshot(&self, snapshot_id: &SnapshotId) -> Result<()> {
        let snapshots = self.snapshots.read().await;
        let _snapshot = snapshots.get(snapshot_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Snapshot not found: {}", snapshot_id.as_str())))?;

        // In a real implementation, this would restore the database state
        // For now, we'll just validate the snapshot exists
        Ok(())
    }

    /// Get statistics for this version store
    pub async fn get_statistics(&self) -> Result<DatabaseVersionStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Clean up old versions based on retention policy
    pub async fn cleanup_versions(&self, policy: RetentionPolicy) -> Result<CleanupResult> {
        let start_time = SystemTime::now();
        let mut versions_removed = 0;
        let mut storage_freed = 0;
        let mut entities_affected = 0;

        let mut entity_versions = self.entity_versions.write().await;
        let mut version_index = self.version_index.write().await;

        for (_entity_id, versions) in entity_versions.iter_mut() {
            let _original_count = versions.len();
            
            // Apply retention policy
            let mut keep_versions = Vec::new();
            let mut remove_versions = Vec::new();

            // Always keep minimum versions
            let min_to_keep = policy.min_versions_to_keep.max(1);
            
            if versions.len() > min_to_keep {
                // Sort by timestamp (newest first)
                versions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                
                for (i, version) in versions.iter().enumerate() {
                    let should_keep = i < min_to_keep ||
                        (policy.preserve_anchors && version.is_anchor) ||
                        (policy.preserve_tagged_versions && !version.metadata.tags.is_empty()) ||
                        (policy.max_versions_per_entity.is_none_or(|max| i < max)) ||
                        (policy.max_age_days.is_none_or(|max_days| {
                            version.timestamp.elapsed().unwrap_or_default().as_secs() < (max_days as u64 * 24 * 3600)
                        }));

                    if should_keep {
                        keep_versions.push(version.clone());
                    } else {
                        remove_versions.push(version.clone());
                        storage_freed += version.compressed_size;
                    }
                }

                if !remove_versions.is_empty() {
                    *versions = keep_versions;
                    versions_removed += remove_versions.len();
                    entities_affected += 1;

                    // Remove from version index
                    for version in remove_versions {
                        version_index.remove(&version.version_id);
                    }
                }
            }
        }

        let cleanup_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        Ok(CleanupResult {
            versions_removed,
            storage_freed_bytes: storage_freed,
            entities_affected,
            cleanup_time_ms: cleanup_time,
        })
    }

    // Helper methods

    async fn get_parent_versions(&self, entity_id: &str) -> Result<Vec<VersionId>> {
        let entity_versions = self.entity_versions.read().await;
        if let Some(versions) = entity_versions.get(entity_id) {
            if let Some(latest) = versions.last() {
                Ok(vec![latest.version_id.clone()])
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
    }

    fn estimate_compressed_size(&self, changes: &[FieldChange]) -> usize {
        // Simple estimation based on serialized size
        serde_json::to_string(changes).map(|s| s.len()).unwrap_or(0)
    }

    fn calculate_checksum(&self, changes: &[FieldChange]) -> String {
        // Simple checksum using hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for change in changes {
            change.field_name.hash(&mut hasher);
            if let Ok(json_str) = serde_json::to_string(&change.new_value) {
                json_str.hash(&mut hasher);
            }
        }
        format!("{:x}", hasher.finish())
    }

    async fn update_stats(&self, is_anchor: bool) {
        let mut stats = self.stats.write().await;
        stats.total_versions += 1;
        if is_anchor {
            stats.anchor_versions += 1;
        } else {
            stats.delta_versions += 1;
        }
        
        // Update entities count
        let entity_versions = self.entity_versions.read().await;
        stats.total_entities = entity_versions.len();
    }

    fn calculate_differences(&self, v1: &VersionEntry, v2: &VersionEntry) -> Vec<FieldDifference> {
        let mut differences = Vec::new();
        
        // Create field maps for comparison
        let mut v1_fields: HashMap<String, &serde_json::Value> = HashMap::new();
        let mut v2_fields: HashMap<String, &serde_json::Value> = HashMap::new();
        
        for change in &v1.changes {
            v1_fields.insert(change.field_name.clone(), &change.new_value);
        }
        
        for change in &v2.changes {
            v2_fields.insert(change.field_name.clone(), &change.new_value);
        }
        
        // Find all unique field names
        let mut all_fields: std::collections::HashSet<&String> = std::collections::HashSet::new();
        all_fields.extend(v1_fields.keys());
        all_fields.extend(v2_fields.keys());
        
        for field_name in all_fields {
            let v1_value = v1_fields.get(field_name);
            let v2_value = v2_fields.get(field_name);
            
            let difference_type = match (v1_value, v2_value) {
                (Some(_), None) => DifferenceType::FieldRemoved,
                (None, Some(_)) => DifferenceType::FieldAdded,
                (Some(val1), Some(val2)) if val1 != val2 => DifferenceType::ValueChanged,
                _ => continue, // No difference
            };
            
            differences.push(FieldDifference {
                field_name: field_name.clone(),
                difference_type,
                version1_value: v1_value.cloned().cloned(),
                version2_value: v2_value.cloned().cloned(),
                similarity_score: 0.5, // Simplified calculation
            });
        }
        
        differences
    }

    fn calculate_similarity(&self, differences: &[FieldDifference]) -> f32 {
        if differences.is_empty() {
            return 1.0;
        }
        
        let total_score: f32 = differences.iter().map(|d| d.similarity_score).sum();
        total_score / differences.len() as f32
    }

    fn create_change_summary(&self, differences: &[FieldDifference]) -> ChangeSummary {
        let mut fields_added = 0;
        let mut fields_modified = 0;
        let mut fields_removed = 0;
        let mut change_categories = HashMap::new();
        
        for diff in differences {
            match diff.difference_type {
                DifferenceType::FieldAdded => fields_added += 1,
                DifferenceType::ValueChanged => fields_modified += 1,
                DifferenceType::FieldRemoved => fields_removed += 1,
                _ => {}
            }
            
            let category = format!("{:?}", diff.difference_type);
            *change_categories.entry(category).or_insert(0) += 1;
        }
        
        ChangeSummary {
            fields_added,
            fields_modified,
            fields_removed,
            major_changes: Vec::new(), // Could be enhanced with smart detection
            change_categories,
        }
    }
}

/// Snapshot data structure
#[derive(Clone)]
struct SnapshotData {
    id: SnapshotId,
    name: String,
    description: Option<String>,
    created_at: SystemTime,
    entity_states: HashMap<String, VersionId>,
}