//! Temporal tracking for time travel queries
//! Pure data structure implementation - no AI

use crate::core::triple::Triple;
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock as StdRwLock};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Temporal wrapper for triples with version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTriple {
    pub triple: Triple,
    pub timestamp: DateTime<Utc>,
    pub version: u32,
    pub operation: TemporalOperation,
    pub previous_value: Option<String>, // For updates
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TemporalOperation {
    Create,
    Update,
    Delete,
}

/// Temporal index for efficient time-based queries
pub struct TemporalIndex {
    // Entity -> Timeline of changes
    entity_timelines: Arc<StdRwLock<HashMap<String, BTreeMap<DateTime<Utc>, Vec<TemporalTriple>>>>>,
    // Global timeline of all changes
    global_timeline: Arc<StdRwLock<BTreeMap<DateTime<Utc>, Vec<TemporalTriple>>>>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            entity_timelines: Arc::new(StdRwLock::new(HashMap::new())),
            global_timeline: Arc::new(StdRwLock::new(BTreeMap::new())),
        }
    }
    
    /// Record a triple operation with timestamp
    pub fn record_operation(&self, triple: Triple, operation: TemporalOperation, previous_value: Option<String>) {
        let timestamp = Utc::now();
        let temporal_triple = TemporalTriple {
            triple: triple.clone(),
            timestamp,
            version: self.get_next_version(&triple.subject),
            operation,
            previous_value,
        };
        
        // Update entity timeline
        {
            let mut timelines = self.entity_timelines.write().unwrap();
            timelines.entry(triple.subject.clone())
                .or_insert_with(BTreeMap::new)
                .entry(timestamp)
                .or_insert_with(Vec::new)
                .push(temporal_triple.clone());
            
            // Also track object if it's an entity
            if triple.object.chars().next().map_or(false, |c| c.is_uppercase()) {
                timelines.entry(triple.object.clone())
                    .or_insert_with(BTreeMap::new)
                    .entry(timestamp)
                    .or_insert_with(Vec::new)
                    .push(temporal_triple.clone());
            }
        }
        
        // Update global timeline
        {
            let mut global = self.global_timeline.write().unwrap();
            global.entry(timestamp)
                .or_insert_with(Vec::new)
                .push(temporal_triple);
        }
    }
    
    fn get_next_version(&self, entity: &str) -> u32 {
        let timelines = self.entity_timelines.read().unwrap();
        timelines.get(entity)
            .map(|timeline| timeline.len() as u32 + 1)
            .unwrap_or(1)
    }
}

/// Query result for temporal operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQueryResult {
    pub query_type: String,
    pub results: Vec<TemporalResultItem>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub total_changes: usize,
    pub insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalResultItem {
    pub timestamp: DateTime<Utc>,
    pub entity: String,
    pub changes: Vec<TemporalChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalChange {
    pub triple: Triple,
    pub operation: TemporalOperation,
    pub version: u32,
    pub previous_value: Option<String>,
}

/// Execute point-in-time query
pub async fn query_point_in_time(
    temporal_index: &TemporalIndex,
    entity: &str,
    timestamp: DateTime<Utc>,
) -> TemporalQueryResult {
    let timelines = temporal_index.entity_timelines.read().unwrap();
    
    let mut results = Vec::new();
    let mut total_changes = 0;
    
    if let Some(timeline) = timelines.get(entity) {
        // Get all changes up to the specified timestamp
        let mut current_state = HashMap::new();
        
        for (_ts, changes) in timeline.range(..=timestamp) {
            for temporal_triple in changes {
                match temporal_triple.operation {
                    TemporalOperation::Create | TemporalOperation::Update => {
                        let key = format!("{}-{}", temporal_triple.triple.predicate, temporal_triple.triple.object);
                        current_state.insert(key, temporal_triple.clone());
                    }
                    TemporalOperation::Delete => {
                        let key = format!("{}-{}", temporal_triple.triple.predicate, temporal_triple.triple.object);
                        current_state.remove(&key);
                    }
                }
                total_changes += 1;
            }
        }
        
        // Convert current state to results
        let changes: Vec<TemporalChange> = current_state.into_iter()
            .map(|(_, t)| TemporalChange {
                triple: t.triple.clone(),
                operation: t.operation.clone(),
                version: t.version,
                previous_value: t.previous_value.clone(),
            })
            .collect();
        
        if !changes.is_empty() {
            results.push(TemporalResultItem {
                timestamp,
                entity: entity.to_string(),
                changes,
            });
        }
    }
    
    let insights = vec![
        format!("{} had {} active facts at {}", entity, results.len(), timestamp.format("%Y-%m-%d %H:%M:%S")),
        format!("Total historical changes: {}", total_changes),
    ];
    
    TemporalQueryResult {
        query_type: "point_in_time".to_string(),
        results,
        time_range: Some((timestamp, timestamp)),
        total_changes,
        insights,
    }
}

/// Track evolution of an entity over time
pub async fn track_entity_evolution(
    temporal_index: &TemporalIndex,
    entity: &str,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
) -> TemporalQueryResult {
    let timelines = temporal_index.entity_timelines.read().unwrap();
    let mut results = Vec::new();
    let mut total_changes = 0;
    
    if let Some(timeline) = timelines.get(entity) {
        let start = start_time.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
        let end = end_time.unwrap_or_else(Utc::now);
        
        for (timestamp, changes) in timeline.range(start..=end) {
            let temporal_changes: Vec<TemporalChange> = changes.iter()
                .map(|t| TemporalChange {
                    triple: t.triple.clone(),
                    operation: t.operation.clone(),
                    version: t.version,
                    previous_value: t.previous_value.clone(),
                })
                .collect();
            
            total_changes += temporal_changes.len();
            
            results.push(TemporalResultItem {
                timestamp: *timestamp,
                entity: entity.to_string(),
                changes: temporal_changes,
            });
        }
    }
    
    // Generate insights about evolution
    let insights = generate_evolution_insights(&results, entity);
    
    TemporalQueryResult {
        query_type: "evolution_tracking".to_string(),
        results,
        time_range: Some((
            start_time.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            end_time.unwrap_or_else(Utc::now)
        )),
        total_changes,
        insights,
    }
}

/// Detect changes in a time period
pub async fn detect_changes(
    temporal_index: &TemporalIndex,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    entity_filter: Option<&str>,
) -> TemporalQueryResult {
    let global_timeline = temporal_index.global_timeline.read().unwrap();
    let mut results = Vec::new();
    let mut total_changes = 0;
    let mut entities_affected = std::collections::HashSet::new();
    
    for (timestamp, changes) in global_timeline.range(start_time..=end_time) {
        let filtered_changes: Vec<TemporalChange> = changes.iter()
            .filter(|t| entity_filter.map_or(true, |e| t.triple.subject == e || t.triple.object == e))
            .map(|t| {
                entities_affected.insert(t.triple.subject.clone());
                entities_affected.insert(t.triple.object.clone());
                
                TemporalChange {
                    triple: t.triple.clone(),
                    operation: t.operation.clone(),
                    version: t.version,
                    previous_value: t.previous_value.clone(),
                }
            })
            .collect();
        
        if !filtered_changes.is_empty() {
            total_changes += filtered_changes.len();
            
            // Group by entity
            let mut entity_changes: HashMap<String, Vec<TemporalChange>> = HashMap::new();
            for change in filtered_changes {
                entity_changes.entry(change.triple.subject.clone())
                    .or_insert_with(Vec::new)
                    .push(change);
            }
            
            for (entity, changes) in entity_changes {
                results.push(TemporalResultItem {
                    timestamp: *timestamp,
                    entity,
                    changes,
                });
            }
        }
    }
    
    let insights = vec![
        format!("Detected {} changes in time period", total_changes),
        format!("{} entities were affected", entities_affected.len()),
        format!("Most active period: {}", find_most_active_period(&results)),
    ];
    
    TemporalQueryResult {
        query_type: "change_detection".to_string(),
        results,
        time_range: Some((start_time, end_time)),
        total_changes,
        insights,
    }
}

/// Generate insights about entity evolution
fn generate_evolution_insights(results: &[TemporalResultItem], entity: &str) -> Vec<String> {
    let mut insights = Vec::new();
    
    if results.is_empty() {
        insights.push(format!("No historical data found for {}", entity));
        return insights;
    }
    
    // Count operations
    let mut creates = 0;
    let mut updates = 0;
    let mut deletes = 0;
    
    for item in results {
        for change in &item.changes {
            match change.operation {
                TemporalOperation::Create => creates += 1,
                TemporalOperation::Update => updates += 1,
                TemporalOperation::Delete => deletes += 1,
            }
        }
    }
    
    insights.push(format!("{} evolved through {} creates, {} updates, {} deletes", 
        entity, creates, updates, deletes));
    
    // Find most changed predicate
    let mut predicate_counts: HashMap<String, usize> = HashMap::new();
    for item in results {
        for change in &item.changes {
            *predicate_counts.entry(change.triple.predicate.clone()).or_insert(0) += 1;
        }
    }
    
    if let Some((pred, count)) = predicate_counts.iter().max_by_key(|(_, c)| *c) {
        insights.push(format!("Most frequently changed attribute: '{}' ({} times)", pred, count));
    }
    
    // Time span
    if let (Some(first), Some(last)) = (results.first(), results.last()) {
        let duration = last.timestamp - first.timestamp;
        insights.push(format!("Evolution span: {} days", duration.num_days()));
    }
    
    insights
}

/// Find the most active time period
fn find_most_active_period(results: &[TemporalResultItem]) -> String {
    if results.is_empty() {
        return "No activity".to_string();
    }
    
    // Group by hour
    let mut hourly_counts: HashMap<String, usize> = HashMap::new();
    
    for item in results {
        let hour_key = item.timestamp.format("%Y-%m-%d %H:00").to_string();
        *hourly_counts.entry(hour_key).or_insert(0) += item.changes.len();
    }
    
    if let Some((period, _)) = hourly_counts.iter().max_by_key(|(_, c)| *c) {
        period.clone()
    } else {
        "Unknown".to_string()
    }
}

lazy_static::lazy_static! {
    pub static ref TEMPORAL_INDEX: TemporalIndex = TemporalIndex::new();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_recording() {
        let index = TemporalIndex::new();
        
        let triple = Triple {
            subject: "Einstein".to_string(),
            predicate: "born_in".to_string(),
            object: "1879".to_string(),
            confidence: 1.0,
            source: None,
        };
        
        index.record_operation(triple.clone(), TemporalOperation::Create, None);
        
        let timelines = index.entity_timelines.read().unwrap();
        assert!(timelines.contains_key("Einstein"));
        assert_eq!(timelines["Einstein"].len(), 1);
    }
    
    #[tokio::test]
    async fn test_point_in_time_query() {
        let index = TemporalIndex::new();
        
        // Record some operations
        let triple1 = Triple {
            subject: "Einstein".to_string(),
            predicate: "occupation".to_string(),
            object: "physicist".to_string(),
            confidence: 1.0,
            source: None,
        };
        
        index.record_operation(triple1, TemporalOperation::Create, None);
        
        // Query current state
        let result = query_point_in_time(&index, "Einstein", Utc::now()).await;
        
        assert_eq!(result.query_type, "point_in_time");
        assert!(!result.results.is_empty());
        assert_eq!(result.total_changes, 1);
    }
}