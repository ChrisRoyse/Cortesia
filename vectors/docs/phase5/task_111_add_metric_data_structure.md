# Task 117: Add MetricData Structure

## Prerequisites Check
- [ ] Task 116 completed: MetricType enum implemented
- [ ] Monitoring module created with imports
- [ ] Run: `cargo check` (should pass)

## Context
Add core MetricData structure for individual performance data points.

## Task Objective
Implement MetricData struct with timestamps, labels, and utility methods.

## Steps
1. Add metric data structure:
   ```rust
   /// Performance metric data point
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct MetricData {
       /// Metric type
       pub metric_type: MetricType,
       /// Metric value
       pub value: f64,
       /// Timestamp
       pub timestamp: Instant,
       /// Component that generated the metric
       pub component: String,
       /// Additional labels/tags
       pub labels: HashMap<String, String>,
       /// Metric ID for tracking
       pub id: String,
   }
   
   impl MetricData {
       /// Create new metric data point
       pub fn new(
           metric_type: MetricType,
           value: f64,
           component: String,
       ) -> Self {
           Self {
               metric_type,
               value,
               timestamp: Instant::now(),
               component,
               labels: HashMap::new(),
               id: Uuid::new_v4().to_string(),
           }
       }
       
       /// Add label to metric
       pub fn with_label(mut self, key: String, value: String) -> Self {
           self.labels.insert(key, value);
           self
       }
   }
   ```
2. Verify compilation

## Success Criteria
- [ ] MetricData structure with all required fields
- [ ] Constructor method with automatic timestamp and ID generation
- [ ] Label support for metric categorization
- [ ] Unique metric ID generation using UUID
- [ ] Compiles without errors

## Time: 2 minutes

## Next Task
Task 118 will add MetricStats aggregation.

## Notes
MetricData provides complete data point structure with automatic timestamping and unique identification.