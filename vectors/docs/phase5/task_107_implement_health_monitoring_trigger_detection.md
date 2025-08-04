# Task 114: Implement Health Monitoring and Automatic Trigger Detection

## Prerequisites Check
- [ ] Task 113 completed: repair execution engine implemented
- [ ] Repair operations are working with timeout handling
- [ ] Run: `cargo check` (should pass)

## Context
Implement health monitoring that automatically detects when repairs are needed and triggers them.

## Task Objective
Create health monitoring system that detects consistency issues and automatically schedules repairs.

## Steps
1. Add health monitoring struct:
   ```rust
   /// Health monitoring and trigger detection system
   pub struct HealthMonitor {
       /// Configuration
       config: AutoRepairConfig,
       /// Health check history
       health_history: Arc<RwLock<Vec<HealthCheckResult>>>,
       /// Last trigger detection timestamp
       last_trigger_check: Arc<RwLock<Instant>>,
       /// Detected triggers queue
       detected_triggers: Arc<RwLock<Vec<DetectedTrigger>>>,
   }
   
   /// Result of health check operation
   #[derive(Debug, Clone)]
   pub struct HealthCheckResult {
       /// Check timestamp
       pub timestamp: Instant,
       /// Overall system health score (0.0 to 1.0)
       pub health_score: f64,
       /// Consistency ratio
       pub consistency_ratio: f64,
       /// Number of inconsistent documents
       pub inconsistent_count: usize,
       /// System response times in milliseconds
       pub response_times: HashMap<String, u64>,
       /// Critical issues detected
       pub critical_issues: Vec<String>,
   }
   
   /// Detected trigger for automatic repair
   #[derive(Debug, Clone)]
   pub struct DetectedTrigger {
       /// Document ID requiring repair
       pub doc_id: String,
       /// Trigger type
       pub trigger: RepairTrigger,
       /// Priority level
       pub priority: RepairPriority,
       /// Suggested strategy
       pub strategy: SyncStrategy,
       /// Detection timestamp
       pub detected_at: Instant,
       /// Additional context
       pub context: String,
   }
   ```
2. Add constructor and basic methods:
   ```rust
   impl HealthMonitor {
       /// Create new health monitor
       pub fn new(config: AutoRepairConfig) -> Self {
           Self {
               config,
               health_history: Arc::new(RwLock::new(Vec::new())),
               last_trigger_check: Arc::new(RwLock::new(Instant::now())),
               detected_triggers: Arc::new(RwLock::new(Vec::new())),
           }
       }
       
       /// Get detected triggers
       pub async fn get_detected_triggers(&self) -> Vec<DetectedTrigger> {
           let mut triggers = self.detected_triggers.write().await;
           triggers.drain(..).collect()
       }
       
       /// Get latest health check result
       pub async fn get_latest_health(&self) -> Option<HealthCheckResult> {
           let history = self.health_history.read().await;
           history.last().cloned()
       }
   }
   ```
3. Add health checking methods:
   ```rust
   impl HealthMonitor {
       /// Perform comprehensive health check
       pub async fn perform_health_check(&self) -> HealthCheckResult {
           let start_time = Instant::now();
           
           // Simulate health check operations
           let consistency_ratio = self.check_consistency_ratio().await;
           let inconsistent_count = self.count_inconsistent_documents().await;
           let response_times = self.measure_response_times().await;
           let critical_issues = self.detect_critical_issues(&response_times, consistency_ratio).await;
           
           // Calculate overall health score
           let health_score = self.calculate_health_score(consistency_ratio, &response_times, &critical_issues);
           
           let result = HealthCheckResult {
               timestamp: start_time,
               health_score,
               consistency_ratio,
               inconsistent_count,
               response_times,
               critical_issues,
           };
           
           // Store in history
           {
               let mut history = self.health_history.write().await;
               history.push(result.clone());
               
               // Keep only recent history (last 100 checks)
               if history.len() > 100 {
                   history.remove(0);
               }
           }
           
           result
       }
       
       /// Check consistency ratio across systems
       async fn check_consistency_ratio(&self) -> f64 {
           // Simulate consistency check
           tokio::time::sleep(Duration::from_millis(50)).await;
           0.85 // Simulated consistency ratio
       }
       
       /// Count inconsistent documents
       async fn count_inconsistent_documents(&self) -> usize {
           tokio::time::sleep(Duration::from_millis(30)).await;
           5 // Simulated inconsistent count
       }
       
       /// Measure system response times
       async fn measure_response_times(&self) -> HashMap<String, u64> {
           let mut response_times = HashMap::new();
           
           // Simulate measuring each system
           response_times.insert("text_search".to_string(), 120);
           response_times.insert("vector_store".to_string(), 85);
           response_times.insert("cache".to_string(), 15);
           
           response_times
       }
       
       /// Detect critical issues
       async fn detect_critical_issues(
           &self,
           response_times: &HashMap<String, u64>,
           consistency_ratio: f64,
       ) -> Vec<String> {
           let mut issues = Vec::new();
           
           // Check consistency threshold
           if consistency_ratio < 0.8 {
               issues.push("Low consistency ratio detected".to_string());
           }
           
           // Check response time thresholds
           for (system, time) in response_times {
               if *time > 1000 {
                   issues.push(format!("{} system slow response: {}ms", system, time));
               }
           }
           
           issues
       }
       
       /// Calculate overall health score
       fn calculate_health_score(
           &self,
           consistency_ratio: f64,
           response_times: &HashMap<String, u64>,
           critical_issues: &[String],
       ) -> f64 {
           let consistency_weight = 0.5;
           let performance_weight = 0.3;
           let issue_weight = 0.2;
           
           // Performance score based on response times
           let avg_response_time = response_times.values().sum::<u64>() as f64 / response_times.len() as f64;
           let performance_score = if avg_response_time < 100.0 {
               1.0
           } else if avg_response_time < 500.0 {
               1.0 - (avg_response_time - 100.0) / 400.0 * 0.5
           } else {
               0.5
           };
           
           // Issue score (fewer issues = better score)
           let issue_score = if critical_issues.is_empty() {
               1.0
           } else {
               (5.0 - critical_issues.len().min(5) as f64) / 5.0
           };
           
           consistency_ratio * consistency_weight + 
           performance_score * performance_weight + 
           issue_score * issue_weight
       }
   }
   ```
4. Add automatic trigger detection:
   ```rust
   impl HealthMonitor {
       /// Detect and queue repair triggers
       pub async fn detect_repair_triggers(&self) {
           let health_result = self.perform_health_check().await;
           let mut triggers = self.detected_triggers.write().await;
           
           // Health-based triggers
           if health_result.health_score < 0.5 {
               for _ in 0..health_result.inconsistent_count.min(10) {
                   triggers.push(DetectedTrigger {
                       doc_id: format!("doc_{}", Uuid::new_v4()),
                       trigger: RepairTrigger::HealthCheck,
                       priority: RepairPriority::High,
                       strategy: SyncStrategy::MostRecent,
                       detected_at: Instant::now(),
                       context: format!("Health score: {:.2}", health_result.health_score),
                   });
               }
           }
           
           // Consistency-based triggers
           if health_result.consistency_ratio < 0.8 {
               for _ in 0..5 {
                   triggers.push(DetectedTrigger {
                       doc_id: format!("doc_{}", Uuid::new_v4()),
                       trigger: RepairTrigger::ConsistencyCheck,
                       priority: RepairPriority::Medium,
                       strategy: SyncStrategy::TextSourceOfTruth,
                       detected_at: Instant::now(),
                       context: format!("Consistency ratio: {:.2}", health_result.consistency_ratio),
                   });
               }
           }
           
           // Performance-based triggers
           for (system, response_time) in &health_result.response_times {
               if *response_time > 1000 {
                   triggers.push(DetectedTrigger {
                       doc_id: format!("perf_{}", system),
                       trigger: RepairTrigger::SearchFailure,
                       priority: RepairPriority::Low,
                       strategy: SyncStrategy::VectorSourceOfTruth,
                       detected_at: Instant::now(),
                       context: format!("{} slow: {}ms", system, response_time),
                   });
               }
           }
       }
       
       /// Check if trigger detection is needed
       pub async fn needs_trigger_detection(&self) -> bool {
           let last_check = self.last_trigger_check.read().await;
           last_check.elapsed().as_secs() >= self.config.health_check_interval
       }
       
       /// Update last trigger check timestamp
       pub async fn update_trigger_check_timestamp(&self) {
           let mut last_check = self.last_trigger_check.write().await;
           *last_check = Instant::now();
       }
   }
   ```
5. Verify compilation

## Success Criteria
- [ ] HealthMonitor with comprehensive health checking
- [ ] Health score calculation with multiple factors
- [ ] Automatic trigger detection based on health metrics
- [ ] Response time monitoring for all systems
- [ ] Critical issue detection and reporting
- [ ] Health history tracking with size limits
- [ ] Performance-based and consistency-based triggers
- [ ] Configurable health check intervals
- [ ] Compiles without errors

## Time: 8 minutes

## Next Task
Task 115 will integrate all repair components into a unified system.

## Notes
Health monitoring provides proactive detection of system issues and automatically schedules appropriate repairs based on observed conditions.