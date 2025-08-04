# Task 117: Add Performance Optimization Recommendations and Alerts

## Prerequisites Check
- [ ] Task 118 completed: system health dashboard and reporting implemented
- [ ] HealthDashboard is functional with health scoring
- [ ] Run: `cargo check` (should pass)

## Context
Add intelligent performance optimization recommendations and proactive alerting system.

## Task Objective
Implement optimization recommendation engine and advanced alerting for proactive system management.

## Steps
1. Add optimization recommendation structures:
   ```rust
   /// Performance optimization recommendation
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct OptimizationRecommendation {
       /// Recommendation ID
       pub id: String,
       /// Component this applies to
       pub component: String,
       /// Recommendation type
       pub recommendation_type: RecommendationType,
       /// Priority level
       pub priority: RecommendationPriority,
       /// Description of the issue
       pub issue_description: String,
       /// Recommended action
       pub recommended_action: String,
       /// Expected impact
       pub expected_impact: String,
       /// Supporting metrics
       pub supporting_metrics: HashMap<MetricType, f64>,
       /// Creation timestamp
       pub created_at: Instant,
       /// Estimated implementation effort
       pub effort_level: EffortLevel,
   }
   
   /// Types of optimization recommendations
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum RecommendationType {
       /// Performance tuning
       PerformanceTuning,
       /// Resource scaling
       ResourceScaling,
       /// Configuration adjustment
       ConfigurationAdjustment,
       /// Index optimization
       IndexOptimization,
       /// Cache optimization
       CacheOptimization,
       /// Query optimization
       QueryOptimization,
       /// System maintenance
       SystemMaintenance,
   }
   
   /// Recommendation priority levels
   #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
   pub enum RecommendationPriority {
       Critical = 0,
       High = 1,
       Medium = 2,
       Low = 3,
   }
   
   /// Implementation effort levels
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum EffortLevel {
       /// Quick fix (< 1 hour)
       Quick,
       /// Moderate effort (1-8 hours)
       Moderate,
       /// Significant effort (1-3 days)
       Significant,
       /// Major project (> 3 days)
       Major,
   }
   ```
2. Add recommendation engine:
   ```rust
   /// Performance optimization recommendation engine
   pub struct RecommendationEngine {
       /// Performance monitor reference
       performance_monitor: Arc<PerformanceMonitor>,
       /// Generated recommendations
       recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
       /// Recommendation rules
       rules: Vec<RecommendationRule>,
   }
   
   /// Rule for generating recommendations
   #[derive(Debug, Clone)]
   pub struct RecommendationRule {
       /// Rule name
       pub name: String,
       /// Metric type to check
       pub metric_type: MetricType,
       /// Threshold value
       pub threshold: f64,
       /// Comparison operator
       pub operator: ComparisonOperator,
       /// Recommendation to generate
       pub recommendation_type: RecommendationType,
       /// Priority to assign
       pub priority: RecommendationPriority,
       /// Rule description
       pub description: String,
   }
   
   /// Comparison operators for rules
   #[derive(Debug, Clone)]
   pub enum ComparisonOperator {
       GreaterThan,
       LessThan,
       Equals,
   }
   ```
3. Add recommendation engine implementation:
   ```rust
   impl RecommendationEngine {
       /// Create new recommendation engine
       pub fn new(performance_monitor: Arc<PerformanceMonitor>) -> Self {
           let rules = Self::create_default_rules();
           
           Self {
               performance_monitor,
               recommendations: Arc::new(RwLock::new(Vec::new())),
               rules,
           }
       }
       
       /// Create default recommendation rules
       fn create_default_rules() -> Vec<RecommendationRule> {
           vec![
               RecommendationRule {
                   name: "High Response Time".to_string(),
                   metric_type: MetricType::QueryResponseTime,
                   threshold: 1000.0,
                   operator: ComparisonOperator::GreaterThan,
                   recommendation_type: RecommendationType::PerformanceTuning,
                   priority: RecommendationPriority::High,
                   description: "Query response times are above acceptable threshold".to_string(),
               },
               RecommendationRule {
                   name: "Low Cache Hit Rate".to_string(),
                   metric_type: MetricType::CacheHitRatio,
                   threshold: 0.8,
                   operator: ComparisonOperator::LessThan,
                   recommendation_type: RecommendationType::CacheOptimization,
                   priority: RecommendationPriority::Medium,
                   description: "Cache hit ratio is below optimal level".to_string(),
               },
               RecommendationRule {
                   name: "High Error Rate".to_string(),
                   metric_type: MetricType::ErrorRate,
                   threshold: 0.05,
                   operator: ComparisonOperator::GreaterThan,
                   recommendation_type: RecommendationType::SystemMaintenance,
                   priority: RecommendationPriority::Critical,
                   description: "Error rate is above acceptable threshold".to_string(),
               },
               RecommendationRule {
                   name: "Low Search Accuracy".to_string(),
                   metric_type: MetricType::SearchAccuracy,
                   threshold: 0.9,
                   operator: ComparisonOperator::LessThan,
                   recommendation_type: RecommendationType::IndexOptimization,
                   priority: RecommendationPriority::High,
                   description: "Search accuracy is below target level".to_string(),
               },
           ]
       }
       
       /// Generate recommendations based on current metrics
       pub async fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
           let mut new_recommendations = Vec::new();
           
           // Get all monitored components
           let components = vec!["unified_search", "vector_store", "text_search", "cache"];
           
           for component in components {
               for rule in &self.rules {
                   if let Some(recommendation) = self.evaluate_rule(rule, component).await {
                       new_recommendations.push(recommendation);
                   }
               }
           }
           
           // Store new recommendations
           {
               let mut recommendations = self.recommendations.write().await;
               recommendations.extend(new_recommendations.clone());
               
               // Keep only recent recommendations (last 50)
               if recommendations.len() > 50 {
                   recommendations.sort_by_key(|r| r.created_at);
                   recommendations.truncate(50);
               }
           }
           
           new_recommendations
       }
       
       /// Evaluate a single rule against a component
       async fn evaluate_rule(&self, rule: &RecommendationRule, component: &str) -> Option<OptimizationRecommendation> {
           if let Some(stats) = self.performance_monitor.get_stats(&rule.metric_type, component).await {
               let metric_value = stats.avg;
               
               let rule_triggered = match rule.operator {
                   ComparisonOperator::GreaterThan => metric_value > rule.threshold,
                   ComparisonOperator::LessThan => metric_value < rule.threshold,
                   ComparisonOperator::Equals => (metric_value - rule.threshold).abs() < 0.001,
               };
               
               if rule_triggered {
                   return Some(self.create_recommendation(rule, component, metric_value));
               }
           }
           
           None
       }
       
       /// Create recommendation from rule
       fn create_recommendation(&self, rule: &RecommendationRule, component: &str, metric_value: f64) -> OptimizationRecommendation {
           let (recommended_action, expected_impact, effort_level) = match rule.recommendation_type {
               RecommendationType::PerformanceTuning => (
                   "Consider optimizing query execution or adding indexes".to_string(),
                   "50-70% improvement in response times".to_string(),
                   EffortLevel::Moderate,
               ),
               RecommendationType::CacheOptimization => (
                   "Increase cache size or adjust TTL settings".to_string(),
                   "20-40% improvement in response times".to_string(),
                   EffortLevel::Quick,
               ),
               RecommendationType::SystemMaintenance => (
                   "Investigate and fix underlying system issues".to_string(),
                   "Significant reduction in error rates".to_string(),
                   EffortLevel::Significant,
               ),
               RecommendationType::IndexOptimization => (
                   "Rebuild or optimize search indexes".to_string(),
                   "15-30% improvement in search accuracy".to_string(),
                   EffortLevel::Moderate,
               ),
               _ => (
                   "General optimization recommended".to_string(),
                   "Variable improvement expected".to_string(),
                   EffortLevel::Moderate,
               ),
           };
           
           let mut supporting_metrics = HashMap::new();
           supporting_metrics.insert(rule.metric_type.clone(), metric_value);
           
           OptimizationRecommendation {
               id: Uuid::new_v4().to_string(),
               component: component.to_string(),
               recommendation_type: rule.recommendation_type.clone(),
               priority: rule.priority.clone(),
               issue_description: format!("{} for {}: {:.2}", rule.description, component, metric_value),
               recommended_action,
               expected_impact,
               supporting_metrics,
               created_at: Instant::now(),
               effort_level,
           }
       }
   }
   ```
4. Add advanced alerting system:
   ```rust
   /// Advanced alerting system
   pub struct AdvancedAlertSystem {
       /// Performance monitor reference
       performance_monitor: Arc<PerformanceMonitor>,
       /// Alert configuration
       config: AlertConfig,
       /// Active alert suppressions
       suppressions: Arc<RwLock<HashMap<String, Instant>>>,
       /// Alert escalation history
       escalation_history: Arc<RwLock<Vec<AlertEscalation>>>,
   }
   
   /// Alert configuration
   #[derive(Debug, Clone)]
   pub struct AlertConfig {
       /// Enable alert suppression
       pub enable_suppression: bool,
       /// Suppression duration in seconds
       pub suppression_duration: u64,
       /// Enable alert escalation
       pub enable_escalation: bool,
       /// Escalation threshold (number of similar alerts)
       pub escalation_threshold: usize,
       /// Escalation time window in seconds
       pub escalation_window: u64,
   }
   
   /// Alert escalation record
   #[derive(Debug, Clone)]
   pub struct AlertEscalation {
       /// Original alert
       pub original_alert: PerformanceAlert,
       /// Escalation level
       pub escalation_level: EscalationLevel,
       /// Escalation timestamp
       pub escalated_at: Instant,
       /// Number of similar alerts that triggered escalation
       pub trigger_count: usize,
   }
   
   /// Escalation levels
   #[derive(Debug, Clone, PartialEq)]
   pub enum EscalationLevel {
       Low,
       Medium,
       High,
       Critical,
   }
   
   impl AdvancedAlertSystem {
       /// Create new advanced alert system
       pub fn new(performance_monitor: Arc<PerformanceMonitor>, config: AlertConfig) -> Self {
           Self {
               performance_monitor,
               config,
               suppressions: Arc::new(RwLock::new(HashMap::new())),
               escalation_history: Arc::new(RwLock::new(Vec::new())),
           }
       }
       
       /// Process alerts with suppression and escalation
       pub async fn process_alerts(&self) {
           let recent_alerts = self.performance_monitor.get_recent_alerts(Some(20)).await;
           
           for alert in recent_alerts {
               if !self.is_alert_suppressed(&alert).await {
                   self.check_escalation(&alert).await;
               }
           }
       }
       
       /// Check if alert should be suppressed
       async fn is_alert_suppressed(&self, alert: &PerformanceAlert) -> bool {
           if !self.config.enable_suppression {
               return false;
           }
           
           let suppressions = self.suppressions.read().await;
           let key = format!("{}:{:?}", alert.component, alert.metric_type);
           
           if let Some(suppressed_until) = suppressions.get(&key) {
               Instant::now() < *suppressed_until
           } else {
               false
           }
       }
       
       /// Check for alert escalation
       async fn check_escalation(&self, alert: &PerformanceAlert) {
           if !self.config.enable_escalation {
               return;
           }
           
           // Count similar recent alerts
           let all_alerts = self.performance_monitor.get_recent_alerts(Some(100)).await;
           let similar_alerts = all_alerts.iter()
               .filter(|a| {
                   a.component == alert.component &&
                   a.metric_type == alert.metric_type &&
                   a.timestamp.elapsed().as_secs() <= self.config.escalation_window
               })
               .count();
           
           if similar_alerts >= self.config.escalation_threshold {
               self.escalate_alert(alert, similar_alerts).await;
           }
       }
       
       /// Escalate alert to higher priority
       async fn escalate_alert(&self, alert: &PerformanceAlert, trigger_count: usize) {
           let escalation_level = match trigger_count {
               0..=2 => EscalationLevel::Low,
               3..=5 => EscalationLevel::Medium,
               6..=10 => EscalationLevel::High,
               _ => EscalationLevel::Critical,
           };
           
           let escalation = AlertEscalation {
               original_alert: alert.clone(),
               escalation_level,
               escalated_at: Instant::now(),
               trigger_count,
           };
           
           let mut history = self.escalation_history.write().await;
           history.push(escalation);
           
           // Keep only recent escalations
           if history.len() > 100 {
               history.remove(0);
           }
           
           println!("Alert escalated to {:?} level for {} (trigger count: {})", 
                   escalation_level, alert.component, trigger_count);
       }
   }
   ```
5. Add integration methods:
   ```rust
   impl RecommendationEngine {
       /// Get current recommendations
       pub async fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
           let recommendations = self.recommendations.read().await;
           recommendations.clone()
       }
       
       /// Get recommendations by priority
       pub async fn get_recommendations_by_priority(&self, priority: RecommendationPriority) -> Vec<OptimizationRecommendation> {
           let recommendations = self.recommendations.read().await;
           recommendations.iter()
               .filter(|r| r.priority == priority)
               .cloned()
               .collect()
       }
       
       /// Clear implemented recommendations
       pub async fn mark_recommendation_implemented(&self, recommendation_id: &str) {
           let mut recommendations = self.recommendations.write().await;
           recommendations.retain(|r| r.id != recommendation_id);
       }
   }
   
   impl AdvancedAlertSystem {
       /// Get escalation history
       pub async fn get_escalation_history(&self) -> Vec<AlertEscalation> {
           let history = self.escalation_history.read().await;
           history.clone()
       }
       
       /// Suppress alerts for component/metric combination
       pub async fn suppress_alerts(&self, component: &str, metric_type: &MetricType, duration_seconds: u64) {
           let mut suppressions = self.suppressions.write().await;
           let key = format!("{}:{:?}", component, metric_type);
           let suppress_until = Instant::now() + Duration::from_secs(duration_seconds);
           suppressions.insert(key, suppress_until);
       }
   }
   ```
6. Verify compilation

## Success Criteria
- [ ] OptimizationRecommendation with comprehensive recommendation data
- [ ] RecommendationEngine with rule-based recommendation generation
- [ ] Default rules for common performance issues
- [ ] AdvancedAlertSystem with suppression and escalation
- [ ] Alert escalation based on frequency and severity
- [ ] Recommendation priority and effort level classification
- [ ] Integration methods for managing recommendations and alerts
- [ ] Configurable thresholds and time windows
- [ ] Compiles without errors

## Time: 9 minutes

## Next Task
Task 120 will implement query optimization and performance tuning.

## Notes
Advanced alerting and recommendations provide proactive system management with intelligent suppression and escalation to prevent alert fatigue while ensuring critical issues get attention.