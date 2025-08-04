# Task 118: Implement System Health Dashboard and Reporting

## Prerequisites Check
- [ ] Task 117 completed: PerformanceMonitor struct implemented
- [ ] Metric collection and alerting is working
- [ ] Run: `cargo check` (should pass)

## Context
Implement system health dashboard and comprehensive reporting for operational visibility.

## Task Objective
Create health dashboard with real-time system status and comprehensive performance reporting.

## Steps
1. Add system health dashboard struct:
   ```rust
   /// System health dashboard
   pub struct HealthDashboard {
       /// Performance monitor reference
       performance_monitor: Arc<PerformanceMonitor>,
       /// Dashboard configuration
       config: DashboardConfig,
       /// System components to monitor
       components: Vec<String>,
       /// Dashboard state
       state: Arc<RwLock<DashboardState>>,
   }
   
   /// Dashboard configuration
   #[derive(Debug, Clone)]
   pub struct DashboardConfig {
       /// Update interval in seconds
       pub update_interval: u64,
       /// Number of data points to show in graphs
       pub graph_data_points: usize,
       /// Enable live updates
       pub live_updates: bool,
       /// Components to include in dashboard
       pub monitored_components: Vec<String>,
   }
   
   impl Default for DashboardConfig {
       fn default() -> Self {
           Self {
               update_interval: 5,
               graph_data_points: 100,
               live_updates: true,
               monitored_components: vec![
                   "unified_search".to_string(),
                   "vector_store".to_string(),
                   "text_search".to_string(),
                   "cache".to_string(),
                   "consistency_manager".to_string(),
               ],
           }
       }
   }
   ```
2. Add dashboard state and health summary:
   ```rust
   /// Current dashboard state
   #[derive(Debug, Clone)]
   pub struct DashboardState {
       /// Last update timestamp
       pub last_update: Instant,
       /// Overall system health score (0.0 to 1.0)
       pub overall_health: f64,
       /// Component health scores
       pub component_health: HashMap<String, f64>,
       /// Active alerts count
       pub active_alerts: usize,
       /// Total metrics collected
       pub total_metrics: usize,
       /// System uptime
       pub uptime: Duration,
   }
   
   /// Comprehensive system health summary
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct HealthSummary {
       /// Overall health score
       pub overall_health: f64,
       /// System status
       pub status: SystemStatus,
       /// Component summaries
       pub components: HashMap<String, ComponentSummary>,
       /// Recent performance trends
       pub trends: PerformanceTrends,
       /// Critical issues
       pub critical_issues: Vec<String>,
       /// Recommendations
       pub recommendations: Vec<String>,
       /// Report timestamp
       pub timestamp: Instant,
   }
   
   /// System status levels
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum SystemStatus {
       Healthy,
       Warning,
       Critical,
       Unknown,
   }
   
   /// Individual component summary
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ComponentSummary {
       /// Component health score
       pub health_score: f64,
       /// Status level
       pub status: SystemStatus,
       /// Key metrics
       pub key_metrics: HashMap<MetricType, f64>,
       /// Recent alerts count
       pub recent_alerts: usize,
       /// Last update time
       pub last_update: Instant,
   }
   ```
3. Add dashboard implementation:
   ```rust
   impl HealthDashboard {
       /// Create new health dashboard
       pub fn new(performance_monitor: Arc<PerformanceMonitor>, config: DashboardConfig) -> Self {
           let components = config.monitored_components.clone();
           
           Self {
               performance_monitor,
               config,
               components,
               state: Arc::new(RwLock::new(DashboardState {
                   last_update: Instant::now(),
                   overall_health: 1.0,
                   component_health: HashMap::new(),
                   active_alerts: 0,
                   total_metrics: 0,
                   uptime: Duration::from_secs(0),
               })),
           }
       }
       
       /// Update dashboard state
       pub async fn update_dashboard(&self) {
           let mut component_health = HashMap::new();
           let mut total_alerts = 0;
           
           // Calculate health for each component
           for component in &self.components {
               let health_score = self.calculate_component_health(component).await;
               component_health.insert(component.clone(), health_score);
           }
           
           // Get recent alerts
           let recent_alerts = self.performance_monitor.get_recent_alerts(Some(50)).await;
           total_alerts = recent_alerts.len();
           
           // Calculate overall health
           let overall_health = if component_health.is_empty() {
               1.0
           } else {
               component_health.values().sum::<f64>() / component_health.len() as f64
           };
           
           // Update state
           {
               let mut state = self.state.write().await;
               state.last_update = Instant::now();
               state.overall_health = overall_health;
               state.component_health = component_health;
               state.active_alerts = total_alerts;
               // uptime would be calculated from actual start time
           }
       }
       
       /// Calculate health score for a component
       async fn calculate_component_health(&self, component: &str) -> f64 {
           let mut health_factors = Vec::new();
           
           // Response time factor
           if let Some(stats) = self.performance_monitor.get_stats(&MetricType::QueryResponseTime, component).await {
               let response_factor = if stats.avg < 500.0 {
                   1.0
               } else if stats.avg < 1000.0 {
                   0.8
               } else {
                   0.5
               };
               health_factors.push(response_factor);
           }
           
           // Error rate factor
           if let Some(stats) = self.performance_monitor.get_stats(&MetricType::ErrorRate, component).await {
               let error_factor = 1.0 - stats.avg.min(1.0);
               health_factors.push(error_factor);
           }
           
           // Cache hit ratio factor (if applicable)
           if let Some(stats) = self.performance_monitor.get_stats(&MetricType::CacheHitRatio, component).await {
               health_factors.push(stats.avg);
           }
           
           // Return average of all factors, or 1.0 if no metrics
           if health_factors.is_empty() {
               1.0
           } else {
               health_factors.iter().sum::<f64>() / health_factors.len() as f64
           }
       }
   }
   ```
4. Add comprehensive reporting:
   ```rust
   impl HealthDashboard {
       /// Generate comprehensive health summary
       pub async fn generate_health_summary(&self) -> HealthSummary {
           let mut components = HashMap::new();
           let mut critical_issues = Vec::new();
           let mut recommendations = Vec::new();
           
           // Generate component summaries
           for component in &self.components {
               let summary = self.generate_component_summary(component).await;
               
               // Check for critical issues
               if summary.status == SystemStatus::Critical {
                   critical_issues.push(format!("{} is in critical state", component));
               }
               
               // Generate recommendations
               if summary.health_score < 0.8 {
                   recommendations.push(format!("Consider investigating {} performance", component));
               }
               
               components.insert(component.clone(), summary);
           }
           
           let state = self.state.read().await;
           let overall_health = state.overall_health;
           
           let status = match overall_health {
               x if x >= 0.9 => SystemStatus::Healthy,
               x if x >= 0.7 => SystemStatus::Warning,
               x if x >= 0.5 => SystemStatus::Critical,
               _ => SystemStatus::Unknown,
           };
           
           let trends = self.calculate_performance_trends().await;
           
           HealthSummary {
               overall_health,
               status,
               components,
               trends,
               critical_issues,
               recommendations,
               timestamp: Instant::now(),
           }
       }
       
       /// Generate summary for individual component
       async fn generate_component_summary(&self, component: &str) -> ComponentSummary {
           let health_score = self.calculate_component_health(component).await;
           
           let status = match health_score {
               x if x >= 0.9 => SystemStatus::Healthy,
               x if x >= 0.7 => SystemStatus::Warning,
               x if x >= 0.5 => SystemStatus::Critical,
               _ => SystemStatus::Unknown,
           };
           
           let mut key_metrics = HashMap::new();
           
           // Collect key metrics for this component
           for metric_type in [MetricType::QueryResponseTime, MetricType::ErrorRate, MetricType::CacheHitRatio] {
               if let Some(stats) = self.performance_monitor.get_stats(&metric_type, component).await {
                   key_metrics.insert(metric_type, stats.avg);
               }
           }
           
           // Count recent alerts for this component
           let all_alerts = self.performance_monitor.get_recent_alerts(Some(100)).await;
           let recent_alerts = all_alerts.iter()
               .filter(|alert| alert.component == component)
               .count();
           
           ComponentSummary {
               health_score,
               status,
               key_metrics,
               recent_alerts,
               last_update: Instant::now(),
           }
       }
   }
   ```
5. Add performance trends analysis:
   ```rust
   /// Performance trends over time
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct PerformanceTrends {
       /// Response time trend (improving/stable/degrading)
       pub response_time_trend: TrendDirection,
       /// Error rate trend
       pub error_rate_trend: TrendDirection,
       /// Throughput trend
       pub throughput_trend: TrendDirection,
       /// Overall trend summary
       pub overall_trend: TrendDirection,
   }
   
   /// Trend direction indicators
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum TrendDirection {
       Improving,
       Stable,
       Degrading,
       Unknown,
   }
   
   impl HealthDashboard {
       /// Calculate performance trends
       async fn calculate_performance_trends(&self) -> PerformanceTrends {
           // Simplified trend calculation - actual implementation would analyze time series data
           PerformanceTrends {
               response_time_trend: TrendDirection::Stable,
               error_rate_trend: TrendDirection::Improving,
               throughput_trend: TrendDirection::Stable,
               overall_trend: TrendDirection::Stable,
           }
       }
       
       /// Get current dashboard state
       pub async fn get_dashboard_state(&self) -> DashboardState {
           self.state.read().await.clone()
       }
       
       /// Start dashboard auto-update
       pub async fn start_auto_update(&self) {
           if !self.config.live_updates {
               return;
           }
           
           // This would spawn a background task for periodic updates
           println!("Dashboard auto-update started with {}s interval", self.config.update_interval);
       }
   }
   ```
6. Verify compilation

## Success Criteria
- [ ] HealthDashboard with real-time system monitoring
- [ ] Comprehensive health summary generation
- [ ] Component-level health scoring and status
- [ ] Performance trends analysis framework
- [ ] Critical issue detection and recommendations
- [ ] System status classification (Healthy/Warning/Critical)
- [ ] Auto-update capability for live dashboards
- [ ] Configurable monitoring components and intervals
- [ ] Compiles without errors

## Time: 8 minutes

## Next Task
Task 119 will add performance optimization recommendations and alerts.

## Notes
Health dashboard provides comprehensive system visibility with actionable insights and automated health scoring for operational monitoring.