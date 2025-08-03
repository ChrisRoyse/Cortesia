# Task 19f: Integrate Monitoring with Core

**Estimated Time**: 5 minutes  
**Dependencies**: 19e  
**Stage**: Performance Monitoring  

## Objective
Integrate monitoring system with the core brain enhanced graph.

## Implementation Steps

1. Update `src/core/brain_enhanced_graph/brain_graph_core.rs`:
```rust
// Add to existing imports
use crate::monitoring::{
    PerformanceMonitor, BasicMetrics, SystemMonitor, AlertManager, HealthChecker
};
use std::time::Instant;

// Add to BrainEnhancedGraphCore struct
pub struct BrainEnhancedGraphCore {
    // ... existing fields ...
    performance_monitor: Option<Arc<PerformanceMonitor>>,
    basic_metrics: Option<Arc<BasicMetrics>>,
    system_monitor: Option<Arc<SystemMonitor>>,
    alert_manager: Option<Arc<RwLock<AlertManager>>>,
    health_checker: Option<Arc<HealthChecker>>,
}

impl BrainEnhancedGraphCore {
    // Add monitoring initialization method
    pub fn enable_monitoring(&mut self, config: MonitoringConfig) -> Result<(), GraphError> {
        let performance_monitor = Arc::new(PerformanceMonitor::new(config.clone()));
        let registry = performance_monitor.get_registry();
        
        let basic_metrics = Arc::new(
            BasicMetrics::new(&registry)
                .map_err(|e| GraphError::MonitoringError(format!("Failed to create metrics: {}", e)))?
        );
        
        let system_monitor = Arc::new(SystemMonitor::new(config.collection_interval_ms));
        
        let (alert_manager, alert_receiver) = AlertManager::new();
        let alert_manager = Arc::new(RwLock::new(alert_manager));
        
        let health_checker = Arc::new(HealthChecker::new());
        
        // Start background monitoring tasks
        tokio::spawn(crate::monitoring::alerts::handle_alerts(alert_receiver));
        
        let system_monitor_clone = system_monitor.clone();
        tokio::spawn(async move {
            system_monitor_clone.start_monitoring().await;
        });
        
        self.performance_monitor = Some(performance_monitor);
        self.basic_metrics = Some(basic_metrics);
        self.system_monitor = Some(system_monitor);
        self.alert_manager = Some(alert_manager);
        self.health_checker = Some(health_checker);
        
        Ok(())
    }
    
    // Update allocate_memory_with_cortical_coordination to include monitoring
    pub async fn allocate_memory_with_cortical_coordination(
        &self,
        request: MemoryAllocationRequest,
    ) -> Result<MemoryAllocationResult, AllocationError> {
        let start_time = Instant::now();
        
        // Record the allocation attempt
        if let Some(metrics) = &self.basic_metrics {
            // Will record completion time later
        }
        
        // Perform the actual allocation
        let result = self.allocate_memory_internal(request).await;
        
        // Record metrics and check alerts
        let duration_seconds = start_time.elapsed().as_secs_f64();
        
        if let Some(metrics) = &self.basic_metrics {
            metrics.record_memory_allocation(duration_seconds);
        }
        
        if let Some(alert_manager) = &self.alert_manager {
            let mut alerts = alert_manager.write().await;
            
            // Check for slow allocation
            if duration_seconds > 1.0 {
                alerts.check_metric(
                    crate::monitoring::alerts::AlertType::SlowResponseTime,
                    duration_seconds * 1000.0, // Convert to ms
                    "memory_allocation"
                );
            }
        }
        
        result
    }
    
    // Update search methods to include monitoring
    pub async fn search_memory_with_semantic_similarity(
        &self,
        request: SearchRequest,
    ) -> Result<SearchResult, SearchError> {
        let start_time = Instant::now();
        
        let result = self.search_memory_internal(request).await;
        
        let duration_seconds = start_time.elapsed().as_secs_f64();
        
        if let Some(metrics) = &self.basic_metrics {
            let result_count = result.as_ref()
                .map(|r| r.concepts.len())
                .unwrap_or(0);
            metrics.record_search_operation(duration_seconds, result_count);
        }
        
        if let Some(alert_manager) = &self.alert_manager {
            let mut alerts = alert_manager.write().await;
            
            // Check for slow search
            if duration_seconds > 2.0 {
                alerts.check_metric(
                    crate::monitoring::alerts::AlertType::SlowResponseTime,
                    duration_seconds * 1000.0,
                    "semantic_search"
                );
            }
        }
        
        result
    }
    
    // Add health status method
    pub async fn get_health_status(&self) -> Result<crate::monitoring::health::HealthStatus, GraphError> {
        if let Some(health_checker) = &self.health_checker {
            Ok(health_checker.check_overall_health().await)
        } else {
            Err(GraphError::MonitoringError("Monitoring not enabled".to_string()))
        }
    }
    
    // Add metrics collection method
    pub async fn get_comprehensive_metrics(&self) -> Result<ComprehensiveMetrics, GraphError> {
        let mut metrics = ComprehensiveMetrics::default();
        
        if let Some(system_monitor) = &self.system_monitor {
            let system_metrics = system_monitor.get_current_metrics().await;
            metrics.system_metrics = Some(system_metrics);
        }
        
        if let Some(basic_metrics) = &self.basic_metrics {
            // Get Prometheus metrics values
            metrics.allocation_count = basic_metrics.memory_allocations_total.get() as u64;
            metrics.search_count = basic_metrics.search_operations_total.get() as u64;
            metrics.cache_hit_rate = basic_metrics.cache_hit_rate.get();
        }
        
        if let Some(alert_manager) = &self.alert_manager {
            let alerts = alert_manager.read().await;
            metrics.active_alerts = alerts.get_active_alerts().len() as u32;
        }
        
        Ok(metrics)
    }
    
    // Add periodic monitoring task
    pub async fn start_periodic_monitoring(&self) {
        if let (Some(system_monitor), Some(alert_manager), Some(basic_metrics)) = 
            (&self.system_monitor, &self.alert_manager, &self.basic_metrics) {
            
            let system_monitor = system_monitor.clone();
            let alert_manager = alert_manager.clone();
            let basic_metrics = basic_metrics.clone();
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                
                loop {
                    interval.tick().await;
                    
                    let system_metrics = system_monitor.get_current_metrics().await;
                    let mut alerts = alert_manager.write().await;
                    
                    // Update basic metrics with system info
                    basic_metrics.update_system_metrics(
                        system_metrics.cpu_usage_percent as f64,
                        system_metrics.memory_used_mb,
                        10 // Placeholder for connections
                    );
                    
                    // Check alert thresholds
                    alerts.check_metric(
                        crate::monitoring::alerts::AlertType::HighMemoryUsage,
                        system_metrics.memory_usage_percent,
                        "system"
                    );
                    
                    alerts.check_metric(
                        crate::monitoring::alerts::AlertType::HighCpuUsage,
                        system_metrics.cpu_usage_percent as f64,
                        "system"
                    );
                }
            });
        }
    }
}

#[derive(Debug, Default)]
pub struct ComprehensiveMetrics {
    pub system_metrics: Option<crate::monitoring::system::SystemMetrics>,
    pub allocation_count: u64,
    pub search_count: u64,
    pub cache_hit_rate: f64,
    pub active_alerts: u32,
}
```

## Acceptance Criteria
- [ ] Monitoring integrated with core operations
- [ ] Metrics recorded for allocations and searches
- [ ] Alert checking integrated
- [ ] Health status available

## Success Metrics
- All core operations include monitoring
- Performance impact < 5%
- Alerts trigger appropriately

## Next Task
19g_create_monitoring_api_endpoints.md