//! Configuration and constraints for Phase 4 Learning System

use crate::learning::adaptive_learning::ResourceRequirement;
use std::time::Duration;

/// Configuration for Phase 4 learning system
#[derive(Debug, Clone)]
pub struct Phase4Config {
    pub learning_aggressiveness: f32,
    pub integration_depth: IntegrationDepth,
    pub performance_targets: PerformanceTargets,
    pub safety_constraints: SafetyConstraints,
    pub resource_limits: ResourceLimits,
}

/// Depth of integration with existing systems
#[derive(Debug, Clone)]
pub enum IntegrationDepth {
    Minimal,     // Basic integration
    Standard,    // Full integration
    Deep,        // Tight coupling with Phase 3
    Experimental, // Advanced experimental features
}

/// Performance targets for the learning system
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub learning_efficiency_target: f32,
    pub adaptation_speed_target: Duration,
    pub memory_overhead_limit: f32,
    pub performance_degradation_limit: f32,
    pub user_satisfaction_target: f32,
}

/// Safety constraints for learning operations
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    pub max_concurrent_learning_sessions: usize,
    pub rollback_capability_required: bool,
    pub performance_monitoring_required: bool,
    pub emergency_protocols_enabled: bool,
    pub user_intervention_threshold: f32,
    pub max_learning_impact_per_session: f32,
}

/// Resource limits for learning operations
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory_usage_mb: f32,
    pub max_cpu_usage_percentage: f32,
    pub max_storage_usage_mb: f32,
    pub max_network_bandwidth_mbps: f32,
    pub max_session_duration: Duration,
    pub max_daily_learning_time: Duration,
}

impl Phase4Config {
    /// Create conservative configuration for production use
    pub fn conservative() -> Self {
        Self {
            learning_aggressiveness: 0.3,
            integration_depth: IntegrationDepth::Standard,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.7,
                adaptation_speed_target: Duration::from_secs(600),
                memory_overhead_limit: 0.15,
                performance_degradation_limit: 0.05,
                user_satisfaction_target: 0.85,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 2,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.6,
                max_learning_impact_per_session: 0.1,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 512.0,
                max_cpu_usage_percentage: 30.0,
                max_storage_usage_mb: 100.0,
                max_network_bandwidth_mbps: 10.0,
                max_session_duration: Duration::from_secs(300),
                max_daily_learning_time: Duration::from_secs(3600),
            },
        }
    }
    
    /// Create aggressive configuration for development/testing
    pub fn aggressive() -> Self {
        Self {
            learning_aggressiveness: 0.8,
            integration_depth: IntegrationDepth::Deep,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.85,
                adaptation_speed_target: Duration::from_secs(120),
                memory_overhead_limit: 0.25,
                performance_degradation_limit: 0.15,
                user_satisfaction_target: 0.9,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 5,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.4,
                max_learning_impact_per_session: 0.25,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 2048.0,
                max_cpu_usage_percentage: 70.0,
                max_storage_usage_mb: 500.0,
                max_network_bandwidth_mbps: 50.0,
                max_session_duration: Duration::from_secs(900),
                max_daily_learning_time: Duration::from_secs(7200),
            },
        }
    }
    
    /// Create experimental configuration for research
    pub fn experimental() -> Self {
        Self {
            learning_aggressiveness: 0.9,
            integration_depth: IntegrationDepth::Experimental,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.95,
                adaptation_speed_target: Duration::from_secs(60),
                memory_overhead_limit: 0.4,
                performance_degradation_limit: 0.3,
                user_satisfaction_target: 0.8,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 10,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.2,
                max_learning_impact_per_session: 0.5,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 4096.0,
                max_cpu_usage_percentage: 90.0,
                max_storage_usage_mb: 1000.0,
                max_network_bandwidth_mbps: 100.0,
                max_session_duration: Duration::from_secs(1800),
                max_daily_learning_time: Duration::from_secs(14400),
            },
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        // Validate learning aggressiveness
        if self.learning_aggressiveness < 0.0 || self.learning_aggressiveness > 1.0 {
            return Err("Learning aggressiveness must be between 0.0 and 1.0".to_string());
        }
        
        // Validate performance targets
        if self.performance_targets.learning_efficiency_target < 0.0 || 
           self.performance_targets.learning_efficiency_target > 1.0 {
            return Err("Learning efficiency target must be between 0.0 and 1.0".to_string());
        }
        
        if self.performance_targets.memory_overhead_limit < 0.0 ||
           self.performance_targets.memory_overhead_limit > 1.0 {
            return Err("Memory overhead limit must be between 0.0 and 1.0".to_string());
        }
        
        // Validate safety constraints
        if self.safety_constraints.max_concurrent_learning_sessions == 0 {
            return Err("Must allow at least one concurrent learning session".to_string());
        }
        
        if self.safety_constraints.user_intervention_threshold < 0.0 ||
           self.safety_constraints.user_intervention_threshold > 1.0 {
            return Err("User intervention threshold must be between 0.0 and 1.0".to_string());
        }
        
        // Validate resource limits
        if self.resource_limits.max_memory_usage_mb <= 0.0 {
            return Err("Memory limit must be positive".to_string());
        }
        
        if self.resource_limits.max_cpu_usage_percentage <= 0.0 ||
           self.resource_limits.max_cpu_usage_percentage > 100.0 {
            return Err("CPU usage percentage must be between 0.0 and 100.0".to_string());
        }
        
        Ok(())
    }
    
    /// Get resource requirement based on configuration
    pub fn get_base_resource_requirement(&self) -> ResourceRequirement {
        let factor = self.learning_aggressiveness;
        
        ResourceRequirement {
            memory_mb: self.resource_limits.max_memory_usage_mb * factor * 0.5,
            cpu_cores: (self.resource_limits.max_cpu_usage_percentage / 100.0) * factor,
            storage_mb: self.resource_limits.max_storage_usage_mb * factor * 0.3,
            network_bandwidth_mbps: self.resource_limits.max_network_bandwidth_mbps * factor * 0.2,
        }
    }
    
    /// Check if operation is within safety constraints
    pub fn check_safety_constraints(&self, resource_usage: &ResourceRequirement, session_count: usize) -> Result<(), String> {
        if session_count > self.safety_constraints.max_concurrent_learning_sessions {
            return Err("Too many concurrent learning sessions".to_string());
        }
        
        if resource_usage.memory_mb > self.resource_limits.max_memory_usage_mb {
            return Err("Memory usage exceeds limit".to_string());
        }
        
        if resource_usage.cpu_cores > (self.resource_limits.max_cpu_usage_percentage / 100.0) {
            return Err("CPU usage exceeds limit".to_string());
        }
        
        Ok(())
    }
}

impl Default for Phase4Config {
    fn default() -> Self {
        Self::conservative()
    }
}