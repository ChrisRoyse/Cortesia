//! Safety validation for optimization operations

use super::types::*;
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::HashMap;
use std::time::Instant;

impl SafetyValidator {
    /// Create new safety validator
    pub fn new() -> Self {
        Self {
            validation_rules: Self::default_validation_rules(),
            safety_threshold: 0.8,
            validation_history: Vec::new(),
        }
    }

    /// Create default validation rules
    fn default_validation_rules() -> Vec<SafetyRule> {
        vec![
            SafetyRule {
                rule_id: "data_integrity".to_string(),
                rule_type: SafetyRuleType::DataIntegrity,
                severity: SafetySeverity::Critical,
                validation_fn: "validate_data_integrity".to_string(),
            },
            SafetyRule {
                rule_id: "performance_regression".to_string(),
                rule_type: SafetyRuleType::PerformanceRegression,
                severity: SafetySeverity::High,
                validation_fn: "validate_performance_regression".to_string(),
            },
            SafetyRule {
                rule_id: "resource_limits".to_string(),
                rule_type: SafetyRuleType::ResourceLimits,
                severity: SafetySeverity::Medium,
                validation_fn: "validate_resource_limits".to_string(),
            },
            SafetyRule {
                rule_id: "concurrent_access".to_string(),
                rule_type: SafetyRuleType::ConcurrentAccess,
                severity: SafetySeverity::High,
                validation_fn: "validate_concurrent_access".to_string(),
            },
            SafetyRule {
                rule_id: "transaction_safety".to_string(),
                rule_type: SafetyRuleType::TransactionSafety,
                severity: SafetySeverity::Critical,
                validation_fn: "validate_transaction_safety".to_string(),
            },
        ]
    }

    /// Validate optimization safety
    pub async fn validate_optimization_safety(
        &mut self,
        graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<f32> {
        let mut safety_score = 1.0;
        let mut validation_results = Vec::new();
        
        // Run all validation rules
        for rule in &self.validation_rules {
            let result = self.run_validation_rule(graph, opportunity, current_metrics, rule).await?;
            validation_results.push(result.clone());
            
            // Apply severity weighting
            let weight = self.get_severity_weight(&rule.severity);
            if !result.passed {
                safety_score -= (1.0 - result.score) * weight;
            }
        }
        
        // Store validation history
        self.validation_history.extend(validation_results);
        
        // Maintain history size
        if self.validation_history.len() > 1000 {
            self.validation_history.drain(0..100);
        }
        
        Ok(safety_score.max(0.0))
    }

    /// Run a specific validation rule
    async fn run_validation_rule(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
        rule: &SafetyRule,
    ) -> Result<ValidationResult> {
        let validation_time = Instant::now();
        
        let (passed, score, details) = match rule.validation_fn.as_str() {
            "validate_data_integrity" => {
                self.validate_data_integrity(graph, opportunity, current_metrics).await?
            }
            "validate_performance_regression" => {
                self.validate_performance_regression(graph, opportunity, current_metrics).await?
            }
            "validate_resource_limits" => {
                self.validate_resource_limits(graph, opportunity, current_metrics).await?
            }
            "validate_concurrent_access" => {
                self.validate_concurrent_access(graph, opportunity, current_metrics).await?
            }
            "validate_transaction_safety" => {
                self.validate_transaction_safety(graph, opportunity, current_metrics).await?
            }
            _ => (false, 0.0, "Unknown validation function".to_string()),
        };
        
        Ok(ValidationResult {
            rule_id: rule.rule_id.clone(),
            passed,
            score,
            details,
            validation_time,
        })
    }

    /// Validate data integrity
    async fn validate_data_integrity(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        _current_metrics: &PerformanceMetrics,
    ) -> Result<(bool, f32, String)> {
        let mut integrity_score = 1.0;
        let mut issues = Vec::new();
        
        // Check affected entities exist
        for &entity_key in &opportunity.affected_entities {
            if graph.get_entity_data(entity_key).is_none() {
                integrity_score -= 0.2;
                issues.push(format!("Entity {:?} does not exist", entity_key));
            }
        }
        
        // Check for circular dependencies in hierarchy optimizations
        if opportunity.optimization_type == OptimizationType::HierarchyConsolidation {
            if let Some(cycle_issue) = self.check_hierarchy_cycles(graph, &opportunity.affected_entities).await? {
                integrity_score -= 0.3;
                issues.push(cycle_issue);
            }
        }
        
        // Check for orphaned entities
        if let Some(orphan_issue) = self.check_orphaned_entities(graph, &opportunity.affected_entities).await? {
            integrity_score -= 0.1;
            issues.push(orphan_issue);
        }
        
        let passed = integrity_score >= 0.8;
        let details = if issues.is_empty() {
            "Data integrity validation passed".to_string()
        } else {
            issues.join("; ")
        };
        
        Ok((passed, integrity_score, details))
    }

    /// Validate performance regression
    async fn validate_performance_regression(
        &self,
        _graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<(bool, f32, String)> {
        let mut regression_score = 1.0;
        let mut issues = Vec::new();
        
        // Check if current performance is already poor
        if current_metrics.query_latency > std::time::Duration::from_millis(500) {
            regression_score -= 0.1;
            issues.push("Current query latency is high".to_string());
        }
        
        if current_metrics.memory_usage > 1_000_000_000 {
            regression_score -= 0.1;
            issues.push("Current memory usage is high".to_string());
        }
        
        if current_metrics.error_rate > 0.05 {
            regression_score -= 0.2;
            issues.push("Current error rate is high".to_string());
        }
        
        // Check optimization risk level
        match opportunity.risk_level {
            RiskLevel::Critical => {
                regression_score -= 0.4;
                issues.push("Critical risk level optimization".to_string());
            }
            RiskLevel::High => {
                regression_score -= 0.2;
                issues.push("High risk level optimization".to_string());
            }
            RiskLevel::Medium => {
                regression_score -= 0.1;
                issues.push("Medium risk level optimization".to_string());
            }
            RiskLevel::Low => {} // No penalty
        }
        
        let passed = regression_score >= 0.7;
        let details = if issues.is_empty() {
            "Performance regression validation passed".to_string()
        } else {
            issues.join("; ")
        };
        
        Ok((passed, regression_score, details))
    }

    /// Validate resource limits
    async fn validate_resource_limits(
        &self,
        _graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<(bool, f32, String)> {
        let mut resource_score = 1.0;
        let mut issues = Vec::new();
        
        // Check current resource utilization
        if current_metrics.resource_utilization > 0.9 {
            resource_score -= 0.3;
            issues.push("High resource utilization".to_string());
        }
        
        // Check memory usage
        if current_metrics.memory_usage > 800_000_000 {
            resource_score -= 0.2;
            issues.push("High memory usage".to_string());
        }
        
        // Check implementation cost
        if opportunity.implementation_cost > 0.5 {
            resource_score -= 0.1;
            issues.push("High implementation cost".to_string());
        }
        
        // Check number of affected entities
        if opportunity.affected_entities.len() > 1000 {
            resource_score -= 0.1;
            issues.push("Large number of affected entities".to_string());
        }
        
        let passed = resource_score >= 0.8;
        let details = if issues.is_empty() {
            "Resource limits validation passed".to_string()
        } else {
            issues.join("; ")
        };
        
        Ok((passed, resource_score, details))
    }

    /// Validate concurrent access safety
    async fn validate_concurrent_access(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<(bool, f32, String)> {
        let mut access_score = 1.0;
        let mut issues = Vec::new();
        
        // Check if system is under high load
        if current_metrics.resource_utilization > 0.8 {
            access_score -= 0.2;
            issues.push("High system load increases concurrency risk".to_string());
        }
        
        // Check for high-traffic entities
        let high_traffic_entities = self.identify_high_traffic_entities(graph, &opportunity.affected_entities).await?;
        if !high_traffic_entities.is_empty() {
            access_score -= 0.1;
            issues.push(format!("High-traffic entities affected: {}", high_traffic_entities.len()));
        }
        
        // Check optimization type risks
        match opportunity.optimization_type {
            OptimizationType::HierarchyConsolidation => {
                access_score -= 0.1;
                issues.push("Hierarchy changes can affect concurrent access".to_string());
            }
            OptimizationType::SubgraphFactorization => {
                access_score -= 0.15;
                issues.push("Subgraph changes can cause access conflicts".to_string());
            }
            _ => {} // Other types have lower concurrency risk
        }
        
        let passed = access_score >= 0.7;
        let details = if issues.is_empty() {
            "Concurrent access validation passed".to_string()
        } else {
            issues.join("; ")
        };
        
        Ok((passed, access_score, details))
    }

    /// Validate transaction safety
    async fn validate_transaction_safety(
        &self,
        _graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<(bool, f32, String)> {
        let mut transaction_score = 1.0;
        let mut issues = Vec::new();
        
        // Check error rate
        if current_metrics.error_rate > 0.02 {
            transaction_score -= 0.2;
            issues.push("High error rate affects transaction safety".to_string());
        }
        
        // Check optimization complexity
        if opportunity.affected_entities.len() > 100 {
            transaction_score -= 0.1;
            issues.push("Large-scale optimization increases transaction complexity".to_string());
        }
        
        // Check prerequisites
        if opportunity.prerequisites.is_empty() {
            transaction_score -= 0.1;
            issues.push("No prerequisites defined for optimization".to_string());
        }
        
        // Check for atomic operations
        match opportunity.optimization_type {
            OptimizationType::HierarchyConsolidation | OptimizationType::SubgraphFactorization => {
                if opportunity.affected_entities.len() > 10 {
                    transaction_score -= 0.15;
                    issues.push("Complex structural changes may not be atomic".to_string());
                }
            }
            _ => {} // Other types are generally more atomic
        }
        
        let passed = transaction_score >= 0.8;
        let details = if issues.is_empty() {
            "Transaction safety validation passed".to_string()
        } else {
            issues.join("; ")
        };
        
        Ok((passed, transaction_score, details))
    }

    /// Check for hierarchy cycles
    async fn check_hierarchy_cycles(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        affected_entities: &[EntityKey],
    ) -> Result<Option<String>> {
        // Simplified cycle detection
        for i in 0..affected_entities.len() {
            for j in i + 1..affected_entities.len() {
                let entity1 = affected_entities[i];
                let entity2 = affected_entities[j];
                
                // Check if there's a path from entity1 to entity2 and back
                if graph.has_path(entity1, entity2) && graph.has_path(entity2, entity1) {
                    return Ok(Some(format!("Potential cycle detected between {:?} and {:?}", entity1, entity2)));
                }
            }
        }
        
        Ok(None)
    }

    /// Check for orphaned entities
    async fn check_orphaned_entities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        affected_entities: &[EntityKey],
    ) -> Result<Option<String>> {
        let mut orphaned_count = 0;
        
        for &entity_key in affected_entities {
            let neighbors = graph.get_neighbors(entity_key).await;
            if neighbors.is_empty() {
                orphaned_count += 1;
            }
        }
        
        if orphaned_count > 0 {
            Ok(Some(format!("{} potentially orphaned entities", orphaned_count)))
        } else {
            Ok(None)
        }
    }

    /// Identify high-traffic entities
    async fn identify_high_traffic_entities(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        affected_entities: &[EntityKey],
    ) -> Result<Vec<EntityKey>> {
        let mut high_traffic = Vec::new();
        
        for &entity_key in affected_entities {
            let neighbors = graph.get_neighbors(entity_key).await;
            if neighbors.len() > 50 {
                high_traffic.push(entity_key);
            }
        }
        
        Ok(high_traffic)
    }

    /// Get severity weight for safety calculations
    fn get_severity_weight(&self, severity: &SafetySeverity) -> f32 {
        match severity {
            SafetySeverity::Critical => 0.4,
            SafetySeverity::High => 0.3,
            SafetySeverity::Medium => 0.2,
            SafetySeverity::Low => 0.1,
        }
    }

    /// Check if optimization passes safety threshold
    pub fn passes_safety_threshold(&self, safety_score: f32) -> bool {
        safety_score >= self.safety_threshold
    }

    /// Get validation statistics
    pub fn get_validation_statistics(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        if self.validation_history.is_empty() {
            return stats;
        }
        
        let total_validations = self.validation_history.len() as f32;
        let passed_validations = self.validation_history.iter().filter(|v| v.passed).count() as f32;
        let avg_score = self.validation_history.iter().map(|v| v.score).sum::<f32>() / total_validations;
        
        stats.insert("total_validations".to_string(), total_validations);
        stats.insert("pass_rate".to_string(), passed_validations / total_validations);
        stats.insert("average_score".to_string(), avg_score);
        
        stats
    }

    /// Get recent validation failures
    pub fn get_recent_failures(&self) -> Vec<&ValidationResult> {
        self.validation_history.iter()
            .filter(|v| !v.passed)
            .rev()
            .take(10)
            .collect()
    }

    /// Update safety threshold
    pub fn update_safety_threshold(&mut self, new_threshold: f32) {
        self.safety_threshold = new_threshold.clamp(0.0, 1.0);
    }

    /// Add custom validation rule
    pub fn add_validation_rule(&mut self, rule: SafetyRule) {
        self.validation_rules.push(rule);
    }

    /// Remove validation rule
    pub fn remove_validation_rule(&mut self, rule_id: &str) {
        self.validation_rules.retain(|rule| rule.rule_id != rule_id);
    }
}

impl Default for SafetyValidator {
    fn default() -> Self {
        Self::new()
    }
}