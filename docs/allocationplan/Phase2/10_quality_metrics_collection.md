# Task 10: Quality Metrics Collection

## Metadata
- **Micro-Phase**: 2.10
- **Duration**: 15-20 minutes
- **Dependencies**: Task 09 (quality_gate_decision)
- **Output**: `src/quality_integration/metrics_collector.rs`

## Description
Create the QualityMetricsCollector that tracks and reports comprehensive metrics about the quality gate system performance, decision patterns, and processing statistics. This enables monitoring and optimization of the quality gate pipeline.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{
        QualityGateDecisionResult, AllocationDecision, DecisionStrategy,
        ValidatedFact, FactContent, ConfidenceComponents
    };

    #[test]
    fn test_metrics_collector_creation() {
        let collector = QualityMetricsCollector::new();
        assert!(collector.is_enabled);
        assert_eq!(collector.decision_history.len(), 0);
        assert_eq!(collector.processing_stats.total_decisions, 0);
    }
    
    #[test]
    fn test_single_decision_recording() {
        let mut collector = QualityMetricsCollector::new();
        
        // Create a mock decision result
        let fact_content = FactContent::new("Test fact for metrics");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let fact = ValidatedFact::new(fact_content, confidence);
        
        let decision_result = create_mock_decision_result(
            AllocationDecision::Approve,
            0.87,
            DecisionStrategy::Conservative
        );
        
        collector.record_decision(&decision_result, &fact);
        
        assert_eq!(collector.processing_stats.total_decisions, 1);
        assert_eq!(collector.processing_stats.approved_decisions, 1);
        assert_eq!(collector.processing_stats.rejected_decisions, 0);
        assert!(collector.processing_stats.average_quality_score > 0.8);
        assert_eq!(collector.decision_history.len(), 1);
    }
    
    #[test]
    fn test_multiple_decisions_aggregation() {
        let mut collector = QualityMetricsCollector::new();
        
        // Record multiple decisions
        for i in 0..10 {
            let decision = if i < 7 { AllocationDecision::Approve } else { AllocationDecision::Reject };
            let quality_score = if i < 7 { 0.85 + (i as f32 * 0.01) } else { 0.65 };
            
            let fact_content = FactContent::new(&format!("Test fact {}", i));
            let confidence = ConfidenceComponents::new(0.8, 0.8, 0.8);
            let fact = ValidatedFact::new(fact_content, confidence);
            
            let decision_result = create_mock_decision_result(decision, quality_score, DecisionStrategy::Balanced);
            collector.record_decision(&decision_result, &fact);
        }
        
        assert_eq!(collector.processing_stats.total_decisions, 10);
        assert_eq!(collector.processing_stats.approved_decisions, 7);
        assert_eq!(collector.processing_stats.rejected_decisions, 3);
        assert!(collector.processing_stats.approval_rate() > 0.6);
        assert!(collector.processing_stats.average_quality_score > 0.7);
    }
    
    #[test]
    fn test_performance_metrics_tracking() {
        let mut collector = QualityMetricsCollector::new();
        
        // Record decisions with varying processing times
        for i in 0..5 {
            let mut decision_result = create_mock_decision_result(
                AllocationDecision::Approve,
                0.85,
                DecisionStrategy::Conservative
            );
            
            // Simulate different processing times
            decision_result.processing_metadata.total_processing_time = (i + 1) * 100; // 100-500ms
            decision_result.processing_metadata.check_durations.insert(
                "thresholds".to_string(), 50 + i * 10
            );
            
            let fact_content = FactContent::new(&format!("Performance test {}", i));
            let confidence = ConfidenceComponents::new(0.8, 0.8, 0.8);
            let fact = ValidatedFact::new(fact_content, confidence);
            
            collector.record_decision(&decision_result, &fact);
        }
        
        let performance_report = collector.generate_performance_report();
        assert!(performance_report.average_processing_time > 0.0);
        assert!(performance_report.max_processing_time >= performance_report.min_processing_time);
        assert!(!performance_report.check_performance.is_empty());
    }
    
    #[test]
    fn test_quality_distribution_analysis() {
        let mut collector = QualityMetricsCollector::new();
        
        // Record decisions across quality spectrum
        let quality_scores = vec![0.95, 0.87, 0.82, 0.75, 0.68, 0.55, 0.42, 0.38, 0.25, 0.15];
        
        for (i, &score) in quality_scores.iter().enumerate() {
            let decision = if score >= 0.75 { AllocationDecision::Approve } else { AllocationDecision::Reject };
            
            let fact_content = FactContent::new(&format!("Quality test {}", i));
            let confidence = ConfidenceComponents::new(score, score, score);
            let fact = ValidatedFact::new(fact_content, confidence);
            
            let decision_result = create_mock_decision_result(decision, score, DecisionStrategy::Balanced);
            collector.record_decision(&decision_result, &fact);
        }
        
        let quality_report = collector.generate_quality_distribution_report();
        assert!(quality_report.quality_buckets.len() > 0);
        assert!(quality_report.high_quality_percentage + quality_report.low_quality_percentage <= 100.0);
    }
    
    #[test]
    fn test_trend_analysis() {
        let mut collector = QualityMetricsCollector::new();
        
        // Simulate improving quality over time
        for i in 0..20 {
            let base_quality = 0.6 + (i as f32 * 0.02); // Gradually improving from 0.6 to 0.98
            let decision = if base_quality >= 0.75 { AllocationDecision::Approve } else { AllocationDecision::Reject };
            
            let fact_content = FactContent::new(&format!("Trend test {}", i));
            let confidence = ConfidenceComponents::new(base_quality, base_quality, base_quality);
            let fact = ValidatedFact::new(fact_content, confidence);
            
            let decision_result = create_mock_decision_result(decision, base_quality, DecisionStrategy::Balanced);
            collector.record_decision(&decision_result, &fact);
        }
        
        let trend_report = collector.analyze_quality_trends(10); // Look at last 10 decisions
        assert!(trend_report.quality_trend != QualityTrend::Stable); // Should detect improvement
        assert!(trend_report.recent_average_quality > trend_report.historical_average_quality);
    }
    
    // Helper function to create mock decision results
    fn create_mock_decision_result(
        decision: AllocationDecision,
        quality_score: f32,
        strategy: DecisionStrategy
    ) -> QualityGateDecisionResult {
        use crate::quality_integration::{DecisionMetadata, QualityGateDecisionResult};
        use std::collections::HashMap;
        
        QualityGateDecisionResult {
            decision,
            overall_quality_score: quality_score,
            passed_checks: if decision == AllocationDecision::Approve { 
                vec!["threshold_validation".to_string(), "validation_chain".to_string()] 
            } else { 
                vec![] 
            },
            failed_checks: if decision == AllocationDecision::Reject { 
                vec!["threshold_validation".to_string()] 
            } else { 
                vec![] 
            },
            rejection_reasons: if decision == AllocationDecision::Reject { 
                vec!["Quality below threshold".to_string()] 
            } else { 
                vec![] 
            },
            threshold_result: None,
            validation_result: None,
            ambiguity_result: None,
            decision_confidence: 0.85,
            strategy_used: strategy,
            processing_metadata: DecisionMetadata {
                check_durations: HashMap::new(),
                total_processing_time: 150,
                config_hash: 12345,
                fact_hash: 67890,
            },
            decided_at: current_timestamp(),
        }
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use crate::quality_integration::{QualityGateDecisionResult, AllocationDecision, DecisionStrategy, ValidatedFact};

/// Statistics about processing performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total number of decisions made
    pub total_decisions: u64,
    
    /// Number of approved decisions
    pub approved_decisions: u64,
    
    /// Number of rejected decisions
    pub rejected_decisions: u64,
    
    /// Number of manual review decisions
    pub manual_review_decisions: u64,
    
    /// Average quality score across all decisions
    pub average_quality_score: f32,
    
    /// Average processing time in milliseconds
    pub average_processing_time: f32,
    
    /// Total processing time in milliseconds
    pub total_processing_time: u64,
    
    /// Decision counts by strategy
    pub strategy_counts: HashMap<DecisionStrategy, u64>,
}

impl ProcessingStats {
    /// Calculate approval rate (0.0-1.0)
    pub fn approval_rate(&self) -> f32 {
        if self.total_decisions == 0 {
            0.0
        } else {
            self.approved_decisions as f32 / self.total_decisions as f32
        }
    }
    
    /// Calculate rejection rate (0.0-1.0)
    pub fn rejection_rate(&self) -> f32 {
        if self.total_decisions == 0 {
            0.0
        } else {
            self.rejected_decisions as f32 / self.total_decisions as f32
        }
    }
}

/// Performance metrics for individual checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckPerformanceMetrics {
    /// Average time for each check type
    pub average_check_times: HashMap<String, f32>,
    
    /// Maximum time for each check type
    pub max_check_times: HashMap<String, u64>,
    
    /// Minimum time for each check type
    pub min_check_times: HashMap<String, u64>,
    
    /// Total count of each check type
    pub check_counts: HashMap<String, u64>,
}

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Average processing time across all decisions
    pub average_processing_time: f32,
    
    /// Maximum processing time observed
    pub max_processing_time: u64,
    
    /// Minimum processing time observed
    pub min_processing_time: u64,
    
    /// 95th percentile processing time
    pub p95_processing_time: u64,
    
    /// Performance metrics for individual checks
    pub check_performance: CheckPerformanceMetrics,
    
    /// Timestamp of report generation
    pub generated_at: u64,
}

/// Quality distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistributionReport {
    /// Number of decisions in each quality bucket
    pub quality_buckets: HashMap<String, u64>,
    
    /// Percentage of high-quality decisions (>= 0.8)
    pub high_quality_percentage: f32,
    
    /// Percentage of medium-quality decisions (0.6-0.8)
    pub medium_quality_percentage: f32,
    
    /// Percentage of low-quality decisions (< 0.6)
    pub low_quality_percentage: f32,
    
    /// Standard deviation of quality scores
    pub quality_std_deviation: f32,
    
    /// Timestamp of report generation
    pub generated_at: u64,
}

/// Quality trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTrend {
    Improving,
    Declining,
    Stable,
}

/// Trend analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisReport {
    /// Overall quality trend
    pub quality_trend: QualityTrend,
    
    /// Recent average quality (last N decisions)
    pub recent_average_quality: f32,
    
    /// Historical average quality
    pub historical_average_quality: f32,
    
    /// Trend strength (0.0-1.0)
    pub trend_strength: f32,
    
    /// Number of decisions used for recent analysis
    pub recent_window_size: usize,
    
    /// Timestamp of analysis
    pub analyzed_at: u64,
}

/// Individual decision record for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Allocation decision made
    pub decision: AllocationDecision,
    
    /// Quality score
    pub quality_score: f32,
    
    /// Processing time in milliseconds
    pub processing_time: u64,
    
    /// Strategy used
    pub strategy: DecisionStrategy,
    
    /// Fact content hash for tracking
    pub fact_hash: u64,
    
    /// Timestamp of decision
    pub timestamp: u64,
}

/// Main quality metrics collection and analysis engine
#[derive(Debug, Clone)]
pub struct QualityMetricsCollector {
    /// Whether metrics collection is enabled
    pub is_enabled: bool,
    
    /// Current processing statistics
    pub processing_stats: ProcessingStats,
    
    /// History of recent decisions (limited size)
    pub decision_history: VecDeque<DecisionRecord>,
    
    /// Maximum history size to maintain
    pub max_history_size: usize,
    
    /// Processing time buckets for percentile calculation
    pub processing_time_buckets: Vec<u64>,
    
    /// Quality score buckets for distribution analysis
    pub quality_score_buckets: Vec<f32>,
}

impl QualityMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            is_enabled: true,
            processing_stats: ProcessingStats {
                total_decisions: 0,
                approved_decisions: 0,
                rejected_decisions: 0,
                manual_review_decisions: 0,
                average_quality_score: 0.0,
                average_processing_time: 0.0,
                total_processing_time: 0,
                strategy_counts: HashMap::new(),
            },
            decision_history: VecDeque::new(),
            max_history_size: 10000, // Keep last 10,000 decisions
            processing_time_buckets: Vec::new(),
            quality_score_buckets: Vec::new(),
        }
    }
    
    /// Record a new decision and update metrics
    pub fn record_decision(&mut self, decision_result: &QualityGateDecisionResult, fact: &ValidatedFact) {
        if !self.is_enabled {
            return;
        }
        
        // Update basic counters
        self.processing_stats.total_decisions += 1;
        
        match decision_result.decision {
            AllocationDecision::Approve => self.processing_stats.approved_decisions += 1,
            AllocationDecision::Reject => self.processing_stats.rejected_decisions += 1,
            AllocationDecision::ManualReview => self.processing_stats.manual_review_decisions += 1,
        }
        
        // Update strategy counts
        *self.processing_stats.strategy_counts
            .entry(decision_result.strategy_used)
            .or_insert(0) += 1;
        
        // Update cumulative metrics
        let processing_time = decision_result.processing_metadata.total_processing_time;
        self.processing_stats.total_processing_time += processing_time;
        self.processing_stats.average_processing_time = 
            self.processing_stats.total_processing_time as f32 / self.processing_stats.total_decisions as f32;
        
        // Update quality score average
        let total_quality = self.processing_stats.average_quality_score * (self.processing_stats.total_decisions - 1) as f32;
        self.processing_stats.average_quality_score = 
            (total_quality + decision_result.overall_quality_score) / self.processing_stats.total_decisions as f32;
        
        // Add to history
        let record = DecisionRecord {
            decision: decision_result.decision,
            quality_score: decision_result.overall_quality_score,
            processing_time,
            strategy: decision_result.strategy_used,
            fact_hash: fact.content.content_hash(),
            timestamp: decision_result.decided_at,
        };
        
        self.decision_history.push_back(record);
        
        // Maintain history size limit
        while self.decision_history.len() > self.max_history_size {
            self.decision_history.pop_front();
        }
        
        // Update buckets for percentile calculations
        self.processing_time_buckets.push(processing_time);
        self.quality_score_buckets.push(decision_result.overall_quality_score);
        
        // Keep bucket sizes reasonable
        if self.processing_time_buckets.len() > self.max_history_size {
            self.processing_time_buckets.drain(0..self.processing_time_buckets.len() - self.max_history_size);
        }
        if self.quality_score_buckets.len() > self.max_history_size {
            self.quality_score_buckets.drain(0..self.quality_score_buckets.len() - self.max_history_size);
        }
    }
    
    /// Generate a comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let mut processing_times: Vec<u64> = self.processing_time_buckets.clone();
        processing_times.sort();
        
        let avg_time = if processing_times.is_empty() {
            0.0
        } else {
            processing_times.iter().sum::<u64>() as f32 / processing_times.len() as f32
        };
        
        let max_time = processing_times.last().copied().unwrap_or(0);
        let min_time = processing_times.first().copied().unwrap_or(0);
        
        // Calculate 95th percentile
        let p95_index = (processing_times.len() as f32 * 0.95) as usize;
        let p95_time = processing_times.get(p95_index).copied().unwrap_or(0);
        
        // Analyze check performance from decision history
        let mut check_times: HashMap<String, Vec<u64>> = HashMap::new();
        
        // This would need access to individual check durations from decision results
        // For now, create placeholder metrics
        let check_performance = CheckPerformanceMetrics {
            average_check_times: HashMap::new(),
            max_check_times: HashMap::new(),
            min_check_times: HashMap::new(),
            check_counts: HashMap::new(),
        };
        
        PerformanceReport {
            average_processing_time: avg_time,
            max_processing_time: max_time,
            min_processing_time: min_time,
            p95_processing_time: p95_time,
            check_performance,
            generated_at: current_timestamp(),
        }
    }
    
    /// Generate quality distribution analysis
    pub fn generate_quality_distribution_report(&self) -> QualityDistributionReport {
        let mut quality_buckets = HashMap::new();
        let mut high_count = 0;
        let mut medium_count = 0;
        let mut low_count = 0;
        
        for &score in &self.quality_score_buckets {
            if score >= 0.8 {
                high_count += 1;
                *quality_buckets.entry("0.8-1.0".to_string()).or_insert(0) += 1;
            } else if score >= 0.6 {
                medium_count += 1;
                *quality_buckets.entry("0.6-0.8".to_string()).or_insert(0) += 1;
            } else {
                low_count += 1;
                *quality_buckets.entry("0.0-0.6".to_string()).or_insert(0) += 1;
            }
        }
        
        let total = self.quality_score_buckets.len() as f32;
        let high_percentage = if total > 0.0 { high_count as f32 / total * 100.0 } else { 0.0 };
        let medium_percentage = if total > 0.0 { medium_count as f32 / total * 100.0 } else { 0.0 };
        let low_percentage = if total > 0.0 { low_count as f32 / total * 100.0 } else { 0.0 };
        
        // Calculate standard deviation
        let mean = if total > 0.0 {
            self.quality_score_buckets.iter().sum::<f32>() / total
        } else {
            0.0
        };
        
        let variance = if total > 1.0 {
            self.quality_score_buckets.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / (total - 1.0)
        } else {
            0.0
        };
        
        let std_deviation = variance.sqrt();
        
        QualityDistributionReport {
            quality_buckets,
            high_quality_percentage: high_percentage,
            medium_quality_percentage: medium_percentage,
            low_quality_percentage: low_percentage,
            quality_std_deviation: std_deviation,
            generated_at: current_timestamp(),
        }
    }
    
    /// Analyze quality trends over time
    pub fn analyze_quality_trends(&self, recent_window: usize) -> TrendAnalysisReport {
        let recent_decisions: Vec<&DecisionRecord> = self.decision_history
            .iter()
            .rev()
            .take(recent_window)
            .collect();
        
        let recent_average = if recent_decisions.is_empty() {
            0.0
        } else {
            recent_decisions.iter().map(|d| d.quality_score).sum::<f32>() / recent_decisions.len() as f32
        };
        
        let historical_average = self.processing_stats.average_quality_score;
        
        // Determine trend
        let difference = recent_average - historical_average;
        let trend = if difference.abs() < 0.02 {
            QualityTrend::Stable
        } else if difference > 0.0 {
            QualityTrend::Improving
        } else {
            QualityTrend::Declining
        };
        
        let trend_strength = (difference.abs() / historical_average.max(0.1)).min(1.0);
        
        TrendAnalysisReport {
            quality_trend: trend,
            recent_average_quality: recent_average,
            historical_average_quality: historical_average,
            trend_strength,
            recent_window_size: recent_decisions.len(),
            analyzed_at: current_timestamp(),
        }
    }
    
    /// Get current processing statistics
    pub fn get_current_stats(&self) -> &ProcessingStats {
        &self.processing_stats
    }
    
    /// Reset all metrics (useful for testing or fresh starts)
    pub fn reset_metrics(&mut self) {
        self.processing_stats = ProcessingStats {
            total_decisions: 0,
            approved_decisions: 0,
            rejected_decisions: 0,
            manual_review_decisions: 0,
            average_quality_score: 0.0,
            average_processing_time: 0.0,
            total_processing_time: 0,
            strategy_counts: HashMap::new(),
        };
        self.decision_history.clear();
        self.processing_time_buckets.clear();
        self.quality_score_buckets.clear();
    }
    
    /// Enable or disable metrics collection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }
    
    /// Set maximum history size
    pub fn set_max_history_size(&mut self, size: usize) {
        self.max_history_size = size;
        
        // Trim current history if needed
        while self.decision_history.len() > self.max_history_size {
            self.decision_history.pop_front();
        }
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for QualityMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create QualityMetricsCollector with comprehensive tracking capabilities
2. Implement decision recording with automatic metric updates
3. Add performance analysis with percentile calculations
4. Implement quality distribution and trend analysis
5. Ensure all metrics are accurate and efficiently calculated

## Success Criteria
- [ ] QualityMetricsCollector compiles without errors
- [ ] Decision recording correctly updates all statistics
- [ ] Performance reports provide meaningful insights
- [ ] Quality distribution analysis is accurate
- [ ] All tests pass with realistic metric scenarios