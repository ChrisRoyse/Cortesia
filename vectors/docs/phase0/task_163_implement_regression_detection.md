# Micro-Task 163: Implement Regression Detection

## Objective
Implement automated regression detection to identify performance degradations.

## Prerequisites
- Task 162 completed (Continuous monitoring setup)

## Time Estimate
8 minutes

## Instructions
1. Create regression detector `regression_detector.rs`:
   ```rust
   use std::collections::VecDeque;
   use std::time::{SystemTime, UNIX_EPOCH};
   
   pub struct RegressionDetector {
       window_size: usize,
       threshold_percent: f64,
       measurements: VecDeque<(u64, f64)>, // (timestamp, value)
   }
   
   impl RegressionDetector {
       pub fn new(window_size: usize, threshold_percent: f64) -> Self {
           Self {
               window_size,
               threshold_percent,
               measurements: VecDeque::new(),
           }
       }
       
       pub fn add_measurement(&mut self, value: f64) -> Option<RegressionAlert> {
           let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
           
           self.measurements.push_back((timestamp, value));
           
           if self.measurements.len() > self.window_size {
               self.measurements.pop_front();
           }
           
           if self.measurements.len() >= self.window_size {
               self.detect_regression()
           } else {
               None
           }
       }
       
       fn detect_regression(&self) -> Option<RegressionAlert> {
           let mid_point = self.window_size / 2;
           let early_avg = self.calculate_average(0, mid_point);
           let recent_avg = self.calculate_average(mid_point, self.window_size);
           
           let change_percent = ((recent_avg - early_avg) / early_avg) * 100.0;
           
           if change_percent > self.threshold_percent {
               Some(RegressionAlert {
                   metric: "performance".to_string(),
                   early_average: early_avg,
                   recent_average: recent_avg,
                   change_percent,
                   timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
               })
           } else {
               None
           }
       }
       
       fn calculate_average(&self, start: usize, end: usize) -> f64 {
           let values: Vec<f64> = self.measurements
               .iter()
               .skip(start)
               .take(end - start)
               .map(|(_, value)| *value)
               .collect();
           
           values.iter().sum::<f64>() / values.len() as f64
       }
   }
   
   #[derive(Debug)]
   pub struct RegressionAlert {
       pub metric: String,
       pub early_average: f64,
       pub recent_average: f64,
       pub change_percent: f64,
       pub timestamp: u64,
   }
   
   fn main() {
       println!("Testing regression detection...");
       
       let mut detector = RegressionDetector::new(10, 20.0); // 10 samples, 20% threshold
       
       // Simulate gradual performance degradation
       let base_performance = 2.0;
       for i in 0..20 {
           let degradation = if i > 10 { (i - 10) as f64 * 0.3 } else { 0.0 };
           let measurement = base_performance + degradation;
           
           if let Some(alert) = detector.add_measurement(measurement) {
               println!("⚠ REGRESSION DETECTED:");
               println!("  Early avg: {:.3}ms", alert.early_average);
               println!("  Recent avg: {:.3}ms", alert.recent_average);
               println!("  Change: +{:.1}%", alert.change_percent);
               break;
           }
           
           println!("Measurement {}: {:.3}ms", i + 1, measurement);
       }
   }
   ```
2. Run: `cargo run --release --bin regression_detector`
3. Commit: `git add src/bin/regression_detector.rs && git commit -m "Implement automated regression detection"`

## Success Criteria
- [ ] Regression detector implemented
- [ ] Statistical analysis working
- [ ] Alert generation functional
- [ ] Test simulation successful
- [ ] Detector committed

## Next Task
task_164_create_performance_dashboard.md