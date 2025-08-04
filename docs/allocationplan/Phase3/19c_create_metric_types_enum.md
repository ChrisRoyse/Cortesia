# Task 19c: Create Metric Types Enum
**Time**: 3 minutes (45 sec read, 1.5 min implement, 45 sec verify)
**Dependencies**: 19b
**Stage**: Performance Foundation

## Objective
Define enum for different types of metrics we'll collect.

## Implementation
Add to `src/inheritance/monitoring/performance_monitor.rs`:
```rust
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum MetricType {
    Inheritance,
    Query,
    Cache,
    System,
    Business,
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timing(std::time::Duration),
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::Inheritance => write!(f, "inheritance"),
            MetricType::Query => write!(f, "query"),
            MetricType::Cache => write!(f, "cache"),
            MetricType::System => write!(f, "system"),
            MetricType::Business => write!(f, "business"),
        }
    }
}
```

## Success Criteria
- MetricType enum created
- MetricValue enum for different value types
- Display trait implemented
- All variants compile

**Next**: 19d