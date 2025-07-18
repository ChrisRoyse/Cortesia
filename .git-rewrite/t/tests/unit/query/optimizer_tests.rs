//! Query Optimizer Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::query::optimizer::*;

#[cfg(test)]
mod optimizer_tests {
    use super::*;

    #[test]
    fn test_query_optimization() {
        let optimizer = QueryOptimizer::new();
        
        // Test basic query optimization
        let query = "SELECT * FROM entities WHERE type = 'paper'";
        let optimized = optimizer.optimize(query).unwrap();
        
        assert!(!optimized.is_empty());
        assert!(optimized.execution_plan.len() > 0);
    }
}