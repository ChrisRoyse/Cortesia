//! Federation Router Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::federation::router::*;

#[cfg(test)]
mod router_tests {
    use super::*;

    #[test]
    fn test_query_routing() {
        let mut router = QueryRouter::new();
        
        // Add routing rules
        router.add_rule("entities", "node1").unwrap();
        router.add_rule("relationships", "node2").unwrap();
        
        // Test routing decisions
        let entity_query = "SELECT * FROM entities";
        let route = router.route(entity_query).unwrap();
        assert_eq!(route, "node1");
        
        let relationship_query = "SELECT * FROM relationships";
        let route = router.route(relationship_query).unwrap();
        assert_eq!(route, "node2");
    }
}