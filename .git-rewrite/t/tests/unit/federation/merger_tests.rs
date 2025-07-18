//! Federation Merger Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::federation::merger::*;

#[cfg(test)]
mod merger_tests {
    use super::*;

    #[test]
    fn test_graph_merging() {
        let graph1 = create_test_graph(50, 75);
        let graph2 = create_test_graph(50, 75);
        
        let merger = GraphMerger::new();
        let merged = merger.merge(&[graph1, graph2]).unwrap();
        
        // Should have combined entities (with possible deduplication)
        assert!(merged.entity_count() <= 100);
        assert!(merged.relationship_count() <= 150);
    }
}