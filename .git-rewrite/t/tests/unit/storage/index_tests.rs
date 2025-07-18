//! Index Unit Tests
//!
//! Tests for various indexing structures including B-trees,
//! hash indexes, and composite indexes.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::storage::index::*;

#[cfg(test)]
mod index_tests {
    use super::*;

    #[test]
    fn test_btree_index_basic_operations() {
        let mut index = BTreeIndex::new();
        
        // Test insertion
        index.insert("key1", "value1").unwrap();
        index.insert("key2", "value2").unwrap();
        index.insert("key3", "value3").unwrap();
        
        assert_eq!(index.len(), 3);
        
        // Test lookup
        assert_eq!(index.get("key1"), Some("value1"));
        assert_eq!(index.get("key2"), Some("value2"));
        assert_eq!(index.get("key3"), Some("value3"));
        assert_eq!(index.get("nonexistent"), None);
        
        // Test update
        index.insert("key2", "updated_value2").unwrap();
        assert_eq!(index.get("key2"), Some("updated_value2"));
        assert_eq!(index.len(), 3); // No size change
        
        // Test removal
        let removed = index.remove("key1").unwrap();
        assert_eq!(removed, Some("value1"));
        assert_eq!(index.len(), 2);
        assert_eq!(index.get("key1"), None);
        
        // Test range queries
        let range_results = index.range("key1".."key3");
        assert_eq!(range_results.len(), 1);
        assert_eq!(range_results[0], ("key2", "updated_value2"));
    }

    #[test]
    fn test_hash_index_performance() {
        let mut index = HashIndex::new();
        let item_count = 100000;
        
        // Test insertion performance
        let (_, insert_time) = measure_execution_time(|| {
            for i in 0..item_count {
                let key = format!("key_{}", i);
                let value = format!("value_{}", i);
                index.insert(&key, &value).unwrap();
            }
        });
        
        println!("Hash index insertion time for {} items: {:?}", item_count, insert_time);
        
        // Test lookup performance
        let (_, lookup_time) = measure_execution_time(|| {
            for i in 0..item_count {
                let key = format!("key_{}", i);
                let _ = index.get(&key);
            }
        });
        
        println!("Hash index lookup time for {} items: {:?}", item_count, lookup_time);
        
        // Hash index should be very fast
        let lookups_per_second = item_count as f64 / lookup_time.as_secs_f64();
        assert!(lookups_per_second > 1_000_000.0, "Hash index too slow: {:.0} ops/sec", lookups_per_second);
    }

    #[test]
    fn test_composite_index() {
        let mut index = CompositeIndex::new(vec!["field1", "field2", "field3"]);
        
        // Insert records with multiple fields
        let record1 = vec![("field1", "value1"), ("field2", "value2"), ("field3", "value3")];
        let record2 = vec![("field1", "value1"), ("field2", "different"), ("field3", "value3")];
        let record3 = vec![("field1", "other"), ("field2", "value2"), ("field3", "value3")];
        
        index.insert("record1", &record1).unwrap();
        index.insert("record2", &record2).unwrap();
        index.insert("record3", &record3).unwrap();
        
        // Test single field queries
        let field1_results = index.query_by_field("field1", "value1");
        assert_eq!(field1_results.len(), 2);
        assert!(field1_results.contains(&"record1"));
        assert!(field1_results.contains(&"record2"));
        
        // Test multi-field queries
        let multi_query = vec![("field1", "value1"), ("field3", "value3")];
        let multi_results = index.query_by_fields(&multi_query);
        assert_eq!(multi_results.len(), 2);
        
        // Test exact match
        let exact_results = index.query_exact(&record1);
        assert_eq!(exact_results.len(), 1);
        assert_eq!(exact_results[0], "record1");
    }
}