//! Memory-Mapped Storage Unit Tests
//!
//! Tests for memory-mapped file storage, including safety,
//! performance, and concurrent access patterns.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::storage::mmap_storage::*;
use std::fs;
use std::path::PathBuf;

#[cfg(test)]
mod mmap_tests {
    use super::*;

    #[test]
    fn test_mmap_storage_basic_operations() {
        let temp_dir = create_temp_directory();
        let storage_path = temp_dir.join("test_storage.dat");
        
        {
            let mut storage = MmapStorage::create(&storage_path, 1024 * 1024).unwrap(); // 1MB
            
            // Test write operations
            let data1 = b"Hello, World!";
            let offset1 = storage.write(data1).unwrap();
            
            let data2 = b"This is a test of mmap storage";
            let offset2 = storage.write(data2).unwrap();
            
            assert_ne!(offset1, offset2);
            
            // Test read operations
            let read_data1 = storage.read(offset1, data1.len()).unwrap();
            assert_eq!(read_data1, data1);
            
            let read_data2 = storage.read(offset2, data2.len()).unwrap();
            assert_eq!(read_data2, data2);
            
            // Test in-place update
            let updated_data = b"Updated data!";
            storage.write_at(offset1, updated_data).unwrap();
            
            let read_updated = storage.read(offset1, updated_data.len()).unwrap();
            assert_eq!(read_updated, updated_data);
            
            // Test storage info
            assert!(storage.size() >= data1.len() + data2.len());
            assert!(storage.capacity() >= 1024 * 1024);
        }
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_mmap_storage_persistence() {
        let temp_dir = create_temp_directory();
        let storage_path = temp_dir.join("persistent_storage.dat");
        
        let test_data = b"Persistent test data";
        let offset;
        
        // Write data and close
        {
            let mut storage = MmapStorage::create(&storage_path, 1024).unwrap();
            offset = storage.write(test_data).unwrap();
            storage.sync().unwrap(); // Ensure data is written to disk
        }
        
        // Reopen and verify data persists
        {
            let storage = MmapStorage::open(&storage_path).unwrap();
            let read_data = storage.read(offset, test_data.len()).unwrap();
            assert_eq!(read_data, test_data);
        }
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_mmap_storage_growth() {
        let temp_dir = create_temp_directory();
        let storage_path = temp_dir.join("growing_storage.dat");
        
        let mut storage = MmapStorage::create(&storage_path, 1024).unwrap(); // Start small
        let initial_capacity = storage.capacity();
        
        // Write data exceeding initial capacity
        let large_data = vec![0u8; 2048]; // Larger than initial capacity
        let offset = storage.write(&large_data).unwrap();
        
        // Storage should have grown
        assert!(storage.capacity() > initial_capacity);
        
        // Data should be readable
        let read_data = storage.read(offset, large_data.len()).unwrap();
        assert_eq!(read_data, large_data);
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_mmap_storage_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;
        
        let temp_dir = create_temp_directory();
        let storage_path = temp_dir.join("concurrent_storage.dat");
        
        // Prepare test data
        let test_data: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("Test data item {}", i).into_bytes())
            .collect();
        
        let offsets: Vec<u64>;
        
        // Write all data
        {
            let mut storage = MmapStorage::create(&storage_path, 64 * 1024).unwrap();
            offsets = test_data.iter()
                .map(|data| storage.write(data).unwrap())
                .collect();
            storage.sync().unwrap();
        }
        
        // Concurrent read test
        {
            let storage = Arc::new(MmapStorage::open(&storage_path).unwrap());
            let thread_count = 4;
            let reads_per_thread = 1000;
            
            let mut handles = Vec::new();
            
            for thread_id in 0..thread_count {
                let storage_clone = Arc::clone(&storage);
                let offsets_clone = offsets.clone();
                let test_data_clone = test_data.clone();
                
                let handle = thread::spawn(move || {
                    let mut rng = DeterministicRng::new(thread_id as u64);
                    
                    for _ in 0..reads_per_thread {
                        let index = rng.gen_range(0..test_data_clone.len());
                        let offset = offsets_clone[index];
                        let expected_data = &test_data_clone[index];
                        
                        let read_data = storage_clone.read(offset, expected_data.len()).unwrap();
                        assert_eq!(read_data, *expected_data);
                    }
                });
                
                handles.push(handle);
            }
            
            // Wait for all threads
            for handle in handles {
                handle.join().unwrap();
            }
        }
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_mmap_storage_performance() {
        let temp_dir = create_temp_directory();
        let storage_path = temp_dir.join("performance_storage.dat");
        
        let mut storage = MmapStorage::create(&storage_path, 10 * 1024 * 1024).unwrap(); // 10MB
        
        let write_count = 10000;
        let data_size = 256;
        let test_data = vec![0xAB; data_size];
        
        // Test write performance
        let (offsets, write_time) = measure_execution_time(|| {
            let mut offsets = Vec::new();
            for _ in 0..write_count {
                let offset = storage.write(&test_data).unwrap();
                offsets.push(offset);
            }
            offsets
        });
        
        println!("Mmap write time for {} operations: {:?}", write_count, write_time);
        let writes_per_second = write_count as f64 / write_time.as_secs_f64();
        assert!(writes_per_second > 10000.0, "Write performance too slow: {:.0} ops/sec", writes_per_second);
        
        // Test read performance
        let (_, read_time) = measure_execution_time(|| {
            for &offset in &offsets {
                let _data = storage.read(offset, data_size).unwrap();
            }
        });
        
        println!("Mmap read time for {} operations: {:?}", write_count, read_time);
        let reads_per_second = write_count as f64 / read_time.as_secs_f64();
        assert!(reads_per_second > 50000.0, "Read performance too slow: {:.0} ops/sec", reads_per_second);
        
        // Test throughput
        let total_bytes = write_count * data_size;
        let write_throughput = total_bytes as f64 / write_time.as_secs_f64() / (1024.0 * 1024.0); // MB/s
        let read_throughput = total_bytes as f64 / read_time.as_secs_f64() / (1024.0 * 1024.0); // MB/s
        
        println!("Write throughput: {:.2} MB/s", write_throughput);
        println!("Read throughput: {:.2} MB/s", read_throughput);
        
        assert!(write_throughput > 10.0, "Write throughput too low: {:.2} MB/s", write_throughput);
        assert!(read_throughput > 100.0, "Read throughput too low: {:.2} MB/s", read_throughput);
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_mmap_storage_error_conditions() {
        let temp_dir = create_temp_directory();
        
        // Test creating storage with invalid path
        let invalid_path = PathBuf::from("/invalid/path/storage.dat");
        let result = MmapStorage::create(&invalid_path, 1024);
        assert!(result.is_err());
        
        // Test opening non-existent file
        let nonexistent_path = temp_dir.join("nonexistent.dat");
        let result = MmapStorage::open(&nonexistent_path);
        assert!(result.is_err());
        
        // Test reading beyond bounds
        let storage_path = temp_dir.join("bounds_test.dat");
        let mut storage = MmapStorage::create(&storage_path, 1024).unwrap();
        
        let data = b"test data";
        let offset = storage.write(data).unwrap();
        
        // Try to read more data than was written
        let result = storage.read(offset, data.len() + 100);
        assert!(result.is_err());
        
        // Try to read from invalid offset
        let result = storage.read(u64::MAX, 10);
        assert!(result.is_err());
        
        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }
}

fn create_temp_directory() -> PathBuf {
    let temp_dir = std::env::temp_dir().join(format!("llmkg_test_{}", 
        std::process::id()));
    fs::create_dir_all(&temp_dir).unwrap();
    temp_dir
}