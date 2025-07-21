use bumpalo::Bump;
use parking_lot::RwLock;
use slotmap::SlotMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::core::types::{EntityKey, EntityData};

pub struct GraphArena {
    bump_allocator: Mutex<Bump>,
    entity_pool: SlotMap<EntityKey, EntityData>,
    generation_counter: AtomicU32,
}

impl GraphArena {
    pub fn new() -> Self {
        Self {
            bump_allocator: Mutex::new(Bump::new()),
            entity_pool: SlotMap::with_key(),
            generation_counter: AtomicU32::new(0),
        }
    }
    
    pub fn allocate_entity(&mut self, data: EntityData) -> EntityKey {
        self.entity_pool.insert(data)
    }
    
    pub fn get_entity(&self, key: EntityKey) -> Option<&EntityData> {
        self.entity_pool.get(key)
    }
    
    pub fn get_entity_mut(&mut self, key: EntityKey) -> Option<&mut EntityData> {
        self.entity_pool.get_mut(key)
    }
    
    pub fn remove_entity(&mut self, key: EntityKey) -> Option<EntityData> {
        self.entity_pool.remove(key)
    }
    
    pub fn reset_generation(&mut self) {
        self.bump_allocator.lock().unwrap().reset();
        self.generation_counter.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn entity_count(&self) -> usize {
        self.entity_pool.len()
    }
    
    pub fn memory_usage(&self) -> usize {
        self.bump_allocator.lock().unwrap().allocated_bytes() + 
        self.entity_pool.capacity() * std::mem::size_of::<EntityData>()
    }
    
    /// Get the capacity of the arena
    pub fn capacity(&self) -> usize {
        self.entity_pool.capacity()
    }
    
    /// Add edge (not applicable - GraphArena stores entities, not edges)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> crate::error::Result<()> {
        Err(crate::error::GraphError::UnsupportedOperation(
            "GraphArena stores entities, not edges. Use CSRGraph for edge storage.".to_string()
        ))
    }
    
    /// Update an entity
    pub fn update_entity(&mut self, key: EntityKey, data: EntityData) -> crate::error::Result<()> {
        if let Some(entity) = self.entity_pool.get_mut(key) {
            *entity = data;
            Ok(())
        } else {
            Err(crate::error::GraphError::EntityKeyNotFound { key })
        }
    }
    
    /// Remove method (alias for remove_entity)
    pub fn remove(&mut self, key: EntityKey) -> Option<EntityData> {
        self.remove_entity(key)
    }
    
    /// Check if arena contains an entity
    pub fn contains_entity(&self, key: EntityKey) -> bool {
        self.entity_pool.contains_key(key)
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        // Approximate size for serialization
        std::mem::size_of::<u32>() + // generation counter
        self.entity_pool.len() * (std::mem::size_of::<EntityKey>() + std::mem::size_of::<EntityData>()) +
        self.bump_allocator.lock().unwrap().allocated_bytes()
    }
}

// Safety: GraphArena is safe to Send/Sync because:
// - Mutex protects the bump allocator
// - SlotMap operations are handled through mutable references only
// - AtomicU32 is already Send + Sync
unsafe impl Send for GraphArena {}
unsafe impl Sync for GraphArena {}

pub struct EpochManager {
    global_epoch: AtomicU64,
    thread_epochs: Vec<AtomicU64>,
    retired_objects: RwLock<Vec<RetiredObject>>,
}

struct RetiredObject {
    ptr: *mut u8,
    size: usize,
    retired_epoch: u64,
}

impl EpochManager {
    pub fn new(num_threads: usize) -> Self {
        let mut thread_epochs = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            thread_epochs.push(AtomicU64::new(0));
        }
        
        Self {
            global_epoch: AtomicU64::new(0),
            thread_epochs,
            retired_objects: RwLock::new(Vec::new()),
        }
    }
    
    pub fn enter(&self, thread_id: usize) -> EpochGuard {
        let current_epoch = self.global_epoch.load(Ordering::Acquire);
        self.thread_epochs[thread_id].store(current_epoch, Ordering::Release);
        EpochGuard {
            thread_id,
            manager: self,
        }
    }
    
    pub fn retire_object(&self, ptr: *mut u8, size: usize) {
        let epoch = self.global_epoch.load(Ordering::Acquire);
        let retired = RetiredObject {
            ptr,
            size,
            retired_epoch: epoch,
        };
        self.retired_objects.write().push(retired);
    }
    
    pub fn advance_epoch(&self) {
        self.global_epoch.fetch_add(1, Ordering::AcqRel);
        self.collect_garbage();
    }
    
    fn collect_garbage(&self) {
        let min_epoch = self.thread_epochs.iter()
            .map(|e| e.load(Ordering::Acquire))
            .min()
            .unwrap_or(0);
        
        let mut retired = self.retired_objects.write();
        retired.retain(|obj| {
            if obj.retired_epoch < min_epoch {
                unsafe {
                    let layout = std::alloc::Layout::from_size_align_unchecked(obj.size, 8);
                    std::alloc::dealloc(obj.ptr, layout);
                }
                false
            } else {
                true
            }
        });
    }
}

pub struct EpochGuard<'a> {
    thread_id: usize,
    manager: &'a EpochManager,
}

impl<'a> Drop for EpochGuard<'a> {
    fn drop(&mut self) {
        self.manager.thread_epochs[self.thread_id].store(u64::MAX, Ordering::Release);
    }
}

unsafe impl Send for RetiredObject {}
unsafe impl Sync for RetiredObject {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
    use crate::core::types::{EntityData, AttributeValue};
    use std::collections::HashMap;

    fn create_test_entity_data(id: u16) -> EntityData {
        EntityData {
            type_id: id,
            properties: format!("test_entity_{}", id),
            embedding: vec![0.1 * id as f32; 128],
        }
    }

    #[tokio::test]
    async fn test_graph_arena_basic_operations() {
        let mut arena = GraphArena::new();
        
        // Test initial state
        assert_eq!(arena.entity_count(), 0);
        assert!(arena.capacity() >= 0);
        
        // Test entity allocation
        let entity1 = create_test_entity_data(1);
        let key1 = arena.allocate_entity(entity1.clone());
        assert_eq!(arena.entity_count(), 1);
        
        // Test entity retrieval
        let retrieved = arena.get_entity(key1).unwrap();
        assert_eq!(retrieved.type_id, 1);
        assert_eq!(retrieved.properties, "test_entity_1");
        assert_eq!(retrieved.embedding.len(), 128);
        
        // Test entity contains
        assert!(arena.contains_entity(key1));
        
        // Test entity update
        let updated_entity = EntityData {
            type_id: 2,
            properties: "updated_entity".to_string(),
            embedding: vec![0.5; 64],
        };
        
        assert!(arena.update_entity(key1, updated_entity.clone()).is_ok());
        let retrieved_updated = arena.get_entity(key1).unwrap();
        assert_eq!(retrieved_updated.type_id, 2);
        assert_eq!(retrieved_updated.properties, "updated_entity");
        assert_eq!(retrieved_updated.embedding.len(), 64);
        
        // Test entity removal
        let removed = arena.remove_entity(key1).unwrap();
        assert_eq!(removed.type_id, 2);
        assert_eq!(arena.entity_count(), 0);
        assert!(!arena.contains_entity(key1));
        
        // Test get non-existent entity
        assert!(arena.get_entity(key1).is_none());
    }

    #[tokio::test]
    async fn test_graph_arena_large_allocations() {
        let mut arena = GraphArena::new();
        let mut keys = Vec::new();
        
        // Allocate large number of entities
        const NUM_ENTITIES: usize = 10000;
        for i in 0..NUM_ENTITIES {
            let entity = create_test_entity_data(i as u16);
            let key = arena.allocate_entity(entity);
            keys.push(key);
        }
        
        assert_eq!(arena.entity_count(), NUM_ENTITIES);
        
        // Verify all entities can be retrieved
        for (i, &key) in keys.iter().enumerate() {
            let entity = arena.get_entity(key).unwrap();
            assert_eq!(entity.type_id, i as u16);
            assert_eq!(entity.properties, format!("test_entity_{}", i));
        }
        
        // Remove half the entities
        for i in (0..NUM_ENTITIES).step_by(2) {
            assert!(arena.remove_entity(keys[i]).is_some());
        }
        
        assert_eq!(arena.entity_count(), NUM_ENTITIES / 2);
        
        // Verify remaining entities are still accessible
        for i in (1..NUM_ENTITIES).step_by(2) {
            assert!(arena.get_entity(keys[i]).is_some());
        }
        
        // Verify removed entities are not accessible
        for i in (0..NUM_ENTITIES).step_by(2) {
            assert!(arena.get_entity(keys[i]).is_none());
        }
    }

    #[tokio::test]
    async fn test_graph_arena_memory_management() {
        let mut arena = GraphArena::new();
        
        // Test initial memory usage
        let initial_memory = arena.memory_usage();
        assert!(initial_memory >= 0);
        
        // Add entities and check memory growth
        let mut keys = Vec::new();
        for i in 0..100 {
            let entity = create_test_entity_data(i);
            let key = arena.allocate_entity(entity);
            keys.push(key);
        }
        
        let memory_after_allocations = arena.memory_usage();
        assert!(memory_after_allocations > initial_memory);
        
        // Test generation reset
        let generation_before = arena.generation_counter.load(Ordering::Relaxed);
        arena.reset_generation();
        let generation_after = arena.generation_counter.load(Ordering::Relaxed);
        assert_eq!(generation_after, generation_before + 1);
        
        // Test encoded size
        let encoded_size = arena.encoded_size();
        assert!(encoded_size > 0);
    }

    #[tokio::test]
    async fn test_graph_arena_edge_cases() {
        let mut arena = GraphArena::new();
        
        // Test update non-existent entity
        let dummy_key = slotmap::KeyData::from_ffi(0).into();
        let dummy_entity = create_test_entity_data(1);
        let result = arena.update_entity(dummy_key, dummy_entity);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::GraphError::EntityKeyNotFound { key } => {
                assert_eq!(key, dummy_key);
            }
            _ => panic!("Expected EntityKeyNotFound error"),
        }
        
        // Test add_edge (should fail as unsupported)
        let edge_result = arena.add_edge(1, 2, 0.5);
        assert!(edge_result.is_err());
        match edge_result.unwrap_err() {
            crate::error::GraphError::UnsupportedOperation(msg) => {
                assert!(msg.contains("GraphArena stores entities, not edges"));
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
        
        // Test remove alias method
        let entity = create_test_entity_data(1);
        let key = arena.allocate_entity(entity.clone());
        let removed = arena.remove(key).unwrap();
        assert_eq!(removed.type_id, entity.type_id);
    }

    #[tokio::test]
    async fn test_epoch_manager_basic_operations() {
        const NUM_THREADS: usize = 4;
        let manager = EpochManager::new(NUM_THREADS);
        
        // Test initial state
        assert_eq!(manager.global_epoch.load(Ordering::Acquire), 0);
        assert_eq!(manager.thread_epochs.len(), NUM_THREADS);
        
        // Test entering epoch for different threads
        let guard0 = manager.enter(0);
        assert_eq!(manager.thread_epochs[0].load(Ordering::Acquire), 0);
        
        let guard1 = manager.enter(1);
        assert_eq!(manager.thread_epochs[1].load(Ordering::Acquire), 0);
        
        // Test advancing epoch
        manager.advance_epoch();
        assert_eq!(manager.global_epoch.load(Ordering::Acquire), 1);
        
        // Test new guard sees advanced epoch
        drop(guard0);
        drop(guard1);
        let guard2 = manager.enter(2);
        assert_eq!(manager.thread_epochs[2].load(Ordering::Acquire), 1);
        
        drop(guard2);
    }

    #[tokio::test]
    async fn test_epoch_manager_object_retirement() {
        const NUM_THREADS: usize = 2;
        let manager = EpochManager::new(NUM_THREADS);
        
        // Simulate memory allocation
        let layout = std::alloc::Layout::from_size_align(64, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        
        // Enter epoch and retire object
        let _guard = manager.enter(0);
        manager.retire_object(ptr, 64);
        
        // Check retired objects list
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 1);
            assert_eq!(retired[0].size, 64);
            assert_eq!(retired[0].retired_epoch, 0);
        }
        
        // Advance epoch to trigger garbage collection
        // Object should not be collected yet because thread 0 is still in epoch 0
        manager.advance_epoch();
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 1); // Object still present
        }
        
        // Drop guard to allow thread to exit epoch
        drop(_guard);
        
        // Enter new epoch for thread 0
        let _guard = manager.enter(0);
        
        // Advance epoch again - now object should be collected
        manager.advance_epoch();
        
        // Object should be deallocated (we can't directly verify deallocation,
        // but the retired objects list should be empty)
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 0);
        }
        
        drop(_guard);
    }

    #[tokio::test]
    async fn test_epoch_guard_drop_behavior() {
        const NUM_THREADS: usize = 2;
        let manager = EpochManager::new(NUM_THREADS);
        
        // Enter epoch
        let guard = manager.enter(0);
        assert_eq!(manager.thread_epochs[0].load(Ordering::Acquire), 0);
        
        // Drop guard should set thread epoch to MAX
        drop(guard);
        assert_eq!(manager.thread_epochs[0].load(Ordering::Acquire), u64::MAX);
    }

    #[tokio::test]
    async fn test_epoch_manager_concurrent_access() {
        const NUM_THREADS: usize = 8;
        const OPERATIONS_PER_THREAD: usize = 100;
        
        let manager = Arc::new(EpochManager::new(NUM_THREADS));
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let mut handles = Vec::new();
        
        for thread_id in 0..NUM_THREADS {
            let manager_clone = Arc::clone(&manager);
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for i in 0..OPERATIONS_PER_THREAD {
                    // Enter epoch
                    let _guard = manager_clone.enter(thread_id);
                    
                    // Simulate some work
                    thread::sleep(Duration::from_nanos(100));
                    
                    // Retire some objects periodically
                    if i % 10 == 0 {
                        let layout = std::alloc::Layout::from_size_align(32, 8).unwrap();
                        let ptr = unsafe { std::alloc::alloc(layout) };
                        if !ptr.is_null() {
                            manager_clone.retire_object(ptr, 32);
                        }
                    }
                    
                    // Advance epoch periodically
                    if thread_id == 0 && i % 5 == 0 {
                        manager_clone.advance_epoch();
                    }
                    
                    // Guard drops automatically at end of scope
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Final cleanup
        manager.advance_epoch();
        manager.advance_epoch();
        
        // All threads should be out of epochs
        for i in 0..NUM_THREADS {
            assert_eq!(manager.thread_epochs[i].load(Ordering::Acquire), u64::MAX);
        }
    }

    #[tokio::test]
    async fn test_epoch_manager_garbage_collection_edge_cases() {
        const NUM_THREADS: usize = 3;
        let manager = EpochManager::new(NUM_THREADS);
        
        // Test garbage collection with no active threads
        let layout = std::alloc::Layout::from_size_align(128, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        manager.retire_object(ptr, 128);
        
        // Advance epoch and collect garbage
        manager.advance_epoch();
        
        // All threads are at u64::MAX initially, so min_epoch should be 0
        // and object should be collected immediately
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 0);
        }
        
        // Test with mixed thread states
        let _guard0 = manager.enter(0); // Thread 0 in current epoch
        let _guard1 = manager.enter(1); // Thread 1 in current epoch
        // Thread 2 remains at u64::MAX
        
        let ptr2 = unsafe { std::alloc::alloc(layout) };
        manager.retire_object(ptr2, 128);
        
        manager.advance_epoch();
        
        // Object should not be collected because threads 0 and 1 are still active
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 1);
        }
        
        drop(_guard0);
        drop(_guard1);
        
        // Now all threads are inactive, object should be collected
        manager.advance_epoch();
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_graph_arena_concurrent_modifications() {
        let arena = Arc::new(std::sync::Mutex::new(GraphArena::new()));
        const NUM_THREADS: usize = 4;
        const ENTITIES_PER_THREAD: usize = 100;
        
        let mut handles = Vec::new();
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        
        for thread_id in 0..NUM_THREADS {
            let arena_clone = Arc::clone(&arena);
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                let mut keys = Vec::new();
                
                // Each thread allocates entities
                for i in 0..ENTITIES_PER_THREAD {
                    let entity = create_test_entity_data((thread_id * 1000 + i) as u16);
                    let key = {
                        let mut arena_guard = arena_clone.lock().unwrap();
                        arena_guard.allocate_entity(entity)
                    };
                    keys.push(key);
                }
                
                // Verify all allocated entities
                for (i, &key) in keys.iter().enumerate() {
                    let arena_guard = arena_clone.lock().unwrap();
                    let entity = arena_guard.get_entity(key).unwrap();
                    assert_eq!(entity.type_id, (thread_id * 1000 + i) as u16);
                }
                
                // Remove half the entities
                for i in (0..ENTITIES_PER_THREAD).step_by(2) {
                    let mut arena_guard = arena_clone.lock().unwrap();
                    assert!(arena_guard.remove_entity(keys[i]).is_some());
                }
                
                keys
            });
            
            handles.push(handle);
        }
        
        // Collect all keys from all threads
        let mut all_keys = Vec::new();
        for handle in handles {
            let keys = handle.join().unwrap();
            all_keys.extend(keys);
        }
        
        // Verify final state
        {
            let arena_guard = arena.lock().unwrap();
            assert_eq!(arena_guard.entity_count(), NUM_THREADS * ENTITIES_PER_THREAD / 2);
            
            // Verify memory tracking
            assert!(arena_guard.memory_usage() > 0);
            assert!(arena_guard.encoded_size() > 0);
        }
    }

    #[tokio::test]
    async fn test_retired_object_safety() {
        // Test that RetiredObject can be safely sent between threads
        let layout = std::alloc::Layout::from_size_align(64, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        
        let retired = RetiredObject {
            ptr,
            size: 64,
            retired_epoch: 1,
        };
        
        // Test Send trait
        let handle = thread::spawn(move || {
            assert_eq!(retired.size, 64);
            assert_eq!(retired.retired_epoch, 1);
            retired
        });
        
        let retrieved_retired = handle.join().unwrap();
        assert_eq!(retrieved_retired.size, 64);
        
        // Clean up
        unsafe {
            std::alloc::dealloc(retrieved_retired.ptr, layout);
        }
    }

    #[tokio::test]
    async fn test_epoch_manager_stress_test() {
        const NUM_THREADS: usize = 16;
        const STRESS_OPERATIONS: usize = 1000;
        
        let manager = Arc::new(EpochManager::new(NUM_THREADS));
        let mut handles = Vec::new();
        
        for thread_id in 0..NUM_THREADS {
            let manager_clone = Arc::clone(&manager);
            
            let handle = thread::spawn(move || {
                for i in 0..STRESS_OPERATIONS {
                    // Rapidly enter and exit epochs
                    {
                        let _guard = manager_clone.enter(thread_id);
                        
                        // Allocate and retire objects frequently
                        if i % 10 == 0 {
                            let layout = std::alloc::Layout::from_size_align(16, 8).unwrap();
                            let ptr = unsafe { std::alloc::alloc(layout) };
                            if !ptr.is_null() {
                                manager_clone.retire_object(ptr, 16);
                            }
                        }
                        
                        // Some threads advance epochs
                        if thread_id < 4 && i % 50 == 0 {
                            manager_clone.advance_epoch();
                        }
                    }
                    
                    // Brief pause to allow epoch transitions
                    if i % 100 == 0 {
                        thread::sleep(Duration::from_nanos(1000));
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Final cleanup cycles
        for _ in 0..10 {
            manager.advance_epoch();
            thread::sleep(Duration::from_millis(1));
        }
        
        // Verify final state
        assert!(manager.global_epoch.load(Ordering::Acquire) > 0);
        
        // Most retired objects should be cleaned up
        let final_retired_count = manager.retired_objects.read().len();
        assert!(final_retired_count < 100, "Too many objects remain: {}", final_retired_count);
    }

    #[test]
    fn test_graph_arena_mutable_access() {
        let mut arena = GraphArena::new();
        
        // Test mutable entity access
        let entity = create_test_entity_data(1);
        let key = arena.allocate_entity(entity);
        
        {
            let entity_mut = arena.get_entity_mut(key).unwrap();
            entity_mut.type_id = 999;
            entity_mut.properties = "modified".to_string();
            entity_mut.embedding = vec![1.0; 10];
        }
        
        // Verify changes
        let entity = arena.get_entity(key).unwrap();
        assert_eq!(entity.type_id, 999);
        assert_eq!(entity.properties, "modified");
        assert_eq!(entity.embedding.len(), 10);
        assert_eq!(entity.embedding[0], 1.0);
    }

    #[test]
    fn test_collect_garbage_private_method() {
        // Test the private collect_garbage method indirectly through advance_epoch
        const NUM_THREADS: usize = 2;
        let manager = EpochManager::new(NUM_THREADS);
        
        // Create objects in different epochs
        let layout = std::alloc::Layout::from_size_align(32, 8).unwrap();
        
        // Object 1 retired in epoch 0
        let ptr1 = unsafe { std::alloc::alloc(layout) };
        manager.retire_object(ptr1, 32);
        
        manager.advance_epoch(); // Now in epoch 1
        
        // Object 2 retired in epoch 1
        let ptr2 = unsafe { std::alloc::alloc(layout) };
        manager.retire_object(ptr2, 32);
        
        // Both objects should still exist
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 2);
        }
        
        // Enter epoch for one thread but not the other
        let _guard = manager.enter(0);
        // Thread 1 remains at u64::MAX
        
        manager.advance_epoch(); // Now in epoch 2
        
        // Objects should still exist because thread 0 is active in epoch 2
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 2);
        }
        
        drop(_guard); // Thread 0 exits epoch
        
        manager.advance_epoch(); // Now in epoch 3
        
        // Now objects should be collected because min_epoch > their retirement epochs
        {
            let retired = manager.retired_objects.read();
            assert_eq!(retired.len(), 0);
        }
    }
}