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