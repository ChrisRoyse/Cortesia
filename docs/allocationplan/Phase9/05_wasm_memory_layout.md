# Micro-Phase 9.05: WASM Memory-Efficient Data Structures

## Objective
Define and implement memory-efficient data structures optimized for WASM's constraints, focusing on minimal memory footprint and cache efficiency.

## Prerequisites
- Completed micro-phases 9.01-9.04 (core bindings ready)
- Understanding of WASM memory model

## Task Description
Create specialized data structures that minimize memory usage while maintaining performance, using compact representations and efficient layouts.

## Specific Actions

1. **Create compact column representation**:
   ```rust
   // src/memory/column.rs
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   #[repr(C, align(16))] // SIMD alignment
   pub struct WasmCorticalColumn {
       // Pack fields efficiently (4 + 4 + 4 + 4 = 16 bytes base)
       pub id: u32,
       pub allocated_concept_id: u32, // 0 = unallocated
       pub activation_level: f32,
       pub state: u8, // Enum packed into u8
       _padding: [u8; 3], // Explicit padding for alignment
   }
   
   #[wasm_bindgen]
   impl WasmCorticalColumn {
       pub fn new(id: u32) -> Self {
           Self {
               id,
               allocated_concept_id: 0,
               activation_level: 0.0,
               state: 0, // Available
               _padding: [0; 3],
           }
       }
       
       #[inline]
       pub fn is_allocated(&self) -> bool {
           self.allocated_concept_id != 0
       }
       
       #[inline]
       pub fn set_activation(&mut self, level: f32) {
           self.activation_level = level.clamp(0.0, 1.0);
       }
   }
   ```

2. **Create sparse connection matrix**:
   ```rust
   // src/memory/sparse.rs
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   pub struct WasmSparseConnections {
       // CSR format for sparse matrix
       row_pointers: Vec<u32>,    // n+1 elements
       column_indices: Vec<u32>,  // nnz elements
       values: Vec<f32>,          // nnz elements
       num_nodes: u32,
   }
   
   #[wasm_bindgen]
   impl WasmSparseConnections {
       pub fn new(num_nodes: u32) -> Self {
           Self {
               row_pointers: vec![0; (num_nodes + 1) as usize],
               column_indices: Vec::new(),
               values: Vec::new(),
               num_nodes,
           }
       }
       
       pub fn add_connection(&mut self, from: u32, to: u32, weight: f32) {
           // Efficient insertion for CSR format
           let insert_pos = self.find_insert_position(from, to);
           self.column_indices.insert(insert_pos, to);
           self.values.insert(insert_pos, weight);
           
           // Update row pointers
           for i in (from + 1)..=self.num_nodes {
               self.row_pointers[i as usize] += 1;
           }
       }
       
       fn find_insert_position(&self, row: u32, col: u32) -> usize {
           let start = self.row_pointers[row as usize] as usize;
           let end = self.row_pointers[(row + 1) as usize] as usize;
           
           self.column_indices[start..end]
               .binary_search(&col)
               .unwrap_or_else(|pos| start + pos)
       }
       
       pub fn get_connections(&self, node: u32) -> Vec<u32> {
           let start = self.row_pointers[node as usize] as usize;
           let end = self.row_pointers[(node + 1) as usize] as usize;
           self.column_indices[start..end].to_vec()
       }
   }
   ```

3. **Create compact concept storage**:
   ```rust
   // src/memory/concept.rs
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   pub struct CompactConcept {
       id: u32,
       // Store content hash instead of full text
       content_hash: u64,
       // Metadata packed into bitfield
       metadata: u32,
   }
   
   #[wasm_bindgen]
   impl CompactConcept {
       pub fn new(id: u32, content: &str) -> Self {
           Self {
               id,
               content_hash: Self::hash_content(content),
               metadata: 0,
           }
       }
       
       fn hash_content(content: &str) -> u64 {
           // Simple FNV-1a hash for demonstration
           let mut hash: u64 = 0xcbf29ce484222325;
           for byte in content.bytes() {
               hash ^= byte as u64;
               hash = hash.wrapping_mul(0x100000001b3);
           }
           hash
       }
       
       pub fn set_metadata(&mut self, complexity: u8, category: u8) {
           self.metadata = ((complexity as u32) << 8) | (category as u32);
       }
       
       pub fn get_complexity(&self) -> u8 {
           ((self.metadata >> 8) & 0xFF) as u8
       }
       
       pub fn get_category(&self) -> u8 {
           (self.metadata & 0xFF) as u8
       }
   }
   ```

4. **Create memory pool allocator**:
   ```rust
   // src/memory/pool.rs
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   pub struct MemoryPool<T> {
       pool: Vec<T>,
       free_list: Vec<usize>,
       capacity: usize,
   }
   
   impl<T: Default> MemoryPool<T> {
       pub fn new(capacity: usize) -> Self {
           let mut pool = Vec::with_capacity(capacity);
           let mut free_list = Vec::with_capacity(capacity);
           
           // Pre-allocate all objects
           for i in 0..capacity {
               pool.push(T::default());
               free_list.push(capacity - 1 - i);
           }
           
           Self {
               pool,
               free_list,
               capacity,
           }
       }
       
       pub fn allocate(&mut self) -> Option<(usize, &mut T)> {
           self.free_list.pop().map(|idx| {
               (idx, &mut self.pool[idx])
           })
       }
       
       pub fn deallocate(&mut self, idx: usize) {
           if idx < self.capacity {
               self.pool[idx] = T::default();
               self.free_list.push(idx);
           }
       }
       
       pub fn get(&self, idx: usize) -> Option<&T> {
           self.pool.get(idx)
       }
       
       pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
           self.pool.get_mut(idx)
       }
   }
   ```

5. **Create SIMD-aligned vectors**:
   ```rust
   // src/memory/simd_vec.rs
   use wasm_bindgen::prelude::*;
   use std::alloc::{alloc, dealloc, Layout};
   use std::ptr;
   
   #[wasm_bindgen]
   pub struct SimdVec<T> {
       ptr: *mut T,
       len: usize,
       capacity: usize,
   }
   
   impl<T> SimdVec<T> {
       const ALIGNMENT: usize = 16; // 128-bit SIMD alignment
       
       pub fn new(capacity: usize) -> Self {
           let layout = Layout::from_size_align(
               capacity * std::mem::size_of::<T>(),
               Self::ALIGNMENT
           ).unwrap();
           
           let ptr = unsafe { alloc(layout) as *mut T };
           
           Self {
               ptr,
               len: 0,
               capacity,
           }
       }
       
       pub fn push(&mut self, value: T) {
           if self.len < self.capacity {
               unsafe {
                   ptr::write(self.ptr.add(self.len), value);
               }
               self.len += 1;
           }
       }
       
       pub fn as_slice(&self) -> &[T] {
           unsafe {
               std::slice::from_raw_parts(self.ptr, self.len)
           }
       }
   }
   
   impl<T> Drop for SimdVec<T> {
       fn drop(&mut self) {
           let layout = Layout::from_size_align(
               self.capacity * std::mem::size_of::<T>(),
               Self::ALIGNMENT
           ).unwrap();
           
           unsafe {
               dealloc(self.ptr as *mut u8, layout);
           }
       }
   }
   ```

## Expected Outputs
- Memory-efficient column representation (16 bytes aligned)
- Sparse matrix in CSR format for connections
- Compact concept storage with hashing
- Memory pool allocator for object reuse
- SIMD-aligned vector implementation

## Validation
1. Check struct sizes: `std::mem::size_of::<WasmCorticalColumn>() == 16`
2. Memory pools allocate/deallocate correctly
3. Sparse matrix operations maintain CSR invariants
4. SIMD alignment verified (16-byte boundaries)

## Next Steps
- Implement CortexKGWasm struct (micro-phase 9.06)
- Port allocation methods (micro-phase 9.07)