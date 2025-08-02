// Ultra-efficient serialization that avoids memory allocation and copying during read operations
// Designed for maximum performance with direct memory access patterns

use crate::core::types::{EntityData, Relationship};
#[cfg(test)]
use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};
use crate::storage::string_interner::StringInterner;
use std::mem;
use std::slice;
use std::sync::Arc;

/// Magic bytes for zero-copy format validation
const ZERO_COPY_MAGIC: [u8; 8] = *b"ZCLLMKG\0";
const ZERO_COPY_VERSION: u32 = 1;

/// Zero-copy serialization header for format validation and metadata
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ZeroCopyHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub entity_count: u32,
    pub relationship_count: u32,
    pub string_count: u32,
    pub total_size: u64,
    pub entity_section_offset: u64,
    pub relationship_section_offset: u64,
    pub string_section_offset: u64,
    pub checksum: u64,
}

impl Default for ZeroCopyHeader {
    fn default() -> Self {
        Self {
            magic: ZERO_COPY_MAGIC,
            version: ZERO_COPY_VERSION,
            entity_count: 0,
            relationship_count: 0,
            string_count: 0,
            total_size: 0,
            entity_section_offset: 0,
            relationship_section_offset: 0,
            string_section_offset: 0,
            checksum: 0,
        }
    }
}

/// Zero-copy entity representation optimized for direct memory access
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ZeroCopyEntity {
    pub id: u32,
    pub type_id: u16,
    pub degree: u16,
    pub embedding_offset: u32,
    pub property_offset: u32,
    pub property_size: u16,
    pub flags: u16,
    // Followed by variable-length data: embedding bytes, property bytes
}

/// Zero-copy relationship representation for efficient graph traversal
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ZeroCopyRelationship {
    pub from: u32,
    pub to: u32,
    pub rel_type: u8,
    pub flags: u8,
    pub weight: f32,
    pub timestamp: u64,
}

/// Zero-copy string entry for efficient string storage
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ZeroCopyString {
    pub id: u32,
    pub length: u16,
    pub hash: u32,
    // Followed by UTF-8 string bytes
}

/// Zero-copy deserializer for direct memory access without allocation
pub struct ZeroCopyDeserializer<'a> {
    data: &'a [u8],
    header: &'a ZeroCopyHeader,
    entity_section: &'a [u8],
    relationship_section: &'a [u8],
    string_section: &'a [u8],
}

impl<'a> ZeroCopyDeserializer<'a> {
    /// Create a zero-copy deserializer from raw bytes
    /// This performs no memory allocation and provides direct access to the data
    pub unsafe fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < mem::size_of::<ZeroCopyHeader>() {
            return Err(GraphError::InvalidData("Buffer too small for header".into()));
        }

        // Cast raw bytes to header - zero allocation
        let header_ptr = data.as_ptr() as *const ZeroCopyHeader;
        let header = &*header_ptr;

        // Validate magic bytes
        if header.magic != ZERO_COPY_MAGIC {
            return Err(GraphError::InvalidData("Invalid magic bytes".into()));
        }

        if header.version != ZERO_COPY_VERSION {
            return Err(GraphError::InvalidData("Unsupported version".into()));
        }

        if data.len() < header.total_size as usize {
            return Err(GraphError::InvalidData("Buffer smaller than declared size".into()));
        }

        // Calculate section pointers - all zero-copy
        let entity_section = &data[header.entity_section_offset as usize..
                                  (header.entity_section_offset + 
                                   header.entity_count as u64 * mem::size_of::<ZeroCopyEntity>() as u64) as usize];

        let relationship_section = &data[header.relationship_section_offset as usize..
                                        (header.relationship_section_offset + 
                                         header.relationship_count as u64 * mem::size_of::<ZeroCopyRelationship>() as u64) as usize];

        let string_section = &data[header.string_section_offset as usize..];

        Ok(Self {
            data,
            header,
            entity_section,
            relationship_section,
            string_section,
        })
    }

    /// Get entity count without any allocation
    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.header.entity_count
    }

    /// Get relationship count without any allocation
    #[inline]
    pub fn relationship_count(&self) -> u32 {
        self.header.relationship_count
    }

    /// Get entity at index with zero-copy access
    /// Returns a direct reference to the data in the buffer
    #[inline]
    pub fn get_entity(&self, index: u32) -> Option<&ZeroCopyEntity> {
        if index >= self.header.entity_count {
            return None;
        }

        unsafe {
            let entity_ptr = self.entity_section.as_ptr()
                .add(index as usize * mem::size_of::<ZeroCopyEntity>()) as *const ZeroCopyEntity;
            Some(&*entity_ptr)
        }
    }

    /// Get relationship at index with zero-copy access
    #[inline]
    pub fn get_relationship(&self, index: u32) -> Option<&ZeroCopyRelationship> {
        if index >= self.header.relationship_count {
            return None;
        }

        unsafe {
            let rel_ptr = self.relationship_section.as_ptr()
                .add(index as usize * mem::size_of::<ZeroCopyRelationship>()) as *const ZeroCopyRelationship;
            Some(&*rel_ptr)
        }
    }

    /// Get entity embedding as a direct slice reference - zero allocation
    #[inline]
    pub fn get_entity_embedding(&self, entity: &ZeroCopyEntity, embedding_dim: usize) -> &[u8] {
        let start = entity.embedding_offset as usize;
        let end = start + embedding_dim / 8; // Assuming quantized to 1 byte per dimension
        &self.data[start..end]
    }

    /// Get entity properties as a direct string slice - zero allocation
    #[inline]
    pub fn get_entity_properties(&self, entity: &ZeroCopyEntity) -> &str {
        let start = entity.property_offset as usize;
        let end = start + entity.property_size as usize;
        let bytes = &self.data[start..end];
        // Safe because we validate UTF-8 during serialization
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Iterator over all entities with zero allocation
    pub fn iter_entities(&self) -> ZeroCopyEntityIter {
        ZeroCopyEntityIter {
            deserializer: self,
            current: 0,
        }
    }

    /// Iterator over all relationships with zero allocation
    pub fn iter_relationships(&self) -> ZeroCopyRelationshipIter {
        ZeroCopyRelationshipIter {
            deserializer: self,
            current: 0,
        }
    }
}

/// Zero-allocation iterator over entities
pub struct ZeroCopyEntityIter<'a> {
    deserializer: &'a ZeroCopyDeserializer<'a>,
    current: u32,
}

impl<'a> Iterator for ZeroCopyEntityIter<'a> {
    type Item = &'a ZeroCopyEntity;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.deserializer.entity_count() {
            return None;
        }

        let entity = self.deserializer.get_entity(self.current)?;
        self.current += 1;
        Some(entity)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.deserializer.entity_count() - self.current) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ZeroCopyEntityIter<'a> {}

/// Zero-allocation iterator over relationships
pub struct ZeroCopyRelationshipIter<'a> {
    deserializer: &'a ZeroCopyDeserializer<'a>,
    current: u32,
}

impl<'a> Iterator for ZeroCopyRelationshipIter<'a> {
    type Item = &'a ZeroCopyRelationship;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.deserializer.relationship_count() {
            return None;
        }

        let relationship = self.deserializer.get_relationship(self.current)?;
        self.current += 1;
        Some(relationship)
    }
}

/// Zero-copy serializer for efficient data writing
pub struct ZeroCopySerializer {
    buffer: Vec<u8>,
    header: ZeroCopyHeader,
    entity_buffer: Vec<u8>,
    relationship_buffer: Vec<u8>,
    string_buffer: Vec<u8>,
    variable_data_buffer: Vec<u8>,
}

impl Default for ZeroCopySerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCopySerializer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            header: ZeroCopyHeader::default(),
            entity_buffer: Vec::new(),
            relationship_buffer: Vec::new(),
            string_buffer: Vec::new(),
            variable_data_buffer: Vec::new(),
        }
    }

    /// Add entity to the serialization buffer
    pub fn add_entity(&mut self, entity: &EntityData, embedding_dim: usize) -> Result<()> {
        let property_bytes = entity.properties.as_bytes();
        let property_offset = self.variable_data_buffer.len() as u32;
        
        // Store property data in variable buffer
        self.variable_data_buffer.extend_from_slice(property_bytes);
        
        // Store quantized embedding (assuming 1 byte per dimension for this example)
        let embedding_offset = self.variable_data_buffer.len() as u32;
        let quantized_embedding = self.quantize_embedding(&entity.embedding, embedding_dim)?;
        self.variable_data_buffer.extend_from_slice(&quantized_embedding);

        // Create zero-copy entity entry
        let zero_copy_entity = ZeroCopyEntity {
            id: self.header.entity_count,
            type_id: entity.type_id,
            degree: 0, // Will be updated when relationships are added
            embedding_offset,
            property_offset,
            property_size: property_bytes.len() as u16,
            flags: 0,
        };

        // Serialize entity to buffer
        unsafe {
            let entity_bytes = slice::from_raw_parts(
                &zero_copy_entity as *const ZeroCopyEntity as *const u8,
                mem::size_of::<ZeroCopyEntity>()
            );
            self.entity_buffer.extend_from_slice(entity_bytes);
        }

        self.header.entity_count += 1;
        Ok(())
    }

    /// Add relationship to the serialization buffer
    pub fn add_relationship(&mut self, relationship: &Relationship) -> Result<()> {
        // Convert EntityKey to u32 for storage
        use slotmap::{Key, KeyData};
        let from_data: KeyData = relationship.from.data();
        let to_data: KeyData = relationship.to.data();
        
        let zero_copy_rel = ZeroCopyRelationship {
            from: from_data.as_ffi() as u32,
            to: to_data.as_ffi() as u32,
            rel_type: relationship.rel_type,
            flags: 0,
            weight: relationship.weight,
            timestamp: 0, // Current timestamp could be added here
        };

        unsafe {
            let rel_bytes = slice::from_raw_parts(
                &zero_copy_rel as *const ZeroCopyRelationship as *const u8,
                mem::size_of::<ZeroCopyRelationship>()
            );
            self.relationship_buffer.extend_from_slice(rel_bytes);
        }

        self.header.relationship_count += 1;
        Ok(())
    }

    /// Finalize serialization and return the complete buffer
    pub fn finalize(mut self) -> Result<Vec<u8>> {
        // Calculate section offsets
        let header_size = mem::size_of::<ZeroCopyHeader>() as u64;
        
        self.header.entity_section_offset = header_size;
        self.header.relationship_section_offset = self.header.entity_section_offset + self.entity_buffer.len() as u64;
        self.header.string_section_offset = self.header.relationship_section_offset + self.relationship_buffer.len() as u64;
        self.header.total_size = self.header.string_section_offset + self.string_buffer.len() as u64 + self.variable_data_buffer.len() as u64;

        // Calculate checksum
        self.header.checksum = self.calculate_checksum();

        // Build final buffer
        self.buffer.clear();
        
        // Write header
        unsafe {
            let header_bytes = slice::from_raw_parts(
                &self.header as *const ZeroCopyHeader as *const u8,
                mem::size_of::<ZeroCopyHeader>()
            );
            self.buffer.extend_from_slice(header_bytes);
        }

        // Write sections
        self.buffer.extend_from_slice(&self.entity_buffer);
        self.buffer.extend_from_slice(&self.relationship_buffer);
        self.buffer.extend_from_slice(&self.string_buffer);
        self.buffer.extend_from_slice(&self.variable_data_buffer);

        Ok(self.buffer)
    }

    // Helper method to quantize embedding (simplified for this example)
    fn quantize_embedding(&self, embedding: &[f32], dim: usize) -> Result<Vec<u8>> {
        // Simple quantization: convert f32 to u8 by scaling and clamping
        // In practice, this would use the existing ProductQuantizer
        let mut quantized = Vec::with_capacity(dim / 8);
        for chunk in embedding.chunks(8) {
            let mut byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                let quantized_value = ((value + 1.0) * 127.5) as u8;
                if i < 8 {
                    byte |= (quantized_value >> 7) << i;
                }
            }
            quantized.push(byte);
        }
        Ok(quantized)
    }

    fn calculate_checksum(&self) -> u64 {
        // Simple FNV-1a hash for data integrity
        let mut hash = 0xcbf29ce484222325u64;
        for byte in &self.entity_buffer {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for byte in &self.relationship_buffer {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// Zero-copy graph storage that integrates with existing systems
pub struct ZeroCopyGraphStorage {
    data: Vec<u8>,
    deserializer: Option<ZeroCopyDeserializer<'static>>,
    string_interner: Arc<StringInterner>,
}

impl ZeroCopyGraphStorage {
    /// Create new zero-copy storage from serialized data
    pub fn from_data(data: Vec<u8>, string_interner: Arc<StringInterner>) -> Result<Self> {
        // Safety: We're storing the data in the struct, so the lifetime is valid
        let deserializer = unsafe {
            let data_ref: &'static [u8] = std::mem::transmute(data.as_slice());
            Some(ZeroCopyDeserializer::new(data_ref)?)
        };

        Ok(Self {
            data,
            deserializer,
            string_interner,
        })
    }

    /// Get entity with zero-copy access
    #[inline]
    pub fn get_entity(&self, entity_id: u32) -> Option<&ZeroCopyEntity> {
        self.deserializer.as_ref()?.get_entity(entity_id)
    }

    /// Get entity embedding as direct slice - zero allocation
    #[inline]
    pub fn get_entity_embedding(&self, entity: &ZeroCopyEntity, embedding_dim: usize) -> &[u8] {
        self.deserializer.as_ref().unwrap().get_entity_embedding(entity, embedding_dim)
    }

    /// Get entity properties as direct string - zero allocation
    #[inline]
    pub fn get_entity_properties(&self, entity: &ZeroCopyEntity) -> &str {
        self.deserializer.as_ref().unwrap().get_entity_properties(entity)
    }

    /// Fast entity iteration with zero allocation
    pub fn iter_entities(&self) -> impl ExactSizeIterator<Item = &ZeroCopyEntity> {
        self.deserializer.as_ref().unwrap().iter_entities()
    }

    /// Memory usage statistics
    pub fn memory_usage(&self) -> usize {
        self.data.len()
    }

    /// Entity count
    pub fn entity_count(&self) -> u32 {
        self.deserializer.as_ref().map(|d| d.entity_count()).unwrap_or(0)
    }
}

/// Performance metrics for zero-copy operations
#[derive(Debug, Clone)]
pub struct ZeroCopyMetrics {
    pub serialization_time_ns: u64,
    pub deserialization_time_ns: u64,
    pub memory_usage_bytes: u64,
    pub entities_processed: u32,
    pub relationships_processed: u32,
    pub compression_ratio: f32,
}

impl Default for ZeroCopyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCopyMetrics {
    pub fn new() -> Self {
        Self {
            serialization_time_ns: 0,
            deserialization_time_ns: 0,
            memory_usage_bytes: 0,
            entities_processed: 0,
            relationships_processed: 0,
            compression_ratio: 1.0,
        }
    }

    pub fn throughput_entities_per_sec(&self) -> f64 {
        if self.serialization_time_ns == 0 {
            return 0.0;
        }
        (self.entities_processed as f64) / (self.serialization_time_ns as f64 / 1_000_000_000.0)
    }

    pub fn memory_efficiency_bytes_per_entity(&self) -> f64 {
        if self.entities_processed == 0 {
            return 0.0;
        }
        (self.memory_usage_bytes as f64) / (self.entities_processed as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityData;

    #[test]
    fn test_zero_copy_serialization_roundtrip() {
        let mut serializer = ZeroCopySerializer::new();
        
        // Add test entity
        let entity = EntityData {
            type_id: 1,
            properties: "test entity".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        };
        
        serializer.add_entity(&entity, 4).unwrap();
        
        // Add test relationship
        let relationship = Relationship {
            from: EntityKey::new("from_entity".to_string()),
            to: EntityKey::new("to_entity".to_string()),
            rel_type: 1,
            weight: 0.5,
        };
        
        serializer.add_relationship(&relationship).unwrap();
        
        // Finalize
        let data = serializer.finalize().unwrap();
        
        // Deserialize with zero-copy
        let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
        
        assert_eq!(deserializer.entity_count(), 1);
        assert_eq!(deserializer.relationship_count(), 1);
        
        let entity = deserializer.get_entity(0).unwrap();
        let type_id = entity.type_id; // Avoid packed field reference
        assert_eq!(type_id, 1);
        
        let properties = deserializer.get_entity_properties(entity);
        assert_eq!(properties, "test entity");
    }

    #[test]
    fn test_zero_copy_iterator_performance() {
        let mut serializer = ZeroCopySerializer::new();
        
        // Add multiple entities
        for i in 0..1000 {
            let entity = EntityData {
                type_id: i as u16,
                properties: format!("entity_{i}"),
                embedding: vec![i as f32; 96],
            };
            serializer.add_entity(&entity, 96).unwrap();
        }
        
        let data = serializer.finalize().unwrap();
        let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
        
        // Test iteration performance
        let start = std::time::Instant::now();
        let count = deserializer.iter_entities().count();
        let duration = start.elapsed();
        
        assert_eq!(count, 1000);
        // Should be very fast due to zero-copy access
        assert!(duration.as_millis() < 10);
    }
}