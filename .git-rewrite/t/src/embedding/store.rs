use crate::embedding::quantizer::ProductQuantizer;
use crate::error::{GraphError, Result};
use std::sync::Arc;
use parking_lot::RwLock;

pub struct EmbeddingStore {
    quantizer: Arc<RwLock<ProductQuantizer>>,
    quantized_bank: Vec<u8>,
    dimension: usize,
    subvector_count: usize,
}

impl EmbeddingStore {
    pub fn new(dimension: usize, subvector_count: usize) -> Result<Self> {
        Ok(Self {
            quantizer: Arc::new(RwLock::new(ProductQuantizer::new(dimension, subvector_count)?)),
            quantized_bank: Vec::new(),
            dimension,
            subvector_count,
        })
    }
    
    pub fn store_embedding(&mut self, embedding: &[f32]) -> Result<u32> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        let quantizer = self.quantizer.read();
        let quantized = quantizer.encode(embedding)?;
        
        let offset = self.quantized_bank.len() as u32;
        self.quantized_bank.extend_from_slice(&quantized);
        
        Ok(offset)
    }
    
    pub fn get_embedding(&self, offset: u32) -> Result<Vec<f32>> {
        let start = offset as usize;
        let end = start + self.subvector_count;
        
        if end > self.quantized_bank.len() {
            return Err(GraphError::IndexCorruption);
        }
        
        let codes = &self.quantized_bank[start..end];
        let quantizer = self.quantizer.read();
        quantizer.decode(codes)
    }
    
    pub fn asymmetric_distance(&self, query: &[f32], offset: u32) -> Result<f32> {
        let start = offset as usize;
        let end = start + self.subvector_count;
        
        if end > self.quantized_bank.len() {
            return Err(GraphError::IndexCorruption);
        }
        
        let codes = &self.quantized_bank[start..end];
        let quantizer = self.quantizer.read();
        quantizer.asymmetric_distance(query, codes)
    }
    
    pub fn memory_usage(&self) -> usize {
        self.quantized_bank.capacity() + 
        self.quantizer.read().memory_usage()
    }
}