use std::collections::HashSet;
use ahash::AHashMap;
use crate::error::Result;
use crate::core::sdr_types::{SDR, SDRPattern};

/// Similarity index for fast SDR pattern matching
pub(crate) struct SimilarityIndex {
    pattern_bits: AHashMap<String, HashSet<usize>>, // pattern_id -> active bits
    bit_to_patterns: AHashMap<usize, HashSet<String>>, // bit -> pattern_ids
}

impl SimilarityIndex {
    pub fn new() -> Self {
        Self {
            pattern_bits: AHashMap::new(),
            bit_to_patterns: AHashMap::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: &SDRPattern) -> Result<()> {
        let pattern_id = pattern.pattern_id.clone();
        let active_bits = pattern.sdr.active_bits.clone();

        // Store pattern bits
        self.pattern_bits.insert(pattern_id.clone(), active_bits.clone());

        // Update inverted index
        for bit in active_bits {
            self.bit_to_patterns
                .entry(bit)
                .or_default()
                .insert(pattern_id.clone());
        }

        Ok(())
    }

    pub fn remove_patterns(&mut self, pattern_ids: &HashSet<String>) -> Result<()> {
        for pattern_id in pattern_ids {
            if let Some(active_bits) = self.pattern_bits.remove(pattern_id) {
                // Remove from inverted index
                for bit in active_bits {
                    if let Some(patterns) = self.bit_to_patterns.get_mut(&bit) {
                        patterns.remove(pattern_id);
                        if patterns.is_empty() {
                            self.bit_to_patterns.remove(&bit);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn find_similar(
        &self,
        query_sdr: &SDR,
        max_results: usize,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let mut candidate_scores = AHashMap::new();

        // Find candidate patterns using inverted index
        for &bit in &query_sdr.active_bits {
            if let Some(patterns) = self.bit_to_patterns.get(&bit) {
                for pattern_id in patterns {
                    *candidate_scores.entry(pattern_id.clone()).or_insert(0) += 1;
                }
            }
        }

        // Calculate actual similarities for candidates
        let mut similarities = Vec::new();
        
        for (pattern_id, _overlap_count) in candidate_scores {
            if let Some(pattern_bits) = self.pattern_bits.get(&pattern_id) {
                let candidate_sdr = SDR::new(pattern_bits.clone(), query_sdr.total_bits);
                let similarity = query_sdr.jaccard_similarity(&candidate_sdr);
                
                if similarity >= threshold {
                    similarities.push((pattern_id, similarity));
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(max_results);

        Ok(similarities)
    }
}