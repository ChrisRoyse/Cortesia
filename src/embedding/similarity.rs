use std::f32;

/// Auto-dispatching cosine similarity function that uses SIMD when available
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    // Use SIMD if available and beneficial
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 16 {
            return unsafe { simd::cosine_similarity_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") && a.len() >= 8 {
            return unsafe { simd::cosine_similarity_sse41(a, b) };
        }
    }
    
    cosine_similarity_scalar(a, b)
}

/// Scalar fallback implementation
#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}

/// Auto-dispatching euclidean distance function that uses SIMD when available
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    // Use SIMD if available and beneficial
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 16 {
            return unsafe { simd::euclidean_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") && a.len() >= 8 {
            return unsafe { simd::euclidean_distance_sse41(a, b) };
        }
    }
    
    euclidean_distance_scalar(a, b)
}

/// Scalar fallback implementation
#[inline]
pub fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
pub mod simd {
    use std::arch::x86_64::*;
    
    /// Optimized AVX2 cosine similarity with better horizontal summing
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 16 {
            return super::cosine_similarity_scalar(a, b);
        }
        
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        let chunks = a.len() / 8;
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        
        // Process 8 floats per iteration
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr_a.add(i * 8));
            let vb = _mm256_loadu_ps(ptr_b.add(i * 8));
            
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }
        
        // Efficient horizontal sum using hadd
        let dot_total = horizontal_sum_avx2(dot_sum);
        let norm_a_total = horizontal_sum_avx2(norm_a_sum);
        let norm_b_total = horizontal_sum_avx2(norm_b_sum);
        
        // Handle remainder elements
        let mut dot_remainder = dot_total;
        let mut norm_a_remainder = norm_a_total;
        let mut norm_b_remainder = norm_b_total;
        
        for i in (chunks * 8)..a.len() {
            let a_val = *a.get_unchecked(i);
            let b_val = *b.get_unchecked(i);
            dot_remainder += a_val * b_val;
            norm_a_remainder += a_val * a_val;
            norm_b_remainder += b_val * b_val;
        }
        
        let norm_a = norm_a_remainder.sqrt();
        let norm_b = norm_b_remainder.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_remainder / (norm_a * norm_b)
        }
    }
    
    /// SSE4.1 fallback implementation
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn cosine_similarity_sse41(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 8 {
            return super::cosine_similarity_scalar(a, b);
        }
        
        let mut dot_sum = _mm_setzero_ps();
        let mut norm_a_sum = _mm_setzero_ps();
        let mut norm_b_sum = _mm_setzero_ps();
        
        let chunks = a.len() / 4;
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        
        // Process 4 floats per iteration
        for i in 0..chunks {
            let va = _mm_loadu_ps(ptr_a.add(i * 4));
            let vb = _mm_loadu_ps(ptr_b.add(i * 4));
            
            dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
            norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
            norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
        }
        
        // Horizontal sum for SSE
        let dot_total = horizontal_sum_sse(dot_sum);
        let norm_a_total = horizontal_sum_sse(norm_a_sum);
        let norm_b_total = horizontal_sum_sse(norm_b_sum);
        
        // Handle remainder
        let mut dot_remainder = dot_total;
        let mut norm_a_remainder = norm_a_total;
        let mut norm_b_remainder = norm_b_total;
        
        for i in (chunks * 4)..a.len() {
            let a_val = *a.get_unchecked(i);
            let b_val = *b.get_unchecked(i);
            dot_remainder += a_val * b_val;
            norm_a_remainder += a_val * a_val;
            norm_b_remainder += b_val * b_val;
        }
        
        let norm_a = norm_a_remainder.sqrt();
        let norm_b = norm_b_remainder.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_remainder / (norm_a * norm_b)
        }
    }
    
    /// Optimized AVX2 euclidean distance
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 16 {
            return super::euclidean_distance_scalar(a, b);
        }
        
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ptr_a.add(i * 8));
            let vb = _mm256_loadu_ps(ptr_b.add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        let mut total = horizontal_sum_avx2(sum);
        
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += diff * diff;
        }
        
        total.sqrt()
    }
    
    /// SSE4.1 euclidean distance
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn euclidean_distance_sse41(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 8 {
            return super::euclidean_distance_scalar(a, b);
        }
        
        let mut sum = _mm_setzero_ps();
        let chunks = a.len() / 4;
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        
        for i in 0..chunks {
            let va = _mm_loadu_ps(ptr_a.add(i * 4));
            let vb = _mm_loadu_ps(ptr_b.add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        
        let mut total = horizontal_sum_sse(sum);
        
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += diff * diff;
        }
        
        total.sqrt()
    }
    
    /// Efficient horizontal sum for AVX2 vectors
    #[inline]
    unsafe fn horizontal_sum_avx2(vec: __m256) -> f32 {
        let high = _mm256_extractf128_ps(vec, 1);
        let low = _mm256_castps256_ps128(vec);
        let sum128 = _mm_add_ps(high, low);
        horizontal_sum_sse(sum128)
    }
    
    /// Efficient horizontal sum for SSE vectors
    #[inline]
    unsafe fn horizontal_sum_sse(vec: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(vec);
        let sums = _mm_add_ps(vec, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf);
        _mm_cvtss_f32(sums)
    }
    
    /// Batch compute multiple cosine similarities using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_cosine_similarity_avx2(
        query: &[f32], 
        embeddings: &[f32], 
        dimension: usize,
        results: &mut [f32]
    ) {
        let entity_count = embeddings.len() / dimension;
        assert!(results.len() >= entity_count);
        
        if dimension < 16 {
            // Fallback for small dimensions
            for i in 0..entity_count {
                let start = i * dimension;
                let end = start + dimension;
                let embedding = &embeddings[start..end];
                results[i] = super::cosine_similarity_scalar(query, embedding);
            }
            return;
        }
        
        for entity_idx in 0..entity_count {
            let embedding_start = entity_idx * dimension;
            let embedding = &embeddings[embedding_start..embedding_start + dimension];
            results[entity_idx] = cosine_similarity_avx2(query, embedding);
        }
    }
}