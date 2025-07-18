use std::f32;

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
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

#[cfg(target_feature = "avx2")]
pub mod simd {
    use std::arch::x86_64::*;
    
    #[inline]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 8 {
            return super::cosine_similarity(a, b);
        }
        
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }
        
        // Horizontal sum
        let mut dot_result = [0.0f32; 8];
        let mut norm_a_result = [0.0f32; 8];
        let mut norm_b_result = [0.0f32; 8];
        
        _mm256_storeu_ps(dot_result.as_mut_ptr(), dot_sum);
        _mm256_storeu_ps(norm_a_result.as_mut_ptr(), norm_a_sum);
        _mm256_storeu_ps(norm_b_result.as_mut_ptr(), norm_b_sum);
        
        let mut dot_total = dot_result.iter().sum::<f32>();
        let mut norm_a_total = norm_a_result.iter().sum::<f32>();
        let mut norm_b_total = norm_b_result.iter().sum::<f32>();
        
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            dot_total += a[i] * b[i];
            norm_a_total += a[i] * a[i];
            norm_b_total += b[i] * b[i];
        }
        
        let norm_a = norm_a_total.sqrt();
        let norm_b = norm_b_total.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_total / (norm_a * norm_b)
        }
    }
    
    #[inline]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.len() < 8 {
            return super::euclidean_distance(a, b);
        }
        
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        // Horizontal sum
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f32>();
        
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            let diff = a[i] - b[i];
            total += diff * diff;
        }
        
        total.sqrt()
    }
}