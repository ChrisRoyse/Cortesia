use serde::{Serialize, Deserialize};

/// Comprehensive memory information for monitoring and health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: u64,
    pub used: u64,
    pub available: u64,
    pub usage_percent: f64,
    pub swap_total: u64,
    pub swap_free: u64,
}

impl MemoryInfo {
    /// Create a new MemoryInfo with only basic fields
    pub fn basic(total: u64, used: u64, available: u64, usage_percent: f64) -> Self {
        Self {
            total,
            used,
            available,
            usage_percent,
            swap_total: 0,
            swap_free: 0,
        }
    }
    
    /// Create a new MemoryInfo with swap information
    pub fn with_swap(total: u64, available: u64, swap_total: u64, swap_free: u64) -> Self {
        let used = total.saturating_sub(available);
        let usage_percent = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };
        
        Self {
            total,
            used,
            available,
            usage_percent,
            swap_total,
            swap_free,
        }
    }
}