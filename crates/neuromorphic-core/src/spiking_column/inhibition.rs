//! Lateral inhibition network for winner-take-all dynamics

use super::{ColumnId, InhibitoryWeight, SpikingCorticalColumn};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::time::{Duration, Instant};

/// Configuration for lateral inhibition
#[derive(Debug, Clone)]
pub struct InhibitionConfig {
    /// Base inhibition strength
    pub base_strength: InhibitoryWeight,
    
    /// Spatial decay factor (0.0 = no decay, 1.0 = immediate decay)
    pub spatial_decay: f32,
    
    /// Inhibition radius (in grid units)
    pub radius: f32,
    
    /// Minimum activation difference for competition
    pub competition_threshold: f32,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self {
            base_strength: 0.8,
            spatial_decay: 0.3,
            radius: 3.0,
            competition_threshold: 0.05,
        }
    }
}

/// Result of winner-take-all competition
#[derive(Debug, Clone)]
pub struct CompetitionResult {
    pub winner_id: ColumnId,
    pub winner_activation: f32,
    pub suppressed_columns: Vec<ColumnId>,
    pub competition_time: Duration,
}

/// Lateral inhibition network managing column competition
pub struct LateralInhibitionNetwork {
    /// Configuration parameters
    config: InhibitionConfig,
    
    /// Spatial positions of columns (for distance-based inhibition)
    column_positions: DashMap<ColumnId, (f32, f32, f32)>,
    
    /// Active inhibition signals
    inhibition_signals: DashMap<ColumnId, InhibitoryWeight>,
    
    /// Competition history for analysis
    competition_history: RwLock<Vec<CompetitionResult>>,
}

impl LateralInhibitionNetwork {
    /// Create new inhibition network
    pub fn new(config: InhibitionConfig) -> Self {
        Self {
            config,
            column_positions: DashMap::new(),
            inhibition_signals: DashMap::new(),
            competition_history: RwLock::new(Vec::with_capacity(1000)),
        }
    }
    
    /// Register a column with its spatial position
    pub fn register_column(&self, id: ColumnId, position: (f32, f32, f32)) {
        self.column_positions.insert(id, position);
    }
    
    /// Run winner-take-all competition
    pub fn compete(&self, candidates: Vec<(ColumnId, f32)>) -> CompetitionResult {
        let start = Instant::now();
        
        // Find column with highest activation
        let winner = candidates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("No candidates for competition");
        
        // Apply lateral inhibition to other columns
        let mut suppressed = Vec::new();
        
        for (id, activation) in &candidates {
            if *id != winner.0 {
                // Calculate inhibition strength based on:
                // 1. Activation difference
                // 2. Spatial distance (if positions known)
                
                let activation_diff = winner.1 - activation;
                let mut inhibition_strength: InhibitoryWeight = self.config.base_strength;
                
                // Apply spatial decay if positions are known
                if let (Some(winner_pos), Some(target_pos)) = 
                    (self.column_positions.get(&winner.0), self.column_positions.get(id)) {
                    
                    let distance = Self::calculate_distance(*winner_pos, *target_pos);
                    
                    if distance <= self.config.radius {
                        // Decay inhibition with distance
                        let decay = 1.0 - (distance / self.config.radius) * self.config.spatial_decay;
                        inhibition_strength *= decay;
                    } else {
                        // Outside inhibition radius
                        inhibition_strength = 0.0;
                    }
                }
                
                // Only inhibit if activation difference is significant
                if activation_diff > self.config.competition_threshold && inhibition_strength > 0.0 {
                    self.apply_inhibition(*id, inhibition_strength);
                    suppressed.push(*id);
                }
            }
        }
        
        let result = CompetitionResult {
            winner_id: winner.0,
            winner_activation: winner.1,
            suppressed_columns: suppressed,
            competition_time: start.elapsed(),
        };
        
        // Record in history
        self.competition_history.write().push(result.clone());
        
        result
    }
    
    /// Apply inhibition to a specific column
    pub fn apply_inhibition(&self, column_id: ColumnId, strength: InhibitoryWeight) {
        let clamped_strength = strength.clamp(0.0, 1.0);
        
        self.inhibition_signals
            .entry(column_id)
            .and_modify(|s| *s = (*s + clamped_strength).min(1.0))
            .or_insert(clamped_strength);
    }
    
    /// Get current inhibition level for a column
    pub fn get_inhibition(&self, column_id: ColumnId) -> InhibitoryWeight {
        self.inhibition_signals.get(&column_id)
            .map(|s| *s.value())
            .unwrap_or(0.0)
    }
    
    /// Clear inhibition signal for a column
    pub fn clear_inhibition(&self, column_id: ColumnId) {
        self.inhibition_signals.remove(&column_id);
    }
    
    /// Clear all inhibition signals
    pub fn clear_all_inhibition(&self) {
        self.inhibition_signals.clear();
    }
    
    /// Calculate 3D Euclidean distance
    fn calculate_distance(pos1: (f32, f32, f32), pos2: (f32, f32, f32)) -> f32 {
        let dx = pos1.0 - pos2.0;
        let dy = pos1.1 - pos2.1;
        let dz = pos1.2 - pos2.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Get competition statistics
    pub fn get_stats(&self) -> InhibitionStats {
        let history = self.competition_history.read();
        
        if history.is_empty() {
            return InhibitionStats::default();
        }
        
        let total_competitions = history.len();
        let avg_suppressed = history.iter()
            .map(|r| r.suppressed_columns.len())
            .sum::<usize>() as f32 / total_competitions as f32;
        
        let avg_competition_time = history.iter()
            .map(|r| r.competition_time.as_micros() as f32)
            .sum::<f32>() / total_competitions as f32;
        
        InhibitionStats {
            total_competitions,
            average_suppressed_columns: avg_suppressed,
            average_competition_time_us: avg_competition_time,
        }
    }
}

/// Statistics about inhibition network performance
#[derive(Debug, Default)]
pub struct InhibitionStats {
    pub total_competitions: usize,
    pub average_suppressed_columns: f32,
    pub average_competition_time_us: f32,
}

/// Integration with SpikingCorticalColumn
impl SpikingCorticalColumn {
    /// Check if column is currently inhibited
    pub fn is_inhibited_by(&self, network: &LateralInhibitionNetwork) -> bool {
        network.get_inhibition(self.id()) > 0.5
    }
    
    /// Apply inhibition from network
    pub fn apply_network_inhibition(&self, network: &LateralInhibitionNetwork) {
        let inhibition = network.get_inhibition(self.id());
        if inhibition > 0.0 {
            // Reduce activation based on inhibition
            let current = self.activation_level();
            let suppressed = current * (1.0 - inhibition);
            self.activation.set_activation(suppressed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_winner_take_all() {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        // Register columns with positions
        network.register_column(1, (0.0, 0.0, 0.0));
        network.register_column(2, (1.0, 0.0, 0.0));
        network.register_column(3, (2.0, 0.0, 0.0));
        
        // Run competition
        let candidates = vec![
            (1, 0.6),
            (2, 0.9), // Should win
            (3, 0.7),
        ];
        
        let result = network.compete(candidates);
        
        assert_eq!(result.winner_id, 2);
        assert_eq!(result.winner_activation, 0.9);
        assert!(result.suppressed_columns.contains(&1));
        assert!(result.suppressed_columns.contains(&3));
    }
    
    #[test]
    fn test_spatial_inhibition() {
        let config = InhibitionConfig {
            radius: 2.0,
            spatial_decay: 0.5,
            ..Default::default()
        };
        
        let network = LateralInhibitionNetwork::new(config);
        
        // Columns at different distances
        network.register_column(1, (0.0, 0.0, 0.0));
        network.register_column(2, (1.0, 0.0, 0.0)); // Close
        network.register_column(3, (3.0, 0.0, 0.0)); // Far
        
        let candidates = vec![
            (1, 0.9), // Winner
            (2, 0.8), // Should be inhibited strongly
            (3, 0.8), // Should be inhibited weakly or not at all
        ];
        
        let _result = network.compete(candidates);
        
        // Check inhibition levels
        let inhibition_2 = network.get_inhibition(2);
        let inhibition_3 = network.get_inhibition(3);
        
        assert!(inhibition_2 > inhibition_3);
        assert!(inhibition_3 < 0.1); // Outside radius
    }
    
    #[test]
    fn test_performance() {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        // Register 100 columns
        for i in 0..100 {
            let x = (i % 10) as f32;
            let y = (i / 10) as f32;
            network.register_column(i, (x, y, 0.0));
        }
        
        // Create candidates with random activations
        let candidates: Vec<_> = (0..100)
            .map(|i| (i as ColumnId, 0.5 + (i as f32 * 0.001)))
            .collect();
        
        // Measure competition time
        let start = Instant::now();
        let result = network.compete(candidates);
        let elapsed = start.elapsed();
        
        // Should complete in under 2ms
        assert!(elapsed < Duration::from_millis(2));
        assert_eq!(result.winner_id, 99); // Highest activation
    }
    
    #[test]
    fn test_inhibition_clearing() {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        network.apply_inhibition(1, 0.8);
        network.apply_inhibition(2, 0.6);
        
        assert_eq!(network.get_inhibition(1), 0.8);
        assert_eq!(network.get_inhibition(2), 0.6);
        
        network.clear_inhibition(1);
        assert_eq!(network.get_inhibition(1), 0.0);
        assert_eq!(network.get_inhibition(2), 0.6);
        
        network.clear_all_inhibition();
        assert_eq!(network.get_inhibition(2), 0.0);
    }
}