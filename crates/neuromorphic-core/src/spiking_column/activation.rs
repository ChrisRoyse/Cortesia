//! Biologically-inspired activation dynamics for spiking neurons

use std::sync::atomic::{AtomicU32, Ordering};
#[allow(unused_imports)]
use std::sync::RwLock;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

/// Default membrane time constant (tau) in milliseconds
pub const DEFAULT_TAU_MS: f32 = 100.0;

/// Default spike threshold
pub const DEFAULT_SPIKE_THRESHOLD: f32 = 0.7;

/// Minimum refractory period in milliseconds
pub const MIN_REFRACTORY_MS: f32 = 2.0;

/// Activation dynamics for a spiking cortical column
pub struct ActivationDynamics {
    /// Current activation level (0.0 to 1.0) stored as u32 bits
    level: AtomicU32,

    /// Last update timestamp
    last_update: Mutex<Instant>,

    /// Membrane time constant (tau) in milliseconds
    tau_ms: f32,

    /// Spike generation threshold
    spike_threshold: f32,

    /// Spike history for refractory period
    last_spike: Mutex<Option<Instant>>,
}

impl ActivationDynamics {
    /// Create new activation dynamics
    pub fn new() -> Self {
        Self::with_parameters(DEFAULT_TAU_MS, DEFAULT_SPIKE_THRESHOLD)
    }

    /// Create with custom parameters
    pub fn with_parameters(tau_ms: f32, spike_threshold: f32) -> Self {
        Self {
            level: AtomicU32::new(0),
            last_update: Mutex::new(Instant::now()),
            tau_ms,
            spike_threshold,
            last_spike: Mutex::new(None),
        }
    }

    /// Set activation level directly
    pub fn set_activation(&self, level: f32) {
        #[cfg(debug_assertions)]
        if !(0.0..=1.0).contains(&level) {
            eprintln!("Warning: Activation out of bounds: {}, clamping to [0, 1]", level);
        }
        let clamped = level.clamp(0.0, 1.0);
        self.level.store(clamped.to_bits(), Ordering::Release);
        *self.last_update.lock() = Instant::now();
    }

    /// Get current activation level with decay applied
    pub fn get_activation(&self) -> f32 {
        self.apply_decay();
        f32::from_bits(self.level.load(Ordering::Acquire))
    }

    /// Apply exponential decay based on elapsed time
    fn apply_decay(&self) {
        let now = Instant::now();
        let mut last_update = self.last_update.lock();
        let dt_ms = now.duration_since(*last_update).as_secs_f32() * 1000.0;

        // Only decay if significant time has passed
        if dt_ms > 0.1 {
            let current = f32::from_bits(self.level.load(Ordering::Acquire));
            let decay_factor = (-dt_ms / self.tau_ms).exp();
            let _decayed = current * decay_factor;

            // Atomic update with CAS loop
            let mut current_bits = self.level.load(Ordering::Acquire);
            loop {
                let current_val = f32::from_bits(current_bits);
                let new_val = current_val * decay_factor;
                let new_bits = new_val.to_bits();

                match self.level.compare_exchange(
                    current_bits,
                    new_bits,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => current_bits = actual,
                }
            }

            *last_update = now;
        }
    }

    /// Strengthen activation (Hebbian-style)
    pub fn strengthen(&self, amount: f32) {
        let amount = amount.clamp(0.0, 1.0);

        // CAS loop for atomic strengthening
        let mut current_bits = self.level.load(Ordering::Acquire);
        loop {
            let current = f32::from_bits(current_bits);
            // Asymptotic strengthening toward 1.0
            let new_val = current + amount * (1.0 - current);
            let new_bits = new_val.clamp(0.0, 1.0).to_bits();

            match self.level.compare_exchange(
                current_bits,
                new_bits,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current_bits = actual,
            }
        }

        *self.last_update.lock() = Instant::now();
    }

    /// Check if activation exceeds spike threshold
    pub fn should_spike(&self) -> bool {
        let activation = self.get_activation();

        // Check refractory period
        if let Some(last_spike) = *self.last_spike.lock() {
            let elapsed_ms = Instant::now().duration_since(last_spike).as_secs_f32() * 1000.0;
            if elapsed_ms < MIN_REFRACTORY_MS {
                return false;
            }
        }

        activation >= self.spike_threshold
    }

    /// Record a spike event
    pub fn record_spike(&self) {
        *self.last_spike.lock() = Some(Instant::now());
    }

    /// Get time since last spike
    pub fn time_since_spike(&self) -> Option<Duration> {
        self.last_spike
            .lock()
            .map(|t| Instant::now().duration_since(t))
    }

    /// Reset activation to zero
    pub fn reset(&self) {
        self.set_activation(0.0);
        *self.last_spike.lock() = None;
    }
}

impl Default for ActivationDynamics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_activation_bounds() {
        let dynamics = ActivationDynamics::new();

        // Test clamping
        dynamics.set_activation(1.5);
        assert_eq!(dynamics.get_activation(), 1.0);

        dynamics.set_activation(-0.5);
        assert_eq!(dynamics.get_activation(), 0.0);

        dynamics.set_activation(0.5);
        assert!((dynamics.get_activation() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_exponential_decay() {
        let dynamics = ActivationDynamics::with_parameters(100.0, 0.7);
        dynamics.set_activation(1.0);

        // Wait 50ms (half tau)
        thread::sleep(Duration::from_millis(50));
        let decayed = dynamics.get_activation();

        // Should decay to approximately e^(-0.5) ≈ 0.606
        let expected = (-0.5_f32).exp();
        assert!(
            (decayed - expected).abs() < 0.1,
            "Expected ~{:.3}, got {:.3}",
            expected,
            decayed
        );
    }

    #[test]
    fn test_hebbian_strengthening() {
        let dynamics = ActivationDynamics::new();

        dynamics.set_activation(0.5);
        dynamics.strengthen(0.3);

        // Should strengthen asymptotically
        let new_level = dynamics.get_activation();
        assert!(new_level > 0.5);
        assert!(new_level < 0.8);

        // Multiple strengthening approaches 1.0
        dynamics.set_activation(0.5);
        for _ in 0..20 {
            dynamics.strengthen(0.2);
        }
        let final_level = dynamics.get_activation();
        assert!(final_level > 0.95);
        assert!(final_level <= 1.0);
    }

    #[test]
    fn test_spike_threshold() {
        let dynamics = ActivationDynamics::with_parameters(100.0, 0.7);

        dynamics.set_activation(0.6);
        assert!(!dynamics.should_spike());

        dynamics.set_activation(0.8);
        assert!(dynamics.should_spike());
    }

    #[test]
    fn test_refractory_period() {
        let dynamics = ActivationDynamics::new();
        dynamics.set_activation(0.9);

        assert!(dynamics.should_spike());
        dynamics.record_spike();

        // Immediately after spike, should be refractory
        assert!(!dynamics.should_spike());

        // After refractory period
        thread::sleep(Duration::from_millis(3));
        assert!(dynamics.should_spike());
    }

    #[test]
    fn test_concurrent_activation_updates() {
        let dynamics = Arc::new(ActivationDynamics::new());
        let mut handles = vec![];

        // Spawn threads doing different operations
        for i in 0..10 {
            let dyn_clone = dynamics.clone();
            handles.push(thread::spawn(move || {
                if i % 2 == 0 {
                    dyn_clone.strengthen(0.1);
                } else {
                    dyn_clone.set_activation(0.5);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have valid activation after concurrent access
        let final_activation = dynamics.get_activation();
        assert!(final_activation >= 0.0 && final_activation <= 1.0);
    }
}