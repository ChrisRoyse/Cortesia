// Test constants for activation engine tests
// These constants represent biologically meaningful values and common test scenarios

// ==================== Activation Levels ====================
/// Strong activation level typical of highly active neurons
pub const STRONG_ACTIVATION: f32 = 0.8;

/// Medium activation level for moderately active neurons
pub const MEDIUM_ACTIVATION: f32 = 0.5;

/// Weak activation level near threshold
pub const WEAK_ACTIVATION: f32 = 0.3;

/// Very weak activation below most thresholds
pub const MINIMAL_ACTIVATION: f32 = 0.1;

/// Maximum possible activation (saturated neuron)
pub const MAX_ACTIVATION: f32 = 1.0;

/// Zero activation (silent neuron)
pub const ZERO_ACTIVATION: f32 = 0.0;

/// Above normal activation for testing clamping
pub const EXCESSIVE_ACTIVATION: f32 = 2.0;

// ==================== Connection Weights ====================
/// Strong synaptic connection weight
pub const STRONG_WEIGHT: f32 = 0.8;

/// Medium synaptic connection weight
pub const MEDIUM_WEIGHT: f32 = 0.5;

/// Weak synaptic connection weight
pub const WEAK_WEIGHT: f32 = 0.2;

/// Maximum connection weight
pub const MAX_WEIGHT: f32 = 1.0;

/// Zero connection weight (no effect)
pub const ZERO_WEIGHT: f32 = 0.0;

/// Extreme weight for testing edge cases
pub const EXTREME_WEIGHT: f32 = 10.0;

/// Negative weight for testing error handling
pub const NEGATIVE_WEIGHT: f32 = -0.5;

// ==================== Thresholds ====================
/// Standard logic gate activation threshold
pub const GATE_THRESHOLD: f32 = 0.5;

/// Tight convergence threshold for precise tests
pub const TIGHT_CONVERGENCE: f32 = 0.0001;

/// Normal convergence threshold
pub const NORMAL_CONVERGENCE: f32 = 0.01;

/// Loose convergence threshold for quick tests
pub const LOOSE_CONVERGENCE: f32 = 0.1;

// ==================== Iteration Limits ====================
/// Very low iteration limit for testing early termination
pub const MIN_ITERATIONS: usize = 2;

/// Standard iteration limit for most tests
pub const STANDARD_ITERATIONS: usize = 50;

/// High iteration limit for complex networks
pub const MAX_ITERATIONS: usize = 100;

// ==================== Inhibition Parameters ====================
/// Normal inhibition strength
pub const NORMAL_INHIBITION: f32 = 0.7;

/// Strong inhibition strength
pub const STRONG_INHIBITION: f32 = 1.0;

/// Very strong inhibition for testing limits
pub const EXTREME_INHIBITION: f32 = 10.0;

// ==================== Expected Values ====================
/// Expected energy for single activation squared
pub const SINGLE_NODE_ENERGY: f32 = 0.5 * 0.5; // 0.25

/// Epsilon for floating point comparisons
pub const EPSILON: f32 = 0.0001;

// ==================== Timing Constants ====================
/// Expected maximum time for simple propagation (ms)
pub const SIMPLE_PROPAGATION_TIMEOUT: u128 = 10;

/// Expected maximum time for complex propagation (ms)
pub const COMPLEX_PROPAGATION_TIMEOUT: u128 = 50;

// ==================== Network Size Constants ====================
/// Small network size for unit tests
pub const SMALL_NETWORK_SIZE: usize = 10;

/// Medium network size for integration tests
pub const MEDIUM_NETWORK_SIZE: usize = 100;

/// Large network size for performance tests
pub const LARGE_NETWORK_SIZE: usize = 1000;