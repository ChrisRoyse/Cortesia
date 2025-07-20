// Test constants for brain_types tests
// Constants specific to brain-inspired neural computation and entity management

// ==================== Temporal Constants ====================
/// Standard temporal decay rate for neurons
pub const STANDARD_DECAY_RATE: f32 = 0.1;

/// Fast temporal decay rate for rapid forgetting
pub const FAST_DECAY_RATE: f32 = 0.5;

/// Slow temporal decay rate for long-term memory
pub const SLOW_DECAY_RATE: f32 = 0.01;

/// No decay rate for persistent entities
pub const NO_DECAY_RATE: f32 = 0.0;

// ==================== Activation Levels ====================
/// Resting potential for inactive neurons
pub const RESTING_POTENTIAL: f32 = 0.0;

/// Threshold potential for activation
pub const THRESHOLD_POTENTIAL: f32 = 0.5;

/// Action potential (strong activation)
pub const ACTION_POTENTIAL: f32 = 0.8;

/// Maximum activation saturation
pub const SATURATION_LEVEL: f32 = 1.0;

/// Above saturation for testing clamping
pub const ABOVE_SATURATION: f32 = 1.5;

// ==================== Logic Gate Thresholds ====================
/// Standard AND gate threshold
pub const AND_GATE_THRESHOLD: f32 = 0.5;

/// Standard OR gate threshold  
pub const OR_GATE_THRESHOLD: f32 = 0.3;

/// Standard threshold gate limit
pub const THRESHOLD_GATE_LIMIT: f32 = 0.7;

/// Weighted gate threshold
pub const WEIGHTED_GATE_THRESHOLD: f32 = 0.6;

/// Inhibitory gate threshold
pub const INHIBITORY_GATE_THRESHOLD: f32 = 0.4;

// ==================== Connection Weights ====================
/// Strong excitatory connection
pub const STRONG_EXCITATORY: f32 = 0.9;

/// Medium excitatory connection
pub const MEDIUM_EXCITATORY: f32 = 0.6;

/// Weak excitatory connection
pub const WEAK_EXCITATORY: f32 = 0.3;

/// Strong inhibitory connection strength
pub const STRONG_INHIBITORY: f32 = 0.8;

/// Medium inhibitory connection strength
pub const MEDIUM_INHIBITORY: f32 = 0.5;

/// Weak inhibitory connection strength
pub const WEAK_INHIBITORY: f32 = 0.2;

// ==================== Hebbian Learning Parameters ====================
/// Standard learning rate for synaptic plasticity
pub const STANDARD_LEARNING_RATE: f32 = 0.1;

/// Fast learning rate for rapid adaptation
pub const FAST_LEARNING_RATE: f32 = 0.3;

/// Slow learning rate for stable learning
pub const SLOW_LEARNING_RATE: f32 = 0.05;

/// Maximum learning rate
pub const MAX_LEARNING_RATE: f32 = 1.0;

/// Minimum learning rate
pub const MIN_LEARNING_RATE: f32 = 0.01;

// ==================== Test Tolerances ====================
/// Floating point epsilon for activation comparisons
pub const ACTIVATION_EPSILON: f32 = 0.001;

/// Stricter epsilon for precise calculations
pub const STRICT_EPSILON: f32 = 0.0001;

/// Looser tolerance for integration tests
pub const LOOSE_TOLERANCE: f32 = 0.01;

// ==================== Entity Constants ====================
/// Standard entity concept IDs for testing
pub const TEST_CONCEPT_INPUT: &str = "test_input_concept";
pub const TEST_CONCEPT_OUTPUT: &str = "test_output_concept";  
pub const TEST_CONCEPT_GATE: &str = "test_gate_concept";
pub const TEST_CONCEPT_HIDDEN: &str = "test_hidden_concept";

/// Common relationship names
pub const TEST_RELATION_ISA: &str = "is_a_relation";
pub const TEST_RELATION_PART_OF: &str = "part_of_relation";
pub const TEST_RELATION_SIMILAR: &str = "similar_relation";

// ==================== Logic Gate Test Inputs ====================
/// Two-input gate test combinations
pub const GATE_INPUTS_00: [f32; 2] = [0.0, 0.0];
pub const GATE_INPUTS_01: [f32; 2] = [0.0, 0.7];
pub const GATE_INPUTS_10: [f32; 2] = [0.7, 0.0];
pub const GATE_INPUTS_11: [f32; 2] = [0.7, 0.7];

/// Single input for unary gates
pub const SINGLE_INPUT_LOW: [f32; 1] = [0.3];
pub const SINGLE_INPUT_HIGH: [f32; 1] = [0.8];

/// Three-input combinations for multi-input gates
pub const THREE_INPUTS_ALL_HIGH: [f32; 3] = [0.8, 0.9, 0.7];
pub const THREE_INPUTS_MIXED: [f32; 3] = [0.8, 0.2, 0.6];
pub const THREE_INPUTS_ALL_LOW: [f32; 3] = [0.2, 0.1, 0.3];

// ==================== Weight Matrix Constants ====================
/// Standard weight matrix for 3-input weighted gate
pub const WEIGHT_MATRIX_3: [f32; 3] = [0.5, 0.3, 0.2];

/// Balanced weight matrix
pub const BALANCED_WEIGHTS: [f32; 2] = [0.5, 0.5];

/// Unbalanced weight matrix favoring first input
pub const UNBALANCED_WEIGHTS: [f32; 2] = [0.8, 0.2];

// ==================== Performance Test Constants ====================
/// Small pattern size for unit tests
pub const SMALL_PATTERN_SIZE: usize = 10;

/// Medium pattern size for integration tests
pub const MEDIUM_PATTERN_SIZE: usize = 100;

/// Large pattern size for performance tests
pub const LARGE_PATTERN_SIZE: usize = 1000;

/// Maximum acceptable processing time (microseconds)
pub const MAX_PROCESSING_TIME_US: u128 = 1000;

// ==================== Temporal Test Constants ====================
/// Milliseconds to wait for decay testing
pub const DECAY_WAIT_MS: u64 = 100;

/// Expected time steps for multi-step propagation
pub const PROPAGATION_STEPS: usize = 5;

// ==================== Error Test Constants ====================
/// Invalid threshold value (negative)
pub const INVALID_THRESHOLD: f32 = -0.5;

/// Invalid weight value (too large)
pub const INVALID_WEIGHT: f32 = 10.0;

/// Empty string for error testing
pub const EMPTY_CONCEPT_ID: &str = "";