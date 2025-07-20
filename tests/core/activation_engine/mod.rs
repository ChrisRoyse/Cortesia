// Core functionality tests
mod test_empty_pattern;
mod test_convergence;
mod test_disconnected_network;
mod test_cyclic_network;
mod test_inhibitory_connections;
mod test_logic_gates;
mod test_temporal_decay;
mod test_pattern_recognition;

// Unit tests
mod test_propagate_activation_unit;
mod test_apply_inhibitory_unit;

// Edge case tests
mod test_nan_infinity;
mod test_concurrent_access;
mod test_error_handling;

// Advanced tests
mod test_advanced_logic_gates;

// Performance tests
mod test_performance;

// Test utilities
mod test_constants;
mod test_helpers;