// Federation module tests - comprehensive test suite for multi-database federation
//
// This test module provides complete coverage for:
// - Cross-database transactions (2-phase commit)
// - Load balancing and query routing  
// - Health monitoring and failover
// - Database registry management
// - Result merging and aggregation
// - Network partition handling
// - Performance benchmarking

pub mod test_coordinator;
pub mod test_registry;
pub mod test_router;
pub mod test_merger;
pub mod test_federation_manager;
pub mod test_types;
pub mod integration_tests;
pub mod performance_tests;
pub mod stress_tests;