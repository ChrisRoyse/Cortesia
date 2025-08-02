//! Mock Framework for Enhanced Knowledge Storage System
//! 
//! Provides comprehensive mocks for all external dependencies following
//! London School TDD methodology with behavior verification.

pub mod model_mocks;
pub mod storage_mocks;
pub mod processing_mocks;
pub mod embedding_mocks;
pub mod comprehensive_mock_data;

// Re-export commonly used mocks
pub use model_mocks::*;
pub use storage_mocks::*;
pub use processing_mocks::*;
pub use embedding_mocks::*;
pub use comprehensive_mock_data::*;