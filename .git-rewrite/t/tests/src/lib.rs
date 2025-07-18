//! LLMKG Test Library
//! 
//! Independent test library for LLMKG unit testing framework

pub mod infrastructure;
pub mod unit;

// Re-export common types
pub use infrastructure::*;
pub use unit::test_utils;
pub use unit::test_utils::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_basics() {
        // Test that the library loads correctly
        let key = EntityKey::from_hash("test");
        let entity = Entity::new(key, "Test".to_string());
        assert_eq!(entity.name(), "Test");
    }
}