pub mod migration;

pub use migration::{
    // Primary cognitive-federation migration system
    CognitiveFederationMigrator,
    CognitiveMigrationReport,
    CognitiveMigrationStats,
    
    // Cognitive entity types
    CognitiveEntity,
    CognitiveRelationship,
    CognitiveMetadata,
    
    // Legacy storage interface
    LegacyStorage,
    LegacyEntity,
    LegacyRelationship,
    KnowledgeEngineLegacyAdapter,
    
    // Enhanced configuration
    MigrationConfig,
    ValidationLevel,
    
    // Shared types
    ValidationReport,
    MigrationError,
    BackupSnapshot,
    MigrationProgress,
    
    // Legacy aliases for backward compatibility
    MigrationTool,
    MigrationReport,
};