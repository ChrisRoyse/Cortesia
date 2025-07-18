pub mod summarization;
pub mod canonicalization;
pub mod salience;
pub mod neural_server;
pub mod structure_predictor;

pub use summarization::{
    NeuralSummarizer,
};

pub use canonicalization::{
    NeuralCanonicalizer,
    EnhancedNeuralCanonicalizer,
    EntityCanonicalizer,
    EntityDeduplicator,
    CanonicalEntity,
    DeduplicationResult,
};

pub use salience::{
    NeuralSalienceModel,
    ImportanceFilter,
    ImportanceScorer,
    ContentFilter,
};