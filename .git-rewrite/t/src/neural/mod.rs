pub mod summarization;
pub mod canonicalization;
pub mod salience;

pub use summarization::{
    NeuralSummarizer,
    SummarizationModel,
    T5SummarizationModel,
    BARTSummarizationModel,
    SalienceFilter,
    SalienceModel,
    KeywordSalienceModel,
    CacheStats,
};

pub use canonicalization::{
    NeuralCanonicalizer,
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