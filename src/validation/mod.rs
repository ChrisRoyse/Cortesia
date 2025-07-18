pub mod human_loop;
pub mod feedback;

pub use human_loop::{
    HumanValidationInterface,
    ValidationItem,
    ValidationResult,
    ValidationQueue,
    ValidationTask,
    ValidationStatus,
    HumanFeedback,
};

pub use feedback::{
    FeedbackProcessor,
    ActiveLearningEngine,
    ValidationFeedback,
    LearningUpdate,
    ConfidenceModel,
};