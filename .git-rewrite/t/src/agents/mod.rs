pub mod coordination;
pub mod construction;

pub use coordination::{
    MultiAgentCoordinator,
    AgentCoordination,
    ConsensusProtocol,
    ConsensusResult,
    MergeStrategy,
    AgentId,
    CoordinationMessage,
};

pub use construction::{
    KnowledgeAgent,
    ConstructionTask,
    ConstructionResult,
    AgentCapabilities,
    TaskDistribution,
    WorkItem,
    AgentPerformance,
};