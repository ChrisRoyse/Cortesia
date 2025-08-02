# MicroPhase 2: Neuromorphic Core Integration

**Duration**: 4-5 hours  
**Priority**: Critical - Core functionality  
**Prerequisites**: MicroPhase 1 (Foundation)

## Overview

Implement the neuromorphic processing core with 4 cortical columns, TTFS encoding, lateral inhibition, and STDP learning engine. This phase is broken into 26 atomic micro-tasks for maximum clarity and parallel execution.

## AI-Actionable Tasks

### Task 2.1.1: Create ColumnType Enum
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/cortical_column.rs`

**Task Prompt for AI**:
Create the foundational ColumnType enum with debug and clone traits. This enum will be used across all cortical column implementations.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    Semantic,
    Structural,
    Temporal,
    Exception,
}

impl ColumnType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ColumnType::Semantic => "semantic",
            ColumnType::Structural => "structural", 
            ColumnType::Temporal => "temporal",
            ColumnType::Exception => "exception",
        }
    }
}
```

**Verification**: Compiles + enum methods work correctly

### Task 2.1.2: Create ColumnResult Struct 
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/cortical_column.rs`

**Task Prompt for AI**:
Add the ColumnResult struct to hold processing results from cortical columns.

```rust
#[derive(Debug, Clone)]
pub struct ColumnResult {
    pub activation_strength: f32,
    pub confidence: f32,
    pub neural_pathway: Vec<String>,
    pub processing_time_ms: f32,
    pub allocated_memory_id: Option<String>,
}

impl ColumnResult {
    pub fn new(activation_strength: f32, confidence: f32) -> Self {
        Self {
            activation_strength,
            confidence,
            neural_pathway: Vec::new(),
            processing_time_ms: 0.0,
            allocated_memory_id: None,
        }
    }
}
```

**Verification**: Compiles + constructor works

### Task 2.1.3: Create LearningFeedback Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/cortical_column.rs`

**Task Prompt for AI**:
Add the LearningFeedback struct for STDP learning updates.

```rust
#[derive(Debug, Clone)]
pub struct LearningFeedback {
    pub success: bool,
    pub reward_signal: f32,
    pub pathway_trace: Vec<String>,
}

impl LearningFeedback {
    pub fn new_success(reward_signal: f32, pathway: Vec<String>) -> Self {
        Self {
            success: true,
            reward_signal,
            pathway_trace: pathway,
        }
    }
    
    pub fn new_failure(penalty: f32, pathway: Vec<String>) -> Self {
        Self {
            success: false,
            reward_signal: penalty,
            pathway_trace: pathway,
        }
    }
}
```

**Verification**: Compiles + constructors work

### Task 2.1.4: Define CorticalColumn Trait
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/cortical_column.rs`

**Task Prompt for AI**:
Define the core CorticalColumn async trait that all column types will implement.

```rust
use async_trait::async_trait;
use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;

#[async_trait]
pub trait CorticalColumn: Send + Sync {
    async fn process(&self, pattern: &TTFSPattern) -> MCPResult<ColumnResult>;
    async fn get_activation_strength(&self) -> f32;
    async fn update_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()>;
    fn get_column_type(&self) -> ColumnType;
    
    // Default implementation for performance metrics
    fn get_processing_stats(&self) -> ProcessingStats {
        ProcessingStats::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    pub total_processed: u64,
    pub average_processing_time_ms: f32,
    pub last_activation_strength: f32,
}
```

**Verification**: Compiles + trait compiles correctly

### Task 2.2.1: Create NetworkType Enum for Semantic Column
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Create the NetworkType enum for semantic neural networks.

```rust
#[derive(Debug, Clone)]
pub enum NetworkType {
    MLP,
    TiDE,
    DeepAR,
    TSMixer,
}

impl NetworkType {
    pub fn get_optimal_for_pattern_size(size: usize) -> Self {
        match size {
            0..=100 => NetworkType::MLP,
            101..=1000 => NetworkType::TiDE,
            1001..=10000 => NetworkType::DeepAR,
            _ => NetworkType::TSMixer,
        }
    }
}
```

**Verification**: Compiles + enum selector works

### Task 2.2.2: Create SemanticColumn Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Define the main SemanticColumn struct with required fields.

```rust
use super::cortical_column::{CorticalColumn, ColumnResult, ColumnType, LearningFeedback};
use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct SemanticColumn {
    network_pool: Arc<RwLock<NetworkPool>>,
    activation_threshold: f32,
    learning_rate: f32,
    current_activation: Arc<RwLock<f32>>,
}

impl SemanticColumn {
    pub fn new_with_networks(network_types: Vec<NetworkType>) -> Self {
        Self {
            network_pool: Arc::new(RwLock::new(NetworkPool::new(network_types))),
            activation_threshold: 0.7,
            learning_rate: 0.01,
            current_activation: Arc::new(RwLock::new(0.0)),
        }
    }
}
```

**Verification**: Compiles + constructor works

### Task 2.2.3: Implement NetworkPool for Semantic
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Implement the NetworkPool struct with mock neural network selection.

```rust
pub struct NetworkPool {
    network_types: Vec<NetworkType>,
}

impl NetworkPool {
    pub fn new(network_types: Vec<NetworkType>) -> Self {
        Self { network_types }
    }
    
    pub async fn select_optimal_for_pattern(&self, pattern: &TTFSPattern) -> MCPResult<MockNetwork> {
        let pattern_size = pattern.spikes.len();
        let optimal_type = if self.network_types.is_empty() {
            NetworkType::MLP
        } else {
            NetworkType::get_optimal_for_pattern_size(pattern_size)
        };
        Ok(MockNetwork::new(optimal_type))
    }
    
    pub async fn get_feature_extractor(&self) -> MCPResult<MockFeatureExtractor> {
        Ok(MockFeatureExtractor::new())
    }
    
    pub async fn apply_stdp_update(&mut self, _feedback: &LearningFeedback, _learning_rate: f32) -> MCPResult<()> {
        // Mock STDP update implementation
        Ok(())
    }
}
```

**Verification**: Compiles + network selection logic works

### Task 2.2.4: Create Mock Network Implementation
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Implement mock neural network for semantic similarity computation.

```rust
pub struct MockNetwork {
    network_type: NetworkType,
}

impl MockNetwork {
    pub fn new(network_type: NetworkType) -> Self {
        Self { network_type }
    }
    
    pub async fn compute_semantic_similarity(&self, pattern: &TTFSPattern) -> MCPResult<f32> {
        // Mock similarity based on pattern complexity
        let complexity = pattern.spikes.len() as f32 / 100.0;
        let base_similarity = match self.network_type {
            NetworkType::MLP => 0.75,
            NetworkType::TiDE => 0.80,
            NetworkType::DeepAR => 0.85,
            NetworkType::TSMixer => 0.90,
        };
        Ok((base_similarity * complexity).min(0.95))
    }
}

pub struct MockFeatureExtractor;

impl MockFeatureExtractor {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn extract_concepts(&self, pattern: &TTFSPattern) -> MCPResult<Vec<String>> {
        let concepts = pattern.spikes.iter()
            .take(3)
            .enumerate()
            .map(|(i, _)| format!("semantic_concept_{}", i))
            .collect();
        Ok(concepts)
    }
}
```

**Verification**: Compiles + mock similarity computation works

### Task 2.2.5: Add Semantic Analysis Methods
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Add private methods for semantic analysis to the SemanticColumn.

```rust
impl SemanticColumn {
    async fn conceptual_similarity_analysis(&self, pattern: &TTFSPattern) -> MCPResult<f32> {
        let networks = self.network_pool.read().await;
        let optimal_network = networks.select_optimal_for_pattern(pattern).await?;
        let similarity_score = optimal_network.compute_semantic_similarity(pattern).await?;
        Ok(similarity_score)
    }
    
    async fn extract_concept_features(&self, pattern: &TTFSPattern) -> MCPResult<Vec<String>> {
        let networks = self.network_pool.read().await;
        let feature_extractor = networks.get_feature_extractor().await?;
        let features = feature_extractor.extract_concepts(pattern).await?;
        Ok(features)
    }
}
```

**Verification**: Compiles + analysis methods work

### Task 2.2.6: Implement CorticalColumn Trait for Semantic
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/semantic_column.rs`

**Task Prompt for AI**:
Implement the CorticalColumn trait for SemanticColumn.

```rust
use async_trait::async_trait;

#[async_trait]
impl CorticalColumn for SemanticColumn {
    async fn process(&self, pattern: &TTFSPattern) -> MCPResult<ColumnResult> {
        let start_time = std::time::Instant::now();
        
        let similarity_score = self.conceptual_similarity_analysis(pattern).await?;
        let neural_pathway = self.extract_concept_features(pattern).await?;
        
        let activation_strength = if similarity_score > self.activation_threshold {
            similarity_score * 1.2
        } else {
            similarity_score * 0.8
        };
        
        *self.current_activation.write().await = activation_strength;
        let processing_time = start_time.elapsed().as_millis() as f32;
        
        Ok(ColumnResult {
            activation_strength,
            confidence: similarity_score,
            neural_pathway,
            processing_time_ms: processing_time,
            allocated_memory_id: None,
        })
    }
    
    async fn get_activation_strength(&self) -> f32 {
        *self.current_activation.read().await
    }
    
    async fn update_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        let mut networks = self.network_pool.write().await;
        networks.apply_stdp_update(feedback, self.learning_rate).await?;
        Ok(())
    }
    
    fn get_column_type(&self) -> ColumnType {
        ColumnType::Semantic
    }
}
```

**Verification**: Compiles + trait implementation works correctly

### Task 2.3.1: Create GraphNetworkType Enum
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Create the GraphNetworkType enum for structural neural networks.

```rust
#[derive(Debug, Clone)]
pub enum GraphNetworkType {
    StemGNN,
    ITransformer,
    PatchTST,
    TFT,
}

impl GraphNetworkType {
    pub fn get_optimal_for_graph_size(nodes: usize, edges: usize) -> Self {
        let complexity = nodes * edges;
        match complexity {
            0..=1000 => GraphNetworkType::StemGNN,
            1001..=10000 => GraphNetworkType::ITransformer,
            10001..=100000 => GraphNetworkType::PatchTST,
            _ => GraphNetworkType::TFT,
        }
    }
}
```

**Verification**: Compiles + network selection logic works

### Task 2.3.2: Create TopologyResult Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Define the TopologyResult struct for structural analysis results.

```rust
#[derive(Debug, Clone)]
pub struct TopologyResult {
    pub connectivity_score: f32,
    pub confidence: f32,
    pub structural_features: Vec<String>,
    pub node_count: usize,
    pub edge_count: usize,
}

impl TopologyResult {
    pub fn new(connectivity_score: f32, confidence: f32) -> Self {
        Self {
            connectivity_score,
            confidence,
            structural_features: Vec::new(),
            node_count: 0,
            edge_count: 0,
        }
    }
    
    pub fn complexity_score(&self) -> f32 {
        (self.node_count * self.edge_count) as f32 / 1000.0
    }
}
```

**Verification**: Compiles + complexity calculation works

### Task 2.3.3: Create StructuralColumn Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Define the main StructuralColumn struct with required fields.

```rust
use super::cortical_column::{CorticalColumn, ColumnResult, ColumnType, LearningFeedback};
use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct StructuralColumn {
    network_pool: Arc<RwLock<GraphNetworkPool>>,
    topology_analyzer: Arc<RwLock<TopologyAnalyzer>>,
    activation_threshold: f32,
    current_activation: Arc<RwLock<f32>>,
}

impl StructuralColumn {
    pub fn new_with_networks(network_types: Vec<GraphNetworkType>) -> Self {
        Self {
            network_pool: Arc::new(RwLock::new(GraphNetworkPool::new(network_types))),
            topology_analyzer: Arc::new(RwLock::new(TopologyAnalyzer::new())),
            activation_threshold: 0.6,
            current_activation: Arc::new(RwLock::new(0.0)),
        }
    }
}
```

**Verification**: Compiles + constructor works

### Task 2.3.4: Implement TopologyAnalyzer
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Implement the TopologyAnalyzer for graph structure analysis.

```rust
pub struct TopologyAnalyzer;

impl TopologyAnalyzer {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn analyze_structure(&self, pattern: &TTFSPattern) -> MCPResult<TopologyResult> {
        let node_count = pattern.spikes.len();
        let edge_count = self.estimate_edges(&pattern.spikes);
        
        let connectivity_score = if node_count > 0 {
            edge_count as f32 / (node_count as f32 * (node_count - 1) as f32 / 2.0)
        } else {
            0.0
        }.min(1.0);
        
        let confidence = (connectivity_score * 0.8 + 0.2).min(0.95);
        
        let structural_features = vec![
            format!("nodes:{}", node_count),
            format!("edges:{}", edge_count),
            format!("density:{:.2}", connectivity_score),
        ];
        
        Ok(TopologyResult {
            connectivity_score,
            confidence,
            structural_features,
            node_count,
            edge_count,
        })
    }
    
    fn estimate_edges(&self, spikes: &[crate::core::types::TTFSSpike]) -> usize {
        // Mock edge estimation based on spike timing patterns
        spikes.len() / 2
    }
}
```

**Verification**: Compiles + topology analysis works

### Task 2.3.5: Implement GraphNetworkPool
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Implement the GraphNetworkPool with path optimization.

```rust
pub struct GraphNetworkPool {
    network_types: Vec<GraphNetworkType>,
}

impl GraphNetworkPool {
    pub fn new(network_types: Vec<GraphNetworkType>) -> Self {
        Self { network_types }
    }
    
    pub async fn get_path_optimizer(&self) -> MCPResult<PathOptimizer> {
        Ok(PathOptimizer::new())
    }
    
    pub async fn update_topology_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        // Mock topology weight update based on feedback
        if feedback.success {
            // Positive reinforcement logic would go here
        } else {
            // Negative reinforcement logic would go here
        }
        Ok(())
    }
}

pub struct PathOptimizer;

impl PathOptimizer {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn find_optimal_allocation(&self, topology: &TopologyResult) -> MCPResult<Vec<String>> {
        let paths = topology.structural_features.iter()
            .take(3)
            .enumerate()
            .map(|(i, feature)| format!("path_{}_{}", i, feature))
            .collect();
        Ok(paths)
    }
}
```

**Verification**: Compiles + path optimization works

### Task 2.3.6: Add Structural Analysis Methods
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Add private methods for structural analysis to the StructuralColumn.

```rust
impl StructuralColumn {
    async fn analyze_graph_topology(&self, pattern: &TTFSPattern) -> MCPResult<TopologyResult> {
        let analyzer = self.topology_analyzer.read().await;
        let topology = analyzer.analyze_structure(pattern).await?;
        Ok(topology)
    }
    
    async fn optimize_allocation_path(&self, topology: &TopologyResult) -> MCPResult<Vec<String>> {
        let networks = self.network_pool.read().await;
        let optimizer = networks.get_path_optimizer().await?;
        let optimal_path = optimizer.find_optimal_allocation(topology).await?;
        Ok(optimal_path)
    }
}
```

**Verification**: Compiles + analysis methods work

### Task 2.3.7: Implement CorticalColumn Trait for Structural
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/structural_column.rs`

**Task Prompt for AI**:
Implement the CorticalColumn trait for StructuralColumn.

```rust
use async_trait::async_trait;

#[async_trait]
impl CorticalColumn for StructuralColumn {
    async fn process(&self, pattern: &TTFSPattern) -> MCPResult<ColumnResult> {
        let start_time = std::time::Instant::now();
        
        let topology_result = self.analyze_graph_topology(pattern).await?;
        let neural_pathway = self.optimize_allocation_path(&topology_result).await?;
        
        let structural_score = topology_result.connectivity_score;
        let activation_strength = if structural_score > self.activation_threshold {
            structural_score * 1.1
        } else {
            structural_score * 0.9
        };
        
        *self.current_activation.write().await = activation_strength;
        let processing_time = start_time.elapsed().as_millis() as f32;
        
        Ok(ColumnResult {
            activation_strength,
            confidence: topology_result.confidence,
            neural_pathway,
            processing_time_ms: processing_time,
            allocated_memory_id: None,
        })
    }
    
    async fn get_activation_strength(&self) -> f32 {
        *self.current_activation.read().await
    }
    
    async fn update_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        let mut networks = self.network_pool.write().await;
        networks.update_topology_weights(feedback).await?;
        Ok(())
    }
    
    fn get_column_type(&self) -> ColumnType {
        ColumnType::Structural
    }
}
```

**Verification**: Compiles + trait implementation works correctly

### Task 2.4.1: Create TemporalNetworkType Enum
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Create the TemporalNetworkType enum for temporal neural networks.

```rust
#[derive(Debug, Clone)]
pub enum TemporalNetworkType {
    LSTM,
    TCN,
    NBEATS,
    GRU,
}

impl TemporalNetworkType {
    pub fn get_optimal_for_sequence_length(sequence_length: usize) -> Self {
        match sequence_length {
            0..=10 => TemporalNetworkType::GRU,
            11..=50 => TemporalNetworkType::LSTM,
            51..=200 => TemporalNetworkType::TCN,
            _ => TemporalNetworkType::NBEATS,
        }
    }
}
```

**Verification**: Compiles + network selection logic works

### Task 2.4.2: Create TemporalResult Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Define the TemporalResult struct for temporal analysis results.

```rust
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct TemporalResult {
    pub pattern_strength: f32,
    pub confidence: f32,
    pub temporal_features: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub sequence_length: usize,
}

impl TemporalResult {
    pub fn new(pattern_strength: f32, confidence: f32, sequence_length: usize) -> Self {
        Self {
            pattern_strength,
            confidence,
            temporal_features: Vec::new(),
            timestamp: Utc::now(),
            sequence_length,
        }
    }
    
    pub fn temporal_complexity(&self) -> f32 {
        self.sequence_length as f32 / 100.0 * self.pattern_strength
    }
}
```

**Verification**: Compiles + complexity calculation works

### Task 2.4.3: Create TemporalColumn Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Define the main TemporalColumn struct with required fields.

```rust
use super::cortical_column::{CorticalColumn, ColumnResult, ColumnType, LearningFeedback};
use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct TemporalColumn {
    sequence_analyzer: Arc<RwLock<SequenceAnalyzer>>,
    pattern_detector: Arc<RwLock<TemporalPatternDetector>>,
    activation_threshold: f32,
    current_activation: Arc<RwLock<f32>>,
}

impl TemporalColumn {
    pub fn new_with_networks(network_types: Vec<TemporalNetworkType>) -> Self {
        Self {
            sequence_analyzer: Arc::new(RwLock::new(SequenceAnalyzer::new(network_types))),
            pattern_detector: Arc::new(RwLock::new(TemporalPatternDetector::new())),
            activation_threshold: 0.65,
            current_activation: Arc::new(RwLock::new(0.0)),
        }
    }
}
```

**Verification**: Compiles + constructor works

### Task 2.4.4: Implement TemporalPatternDetector
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Implement the TemporalPatternDetector for time series pattern detection.

```rust
pub struct TemporalPatternDetector;

impl TemporalPatternDetector {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn detect_patterns(&self, pattern: &TTFSPattern) -> MCPResult<TemporalResult> {
        let sequence_length = pattern.spikes.len();
        
        // Calculate pattern strength based on temporal regularity
        let pattern_strength = self.calculate_temporal_regularity(&pattern.spikes);
        
        // Calculate confidence based on sequence characteristics
        let confidence = (pattern_strength * 0.9 + 0.1).min(0.95);
        
        let temporal_features = vec![
            format!("sequence_length:{}", sequence_length),
            format!("regularity:{:.2}", pattern_strength),
            format!("timestamp:{}", Utc::now().format("%Y-%m-%d %H:%M:%S")),
        ];
        
        Ok(TemporalResult {
            pattern_strength,
            confidence,
            temporal_features,
            timestamp: Utc::now(),
            sequence_length,
        })
    }
    
    fn calculate_temporal_regularity(&self, spikes: &[crate::core::types::TTFSSpike]) -> f32 {
        if spikes.len() < 2 {
            return 0.1;
        }
        
        // Mock regularity calculation based on timing intervals
        let intervals: Vec<f32> = spikes.windows(2)
            .map(|pair| (pair[1].timestamp - pair[0].timestamp).abs())
            .collect();
        
        if intervals.is_empty() {
            return 0.1;
        }
        
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance = intervals.iter()
            .map(|x| (x - mean_interval).powi(2))
            .sum::<f32>() / intervals.len() as f32;
        
        // Lower variance = higher regularity = higher pattern strength
        (1.0 / (1.0 + variance)).min(0.95)
    }
}
```

**Verification**: Compiles + pattern detection works

### Task 2.4.5: Implement SequenceAnalyzer
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Implement the SequenceAnalyzer for temporal sequence prediction.

```rust
pub struct SequenceAnalyzer {
    network_types: Vec<TemporalNetworkType>,
}

impl SequenceAnalyzer {
    pub fn new(network_types: Vec<TemporalNetworkType>) -> Self {
        Self { network_types }
    }
    
    pub async fn predict_next_elements(&self, pattern: &TTFSPattern) -> MCPResult<Vec<String>> {
        let sequence_length = pattern.spikes.len();
        
        // Select optimal network for this sequence length
        let optimal_network = if self.network_types.is_empty() {
            TemporalNetworkType::LSTM
        } else {
            TemporalNetworkType::get_optimal_for_sequence_length(sequence_length)
        };
        
        // Generate mock predictions based on network type and sequence
        let predictions = match optimal_network {
            TemporalNetworkType::LSTM => self.lstm_predict(pattern),
            TemporalNetworkType::TCN => self.tcn_predict(pattern),
            TemporalNetworkType::NBEATS => self.nbeats_predict(pattern),
            TemporalNetworkType::GRU => self.gru_predict(pattern),
        };
        
        Ok(predictions)
    }
    
    pub async fn update_temporal_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        // Mock temporal weight update based on feedback
        if feedback.success {
            // Positive reinforcement would strengthen temporal patterns
        } else {
            // Negative feedback would adjust temporal predictions
        }
        Ok(())
    }
    
    fn lstm_predict(&self, pattern: &TTFSPattern) -> Vec<String> {
        pattern.spikes.iter().take(3)
            .enumerate()
            .map(|(i, _)| format!("lstm_pred_{}", i))
            .collect()
    }
    
    fn tcn_predict(&self, pattern: &TTFSPattern) -> Vec<String> {
        pattern.spikes.iter().take(2)
            .enumerate()
            .map(|(i, _)| format!("tcn_pred_{}", i))
            .collect()
    }
    
    fn nbeats_predict(&self, pattern: &TTFSPattern) -> Vec<String> {
        pattern.spikes.iter().take(4)
            .enumerate()
            .map(|(i, _)| format!("nbeats_pred_{}", i))
            .collect()
    }
    
    fn gru_predict(&self, pattern: &TTFSPattern) -> Vec<String> {
        pattern.spikes.iter().take(2)
            .enumerate()
            .map(|(i, _)| format!("gru_pred_{}", i))
            .collect()
    }
}
```

**Verification**: Compiles + sequence prediction works

### Task 2.4.6: Add Temporal Analysis Methods
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Add private methods for temporal analysis to the TemporalColumn.

```rust
impl TemporalColumn {
    async fn analyze_temporal_patterns(&self, pattern: &TTFSPattern) -> MCPResult<TemporalResult> {
        let detector = self.pattern_detector.read().await;
        let temporal_analysis = detector.detect_patterns(pattern).await?;
        Ok(temporal_analysis)
    }
    
    async fn predict_sequence_continuation(&self, pattern: &TTFSPattern) -> MCPResult<Vec<String>> {
        let analyzer = self.sequence_analyzer.read().await;
        let predicted_sequence = analyzer.predict_next_elements(pattern).await?;
        Ok(predicted_sequence)
    }
}
```

**Verification**: Compiles + analysis methods work

### Task 2.4.7: Implement CorticalColumn Trait for Temporal
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/temporal_column.rs`

**Task Prompt for AI**:
Implement the CorticalColumn trait for TemporalColumn.

```rust
use async_trait::async_trait;

#[async_trait]
impl CorticalColumn for TemporalColumn {
    async fn process(&self, pattern: &TTFSPattern) -> MCPResult<ColumnResult> {
        let start_time = std::time::Instant::now();
        
        let temporal_result = self.analyze_temporal_patterns(pattern).await?;
        let neural_pathway = self.predict_sequence_continuation(pattern).await?;
        
        let temporal_score = temporal_result.pattern_strength;
        let activation_strength = if temporal_score > self.activation_threshold {
            temporal_score * 1.15 // Strong boost for temporal patterns
        } else {
            temporal_score * 0.85
        };
        
        *self.current_activation.write().await = activation_strength;
        let processing_time = start_time.elapsed().as_millis() as f32;
        
        Ok(ColumnResult {
            activation_strength,
            confidence: temporal_result.confidence,
            neural_pathway,
            processing_time_ms: processing_time,
            allocated_memory_id: None,
        })
    }
    
    async fn get_activation_strength(&self) -> f32 {
        *self.current_activation.read().await
    }
    
    async fn update_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        let mut analyzer = self.sequence_analyzer.write().await;
        analyzer.update_temporal_weights(feedback).await?;
        Ok(())
    }
    
    fn get_column_type(&self) -> ColumnType {
        ColumnType::Temporal
    }
}
```

**Verification**: Compiles + trait implementation works correctly

### Task 2.5.1: Create ExceptionNetworkType Enum
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Create the ExceptionNetworkType enum for exception detection networks.

```rust
#[derive(Debug, Clone)]
pub enum ExceptionNetworkType {
    CascadeCorrelation,
    SparseConnected,
    DLinear,
}

impl ExceptionNetworkType {
    pub fn get_optimal_for_anomaly_type(anomaly_complexity: f32) -> Self {
        match anomaly_complexity {
            x if x < 0.3 => ExceptionNetworkType::DLinear,
            x if x < 0.7 => ExceptionNetworkType::SparseConnected,
            _ => ExceptionNetworkType::CascadeCorrelation,
        }
    }
}
```

**Verification**: Compiles + network selection logic works

### Task 2.5.2: Create ContradictionResult Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Define the ContradictionResult struct for contradiction detection results.

```rust
#[derive(Debug, Clone)]
pub struct ContradictionResult {
    pub has_contradictions: bool,
    pub confidence: f32,
    pub contradiction_details: Vec<String>,
    pub severity_level: ContradictionSeverity,
}

#[derive(Debug, Clone)]
pub enum ContradictionSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl ContradictionResult {
    pub fn new(has_contradictions: bool, confidence: f32) -> Self {
        let severity_level = if !has_contradictions {
            ContradictionSeverity::None
        } else {
            match confidence {
                x if x < 0.3 => ContradictionSeverity::Low,
                x if x < 0.5 => ContradictionSeverity::Medium,
                x if x < 0.8 => ContradictionSeverity::High,
                _ => ContradictionSeverity::Critical,
            }
        };
        
        Self {
            has_contradictions,
            confidence,
            contradiction_details: Vec::new(),
            severity_level,
        }
    }
}
```

**Verification**: Compiles + severity classification works

### Task 2.5.3: Create AnomalyResult Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Define the AnomalyResult struct for anomaly detection results.

```rust
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub anomaly_score: f32,
    pub confidence: f32,
    pub anomaly_features: Vec<String>,
    pub anomaly_type: AnomalyType,
}

#[derive(Debug, Clone)]
pub enum AnomalyType {
    None,
    Statistical,
    Temporal,
    Structural,
    Behavioral,
}

impl AnomalyResult {
    pub fn new(anomaly_score: f32, confidence: f32) -> Self {
        let anomaly_type = match anomaly_score {
            x if x < 0.2 => AnomalyType::None,
            x if x < 0.4 => AnomalyType::Statistical,
            x if x < 0.6 => AnomalyType::Temporal,
            x if x < 0.8 => AnomalyType::Structural,
            _ => AnomalyType::Behavioral,
        };
        
        Self {
            anomaly_score,
            confidence,
            anomaly_features: Vec::new(),
            anomaly_type,
        }
    }
}
```

**Verification**: Compiles + anomaly type classification works

### Task 2.5.4: Create ExceptionColumn Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Define the main ExceptionColumn struct with required fields.

```rust
use super::cortical_column::{CorticalColumn, ColumnResult, ColumnType, LearningFeedback};
use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct ExceptionColumn {
    contradiction_detector: Arc<RwLock<ContradictionDetector>>,
    anomaly_analyzer: Arc<RwLock<AnomalyAnalyzer>>,
    activation_threshold: f32,
    current_activation: Arc<RwLock<f32>>,
}

impl ExceptionColumn {
    pub fn new_with_networks(network_types: Vec<ExceptionNetworkType>) -> Self {
        Self {
            contradiction_detector: Arc::new(RwLock::new(ContradictionDetector::new())),
            anomaly_analyzer: Arc::new(RwLock::new(AnomalyAnalyzer::new(network_types))),
            activation_threshold: 0.9, // High threshold for exceptions
            current_activation: Arc::new(RwLock::new(0.0)),
        }
    }
}
```

**Verification**: Compiles + constructor works

### Task 2.5.5: Implement ContradictionDetector
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Implement the ContradictionDetector for logical contradiction detection.

```rust
pub struct ContradictionDetector;

impl ContradictionDetector {
    pub fn new() -> Self {
        Self
    }
    
    pub async fn analyze_contradictions(&self, pattern: &TTFSPattern) -> MCPResult<ContradictionResult> {
        // Mock contradiction detection based on pattern characteristics
        let pattern_complexity = self.calculate_pattern_complexity(pattern);
        
        // Detect potential contradictions based on spike timing conflicts
        let has_contradictions = self.detect_timing_conflicts(&pattern.spikes);
        
        let confidence = if has_contradictions {
            pattern_complexity * 0.8 + 0.2
        } else {
            0.1
        };
        
        let mut result = ContradictionResult::new(has_contradictions, confidence);
        
        if has_contradictions {
            result.contradiction_details = vec![
                format!("timing_conflict_detected"),
                format!("complexity_score:{:.2}", pattern_complexity),
            ];
        }
        
        Ok(result)
    }
    
    fn calculate_pattern_complexity(&self, pattern: &TTFSPattern) -> f32 {
        if pattern.spikes.is_empty() {
            return 0.0;
        }
        
        // Calculate complexity based on spike distribution
        let unique_neurons: std::collections::HashSet<_> = pattern.spikes.iter()
            .map(|spike| spike.neuron_id)
            .collect();
        
        let neuron_diversity = unique_neurons.len() as f32 / pattern.spikes.len() as f32;
        neuron_diversity.min(1.0)
    }
    
    fn detect_timing_conflicts(&self, spikes: &[crate::core::types::TTFSSpike]) -> bool {
        // Mock conflict detection - check for unreasonably close timestamps
        for window in spikes.windows(2) {
            let time_diff = (window[1].timestamp - window[0].timestamp).abs();
            if time_diff < 0.001 && window[0].neuron_id == window[1].neuron_id {
                return true; // Same neuron firing too quickly
            }
        }
        false
    }
}
```

**Verification**: Compiles + contradiction detection works

### Task 2.5.6: Implement AnomalyAnalyzer
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Implement the AnomalyAnalyzer for statistical anomaly detection.

```rust
pub struct AnomalyAnalyzer {
    network_types: Vec<ExceptionNetworkType>,
}

impl AnomalyAnalyzer {
    pub fn new(network_types: Vec<ExceptionNetworkType>) -> Self {
        Self { network_types }
    }
    
    pub async fn detect_anomalies(&self, pattern: &TTFSPattern) -> MCPResult<AnomalyResult> {
        // Calculate statistical properties of the pattern
        let (mean_timestamp, std_dev) = self.calculate_timing_statistics(&pattern.spikes);
        
        // Detect anomalies based on statistical deviations
        let anomaly_score = self.calculate_anomaly_score(&pattern.spikes, mean_timestamp, std_dev);
        
        let confidence = (anomaly_score * 0.9 + 0.1).min(0.95);
        
        let mut result = AnomalyResult::new(anomaly_score, confidence);
        
        result.anomaly_features = vec![
            format!("timing_mean:{:.3}", mean_timestamp),
            format!("timing_std:{:.3}", std_dev),
            format!("anomaly_score:{:.3}", anomaly_score),
        ];
        
        Ok(result)
    }
    
    pub async fn update_exception_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        // Mock exception weight update based on feedback
        if feedback.success {
            // Successful exception detection - reinforce patterns
        } else {
            // False positive - adjust sensitivity
        }
        Ok(())
    }
    
    fn calculate_timing_statistics(&self, spikes: &[crate::core::types::TTFSSpike]) -> (f32, f32) {
        if spikes.is_empty() {
            return (0.0, 0.0);
        }
        
        let mean = spikes.iter().map(|s| s.timestamp).sum::<f32>() / spikes.len() as f32;
        
        let variance = spikes.iter()
            .map(|s| (s.timestamp - mean).powi(2))
            .sum::<f32>() / spikes.len() as f32;
        
        (mean, variance.sqrt())
    }
    
    fn calculate_anomaly_score(&self, spikes: &[crate::core::types::TTFSSpike], mean: f32, std_dev: f32) -> f32 {
        if spikes.is_empty() || std_dev == 0.0 {
            return 0.0;
        }
        
        // Calculate how many spikes are statistical outliers
        let outlier_count = spikes.iter()
            .filter(|spike| {
                let z_score = (spike.timestamp - mean).abs() / std_dev;
                z_score > 2.0 // More than 2 standard deviations
            })
            .count();
        
        (outlier_count as f32 / spikes.len() as f32).min(1.0)
    }
}
```

**Verification**: Compiles + anomaly detection works

### Task 2.5.7: Add Exception Analysis Methods
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Add private methods for exception analysis to the ExceptionColumn.

```rust
impl ExceptionColumn {
    async fn detect_contradictions(&self, pattern: &TTFSPattern) -> MCPResult<ContradictionResult> {
        let detector = self.contradiction_detector.read().await;
        let contradictions = detector.analyze_contradictions(pattern).await?;
        Ok(contradictions)
    }
    
    async fn analyze_anomalies(&self, pattern: &TTFSPattern) -> MCPResult<AnomalyResult> {
        let analyzer = self.anomaly_analyzer.read().await;
        let anomalies = analyzer.detect_anomalies(pattern).await?;
        Ok(anomalies)
    }
}
```

**Verification**: Compiles + analysis methods work

### Task 2.5.8: Implement CorticalColumn Trait for Exception
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/exception_column.rs`

**Task Prompt for AI**:
Implement the CorticalColumn trait for ExceptionColumn.

```rust
use async_trait::async_trait;

#[async_trait]
impl CorticalColumn for ExceptionColumn {
    async fn process(&self, pattern: &TTFSPattern) -> MCPResult<ColumnResult> {
        let start_time = std::time::Instant::now();
        
        let contradiction_result = self.detect_contradictions(pattern).await?;
        let anomaly_result = self.analyze_anomalies(pattern).await?;
        
        // Combine contradiction and anomaly scores
        let exception_score = (contradiction_result.confidence + anomaly_result.confidence) / 2.0;
        
        // Exception column activates strongly when problems are detected
        let activation_strength = if exception_score > self.activation_threshold {
            0.95 // High activation to flag potential issues
        } else {
            0.1  // Low activation when no exceptions detected
        };
        
        *self.current_activation.write().await = activation_strength;
        
        let neural_pathway = vec![
            format!("contradiction_check:{}", contradiction_result.has_contradictions),
            format!("anomaly_score:{:.2}", anomaly_result.anomaly_score),
            format!("severity:{:?}", contradiction_result.severity_level),
            format!("anomaly_type:{:?}", anomaly_result.anomaly_type),
        ];
        
        let processing_time = start_time.elapsed().as_millis() as f32;
        
        Ok(ColumnResult {
            activation_strength,
            confidence: exception_score,
            neural_pathway,
            processing_time_ms: processing_time,
            allocated_memory_id: None,
        })
    }
    
    async fn get_activation_strength(&self) -> f32 {
        *self.current_activation.read().await
    }
    
    async fn update_weights(&mut self, feedback: &LearningFeedback) -> MCPResult<()> {
        let mut analyzer = self.anomaly_analyzer.write().await;
        analyzer.update_exception_weights(feedback).await?;
        Ok(())
    }
    
    fn get_column_type(&self) -> ColumnType {
        ColumnType::Exception
    }
}
```

**Verification**: Compiles + trait implementation works correctly

### Task 2.6.1: Create CorticalConsensus Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Create the CorticalConsensus struct to hold lateral inhibition results.

```rust
use super::cortical_column::{ColumnResult, ColumnType};
use crate::mcp::errors::{MCPResult, MCPServerError};

#[derive(Debug, Clone)]
pub struct CorticalConsensus {
    pub winning_column: ColumnType,
    pub consensus_strength: f32,
    pub column_activations: Vec<(ColumnType, f32)>,
    pub inhibition_applied: bool,
    pub competition_statistics: CompetitionStats,
}

#[derive(Debug, Clone)]
pub struct CompetitionStats {
    pub total_activations: f32,
    pub max_activation: f32,
    pub min_activation: f32,
    pub activation_variance: f32,
}

impl CompetitionStats {
    pub fn calculate(activations: &[(ColumnType, f32)]) -> Self {
        if activations.is_empty() {
            return Self {
                total_activations: 0.0,
                max_activation: 0.0,
                min_activation: 0.0,
                activation_variance: 0.0,
            };
        }
        
        let values: Vec<f32> = activations.iter().map(|(_, v)| *v).collect();
        let total = values.iter().sum();
        let max_val = values.iter().fold(0.0, |a, b| a.max(*b));
        let min_val = values.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        
        let mean = total / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        
        Self {
            total_activations: total,
            max_activation: max_val,
            min_activation: min_val,
            activation_variance: variance,
        }
    }
}
```

**Verification**: Compiles + statistics calculation works

### Task 2.6.2: Create LateralInhibitionEngine Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Define the main LateralInhibitionEngine struct with configuration parameters.

```rust
pub struct LateralInhibitionEngine {
    inhibition_threshold: f32,
    winner_boost_factor: f32,
    loser_suppression_factor: f32,
    competitive_dynamics: CompetitiveDynamics,
}

#[derive(Debug, Clone)]
pub struct CompetitiveDynamics {
    pub adaptive_threshold: bool,
    pub winner_take_all: bool,
    pub soft_competition: bool,
}

impl Default for CompetitiveDynamics {
    fn default() -> Self {
        Self {
            adaptive_threshold: true,
            winner_take_all: false,
            soft_competition: true,
        }
    }
}

impl LateralInhibitionEngine {
    pub fn new() -> Self {
        Self {
            inhibition_threshold: 0.1,
            winner_boost_factor: 1.2,
            loser_suppression_factor: 0.3,
            competitive_dynamics: CompetitiveDynamics::default(),
        }
    }
    
    pub fn with_config(threshold: f32, boost: f32, suppression: f32) -> Self {
        Self {
            inhibition_threshold: threshold,
            winner_boost_factor: boost,
            loser_suppression_factor: suppression,
            competitive_dynamics: CompetitiveDynamics::default(),
        }
    }
}
```

**Verification**: Compiles + constructors work

### Task 2.6.3: Implement Column Type Determination
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Add method to map column indices to types.

```rust
impl LateralInhibitionEngine {
    fn determine_column_type(&self, index: usize) -> ColumnType {
        match index {
            0 => ColumnType::Semantic,
            1 => ColumnType::Structural,
            2 => ColumnType::Temporal,
            3 => ColumnType::Exception,
            _ => {
                // Cycle through types for additional columns
                match index % 4 {
                    0 => ColumnType::Semantic,
                    1 => ColumnType::Structural,
                    2 => ColumnType::Temporal,
                    _ => ColumnType::Exception,
                }
            }
        }
    }
    
    fn find_winning_column(&self, column_results: &[ColumnResult]) -> (usize, f32) {
        let mut max_activation = 0.0;
        let mut winning_index = 0;
        
        for (i, result) in column_results.iter().enumerate() {
            if result.activation_strength > max_activation {
                max_activation = result.activation_strength;
                winning_index = i;
            }
        }
        
        (winning_index, max_activation)
    }
}
```

**Verification**: Compiles + column mapping works

### Task 2.6.4: Implement Consensus Strength Calculation
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Add methods for calculating consensus strength and competition metrics.

```rust
impl LateralInhibitionEngine {
    fn calculate_consensus_strength(&self, activations: &[(ColumnType, f32)]) -> f32 {
        if activations.is_empty() {
            return 0.0;
        }
        
        let values: Vec<f32> = activations.iter().map(|(_, v)| *v).collect();
        let total_activation: f32 = values.iter().sum();
        let max_activation = values.iter().fold(0.0, f32::max);
        
        if total_activation == 0.0 {
            return 0.0;
        }
        
        // Consensus is stronger when one column dominates
        let dominance_ratio = max_activation / (total_activation / values.len() as f32);
        
        // Apply competitive dynamics modulation
        if self.competitive_dynamics.winner_take_all {
            dominance_ratio * 1.5
        } else if self.competitive_dynamics.soft_competition {
            dominance_ratio * 0.8
        } else {
            dominance_ratio
        }
    }
    
    fn calculate_adaptive_threshold(&self, column_results: &[ColumnResult]) -> f32 {
        if !self.competitive_dynamics.adaptive_threshold {
            return self.inhibition_threshold;
        }
        
        // Adapt threshold based on activation distribution
        let activations: Vec<f32> = column_results.iter()
            .map(|r| r.activation_strength)
            .collect();
        
        if activations.is_empty() {
            return self.inhibition_threshold;
        }
        
        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let std_dev = {
            let variance = activations.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / activations.len() as f32;
            variance.sqrt()
        };
        
        // Adaptive threshold based on standard deviation
        (self.inhibition_threshold + std_dev * 0.5).min(0.5)
    }
}
```

**Verification**: Compiles + consensus calculation works

### Task 2.6.5: Implement Main Lateral Inhibition Logic
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Implement the core lateral inhibition processing logic.

```rust
impl LateralInhibitionEngine {
    pub async fn apply_lateral_inhibition(
        &self,
        column_results: Vec<ColumnResult>
    ) -> MCPResult<CorticalConsensus> {
        if column_results.is_empty() {
            return Err(MCPServerError::ValidationError(
                "No column results provided for lateral inhibition".to_string()
            ));
        }
        
        // 1. Find the winning column (highest activation)
        let (winning_index, max_activation) = self.find_winning_column(&column_results);
        let winning_column = self.determine_column_type(winning_index);
        
        // 2. Calculate adaptive threshold
        let current_threshold = self.calculate_adaptive_threshold(&column_results);
        
        // 3. Apply inhibition based on activation differences
        let mut column_activations = Vec::new();
        let mut inhibition_applied = false;
        
        for (i, result) in column_results.iter().enumerate() {
            let column_type = self.determine_column_type(i);
            let mut adjusted_activation = result.activation_strength;
            
            if i == winning_index {
                // Boost the winner
                adjusted_activation *= self.winner_boost_factor;
            } else if max_activation - result.activation_strength > current_threshold {
                // Suppress losers based on competitive dynamics
                let suppression_factor = if self.competitive_dynamics.winner_take_all {
                    self.loser_suppression_factor * 0.5 // More aggressive suppression
                } else {
                    self.loser_suppression_factor
                };
                adjusted_activation *= suppression_factor;
                inhibition_applied = true;
            }
            
            column_activations.push((column_type, adjusted_activation));
        }
        
        // 4. Calculate consensus strength and statistics
        let consensus_strength = self.calculate_consensus_strength(&column_activations);
        let competition_statistics = CompetitionStats::calculate(&column_activations);
        
        Ok(CorticalConsensus {
            winning_column,
            consensus_strength,
            column_activations,
            inhibition_applied,
            competition_statistics,
        })
    }
}
```

**Verification**: Compiles + lateral inhibition logic works

### Task 2.6.6: Add Unit Tests for Lateral Inhibition
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/lateral_inhibition.rs`

**Task Prompt for AI**:
Add comprehensive unit tests for the lateral inhibition engine.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_mock_column_result(activation: f32, confidence: f32) -> ColumnResult {
        ColumnResult {
            activation_strength: activation,
            confidence,
            neural_pathway: vec!["test_path".to_string()],
            processing_time_ms: 10.0,
            allocated_memory_id: None,
        }
    }
    
    #[tokio::test]
    async fn test_lateral_inhibition_basic() {
        let inhibition_engine = LateralInhibitionEngine::new();
        
        let mock_results = vec![
            create_mock_column_result(0.9, 0.85),  // Semantic - should win
            create_mock_column_result(0.3, 0.4),   // Structural
            create_mock_column_result(0.2, 0.3),   // Temporal
            create_mock_column_result(0.1, 0.2),   // Exception
        ];
        
        let consensus = inhibition_engine.apply_lateral_inhibition(mock_results).await.unwrap();
        
        assert!(matches!(consensus.winning_column, ColumnType::Semantic));
        assert!(consensus.inhibition_applied);
        assert!(consensus.consensus_strength > 1.0);
        assert_eq!(consensus.column_activations.len(), 4);
    }
    
    #[tokio::test]
    async fn test_empty_column_results() {
        let inhibition_engine = LateralInhibitionEngine::new();
        let result = inhibition_engine.apply_lateral_inhibition(vec![]).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_close_competition() {
        let inhibition_engine = LateralInhibitionEngine::new();
        
        let mock_results = vec![
            create_mock_column_result(0.75, 0.8),  // Semantic
            create_mock_column_result(0.73, 0.79), // Structural - close competition
        ];
        
        let consensus = inhibition_engine.apply_lateral_inhibition(mock_results).await.unwrap();
        
        assert!(matches!(consensus.winning_column, ColumnType::Semantic));
        // With close competition, inhibition might not be applied
        assert!(consensus.consensus_strength > 0.0);
    }
    
    #[tokio::test]
    async fn test_consensus_strength_calculation() {
        let inhibition_engine = LateralInhibitionEngine::new();
        let activations = vec![
            (ColumnType::Semantic, 0.9),
            (ColumnType::Structural, 0.1),
            (ColumnType::Temporal, 0.1),
        ];
        
        let strength = inhibition_engine.calculate_consensus_strength(&activations);
        assert!(strength > 1.0); // Strong consensus with clear winner
    }
}
```

**Verification**: Compiles + all tests pass

### Task 2.7.1: Create Module Structure
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/mod.rs`

**Task Prompt for AI**:
Create the main neuromorphic module with all submodule declarations.

```rust
pub mod cortical_column;
pub mod semantic_column;
pub mod structural_column;
pub mod temporal_column;
pub mod exception_column;
pub mod lateral_inhibition;

// Re-export key types for easier access
pub use cortical_column::{CorticalColumn, ColumnResult, ColumnType, LearningFeedback, ProcessingStats};
pub use semantic_column::{SemanticColumn, NetworkType};
pub use structural_column::{StructuralColumn, GraphNetworkType, TopologyResult};
pub use temporal_column::{TemporalColumn, TemporalNetworkType, TemporalResult};
pub use exception_column::{ExceptionColumn, ExceptionNetworkType, ContradictionResult, AnomalyResult};
pub use lateral_inhibition::{LateralInhibitionEngine, CorticalConsensus, CompetitionStats};

use crate::core::types::TTFSPattern;
use crate::mcp::errors::MCPResult;
use tokio::sync::RwLock;
use std::sync::Arc;
```

**Verification**: Compiles + all modules accessible

### Task 2.7.2: Create NeuromorphicCore Struct
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/mod.rs`

**Task Prompt for AI**:
Define the main NeuromorphicCore struct that coordinates all cortical columns.

```rust
pub struct NeuromorphicCore {
    semantic_column: Arc<RwLock<SemanticColumn>>,
    structural_column: Arc<RwLock<StructuralColumn>>,
    temporal_column: Arc<RwLock<TemporalColumn>>,
    exception_column: Arc<RwLock<ExceptionColumn>>,
    lateral_inhibition: LateralInhibitionEngine,
    processing_stats: Arc<RwLock<CoreProcessingStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct CoreProcessingStats {
    pub total_patterns_processed: u64,
    pub average_processing_time_ms: f32,
    pub column_activation_history: Vec<ColumnActivationRecord>,
    pub consensus_strength_history: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ColumnActivationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub semantic_activation: f32,
    pub structural_activation: f32,
    pub temporal_activation: f32,
    pub exception_activation: f32,
    pub winning_column: ColumnType,
}
```

**Verification**: Compiles + struct definition works

### Task 2.7.3: Implement NeuromorphicCore Constructor
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/mod.rs`

**Task Prompt for AI**:
Implement constructor and configuration methods for NeuromorphicCore.

```rust
impl NeuromorphicCore {
    pub fn new() -> Self {
        Self {
            semantic_column: Arc::new(RwLock::new(SemanticColumn::new_with_networks(vec![
                NetworkType::MLP,
                NetworkType::TiDE,
            ]))),
            structural_column: Arc::new(RwLock::new(StructuralColumn::new_with_networks(vec![
                GraphNetworkType::StemGNN,
                GraphNetworkType::ITransformer,
            ]))),
            temporal_column: Arc::new(RwLock::new(TemporalColumn::new_with_networks(vec![
                TemporalNetworkType::LSTM,
                TemporalNetworkType::TCN,
            ]))),
            exception_column: Arc::new(RwLock::new(ExceptionColumn::new_with_networks(vec![
                ExceptionNetworkType::CascadeCorrelation,
                ExceptionNetworkType::SparseConnected,
            ]))),
            lateral_inhibition: LateralInhibitionEngine::new(),
            processing_stats: Arc::new(RwLock::new(CoreProcessingStats::default())),
        }
    }
    
    pub fn with_custom_inhibition(inhibition_engine: LateralInhibitionEngine) -> Self {
        let mut core = Self::new();
        core.lateral_inhibition = inhibition_engine;
        core
    }
    
    pub async fn get_processing_stats(&self) -> CoreProcessingStats {
        self.processing_stats.read().await.clone()
    }
}
```

**Verification**: Compiles + constructor creates all columns

### Task 2.7.4: Implement Parallel Processing Logic
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/mod.rs`

**Task Prompt for AI**:
Implement the core pattern processing logic with parallel column execution.

```rust
impl NeuromorphicCore {
    pub async fn process_pattern(&self, pattern: &TTFSPattern) -> MCPResult<CorticalConsensus> {
        let start_time = std::time::Instant::now();
        
        // Process pattern through all 4 cortical columns in parallel
        let (semantic_result, structural_result, temporal_result, exception_result) = tokio::join!(
            self.semantic_column.read().await.process(pattern),
            self.structural_column.read().await.process(pattern),
            self.temporal_column.read().await.process(pattern),
            self.exception_column.read().await.process(pattern)
        );
        
        // Collect all results, handling any errors
        let column_results = vec![
            semantic_result?,
            structural_result?,
            temporal_result?,
            exception_result?,
        ];
        
        // Apply lateral inhibition to determine consensus
        let consensus = self.lateral_inhibition.apply_lateral_inhibition(column_results).await?;
        
        // Update processing statistics
        let processing_time = start_time.elapsed().as_millis() as f32;
        self.update_processing_stats(&consensus, processing_time).await;
        
        Ok(consensus)
    }
    
    async fn update_processing_stats(&self, consensus: &CorticalConsensus, processing_time: f32) {
        let mut stats = self.processing_stats.write().await;
        
        stats.total_patterns_processed += 1;
        
        // Update running average of processing time
        let count = stats.total_patterns_processed as f32;
        stats.average_processing_time_ms = 
            (stats.average_processing_time_ms * (count - 1.0) + processing_time) / count;
        
        // Record column activations
        let activation_record = ColumnActivationRecord {
            timestamp: chrono::Utc::now(),
            semantic_activation: consensus.column_activations.get(0).map(|(_, a)| *a).unwrap_or(0.0),
            structural_activation: consensus.column_activations.get(1).map(|(_, a)| *a).unwrap_or(0.0),
            temporal_activation: consensus.column_activations.get(2).map(|(_, a)| *a).unwrap_or(0.0),
            exception_activation: consensus.column_activations.get(3).map(|(_, a)| *a).unwrap_or(0.0),
            winning_column: consensus.winning_column.clone(),
        };
        
        stats.column_activation_history.push(activation_record);
        stats.consensus_strength_history.push(consensus.consensus_strength);
        
        // Keep only last 1000 records to prevent unbounded growth
        if stats.column_activation_history.len() > 1000 {
            stats.column_activation_history.remove(0);
        }
        if stats.consensus_strength_history.len() > 1000 {
            stats.consensus_strength_history.remove(0);
        }
    }
}
```

**Verification**: Compiles + parallel processing works + stats updated

### Task 2.7.5: Add Learning and Feedback Methods
**Estimated Time**: 20 minutes  
**Expected Deliverable**: `src/mcp/neuromorphic/mod.rs`

**Task Prompt for AI**:
Add methods for applying learning feedback to all columns.

```rust
impl NeuromorphicCore {
    pub async fn apply_learning_feedback(&self, feedback: &LearningFeedback) -> MCPResult<()> {
        // Apply feedback to all columns in parallel
        let (semantic_result, structural_result, temporal_result, exception_result) = tokio::join!(
            self.semantic_column.write().await.update_weights(feedback),
            self.structural_column.write().await.update_weights(feedback),
            self.temporal_column.write().await.update_weights(feedback),
            self.exception_column.write().await.update_weights(feedback)
        );
        
        // Check that all updates succeeded
        semantic_result?;
        structural_result?;
        temporal_result?;
        exception_result?;
        
        Ok(())
    }
    
    pub async fn get_column_activation_strengths(&self) -> MCPResult<Vec<(ColumnType, f32)>> {
        let (semantic_strength, structural_strength, temporal_strength, exception_strength) = tokio::join!(
            self.semantic_column.read().await.get_activation_strength(),
            self.structural_column.read().await.get_activation_strength(),
            self.temporal_column.read().await.get_activation_strength(),
            self.exception_column.read().await.get_activation_strength()
        );
        
        Ok(vec![
            (ColumnType::Semantic, semantic_strength),
            (ColumnType::Structural, structural_strength),
            (ColumnType::Temporal, temporal_strength),
            (ColumnType::Exception, exception_strength),
        ])
    }
    
    pub async fn reset_column_activations(&self) -> MCPResult<()> {
        // Reset all column activations to zero
        let feedback = LearningFeedback {
            success: true,
            reward_signal: 0.0,
            pathway_trace: vec!["reset".to_string()],
        };
        
        self.apply_learning_feedback(&feedback).await?;
        Ok(())
    }
}
```

**Verification**: Compiles + learning feedback works + activation retrieval works

## Validation Checklist

**Core Structure (Tasks 2.1.1-2.1.4)**:
- [ ] ColumnType enum with string conversion
- [ ] ColumnResult struct with constructor
- [ ] LearningFeedback struct with success/failure constructors
- [ ] CorticalColumn trait with async methods

**Semantic Column (Tasks 2.2.1-2.2.6)**:
- [ ] NetworkType enum with pattern size selector
- [ ] SemanticColumn struct with constructor
- [ ] NetworkPool with network selection logic
- [ ] Mock neural network implementations
- [ ] Conceptual similarity analysis methods
- [ ] CorticalColumn trait implementation

**Structural Column (Tasks 2.3.1-2.3.7)**:
- [ ] GraphNetworkType enum with complexity selector
- [ ] TopologyResult struct with complexity calculation
- [ ] StructuralColumn struct with constructor
- [ ] TopologyAnalyzer with graph analysis
- [ ] GraphNetworkPool with path optimization
- [ ] Structural analysis methods
- [ ] CorticalColumn trait implementation

**Temporal Column (Tasks 2.4.1-2.4.7)**:
- [ ] TemporalNetworkType enum with sequence selector
- [ ] TemporalResult struct with complexity calculation
- [ ] TemporalColumn struct with constructor
- [ ] TemporalPatternDetector with regularity analysis
- [ ] SequenceAnalyzer with prediction methods
- [ ] Temporal analysis methods
- [ ] CorticalColumn trait implementation

**Exception Column (Tasks 2.5.1-2.5.8)**:
- [ ] ExceptionNetworkType enum with anomaly selector
- [ ] ContradictionResult struct with severity levels
- [ ] AnomalyResult struct with type classification
- [ ] ExceptionColumn struct with constructor
- [ ] ContradictionDetector with timing conflict detection
- [ ] AnomalyAnalyzer with statistical outlier detection
- [ ] Exception analysis methods
- [ ] CorticalColumn trait implementation

**Lateral Inhibition (Tasks 2.6.1-2.6.6)**:
- [ ] CorticalConsensus struct with competition statistics
- [ ] LateralInhibitionEngine with adaptive thresholds
- [ ] Column type determination and winner finding
- [ ] Consensus strength calculation with competitive dynamics
- [ ] Main lateral inhibition logic with adaptive processing
- [ ] Comprehensive unit tests covering edge cases

**Integration (Tasks 2.7.1-2.7.5)**:
- [ ] Module structure with proper re-exports
- [ ] NeuromorphicCore struct with processing statistics
- [ ] Constructor with default network configurations
- [ ] Parallel processing logic with statistics tracking
- [ ] Learning feedback and activation management

**Performance Requirements**:
- [ ] Each micro-task compiles independently
- [ ] All unit tests pass
- [ ] Processing time <50ms for typical patterns
- [ ] Memory usage <100MB for core structure
- [ ] Parallel processing shows performance improvement

**Quality Standards**:
- [ ] All structs have proper Debug/Clone derives
- [ ] Error handling with MCPResult throughout
- [ ] Async/await used correctly for I/O operations
- [ ] Mock implementations realistic enough for testing
- [ ] Statistics tracking for performance monitoring

## Next Phase Dependencies

This phase provides neuromorphic processing for:
- **MicroPhase 3**: Tool schemas will reference neuromorphic types
- **MicroPhase 4**: Tool implementation will use cortical column processing
- **MicroPhase 6**: Performance optimization will tune neuromorphic parameters
- **MicroPhase 7**: Testing will validate neuromorphic behavior patterns
- **MicroPhase 8**: Integration will connect neuromorphic core to MCP server

## Estimated Total Time: 4-5 hours (26 micro-tasks × 15-20 minutes each)