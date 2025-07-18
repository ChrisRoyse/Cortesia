use serde::{Deserialize, Serialize};
use std::sync::Arc;
use ahash::AHashMap;

use crate::core::brain_types::{EntityDirection, LogicGateType, GraphOperation, TrainingExample, RelationType};
use crate::neural::neural_server::{NeuralProcessingServer, PredictionResult};
use crate::error::Result;

/// Graph structure predictor using neural networks
#[derive(Clone)]
pub struct GraphStructurePredictor {
    pub model_id: String,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub training_data: Vec<TrainingExample>,
    pub vocabulary: Arc<Vocabulary>,
}

/// Vocabulary for tokenizing text
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub word_to_id: AHashMap<String, usize>,
    pub id_to_word: Vec<String>,
    pub special_tokens: SpecialTokens,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub pad_token: String,
    pub unk_token: String,
    pub cls_token: String,
    pub sep_token: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad_token: "<PAD>".to_string(),
            unk_token: "<UNK>".to_string(),
            cls_token: "<CLS>".to_string(),
            sep_token: "<SEP>".to_string(),
        }
    }
}

/// Training metrics for structure prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub loss: f32,
}

impl GraphStructurePredictor {
    pub fn new(model_id: String, neural_server: Arc<NeuralProcessingServer>) -> Self {
        Self {
            model_id,
            neural_server,
            training_data: Vec::new(),
            vocabulary: Arc::new(Vocabulary::new()),
        }
    }

    /// Predict graph structure from input text
    pub async fn predict_structure(
        &self,
        input_text: &str,
    ) -> Result<Vec<GraphOperation>> {
        // 1. Tokenize and encode input text
        let encoded_input = self.encode_text(input_text)?;
        
        // 2. Get prediction from neural model
        let prediction = self.neural_server.neural_predict(
            &self.model_id,
            encoded_input,
        ).await?;
        
        // 3. Decode prediction into graph operations
        let operations = self.decode_prediction(prediction, input_text)?;
        
        Ok(operations)
    }

    /// Train the structure prediction model
    pub async fn train_from_examples(
        &mut self,
        examples: Vec<TrainingExample>,
    ) -> Result<TrainingMetrics> {
        // Store training examples
        self.training_data.extend(examples.clone());
        
        // Build vocabulary from all training examples
        let training_data_clone = self.training_data.clone();
        self.build_vocabulary(&training_data_clone)?;
        
        // Prepare training data
        let (inputs, targets) = self.prepare_training_data(&examples)?;
        
        // Convert to format expected by neural server
        let training_dataset = serde_json::json!({
            "inputs": inputs,
            "targets": targets,
            "vocab_size": self.vocabulary.word_to_id.len(),
        });
        
        // Train the model
        let training_result = self.neural_server.neural_train(
            &self.model_id,
            &training_dataset.to_string(),
            100, // epochs
        ).await?;
        
        // Calculate metrics
        let metrics = TrainingMetrics {
            accuracy: training_result.metrics.get("accuracy").copied().unwrap_or(0.0),
            precision: training_result.metrics.get("precision").copied().unwrap_or(0.0),
            recall: training_result.metrics.get("recall").copied().unwrap_or(0.0),
            f1_score: training_result.metrics.get("f1_score").copied().unwrap_or(0.0),
            loss: training_result.final_loss,
        };
        
        Ok(metrics)
    }

    /// Encode text to numerical representation
    fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenize(text);
        let mut encoded = Vec::new();
        
        for token in tokens {
            let token_id = self.vocabulary.word_to_id
                .get(&token)
                .or_else(|| self.vocabulary.word_to_id.get(&self.vocabulary.special_tokens.unk_token))
                .copied()
                .unwrap_or(0);
            
            encoded.push(token_id as f32);
        }
        
        // Pad or truncate to fixed length
        let max_length = 128;
        encoded.resize(max_length, 0.0);
        
        Ok(encoded)
    }

    /// Decode neural prediction into graph operations
    fn decode_prediction(
        &self,
        prediction: PredictionResult,
        original_text: &str,
    ) -> Result<Vec<GraphOperation>> {
        let mut operations = Vec::new();
        
        // Parse prediction values
        // This is a simplified decoder - in practice, this would be more sophisticated
        let chunks: Vec<&[f32]> = prediction.prediction.chunks(4).collect();
        
        for chunk in chunks {
            if chunk.len() < 4 {
                continue;
            }
            
            let op_type = chunk[0];
            let confidence = chunk[1];
            
            // Skip low confidence predictions
            if confidence < 0.5 {
                continue;
            }
            
            let operation = self.decode_operation_type(op_type, chunk, original_text)?;
            if let Some(op) = operation {
                operations.push(op);
            }
        }
        
        // If no operations predicted, create basic structure from text
        if operations.is_empty() {
            operations = self.create_default_structure(original_text)?;
        }
        
        Ok(operations)
    }

    /// Decode a single operation from prediction values
    fn decode_operation_type(
        &self,
        op_type: f32,
        chunk: &[f32],
        text: &str,
    ) -> Result<Option<GraphOperation>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        match (op_type * 10.0) as u8 {
            0..=3 => {
                // CreateNode operation
                if let Some(concept) = self.extract_concept_from_text(text, chunk) {
                    let node_type = if chunk[2] > 0.5 {
                        EntityDirection::Output
                    } else {
                        EntityDirection::Input
                    };
                    
                    Ok(Some(GraphOperation::CreateNode {
                        concept,
                        node_type,
                    }))
                } else {
                    Ok(None)
                }
            }
            4..=6 => {
                // CreateLogicGate operation
                let gate_type = match (chunk[2] * 5.0) as u8 {
                    0 => LogicGateType::And,
                    1 => LogicGateType::Or,
                    2 => LogicGateType::Not,
                    3 => LogicGateType::Inhibitory,
                    _ => LogicGateType::Weighted,
                };
                
                // Extract inputs and outputs from text
                let (inputs, outputs) = self.extract_gate_connections(text)?;
                
                if !inputs.is_empty() && !outputs.is_empty() {
                    Ok(Some(GraphOperation::CreateLogicGate {
                        inputs,
                        outputs,
                        gate_type,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => {
                // CreateRelationship operation
                if words.len() >= 3 {
                    let source = words[0].to_string();
                    let relation_type = match words[1] {
                        "is_a" => RelationType::IsA,
                        "has_property" => RelationType::HasProperty,
                        "has_instance" => RelationType::HasInstance,
                        "related_to" => RelationType::RelatedTo,
                        _ => RelationType::RelatedTo,
                    };
                    let target = words[2].to_string();
                    let weight = chunk[3].abs().min(1.0);
                    
                    Ok(Some(GraphOperation::CreateRelationship {
                        source,
                        target,
                        relation_type,
                        weight,
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Create default structure when prediction fails
    fn create_default_structure(&self, text: &str) -> Result<Vec<GraphOperation>> {
        let mut operations = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Example: "Pluto is a dog" -> Create nodes and relationship
        if words.len() >= 3 {
            // Create input node for subject
            operations.push(GraphOperation::CreateNode {
                concept: words[0].to_string(),
                node_type: EntityDirection::Input,
            });
            
            // Create output node for object
            operations.push(GraphOperation::CreateNode {
                concept: words[2].to_string(),
                node_type: EntityDirection::Output,
            });
            
            // Create relationship
            operations.push(GraphOperation::CreateRelationship {
                source: words[0].to_string(),
                target: words[2].to_string(),
                relation_type: match words[1] {
                    "is_a" => RelationType::IsA,
                    "has_property" => RelationType::HasProperty,
                    "has_instance" => RelationType::HasInstance,
                    "related_to" => RelationType::RelatedTo,
                    _ => RelationType::RelatedTo,
                },
                weight: 1.0,
            });
            
            // Create logic gate to represent the relationship
            operations.push(GraphOperation::CreateLogicGate {
                inputs: vec![words[0].to_string(), words[1].to_string()],
                outputs: vec![words[2].to_string()],
                gate_type: LogicGateType::And,
            });
        }
        
        Ok(operations)
    }

    /// Extract concept from text based on prediction
    fn extract_concept_from_text(&self, text: &str, chunk: &[f32]) -> Option<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_index = (chunk[3] * words.len() as f32) as usize;
        
        words.get(word_index).map(|w| w.to_string())
    }

    /// Extract gate connections from text
    fn extract_gate_connections(&self, text: &str) -> Result<(Vec<String>, Vec<String>)> {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Simple heuristic: first word is input, last word is output
        let inputs = if !words.is_empty() {
            vec![words[0].to_string()]
        } else {
            vec![]
        };
        
        let outputs = if words.len() > 1 {
            vec![words[words.len() - 1].to_string()]
        } else {
            vec![]
        };
        
        Ok((inputs, outputs))
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// Build vocabulary from training examples
    fn build_vocabulary(&mut self, examples: &[TrainingExample]) -> Result<()> {
        let mut word_freq = AHashMap::new();
        
        // Count word frequencies
        for example in examples {
            let tokens = self.tokenize(&example.text);
            for token in tokens {
                *word_freq.entry(token).or_insert(0) += 1;
            }
        }
        
        // Create vocabulary with most frequent words
        let mut vocab = Vocabulary::new();
        let mut words: Vec<_> = word_freq.into_iter().collect();
        words.sort_by_key(|(_, freq)| std::cmp::Reverse(*freq));
        
        for (word, _) in words.iter().take(10000) {
            vocab.add_word(word.clone());
        }
        
        self.vocabulary = Arc::new(vocab);
        Ok(())
    }

    /// Prepare training data for neural network
    fn prepare_training_data(
        &self,
        examples: &[TrainingExample],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        for example in examples {
            // Encode input text
            let input = self.encode_text(&example.text)?;
            inputs.push(input);
            
            // Encode target operations
            let target = self.encode_operations(&example.expected_operations)?;
            targets.push(target);
        }
        
        Ok((inputs, targets))
    }

    /// Encode graph operations as target vector
    fn encode_operations(&self, operations: &[GraphOperation]) -> Result<Vec<f32>> {
        let mut encoded = Vec::new();
        
        for op in operations {
            let op_encoding = match op {
                GraphOperation::CreateNode { node_type, .. } => {
                    vec![
                        0.1, // Operation type: CreateNode
                        1.0, // Confidence
                        if matches!(node_type, EntityDirection::Output) { 1.0 } else { 0.0 },
                        0.0, // Padding
                    ]
                }
                GraphOperation::CreateLogicGate { gate_type, .. } => {
                    let gate_value = match gate_type {
                        LogicGateType::And => 0.0,
                        LogicGateType::Or => 0.2,
                        LogicGateType::Not => 0.4,
                        LogicGateType::Xor => 0.1,
                        LogicGateType::Nand => 0.3,
                        LogicGateType::Nor => 0.5,
                        LogicGateType::Xnor => 0.7,
                        LogicGateType::Identity => 0.9,
                        LogicGateType::Threshold => 0.95,
                        LogicGateType::Inhibitory => 0.6,
                        LogicGateType::Weighted => 0.8,
                    };
                    vec![
                        0.5, // Operation type: CreateLogicGate
                        1.0, // Confidence
                        gate_value,
                        0.0, // Padding
                    ]
                }
                GraphOperation::CreateRelationship { weight, .. } => {
                    vec![
                        0.9, // Operation type: CreateRelationship
                        1.0, // Confidence
                        0.5, // Placeholder
                        *weight,
                    ]
                }
            };
            
            encoded.extend(op_encoding);
        }
        
        // Pad to fixed size
        let target_size = 512;
        encoded.resize(target_size, 0.0);
        
        Ok(encoded)
    }
}

impl Vocabulary {
    pub fn new() -> Self {
        let special_tokens = SpecialTokens::default();
        let mut vocab = Self {
            word_to_id: AHashMap::new(),
            id_to_word: vec![],
            special_tokens: special_tokens.clone(),
        };
        
        // Add special tokens
        vocab.add_word(special_tokens.pad_token);
        vocab.add_word(special_tokens.unk_token);
        vocab.add_word(special_tokens.cls_token);
        vocab.add_word(special_tokens.sep_token);
        
        vocab
    }

    pub fn add_word(&mut self, word: String) {
        if !self.word_to_id.contains_key(&word) {
            let id = self.id_to_word.len();
            self.word_to_id.insert(word.clone(), id);
            self.id_to_word.push(word);
        }
    }

    pub fn get_id(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(word).copied()
    }

    pub fn get_word(&self, id: usize) -> Option<&str> {
        self.id_to_word.get(id).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::neural_server::NeuralProcessingServer;

    #[tokio::test]
    async fn test_structure_predictor_creation() {
        let neural_server = Arc::new(
            NeuralProcessingServer::new("localhost:9000".to_string()).await.unwrap()
        );
        let predictor = GraphStructurePredictor::new(
            "structure_model".to_string(),
            neural_server,
        );
        
        assert_eq!(predictor.model_id, "structure_model");
    }

    #[tokio::test]
    async fn test_default_structure_creation() {
        let neural_server = Arc::new(
            NeuralProcessingServer::new("localhost:9000".to_string()).await.unwrap()
        );
        let predictor = GraphStructurePredictor::new(
            "structure_model".to_string(),
            neural_server,
        );
        
        let operations = predictor.create_default_structure("Pluto is a dog").unwrap();
        assert!(!operations.is_empty());
        
        // Check that we have node creation operations
        let has_node_ops = operations.iter().any(|op| {
            matches!(op, GraphOperation::CreateNode { .. })
        });
        assert!(has_node_ops);
    }

    #[test]
    fn test_vocabulary() {
        let mut vocab = Vocabulary::new();
        vocab.add_word("hello".to_string());
        vocab.add_word("world".to_string());
        
        assert_eq!(vocab.get_id("hello"), Some(4)); // After special tokens
        assert_eq!(vocab.get_word(4), Some("hello"));
    }

    #[test]
    fn test_tokenization() {
        let neural_server = Arc::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(NeuralProcessingServer::new("localhost:9000".to_string()))
                .unwrap()
        );
        let predictor = GraphStructurePredictor::new(
            "test".to_string(),
            neural_server,
        );
        
        let tokens = predictor.tokenize("Hello World Test");
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }
}