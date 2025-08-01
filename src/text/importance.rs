//! Heuristic importance scoring using graph-based methods
//!
//! This module provides graph-based importance scoring for knowledge entities.
//! Uses graph metrics like degree centrality, PageRank, and keyword analysis.

use crate::error::Result;
use std::collections::HashMap;

/// Graph metrics for importance calculation
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    pub degree_centrality: f32,
    pub betweenness_centrality: f32,
    pub closeness_centrality: f32,
    pub pagerank_score: f32,
    pub clustering_coefficient: f32,
    pub connection_count: usize,
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            degree_centrality: 0.0,
            betweenness_centrality: 0.0,
            closeness_centrality: 0.0,
            pagerank_score: 0.1, // Minimum PageRank value
            clustering_coefficient: 0.0,
            connection_count: 0,
        }
    }
}

/// Heuristic importance scorer using graph-based analysis
#[derive(Clone)]
pub struct HeuristicImportanceScorer {
    keyword_weights: HashMap<String, f32>,
    graph_weight: f32,
    keyword_weight: f32,
    frequency_weight: f32,
}

impl HeuristicImportanceScorer {
    pub fn new() -> Self {
        Self {
            keyword_weights: Self::create_keyword_weights(),
            graph_weight: 0.4,
            keyword_weight: 0.3,
            frequency_weight: 0.3,
        }
    }

    /// Calculate importance score using graph-based heuristics
    pub fn calculate_importance(&self, text: &str, graph_metrics: Option<GraphMetrics>) -> f32 {
        let keyword_score = self.calculate_keyword_score(text);
        let frequency_score = self.calculate_frequency_score(text);
        let graph_score = self.calculate_graph_score(graph_metrics);
        
        // Weighted combination of scores
        self.keyword_weight * keyword_score +
        self.frequency_weight * frequency_score +
        self.graph_weight * graph_score
    }

    /// Calculate importance using async interface for compatibility
    pub async fn calculate_importance_async(&self, text: &str, graph_metrics: Option<GraphMetrics>) -> Result<f32> {
        Ok(self.calculate_importance(text, graph_metrics))
    }

    /// Calculate batch importance scores
    pub fn calculate_batch_importance(&self, texts: &[String], metrics: &[Option<GraphMetrics>]) -> Result<Vec<f32>> {
        if texts.len() != metrics.len() {
            return Err(crate::error::GraphError::InvalidInput("Texts and metrics arrays must have same length".to_string()));
        }
        
        let scores = texts.iter()
            .zip(metrics.iter())
            .map(|(text, metric)| self.calculate_importance(text, metric.clone()))
            .collect();
        
        Ok(scores)
    }

    /// Calculate contextual importance with related concepts
    pub fn calculate_contextual_importance(&self, text: &str, context: &str, graph_metrics: Option<GraphMetrics>) -> f32 {
        let base_score = self.calculate_importance(text, graph_metrics);
        let context_bonus = self.calculate_context_bonus(text, context);
        
        (base_score + context_bonus).min(1.0)
    }

    /// Calculate relative importance between texts
    pub fn calculate_relative_importance(&self, texts: &[String]) -> Vec<f32> {
        let raw_scores: Vec<f32> = texts.iter()
            .map(|text| self.calculate_importance(text, None))
            .collect();
        
        // Normalize to relative scores (0-1)
        let max_score = raw_scores.iter().fold(0.0f32, |acc, &x| acc.max(x));
        if max_score > 0.0 {
            raw_scores.iter().map(|&score| score / max_score).collect()
        } else {
            vec![0.0; texts.len()]
        }
    }

    /// Calculate keyword-based importance score
    fn calculate_keyword_score(&self, text: &str) -> f32 {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding
            .split_whitespace()
            .collect();
        
        let mut score = 0.0;
        let mut total_weight = 0.0;
        
        for word in words {
            if let Some(&weight) = self.keyword_weights.get(word) {
                score += weight;
                total_weight += 1.0;
            } else if word.len() > 5 {
                // Long words get some importance
                score += 0.2;
                total_weight += 1.0;
            }
        }
        
        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.1 // Minimum baseline score
        }
    }

    /// Calculate frequency-based importance score
    fn calculate_frequency_score(&self, text: &str) -> f32 {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding
            .split_whitespace()
            .collect();
        
        let mut word_freq = HashMap::new();
        for word in &words {
            *word_freq.entry(*word).or_insert(0) += 1;
        }
        
        // Calculate TF-IDF-like score
        let mut score = 0.0;
        let doc_length = words.len() as f32;
        
        for (word, freq) in word_freq {
            if word.len() > 2 {
                let tf = freq as f32 / doc_length;
                let idf = (100.0 / (freq as f32 + 1.0)).ln(); // Simplified IDF
                score += tf * idf;
            }
        }
        
        score.min(1.0) // Cap at 1.0
    }

    /// Calculate graph-based importance score
    fn calculate_graph_score(&self, metrics: Option<GraphMetrics>) -> f32 {
        match metrics {
            Some(m) => {
                // Weighted combination of graph metrics
                0.3 * m.degree_centrality +
                0.2 * m.betweenness_centrality +
                0.2 * m.closeness_centrality +
                0.2 * m.pagerank_score +
                0.1 * m.clustering_coefficient
            },
            None => 0.1 // Default minimal importance
        }
    }

    /// Calculate context bonus for related concepts
    fn calculate_context_bonus(&self, text: &str, context: &str) -> f32 {
        let text_binding = text.to_lowercase();
        let text_words: Vec<&str> = text_binding.split_whitespace().collect();
        let context_binding = context.to_lowercase();
        let context_words: Vec<&str> = context_binding.split_whitespace().collect();
        
        let mut overlap_count = 0;
        for text_word in &text_words {
            if context_words.contains(text_word) && text_word.len() > 3 {
                overlap_count += 1;
            }
        }
        
        if !text_words.is_empty() {
            (overlap_count as f32 / text_words.len() as f32) * 0.2 // 20% max bonus
        } else {
            0.0
        }
    }

    /// Create keyword importance weights
    fn create_keyword_weights() -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        
        // Technical terms
        weights.insert("algorithm".to_string(), 0.9);
        weights.insert("system".to_string(), 0.8);
        weights.insert("framework".to_string(), 0.8);
        weights.insert("architecture".to_string(), 0.8);
        weights.insert("optimization".to_string(), 0.8);
        weights.insert("performance".to_string(), 0.7);
        weights.insert("security".to_string(), 0.8);
        weights.insert("scalability".to_string(), 0.8);
        weights.insert("reliability".to_string(), 0.7);
        weights.insert("efficiency".to_string(), 0.7);
        
        // Knowledge graph terms
        weights.insert("knowledge".to_string(), 0.9);
        weights.insert("graph".to_string(), 0.8);
        weights.insert("node".to_string(), 0.7);
        weights.insert("edge".to_string(), 0.7);
        weights.insert("relationship".to_string(), 0.8);
        weights.insert("entity".to_string(), 0.7);
        weights.insert("semantic".to_string(), 0.8);
        weights.insert("ontology".to_string(), 0.9);
        weights.insert("reasoning".to_string(), 0.8);
        weights.insert("inference".to_string(), 0.8);
        
        // AI/ML terms
        weights.insert("machine".to_string(), 0.8);
        weights.insert("learning".to_string(), 0.8);
        weights.insert("model".to_string(), 0.8);
        weights.insert("network".to_string(), 0.7);
        weights.insert("model".to_string(), 0.7);
        weights.insert("training".to_string(), 0.7);
        weights.insert("prediction".to_string(), 0.7);
        weights.insert("classification".to_string(), 0.7);
        weights.insert("clustering".to_string(), 0.7);
        weights.insert("embedding".to_string(), 0.8);
        
        // Data terms
        weights.insert("database".to_string(), 0.7);
        weights.insert("storage".to_string(), 0.6);
        weights.insert("indexing".to_string(), 0.7);
        weights.insert("query".to_string(), 0.7);
        weights.insert("search".to_string(), 0.6);
        weights.insert("retrieval".to_string(), 0.7);
        weights.insert("processing".to_string(), 0.6);
        weights.insert("analysis".to_string(), 0.6);
        weights.insert("mining".to_string(), 0.7);
        weights.insert("extraction".to_string(), 0.6);
        
        // Business terms
        weights.insert("strategy".to_string(), 0.7);
        weights.insert("solution".to_string(), 0.6);
        weights.insert("implementation".to_string(), 0.6);
        weights.insert("integration".to_string(), 0.6);
        weights.insert("deployment".to_string(), 0.6);
        weights.insert("maintenance".to_string(), 0.5);
        weights.insert("monitoring".to_string(), 0.6);
        weights.insert("analytics".to_string(), 0.7);
        weights.insert("intelligence".to_string(), 0.8);
        weights.insert("automation".to_string(), 0.7);
        
        weights
    }
}

impl Default for HeuristicImportanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_importance_scoring() {
        let scorer = HeuristicImportanceScorer::new();
        
        let high_importance = "machine learning algorithm optimization framework";
        let low_importance = "the quick brown fox jumps";
        
        let high_score = scorer.calculate_importance(high_importance, None);
        let low_score = scorer.calculate_importance(low_importance, None);
        
        assert!(high_score > low_score);
        assert!(high_score > 0.5);
        assert!(low_score < 0.3);
    }

    #[test]
    fn test_graph_metrics_integration() {
        let scorer = HeuristicImportanceScorer::new();
        
        let metrics = GraphMetrics {
            degree_centrality: 0.8,
            pagerank_score: 0.7,
            connection_count: 10,
            ..Default::default()
        };
        
        let text = "important concept";
        let score_without_metrics = scorer.calculate_importance(text, None);
        let score_with_metrics = scorer.calculate_importance(text, Some(metrics));
        
        assert!(score_with_metrics > score_without_metrics);
    }

    #[test]
    fn test_contextual_importance() {
        let scorer = HeuristicImportanceScorer::new();
        
        let text = "decision tree model";
        let context = "machine learning system with pattern recognition components";
        
        let base_score = scorer.calculate_importance(text, None);
        let contextual_score = scorer.calculate_contextual_importance(text, context, None);
        
        assert!(contextual_score >= base_score);
    }

    #[test]
    fn test_batch_importance_scoring() {
        let scorer = HeuristicImportanceScorer::new();
        
        let texts = vec![
            "machine learning algorithm".to_string(),
            "simple text here".to_string(),
            "knowledge graph reasoning".to_string(),
        ];
        
        let metrics = vec![None, None, None];
        let scores = scorer.calculate_batch_importance(&texts, &metrics).unwrap();
        
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]); // ML should score higher than simple text
        assert!(scores[2] > scores[1]); // KG should score higher than simple text
    }

    #[test]
    fn test_relative_importance() {
        let scorer = HeuristicImportanceScorer::new();
        
        let texts = vec![
            "algorithm optimization".to_string(),
            "cat dog".to_string(),
            "decision tree".to_string(),
        ];
        
        let scores = scorer.calculate_relative_importance(&texts);
        assert_eq!(scores.len(), 3);
        
        // Scores should be normalized (highest should be 1.0)
        let max_score = scores.iter().fold(0.0f32, |acc, &x| acc.max(x));
        assert!((max_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_performance() {
        let scorer = HeuristicImportanceScorer::new();
        let text = "machine learning algorithm for knowledge graph reasoning and semantic analysis";
        
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = scorer.calculate_importance(text, None);
        }
        let elapsed = start.elapsed();
        
        // Should process 1000 calculations in under 100ms
        assert!(elapsed.as_millis() < 100);
        println!("1000 importance calculations took: {elapsed:?}");
    }
}