//! Test corpus module for LLMKG accuracy validation
//! 
//! This module provides comprehensive ground truth test data and accuracy measurement
//! tools for validating entity extraction and relationship extraction systems.
//! 
//! # Phase 1 Requirements
//! 
//! This module validates that the LLMKG system meets Phase 1 accuracy requirements:
//! - Entity extraction accuracy > 95%
//! - Relationship extraction accuracy > 85%
//! - Question answering accuracy > 85%
//! 
//! # Usage
//! 
//! ```rust
//! use llmkg::tests::corpus::{AccuracyTester, CorpusGenerator};
//! 
//! // Load test corpus and evaluate accuracy
//! let tester = AccuracyTester::new();
//! let corpus = AccuracyTester::load_test_corpus("tests/corpus/test_corpus.json")?;
//! let report = tester.evaluate_accuracy(&corpus);
//! 
//! println!("Entity Accuracy: {:.2}%", report.entity_metrics.accuracy_percentage);
//! println!("Relationship Accuracy: {:.2}%", report.relationship_metrics.accuracy_percentage);
//! println!("Meets Phase 1 Requirements: {}", report.meets_phase1_requirements);
//! ```

pub mod accuracy_tests;
pub mod corpus_generator;

pub use accuracy_tests::{
    AccuracyTester, ComprehensiveAccuracyReport, EntityAccuracyMetrics, 
    RelationshipAccuracyMetrics, TestCorpus, TestSample, GroundTruthEntity, 
    GroundTruthRelationship, TestCorpusMetadata
};

pub use corpus_generator::{
    CorpusGenerator, TemplatePattern, TemplateVariable, EntityTemplate, RelationshipTemplate
};

/// Convenience function to run a comprehensive accuracy test
pub fn run_accuracy_test(corpus_path: &str) -> Result<ComprehensiveAccuracyReport, Box<dyn std::error::Error>> {
    let tester = AccuracyTester::new();
    let corpus = AccuracyTester::load_test_corpus(corpus_path)?;
    Ok(tester.evaluate_accuracy(&corpus))
}

/// Convenience function to generate additional test samples
pub fn generate_test_samples(num_samples: usize) -> Vec<TestSample> {
    let generator = CorpusGenerator::new();
    generator.generate_samples(num_samples)
}

/// Convenience function to create a balanced corpus across domains
pub fn create_balanced_test_corpus(samples_per_domain: usize) -> TestCorpus {
    let generator = CorpusGenerator::new();
    generator.create_balanced_corpus(samples_per_domain)
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_full_accuracy_pipeline() {
        // Generate a small test corpus
        let corpus = create_balanced_test_corpus(2);
        assert!(!corpus.samples.is_empty());
        
        // Test accuracy evaluation
        let tester = AccuracyTester::new();
        let report = tester.evaluate_accuracy(&corpus);
        
        // Should have meaningful metrics
        assert!(report.entity_metrics.total_ground_truth > 0);
        assert!(report.relationship_metrics.total_ground_truth > 0);
        assert!(report.overall_score >= 0.0 && report.overall_score <= 100.0);
        
        // Generate detailed report
        let detailed_report = tester.generate_detailed_report(&report);
        assert!(!detailed_report.is_empty());
        assert!(detailed_report.contains("Entity Accuracy"));
        assert!(detailed_report.contains("Relationship Accuracy"));
    }

    #[test]
    fn test_corpus_extension() {
        // Create initial corpus
        let mut corpus = create_balanced_test_corpus(1);
        let initial_size = corpus.samples.len();
        
        // Extend corpus
        let generator = CorpusGenerator::new();
        generator.extend_corpus(&mut corpus, 5);
        
        // Should have more samples
        assert!(corpus.samples.len() == initial_size + 5);
        assert_eq!(corpus.metadata.total_samples, corpus.samples.len());
    }

    #[test]
    fn test_corpus_file_operations() {
        use tempfile::NamedTempFile;
        
        // Create test corpus
        let corpus = create_balanced_test_corpus(2);
        
        // Save to temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();
        
        let generator = CorpusGenerator::new();
        generator.save_corpus_to_file(&corpus, temp_path).unwrap();
        
        // Load back and verify
        let loaded_corpus = AccuracyTester::load_test_corpus(temp_path).unwrap();
        assert_eq!(loaded_corpus.samples.len(), corpus.samples.len());
        assert_eq!(loaded_corpus.metadata.total_samples, corpus.metadata.total_samples);
    }

    #[test]
    fn test_accuracy_requirements_validation() {
        // This is an integration test to ensure our accuracy test can detect
        // when systems don't meet Phase 1 requirements
        
        let corpus = create_balanced_test_corpus(3);
        let tester = AccuracyTester::new();
        let report = tester.evaluate_accuracy(&corpus);
        
        // The report should clearly indicate whether requirements are met
        if !report.meets_phase1_requirements {
            assert!(!report.detailed_errors.is_empty());
            
            // Should have specific error messages about accuracy thresholds
            let error_text = report.detailed_errors.join(" ");
            if report.entity_metrics.accuracy_percentage < 95.0 {
                assert!(error_text.contains("Entity extraction accuracy"));
                assert!(error_text.contains("95%"));
            }
            if report.relationship_metrics.accuracy_percentage < 85.0 {
                assert!(error_text.contains("Relationship extraction accuracy"));
                assert!(error_text.contains("85%"));
            }
        }
        
        // Overall score should be reasonable
        assert!(report.overall_score >= 0.0);
        assert!(report.overall_score <= 100.0);
    }
}