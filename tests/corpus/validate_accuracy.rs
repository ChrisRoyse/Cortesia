use std::path::Path;
use std::env;

// Import the corpus modules directly from the test crate
mod accuracy_tests;
mod corpus_generator;

use accuracy_tests::{AccuracyTester, TestCorpus};
use serde_json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== LLMKG Phase 1 Accuracy Validation ===\n");

    // Get corpus path from command line or use default
    let args: Vec<String> = env::args().collect();
    let corpus_path = if args.len() > 1 {
        &args[1]
    } else {
        "tests/corpus/test_corpus.json"
    };

    // Check if corpus file exists
    if !Path::new(corpus_path).exists() {
        eprintln!("❌ Error: Test corpus file not found at: {}", corpus_path);
        eprintln!("Please ensure the test corpus file exists or provide a valid path as an argument.");
        std::process::exit(1);
    }

    println!("📊 Loading test corpus from: {}", corpus_path);

    // Run accuracy test
    let tester = AccuracyTester::new();
    let corpus = AccuracyTester::load_test_corpus(corpus_path)?;
    let report = tester.evaluate_accuracy(&corpus);
    
    // Generate and display detailed report
    let detailed_report = tester.generate_detailed_report(&report);
    println!("{}", detailed_report);

    // Save report to file
    let report_json = serde_json::to_string_pretty(&report)?;
    let report_path = "accuracy_validation_report.json";
    std::fs::write(report_path, &report_json)?;
    println!("\n📄 Detailed report saved to: {}", report_path);

    // Exit with appropriate code
    if report.meets_phase1_requirements {
        println!("\n✅ SUCCESS: All Phase 1 accuracy requirements met!");
        println!("   - Entity extraction: {:.2}% (>= 95% required)", report.entity_metrics.accuracy_percentage);
        println!("   - Relationship extraction: {:.2}% (>= 85% required)", report.relationship_metrics.accuracy_percentage);
        println!("   - Overall score: {:.2}%", report.overall_score);
        std::process::exit(0);
    } else {
        println!("\n❌ FAILURE: Phase 1 accuracy requirements not met!");
        println!("   - Entity extraction: {:.2}% (>= 95% required)", report.entity_metrics.accuracy_percentage);
        println!("   - Relationship extraction: {:.2}% (>= 85% required)", report.relationship_metrics.accuracy_percentage);
        println!("   - Overall score: {:.2}%", report.overall_score);
        
        println!("\n🔧 Issues to address:");
        for error in &report.detailed_errors {
            println!("   • {}", error);
        }
        
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus_generator::{CorpusGenerator};
    use tempfile::NamedTempFile;

    #[test]
    fn test_accuracy_validation_with_generated_corpus() {
        // Create a small test corpus
        let generator = CorpusGenerator::new();
        let corpus = generator.create_balanced_corpus(3);
        assert!(!corpus.samples.is_empty());

        // Save to temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();
        
        generator.save_corpus_to_file(&corpus, temp_path).unwrap();

        // Run accuracy test
        let tester = AccuracyTester::new();
        let loaded_corpus = AccuracyTester::load_test_corpus(temp_path).unwrap();
        let report = tester.evaluate_accuracy(&loaded_corpus);
        
        // Should have reasonable metrics
        assert!(report.entity_metrics.total_ground_truth > 0);
        assert!(report.relationship_metrics.total_ground_truth > 0);
        assert!(report.overall_score >= 0.0);
        assert!(report.overall_score <= 100.0);
    }

    #[test]
    fn test_report_generation() {
        // Create a minimal test corpus
        let generator = CorpusGenerator::new();
        let corpus = generator.create_balanced_corpus(1);
        let tester = AccuracyTester::new();
        let report = tester.evaluate_accuracy(&corpus);
        
        // Generate detailed report
        let detailed_report = tester.generate_detailed_report(&report);
        
        // Should contain key sections
        assert!(detailed_report.contains("ENTITY EXTRACTION METRICS"));
        assert!(detailed_report.contains("RELATIONSHIP EXTRACTION METRICS"));
        assert!(detailed_report.contains("Accuracy:"));
        assert!(detailed_report.contains("Precision:"));
        assert!(detailed_report.contains("Recall:"));
        assert!(detailed_report.contains("F1-Score:"));
    }
}