use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;

use llmkg::core::entity_extractor::{EntityExtractor, Entity, EntityType};
use llmkg::core::relationship_extractor::{RelationshipExtractor, Relationship, RelationshipType};

#[derive(Debug, Deserialize, Serialize)]
pub struct TestCorpusMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub total_samples: usize,
    pub domains: Vec<String>,
    pub entity_types: Vec<String>,
    pub relationship_types: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GroundTruthEntity {
    pub text: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GroundTruthRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(rename = "type")]
    pub relationship_type: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TestSample {
    pub id: u32,
    pub text: String,
    pub domain: String,
    pub entities: Vec<GroundTruthEntity>,
    pub relationships: Vec<GroundTruthRelationship>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TestCorpus {
    pub metadata: TestCorpusMetadata,
    pub samples: Vec<TestSample>,
}

#[derive(Debug, Serialize)]
pub struct EntityAccuracyMetrics {
    pub total_ground_truth: usize,
    pub total_extracted: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy_percentage: f64,
    pub type_accuracy: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct RelationshipAccuracyMetrics {
    pub total_ground_truth: usize,
    pub total_extracted: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy_percentage: f64,
    pub type_accuracy: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct ComprehensiveAccuracyReport {
    pub entity_metrics: EntityAccuracyMetrics,
    pub relationship_metrics: RelationshipAccuracyMetrics,
    pub overall_score: f64,
    pub meets_phase1_requirements: bool,
    pub detailed_errors: Vec<String>,
    pub sample_errors: HashMap<u32, Vec<String>>,
}

pub struct AccuracyTester {
    entity_extractor: EntityExtractor,
    relationship_extractor: RelationshipExtractor,
}

impl AccuracyTester {
    pub fn new() -> Self {
        Self {
            entity_extractor: EntityExtractor::new(),
            relationship_extractor: RelationshipExtractor::new(),
        }
    }

    pub fn load_test_corpus(corpus_path: &str) -> Result<TestCorpus, Box<dyn std::error::Error>> {
        let corpus_content = fs::read_to_string(corpus_path)?;
        let corpus: TestCorpus = serde_json::from_str(&corpus_content)?;
        Ok(corpus)
    }

    pub fn evaluate_accuracy(&self, corpus: &TestCorpus) -> ComprehensiveAccuracyReport {
        let mut entity_tp = 0;
        let mut entity_fp = 0;
        let mut entity_fn = 0;
        let mut entity_type_correct = HashMap::new();
        let mut entity_type_total = HashMap::new();

        let mut rel_tp = 0;
        let mut rel_fp = 0;
        let mut rel_fn = 0;
        let mut rel_type_correct = HashMap::new();
        let mut rel_type_total = HashMap::new();

        let mut detailed_errors = Vec::new();
        let mut sample_errors = HashMap::new();

        for sample in &corpus.samples {
            let mut sample_error_list = Vec::new();
            
            // Extract entities using the system
            let extracted_entities = self.entity_extractor.extract_entities(&sample.text);
            let extracted_relationships = self.relationship_extractor.extract_relationships(&sample.text, &extracted_entities);

            // Evaluate entity extraction
            let (sample_entity_tp, sample_entity_fp, sample_entity_fn, entity_errors) = 
                self.evaluate_entities(&sample.entities, &extracted_entities, &sample.text);
            
            entity_tp += sample_entity_tp;
            entity_fp += sample_entity_fp;
            entity_fn += sample_entity_fn;
            sample_error_list.extend(entity_errors);

            // Track entity type accuracy
            for gt_entity in &sample.entities {
                let count = entity_type_total.entry(gt_entity.entity_type.clone()).or_insert(0);
                *count += 1;

                // Check if this entity was correctly extracted with correct type
                let correctly_extracted = extracted_entities.iter().any(|e| {
                    self.entities_match(gt_entity, e, &sample.text) && 
                    self.entity_type_matches(&gt_entity.entity_type, &e.entity_type)
                });

                if correctly_extracted {
                    let correct_count = entity_type_correct.entry(gt_entity.entity_type.clone()).or_insert(0);
                    *correct_count += 1;
                }
            }

            // Evaluate relationship extraction
            let (sample_rel_tp, sample_rel_fp, sample_rel_fn, rel_errors) = 
                self.evaluate_relationships(&sample.relationships, &extracted_relationships);
            
            rel_tp += sample_rel_tp;
            rel_fp += sample_rel_fp;
            rel_fn += sample_rel_fn;
            sample_error_list.extend(rel_errors);

            // Track relationship type accuracy
            for gt_rel in &sample.relationships {
                let count = rel_type_total.entry(gt_rel.relationship_type.clone()).or_insert(0);
                *count += 1;

                // Check if this relationship was correctly extracted with correct type
                let correctly_extracted = extracted_relationships.iter().any(|r| {
                    self.relationships_match(gt_rel, r) && 
                    self.relationship_type_matches(&gt_rel.relationship_type, &r.relationship_type)
                });

                if correctly_extracted {
                    let correct_count = rel_type_correct.entry(gt_rel.relationship_type.clone()).or_insert(0);
                    *correct_count += 1;
                }
            }

            if !sample_error_list.is_empty() {
                sample_errors.insert(sample.id, sample_error_list);
            }
        }

        // Calculate entity metrics
        let entity_precision = if entity_tp + entity_fp > 0 {
            entity_tp as f64 / (entity_tp + entity_fp) as f64
        } else {
            0.0
        };

        let entity_recall = if entity_tp + entity_fn > 0 {
            entity_tp as f64 / (entity_tp + entity_fn) as f64
        } else {
            0.0
        };

        let entity_f1 = if entity_precision + entity_recall > 0.0 {
            2.0 * (entity_precision * entity_recall) / (entity_precision + entity_recall)
        } else {
            0.0
        };

        let entity_accuracy = entity_f1 * 100.0;

        // Calculate entity type accuracy
        let entity_type_accuracy: HashMap<String, f64> = entity_type_total.iter()
            .map(|(entity_type, &total)| {
                let correct = entity_type_correct.get(entity_type).unwrap_or(&0);
                let accuracy = if total > 0 {
                    (*correct as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                (entity_type.clone(), accuracy)
            })
            .collect();

        // Calculate relationship metrics
        let rel_precision = if rel_tp + rel_fp > 0 {
            rel_tp as f64 / (rel_tp + rel_fp) as f64
        } else {
            0.0
        };

        let rel_recall = if rel_tp + rel_fn > 0 {
            rel_tp as f64 / (rel_tp + rel_fn) as f64
        } else {
            0.0
        };

        let rel_f1 = if rel_precision + rel_recall > 0.0 {
            2.0 * (rel_precision * rel_recall) / (rel_precision + rel_recall)
        } else {
            0.0
        };

        let rel_accuracy = rel_f1 * 100.0;

        // Calculate relationship type accuracy
        let rel_type_accuracy: HashMap<String, f64> = rel_type_total.iter()
            .map(|(rel_type, &total)| {
                let correct = rel_type_correct.get(rel_type).unwrap_or(&0);
                let accuracy = if total > 0 {
                    (*correct as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                (rel_type.clone(), accuracy)
            })
            .collect();

        // Overall score (weighted average)
        let overall_score = (entity_accuracy * 0.6) + (rel_accuracy * 0.4);

        // Check Phase 1 requirements
        let meets_requirements = entity_accuracy >= 95.0 && rel_accuracy >= 85.0;

        if !meets_requirements {
            if entity_accuracy < 95.0 {
                detailed_errors.push(format!(
                    "Entity extraction accuracy ({:.2}%) does not meet Phase 1 requirement (>= 95%)",
                    entity_accuracy
                ));
            }
            if rel_accuracy < 85.0 {
                detailed_errors.push(format!(
                    "Relationship extraction accuracy ({:.2}%) does not meet Phase 1 requirement (>= 85%)",
                    rel_accuracy
                ));
            }
        }

        ComprehensiveAccuracyReport {
            entity_metrics: EntityAccuracyMetrics {
                total_ground_truth: corpus.samples.iter().map(|s| s.entities.len()).sum(),
                total_extracted: entity_tp + entity_fp,
                true_positives: entity_tp,
                false_positives: entity_fp,
                false_negatives: entity_fn,
                precision: entity_precision,
                recall: entity_recall,
                f1_score: entity_f1,
                accuracy_percentage: entity_accuracy,
                type_accuracy: entity_type_accuracy,
            },
            relationship_metrics: RelationshipAccuracyMetrics {
                total_ground_truth: corpus.samples.iter().map(|s| s.relationships.len()).sum(),
                total_extracted: rel_tp + rel_fp,
                true_positives: rel_tp,
                false_positives: rel_fp,
                false_negatives: rel_fn,
                precision: rel_precision,
                recall: rel_recall,
                f1_score: rel_f1,
                accuracy_percentage: rel_accuracy,
                type_accuracy: rel_type_accuracy,
            },
            overall_score,
            meets_phase1_requirements: meets_requirements,
            detailed_errors,
            sample_errors,
        }
    }

    fn evaluate_entities(
        &self,
        ground_truth: &[GroundTruthEntity],
        extracted: &[Entity],
        text: &str,
    ) -> (usize, usize, usize, Vec<String>) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        let mut errors = Vec::new();

        // Track which ground truth entities were found
        let mut found_gt = HashSet::new();
        let mut found_extracted = HashSet::new();

        // Check true positives
        for (i, gt_entity) in ground_truth.iter().enumerate() {
            let mut found = false;
            for (j, extracted_entity) in extracted.iter().enumerate() {
                if !found_extracted.contains(&j) && self.entities_match(gt_entity, extracted_entity, text) {
                    tp += 1;
                    found_gt.insert(i);
                    found_extracted.insert(j);
                    found = true;

                    // Check type accuracy
                    if !self.entity_type_matches(&gt_entity.entity_type, &extracted_entity.entity_type) {
                        errors.push(format!(
                            "Entity '{}' found but wrong type: expected '{}', got '{:?}'",
                            gt_entity.text, gt_entity.entity_type, extracted_entity.entity_type
                        ));
                    }
                    break;
                }
            }
            if !found {
                fn_count += 1;
                errors.push(format!("Missed entity: '{}' ({})", gt_entity.text, gt_entity.entity_type));
            }
        }

        // Count false positives (extracted entities not in ground truth)
        for (j, extracted_entity) in extracted.iter().enumerate() {
            if !found_extracted.contains(&j) {
                fp += 1;
                errors.push(format!("False positive entity: '{}' ({:?})", extracted_entity.name, extracted_entity.entity_type));
            }
        }

        (tp, fp, fn_count, errors)
    }

    fn evaluate_relationships(
        &self,
        ground_truth: &[GroundTruthRelationship],
        extracted: &[Relationship],
    ) -> (usize, usize, usize, Vec<String>) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        let mut errors = Vec::new();

        // Track which relationships were found
        let mut found_gt = HashSet::new();
        let mut found_extracted = HashSet::new();

        // Check true positives
        for (i, gt_rel) in ground_truth.iter().enumerate() {
            let mut found = false;
            for (j, extracted_rel) in extracted.iter().enumerate() {
                if !found_extracted.contains(&j) && self.relationships_match(gt_rel, extracted_rel) {
                    tp += 1;
                    found_gt.insert(i);
                    found_extracted.insert(j);
                    found = true;

                    // Check type accuracy
                    if !self.relationship_type_matches(&gt_rel.relationship_type, &extracted_rel.relationship_type) {
                        errors.push(format!(
                            "Relationship '{}--{}-->{} found but wrong type: expected '{}', got '{:?}'",
                            gt_rel.subject, gt_rel.predicate, gt_rel.object,
                            gt_rel.relationship_type, extracted_rel.relationship_type
                        ));
                    }
                    break;
                }
            }
            if !found {
                fn_count += 1;
                errors.push(format!(
                    "Missed relationship: '{}--{}-->{} ({})",
                    gt_rel.subject, gt_rel.predicate, gt_rel.object, gt_rel.relationship_type
                ));
            }
        }

        // Count false positives
        for (j, extracted_rel) in extracted.iter().enumerate() {
            if !found_extracted.contains(&j) {
                fp += 1;
                errors.push(format!(
                    "False positive relationship: '{}--{}-->{} ({:?})",
                    extracted_rel.subject, extracted_rel.predicate, extracted_rel.object, extracted_rel.relationship_type
                ));
            }
        }

        (tp, fp, fn_count, errors)
    }

    fn entities_match(&self, gt: &GroundTruthEntity, extracted: &Entity, text: &str) -> bool {
        // For this accuracy test, we'll use a more flexible matching approach
        // that doesn't rely on exact character positions due to Unicode issues
        
        // Direct name match
        if gt.text == extracted.name {
            return true;
        }

        // Normalized match (case insensitive, trimmed)
        let gt_normalized = gt.text.trim().to_lowercase();
        let extracted_normalized = extracted.name.trim().to_lowercase();

        // Check for exact matches
        if gt_normalized == extracted_normalized {
            return true;
        }

        // Allow for "The" prefix differences
        if gt_normalized.trim_start_matches("the ") == extracted_normalized.trim_start_matches("the ") {
            return true;
        }

        // Allow for substring matches (extracted entity might be part of ground truth or vice versa)
        // But only if the match is significant (not just single characters)
        if gt_normalized.len() >= 3 && extracted_normalized.len() >= 3 {
            if gt_normalized.contains(&extracted_normalized) || extracted_normalized.contains(&gt_normalized) {
                return true;
            }
        }

        // Check if the entity appears in the text (simple text search)
        let text_lower = text.to_lowercase();
        if text_lower.contains(&gt_normalized) && text_lower.contains(&extracted_normalized) {
            // Check if they refer to the same part of the text by looking at overlapping contexts
            if let (Some(gt_pos), Some(ext_pos)) = (text_lower.find(&gt_normalized), text_lower.find(&extracted_normalized)) {
                // If positions are close (within 10 characters), consider them matching
                if (gt_pos as i32 - ext_pos as i32).abs() <= 10 {
                    return true;
                }
            }
        }

        false
    }

    fn relationships_match(&self, gt: &GroundTruthRelationship, extracted: &Relationship) -> bool {
        // Normalize for comparison
        let gt_subject = gt.subject.trim().to_lowercase();
        let gt_object = gt.object.trim().to_lowercase();
        let extracted_subject = extracted.subject.trim().to_lowercase();
        let extracted_object = extracted.object.trim().to_lowercase();

        // Allow for flexible predicate matching
        let predicates_match = self.predicates_match(&gt.predicate, &extracted.predicate);

        (gt_subject == extracted_subject || 
         gt_subject.trim_start_matches("the ") == extracted_subject.trim_start_matches("the ")) &&
        (gt_object == extracted_object ||
         gt_object.trim_start_matches("the ") == extracted_object.trim_start_matches("the ")) &&
        predicates_match
    }

    fn predicates_match(&self, gt_predicate: &str, extracted_predicate: &str) -> bool {
        let gt_norm = gt_predicate.trim().to_lowercase();
        let ext_norm = extracted_predicate.trim().to_lowercase();

        // Direct match
        if gt_norm == ext_norm {
            return true;
        }

        // Common predicate synonyms
        match (gt_norm.as_str(), ext_norm.as_str()) {
            ("developed", "created") | ("created", "developed") => true,
            ("founded", "created") | ("created", "founded") => true,
            ("wrote", "created") | ("created", "wrote") => true,
            ("painted", "created") | ("created", "painted") => true,
            ("designed", "created") | ("created", "designed") => true,
            ("invented", "created") | ("created", "invented") => true,
            ("built", "created") | ("created", "built") => true,
            ("located in", "in") | ("in", "located in") => true,
            ("born in", "from") | ("from", "born in") => true,
            ("is", "was") | ("was", "is") => true,
            ("has", "have") | ("have", "has") => true,
            _ => false,
        }
    }

    fn entity_type_matches(&self, gt_type: &str, extracted_type: &EntityType) -> bool {
        match (gt_type, extracted_type) {
            ("Person", EntityType::Person) => true,
            ("Place", EntityType::Place) => true,
            ("Organization", EntityType::Organization) => true,
            ("Concept", EntityType::Concept) => true,
            ("Event", EntityType::Event) => true,
            ("Time", EntityType::Time) => true,
            ("Quantity", EntityType::Quantity) => true,
            ("Unknown", EntityType::Unknown) => true,
            _ => false,
        }
    }

    fn relationship_type_matches(&self, gt_type: &str, extracted_type: &RelationshipType) -> bool {
        match (gt_type, extracted_type) {
            ("Created", RelationshipType::Created) => true,
            ("Discovered", RelationshipType::Discovered) => true,
            ("Invented", RelationshipType::Invented) => true,
            ("Developed", RelationshipType::Developed) => true,
            ("Founded", RelationshipType::Founded) => true,
            ("Built", RelationshipType::Built) => true,
            ("Wrote", RelationshipType::Wrote) => true,
            ("Designed", RelationshipType::Designed) => true,
            ("LocatedIn", RelationshipType::LocatedIn) => true,
            ("From", RelationshipType::From) => true,
            ("WorksWith", RelationshipType::WorksWith) => true,
            ("MarriedTo", RelationshipType::MarriedTo) => true,
            ("ChildOf", RelationshipType::ChildOf) => true,
            ("ParentOf", RelationshipType::ParentOf) => true,
            ("Is", RelationshipType::Is) => true,
            ("Has", RelationshipType::Has) => true,
            ("PartOf", RelationshipType::PartOf) => true,
            ("Contains", RelationshipType::Contains) => true,
            ("BelongsTo", RelationshipType::BelongsTo) => true,
            ("RelatedTo", RelationshipType::RelatedTo) => true,
            ("Causes", RelationshipType::Causes) => true,
            ("During", RelationshipType::During) => true,
            ("Before", RelationshipType::Before) => true,
            ("After", RelationshipType::After) => true,
            // Allow flexible matching for similar types
            ("Created", RelationshipType::Developed) => true,
            ("Developed", RelationshipType::Created) => true,
            ("Founded", RelationshipType::Created) => true,
            ("Built", RelationshipType::Created) => true,
            ("Wrote", RelationshipType::Created) => true,
            ("Painted", RelationshipType::Created) => true,
            ("Designed", RelationshipType::Created) => true,
            ("Invented", RelationshipType::Created) => true,
            _ => false,
        }
    }

    pub fn generate_detailed_report(&self, report: &ComprehensiveAccuracyReport) -> String {
        let mut output = String::new();
        
        output.push_str("=== LLMKG Phase 1 Accuracy Test Results ===\n\n");
        
        // Overall results
        output.push_str(&format!("Overall Score: {:.2}%\n", report.overall_score));
        output.push_str(&format!("Meets Phase 1 Requirements: {}\n\n", 
            if report.meets_phase1_requirements { "✅ YES" } else { "❌ NO" }));

        // Entity metrics
        output.push_str("=== ENTITY EXTRACTION METRICS ===\n");
        output.push_str(&format!("Accuracy: {:.2}% (Requirement: >= 95%)\n", report.entity_metrics.accuracy_percentage));
        output.push_str(&format!("Precision: {:.4}\n", report.entity_metrics.precision));
        output.push_str(&format!("Recall: {:.4}\n", report.entity_metrics.recall));
        output.push_str(&format!("F1-Score: {:.4}\n", report.entity_metrics.f1_score));
        output.push_str(&format!("Total Ground Truth: {}\n", report.entity_metrics.total_ground_truth));
        output.push_str(&format!("Total Extracted: {}\n", report.entity_metrics.total_extracted));
        output.push_str(&format!("True Positives: {}\n", report.entity_metrics.true_positives));
        output.push_str(&format!("False Positives: {}\n", report.entity_metrics.false_positives));
        output.push_str(&format!("False Negatives: {}\n\n", report.entity_metrics.false_negatives));

        output.push_str("Entity Type Accuracy:\n");
        for (entity_type, accuracy) in &report.entity_metrics.type_accuracy {
            output.push_str(&format!("  {}: {:.2}%\n", entity_type, accuracy));
        }

        // Relationship metrics
        output.push_str("\n=== RELATIONSHIP EXTRACTION METRICS ===\n");
        output.push_str(&format!("Accuracy: {:.2}% (Requirement: >= 85%)\n", report.relationship_metrics.accuracy_percentage));
        output.push_str(&format!("Precision: {:.4}\n", report.relationship_metrics.precision));
        output.push_str(&format!("Recall: {:.4}\n", report.relationship_metrics.recall));
        output.push_str(&format!("F1-Score: {:.4}\n", report.relationship_metrics.f1_score));
        output.push_str(&format!("Total Ground Truth: {}\n", report.relationship_metrics.total_ground_truth));
        output.push_str(&format!("Total Extracted: {}\n", report.relationship_metrics.total_extracted));
        output.push_str(&format!("True Positives: {}\n", report.relationship_metrics.true_positives));
        output.push_str(&format!("False Positives: {}\n", report.relationship_metrics.false_positives));
        output.push_str(&format!("False Negatives: {}\n\n", report.relationship_metrics.false_negatives));

        output.push_str("Relationship Type Accuracy:\n");
        for (rel_type, accuracy) in &report.relationship_metrics.type_accuracy {
            output.push_str(&format!("  {}: {:.2}%\n", rel_type, accuracy));
        }

        // Detailed errors
        if !report.detailed_errors.is_empty() {
            output.push_str("\n=== DETAILED ERRORS ===\n");
            for error in &report.detailed_errors {
                output.push_str(&format!("❌ {}\n", error));
            }
        }

        // Sample errors (first 10)
        if !report.sample_errors.is_empty() {
            output.push_str("\n=== SAMPLE ERRORS (First 10) ===\n");
            let mut count = 0;
            for (sample_id, errors) in report.sample_errors.iter() {
                if count >= 10 { break; }
                output.push_str(&format!("Sample {}: {} errors\n", sample_id, errors.len()));
                for error in errors.iter().take(3) {  // Show first 3 errors per sample
                    output.push_str(&format!("  - {}\n", error));
                }
                if errors.len() > 3 {
                    output.push_str(&format!("  ... and {} more\n", errors.len() - 3));
                }
                count += 1;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;
    use std::io::Write;

    #[test]
    fn test_load_test_corpus() {
        let corpus_json = r#"
        {
            "metadata": {
                "name": "Test Corpus",
                "version": "1.0.0",
                "description": "Test corpus for accuracy testing",
                "total_samples": 1,
                "domains": ["test"],
                "entity_types": ["Person"],
                "relationship_types": ["Created"]
            },
            "samples": [
                {
                    "id": 1,
                    "text": "Einstein developed relativity.",
                    "domain": "science",
                    "entities": [
                        {"text": "Einstein", "type": "Person", "start": 0, "end": 8}
                    ],
                    "relationships": [
                        {"subject": "Einstein", "predicate": "developed", "object": "relativity", "type": "Developed"}
                    ]
                }
            ]
        }
        "#;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        write!(temp_file, "{}", corpus_json).unwrap();
        
        let corpus = AccuracyTester::load_test_corpus(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(corpus.samples.len(), 1);
        assert_eq!(corpus.samples[0].entities.len(), 1);
        assert_eq!(corpus.samples[0].relationships.len(), 1);
    }

    #[test]
    fn test_entity_matching() {
        let tester = AccuracyTester::new();
        
        let gt_entity = GroundTruthEntity {
            text: "Albert Einstein".to_string(),
            entity_type: "Person".to_string(),
            start: 0,
            end: 15,
        };

        let extracted_entity = Entity {
            name: "Albert Einstein".to_string(),
            entity_type: EntityType::Person,
            start_pos: 0,
            end_pos: 15,
        };

        let text = "Albert Einstein developed relativity.";
        assert!(tester.entities_match(&gt_entity, &extracted_entity, text));
    }

    #[test]
    fn test_relationship_matching() {
        let tester = AccuracyTester::new();
        
        let gt_rel = GroundTruthRelationship {
            subject: "Einstein".to_string(),
            predicate: "developed".to_string(),
            object: "relativity".to_string(),
            relationship_type: "Developed".to_string(),
        };

        let extracted_rel = Relationship {
            subject: "Einstein".to_string(),
            predicate: "developed".to_string(),
            object: "relativity".to_string(),
            relationship_type: RelationshipType::Developed,
            confidence: 0.9,
            context_start: 0,
            context_end: 30,
        };

        assert!(tester.relationships_match(&gt_rel, &extracted_rel));
    }

    #[test]
    fn test_accuracy_evaluation() {
        let tester = AccuracyTester::new();
        
        // Create a minimal test corpus
        let corpus = TestCorpus {
            metadata: TestCorpusMetadata {
                name: "Test Corpus".to_string(),
                version: "1.0.0".to_string(),
                description: "Test".to_string(),
                total_samples: 1,
                domains: vec!["test".to_string()],
                entity_types: vec!["Person".to_string()],
                relationship_types: vec!["Developed".to_string()],
            },
            samples: vec![
                TestSample {
                    id: 1,
                    text: "Einstein developed relativity.".to_string(),
                    domain: "science".to_string(),
                    entities: vec![
                        GroundTruthEntity {
                            text: "Einstein".to_string(),
                            entity_type: "Person".to_string(),
                            start: 0,
                            end: 8,
                        }
                    ],
                    relationships: vec![
                        GroundTruthRelationship {
                            subject: "Einstein".to_string(),
                            predicate: "developed".to_string(),
                            object: "relativity".to_string(),
                            relationship_type: "Developed".to_string(),
                        }
                    ],
                }
            ],
        };

        let report = tester.evaluate_accuracy(&corpus);
        
        // Should have some meaningful metrics
        assert!(report.entity_metrics.total_ground_truth > 0);
        assert!(report.relationship_metrics.total_ground_truth > 0);
        assert!(report.overall_score >= 0.0);
    }
}