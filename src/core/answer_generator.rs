use crate::core::triple::Triple;
use crate::core::knowledge_types::{Answer, QuestionIntent, QuestionType, AnswerType};

pub struct AnswerGenerator;

impl AnswerGenerator {
    pub fn generate_answer(facts: Vec<Triple>, intent: QuestionIntent) -> Answer {
        if facts.is_empty() {
            return Answer {
                text: "I don't have enough information to answer this question.".to_string(),
                confidence: 0.0,
                facts: vec![],
                entities: vec![],
            };
        }

        // Group facts by relevance
        let relevant_facts = Self::filter_relevant_facts(&facts, &intent);
        let confidence = Self::calculate_confidence(&relevant_facts, &intent);
        
        // Generate answer based on question type
        let answer_text = match intent.question_type {
            QuestionType::What => Self::generate_what_answer(&relevant_facts, &intent),
            QuestionType::Who => Self::generate_who_answer(&relevant_facts, &intent),
            QuestionType::When => Self::generate_when_answer(&relevant_facts, &intent),
            QuestionType::Where => Self::generate_where_answer(&relevant_facts, &intent),
            QuestionType::Why => Self::generate_why_answer(&relevant_facts, &intent),
            QuestionType::How => Self::generate_how_answer(&relevant_facts, &intent),
            QuestionType::Which => Self::generate_which_answer(&relevant_facts, &intent),
            QuestionType::Is => Self::generate_is_answer(&relevant_facts, &intent),
        };

        // Extract mentioned entities
        let mut entities = intent.entities.clone();
        for fact in &relevant_facts {
            if !entities.contains(&fact.subject) {
                entities.push(fact.subject.clone());
            }
            if !entities.contains(&fact.object) {
                entities.push(fact.object.clone());
            }
        }

        Answer {
            text: answer_text,
            confidence,
            facts: relevant_facts,
            entities,
        }
    }

    fn filter_relevant_facts(facts: &[Triple], intent: &QuestionIntent) -> Vec<Triple> {
        let mut relevant = Vec::new();
        
        // If no entities were extracted from the question, return all facts
        // This can happen with questions like "Who discovered polonium?" where 
        // "polonium" is lowercase and not recognized as an entity
        if intent.entities.is_empty() {
            return facts.to_vec();
        }
        
        for fact in facts {
            // Check if fact contains any of the entities from the question
            let contains_entity = intent.entities.iter().any(|entity| {
                fact.subject.contains(entity) || fact.object.contains(entity)
            });

            if contains_entity {
                relevant.push(fact.clone());
            }
        }

        // Sort by relevance (simplified - in production would use more sophisticated scoring)
        relevant.sort_by_key(|fact| {
            let mut score = 0;
            for entity in &intent.entities {
                if fact.subject.contains(entity) {
                    score += 2;
                }
                if fact.object.contains(entity) {
                    score += 1;
                }
            }
            std::cmp::Reverse(score)
        });

        relevant
    }

    fn calculate_confidence(facts: &[Triple], intent: &QuestionIntent) -> f32 {
        if facts.is_empty() {
            return 0.0;
        }

        // Base confidence on number of relevant facts and their individual confidence scores
        let avg_fact_confidence: f32 = facts.iter().map(|f| f.confidence).sum::<f32>() / facts.len() as f32;
        let coverage_score = (facts.len() as f32 / 10.0).min(1.0); // More facts = higher confidence, up to 10

        // Check if we have facts that directly answer the question type
        let type_match_score = match intent.question_type {
            QuestionType::Who => {
                if facts.iter().any(|f| matches!(f.predicate.as_str(), "created" | "invented" | "discovered" | "founded" | "wrote")) {
                    1.0
                } else {
                    0.5
                }
            }
            QuestionType::When => {
                if facts.iter().any(|f| f.object.chars().any(|c| c.is_numeric())) {
                    1.0
                } else {
                    0.3
                }
            }
            QuestionType::Where => {
                if facts.iter().any(|f| matches!(f.predicate.as_str(), "located_in" | "from" | "in" | "at")) {
                    1.0
                } else {
                    0.5
                }
            }
            _ => 0.7,
        };

        (avg_fact_confidence * 0.4 + coverage_score * 0.3 + type_match_score * 0.3).min(1.0)
    }

    fn generate_what_answer(facts: &[Triple], _intent: &QuestionIntent) -> String {
        // Check if this is an action-based "what" question (what did X do/create/develop?)
        let action_predicates = ["developed", "created", "invented", "discovered", "founded", "wrote", "designed", "built", "made", "produced"];
        
        // First, look for action-based facts that directly answer "what did X do?"
        if let Some(fact) = facts.iter().find(|f| action_predicates.contains(&f.predicate.as_str())) {
            return fact.object.clone();
        }
        
        // Look for other meaningful predicates that could answer "what" questions
        let meaningful_predicates = ["published", "released", "announced", "proposed", "established", "formed"];
        if let Some(fact) = facts.iter().find(|f| meaningful_predicates.contains(&f.predicate.as_str())) {
            return fact.object.clone();
        }
        
        // Look for definitional facts only if no action facts were found
        if let Some(fact) = facts.iter().find(|f| f.predicate == "is" || f.predicate == "are") {
            return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
        }

        // Look for property facts
        if let Some(fact) = facts.iter().find(|f| f.predicate == "has" || f.predicate == "contains") {
            return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
        }

        // Default to listing relevant facts
        Self::generate_fact_list(facts, 3)
    }

    fn generate_who_answer(facts: &[Triple], _intent: &QuestionIntent) -> String {
        // Look for creation/invention facts
        let creation_predicates = ["created", "invented", "discovered", "founded", "wrote", "designed", "developed"];
        
        // Filter and rank facts by how likely they are to represent actual people
        let mut candidate_facts: Vec<&Triple> = facts.iter()
            .filter(|f| creation_predicates.contains(&f.predicate.as_str()))
            .collect();
        
        // Sort by likelihood of being a person (using heuristics)
        candidate_facts.sort_by_key(|fact| {
            let subject = &fact.subject;
            let mut score = 0;
            
            // Prefer subjects that look like person names
            let words: Vec<&str> = subject.split_whitespace().collect();
            if words.len() == 2 || words.len() == 3 {
                score += 10; // Likely person name pattern
            }
            
            // Penalize subjects that contain concept words
            let concept_indicators = ["Prize", "Prizes", "Award", "Awards", "Medal", "Medals", "Nobel"];
            if words.iter().any(|w| concept_indicators.contains(w)) {
                score -= 20; // Unlikely to be a person
            }
            
            // Prefer names with typical person name patterns
            if words.iter().all(|w| w.chars().next().map_or(false, |c| c.is_uppercase()) && w.len() > 1) {
                score += 5; // Capitalized words likely indicate proper names
            }
            
            // Prefer subjects that contain common name indicators
            let name_indicators = ["Marie", "Maria", "John", "Jane", "Albert", "Isaac", "Charles", "Louis"];
            if words.iter().any(|w| name_indicators.contains(w)) {
                score += 15; // Contains common name
            }
            
            std::cmp::Reverse(score) // Sort in descending order (highest score first)
        });
        
        if let Some(fact) = candidate_facts.first() {
            return fact.subject.clone();
        }

        // Look for "by" relationships
        if let Some(fact) = facts.iter().find(|f| f.predicate.ends_with("by")) {
            return fact.object.clone();
        }

        // Return the most likely person entity (with same filtering logic)
        let mut candidate_subjects: Vec<String> = facts.iter()
            .map(|f| f.subject.clone())
            .collect();
        
        candidate_subjects.sort_by_key(|subject| {
            let words: Vec<&str> = subject.split_whitespace().collect();
            let mut score = 0;
            
            if words.len() == 2 || words.len() == 3 {
                score += 10;
            }
            
            let concept_indicators = ["Prize", "Prizes", "Award", "Awards", "Medal", "Medals", "Nobel"];
            if words.iter().any(|w| concept_indicators.contains(w)) {
                score -= 20;
            }
            
            if words.iter().all(|w| w.chars().next().map_or(false, |c| c.is_uppercase()) && w.len() > 1) {
                score += 5;
            }
            
            let name_indicators = ["Marie", "Maria", "John", "Jane", "Albert", "Isaac", "Charles", "Louis"];
            if words.iter().any(|w| name_indicators.contains(w)) {
                score += 15;
            }
            
            std::cmp::Reverse(score)
        });
        
        if let Some(subject) = candidate_subjects.first() {
            return subject.clone();
        }

        "Unknown person".to_string()
    }

    fn generate_when_answer(facts: &[Triple], _intent: &QuestionIntent) -> String {
        // Look for facts with time-related objects
        if let Some(fact) = facts.iter().find(|f| {
            f.object.chars().any(|c| c.is_numeric()) || 
            f.predicate.contains("date") || 
            f.predicate.contains("time") ||
            f.predicate.contains("year")
        }) {
            return fact.object.clone();
        }

        // Look for "in" relationships with years
        if let Some(fact) = facts.iter().find(|f| {
            f.predicate == "in" && f.object.chars().any(|c| c.is_numeric())
        }) {
            return fact.object.clone();
        }

        "Unknown time".to_string()
    }

    fn generate_where_answer(facts: &[Triple], _intent: &QuestionIntent) -> String {
        // Look for location predicates
        let location_predicates = ["located_in", "in", "at", "from", "based_in", "near"];
        
        if let Some(fact) = facts.iter().find(|f| location_predicates.contains(&f.predicate.as_str())) {
            return fact.object.clone();
        }

        // Return the first location-like entity
        if let Some(fact) = facts.first() {
            return fact.object.clone();
        }

        "Unknown location".to_string()
    }

    fn generate_why_answer(facts: &[Triple], _intent: &QuestionIntent) -> String {
        // Look for causal relationships
        let causal_predicates = ["because", "caused_by", "due_to", "results_from", "leads_to", "causes"];
        
        if let Some(fact) = facts.iter().find(|f| causal_predicates.contains(&f.predicate.as_str())) {
            return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
        }

        // Generate explanation from available facts
        if facts.len() >= 2 {
            return format!(
                "Based on the facts: {} {} {}, and {} {} {}",
                facts[0].subject, facts[0].predicate, facts[0].object,
                facts[1].subject, facts[1].predicate, facts[1].object
            );
        }

        Self::generate_fact_list(facts, 3)
    }

    fn generate_how_answer(facts: &[Triple], intent: &QuestionIntent) -> String {
        // Check if it's a "how many" question
        if intent.expected_answer_type == AnswerType::Number {
            if let Some(fact) = facts.iter().find(|f| f.object.chars().any(|c| c.is_numeric())) {
                return fact.object.clone();
            }
            return format!("{} items found", facts.len());
        }

        // Look for process or method facts
        let process_predicates = ["using", "through", "via", "by", "with"];
        
        if let Some(fact) = facts.iter().find(|f| process_predicates.contains(&f.predicate.as_str())) {
            return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
        }

        Self::generate_fact_list(facts, 3)
    }

    fn generate_which_answer(facts: &[Triple], intent: &QuestionIntent) -> String {
        if intent.expected_answer_type == AnswerType::List {
            // Generate a list of options
            let options: Vec<String> = facts.iter()
                .take(5)
                .map(|f| f.object.clone())
                .collect();
            
            if options.is_empty() {
                return "No options found".to_string();
            }
            
            return options.join(", ");
        }

        // Single selection
        if let Some(fact) = facts.first() {
            return fact.object.clone();
        }

        "No matching option found".to_string()
    }

    fn generate_is_answer(facts: &[Triple], intent: &QuestionIntent) -> String {
        // For boolean questions, look for confirming or denying facts
        for fact in facts {
            // Check if any entity from the question appears as subject
            for entity in &intent.entities {
                if fact.subject.contains(entity) {
                    // Check if the predicate confirms the relationship
                    if fact.predicate == "is" || fact.predicate == "are" {
                        return "Yes".to_string();
                    }
                    if fact.predicate == "is_not" || fact.predicate == "are_not" {
                        return "No".to_string();
                    }
                }
            }
        }

        // If we have relevant facts but can't determine yes/no
        if !facts.is_empty() {
            return format!("Based on available information: {}", Self::generate_fact_list(facts, 1));
        }

        "Cannot determine from available information".to_string()
    }

    fn generate_fact_list(facts: &[Triple], max_facts: usize) -> String {
        let fact_strings: Vec<String> = facts.iter()
            .take(max_facts)
            .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
            .collect();

        if fact_strings.is_empty() {
            return "No relevant facts found".to_string();
        }

        fact_strings.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::knowledge_types::TimeRange;

    fn create_test_intent(question_type: QuestionType, entities: Vec<&str>) -> QuestionIntent {
        QuestionIntent {
            question_type,
            entities: entities.iter().map(|s| s.to_string()).collect(),
            expected_answer_type: AnswerType::Fact,
            temporal_context: None,
        }
    }

    #[test]
    fn test_generate_who_answer() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "invented".to_string(), "E=mc²".to_string()).unwrap(),
            Triple::new("Theory".to_string(), "created_by".to_string(), "Einstein".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::Who, vec!["E=mc²"]);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "Einstein");
        assert!(answer.confidence > 0.5);
    }

    #[test]
    fn test_generate_what_answer() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "is".to_string(), "physicist".to_string()).unwrap(),
            Triple::new("Einstein".to_string(), "created".to_string(), "relativity".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::What, vec!["Einstein"]);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert!(answer.text.contains("Einstein is physicist"));
        assert!(!answer.facts.is_empty());
    }

    #[test]
    fn test_generate_when_answer() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "born_in".to_string(), "1879".to_string()).unwrap(),
            Triple::new("Theory".to_string(), "published_in".to_string(), "1905".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::When, vec!["Einstein"]);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "1879");
    }

    #[test]
    fn test_empty_facts() {
        let facts = vec![];
        let intent = create_test_intent(QuestionType::What, vec!["Unknown"]);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "I don't have enough information to answer this question.");
        assert_eq!(answer.confidence, 0.0);
    }
}