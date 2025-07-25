use std::collections::HashSet;
use regex::Regex;
use lazy_static::lazy_static;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub start_pos: usize,
    pub end_pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Concept,
    Event,
    Time,
    Quantity,
    Unknown,
}

lazy_static! {
    // Common titles and prefixes that indicate person names
    static ref PERSON_TITLES: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("Mr");
        s.insert("Mrs");
        s.insert("Ms");
        s.insert("Miss");
        s.insert("Dr");
        s.insert("Prof");
        s.insert("Professor");
        s.insert("Sir");
        s.insert("Lord");
        s.insert("Lady");
        s.insert("Captain");
        s.insert("General");
        s.insert("Colonel");
        s.insert("Major");
        s.insert("President");
        s.insert("Minister");
        s
    };

    // Common organization indicators
    static ref ORG_INDICATORS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("Inc");
        s.insert("LLC");
        s.insert("Ltd");
        s.insert("Corporation");
        s.insert("Corp");
        s.insert("Company");
        s.insert("Co");
        s.insert("Group");
        s.insert("Foundation");
        s.insert("Institute");
        s.insert("University");
        s.insert("College");
        s.insert("Department");
        s.insert("Agency");
        s.insert("Commission");
        s.insert("Committee");
        s.insert("Association");
        s.insert("Society");
        s.insert("Bank");
        s
    };

    // Place indicators
    static ref PLACE_INDICATORS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("Street");
        s.insert("St");
        s.insert("Avenue");
        s.insert("Ave");
        s.insert("Road");
        s.insert("Rd");
        s.insert("Boulevard");
        s.insert("Blvd");
        s.insert("City");
        s.insert("State");
        s.insert("Country");
        s.insert("Ocean");
        s.insert("Sea");
        s.insert("River");
        s.insert("Mountain");
        s.insert("Mt");
        s.insert("Lake");
        s.insert("Park");
        s.insert("Island");
        s.insert("Tower");
        s.insert("Building");
        s.insert("Bridge");
        s.insert("Center");
        s.insert("Square");
        s.insert("Hall");
        s.insert("Station");
        s.insert("Airport");
        s.insert("Port");
        s.insert("Mall");
        s.insert("Market");
        s.insert("Museum");
        s.insert("Library");
        s.insert("Hospital");
        s
    };

    // Known concepts that should be recognized as concepts
    static ref KNOWN_CONCEPTS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("Prize");
        s.insert("Prizes");
        s.insert("Award");
        s.insert("Awards");
        s.insert("Medal");
        s.insert("Medals");
        s.insert("Trophy");
        s.insert("Trophies");
        s.insert("Certificate");
        s.insert("Certificates");
        s.insert("Diploma");
        s.insert("Diplomas");
        s.insert("Degree");
        s.insert("Degrees");
        s.insert("Theory");
        s.insert("Principle");
        s.insert("Law");
        s.insert("Theorem");
        s.insert("Formula");
        s.insert("Equation");
        s.insert("Method");
        s.insert("Technique");
        s.insert("Process");
        s.insert("System");
        s.insert("Model");
        s.insert("Framework");
        s.insert("Concept");
        s.insert("Idea");
        s.insert("Philosophy");
        s.insert("Science");
        s.insert("Physics");
        s.insert("Chemistry");
        s.insert("Biology");
        s.insert("Mathematics");
        s.insert("Medicine");
        s.insert("Literature");
        s.insert("Peace");
        s.insert("Economics");
        s
    };

    // Known place names that should be recognized as places
    static ref KNOWN_PLACES: HashSet<&'static str> = {
        let mut s = HashSet::new();
        // Countries
        s.insert("Poland");
        s.insert("France");
        s.insert("Germany");
        s.insert("England");
        s.insert("Spain");
        s.insert("Italy");
        s.insert("Russia");
        s.insert("China");
        s.insert("Japan");
        s.insert("India");
        s.insert("Brazil");
        s.insert("Canada");
        s.insert("Australia");
        s.insert("Mexico");
        s.insert("Argentina");
        s.insert("Chile");
        s.insert("Sweden");
        s.insert("Norway");
        s.insert("Denmark");
        s.insert("Finland");
        s.insert("Iceland");
        s.insert("Ireland");
        s.insert("Scotland");
        s.insert("Wales");
        s.insert("Netherlands");
        s.insert("Belgium");
        s.insert("Switzerland");
        s.insert("Austria");
        s.insert("Portugal");
        s.insert("Greece");
        s.insert("Turkey");
        s.insert("Egypt");
        s.insert("Israel");
        s.insert("Jordan");
        s.insert("Iran");
        s.insert("Iraq");
        s.insert("Afghanistan");
        s.insert("Pakistan");
        s.insert("Bangladesh");
        s.insert("Thailand");
        s.insert("Vietnam");
        s.insert("Philippines");
        s.insert("Indonesia");
        s.insert("Malaysia");
        s.insert("Singapore");
        s.insert("Korea");
        s.insert("Taiwan");
        s.insert("Mongolia");
        s.insert("Kazakhstan");
        s.insert("Ukraine");
        s.insert("Romania");
        s.insert("Bulgaria");
        s.insert("Hungary");
        s.insert("Slovakia");
        s.insert("Slovenia");
        s.insert("Croatia");
        s.insert("Serbia");
        s.insert("Bosnia");
        s.insert("Montenegro");
        s.insert("Albania");
        s.insert("Macedonia");
        s.insert("Lithuania");
        s.insert("Latvia");
        s.insert("Estonia");
        s.insert("Belarus");
        s.insert("Moldova");
        s.insert("Georgia");
        s.insert("Armenia");
        s.insert("Azerbaijan");
        s.insert("Uzbekistan");
        s.insert("Turkmenistan");
        s.insert("Kyrgyzstan");
        s.insert("Tajikistan");
        // Major cities
        s.insert("Warsaw");
        s.insert("Paris");
        s.insert("Berlin");
        s.insert("London");
        s.insert("Madrid");
        s.insert("Rome");
        s.insert("Moscow");
        s.insert("Beijing");
        s.insert("Tokyo");
        s.insert("Delhi");
        s.insert("Mumbai");
        s.insert("Shanghai");
        s.insert("Istanbul");
        s.insert("Cairo");
        s.insert("Tehran");
        s.insert("Baghdad");
        s.insert("Kabul");
        s.insert("Islamabad");
        s.insert("Dhaka");
        s.insert("Bangkok");
        s.insert("Hanoi");
        s.insert("Manila");
        s.insert("Jakarta");
        s.insert("Kuala");
        s.insert("Seoul");
        s.insert("Taipei");
        s.insert("Ulaanbaatar");
        s.insert("Almaty");
        s.insert("Kiev");
        s.insert("Bucharest");
        s.insert("Sofia");
        s.insert("Budapest");
        s.insert("Bratislava");
        s.insert("Ljubljana");
        s.insert("Zagreb");
        s.insert("Belgrade");
        s.insert("Sarajevo");
        s.insert("Podgorica");
        s.insert("Tirana");
        s.insert("Skopje");
        s.insert("Vilnius");
        s.insert("Riga");
        s.insert("Tallinn");
        s.insert("Minsk");
        s.insert("Chisinau");
        s.insert("Tbilisi");
        s.insert("Yerevan");
        s.insert("Baku");
        s.insert("Tashkent");
        s.insert("Ashgabat");
        s.insert("Bishkek");
        s.insert("Dushanbe");
        s.insert("Prague");
        s.insert("Vienna");
        s.insert("Zurich");
        s.insert("Geneva");
        s.insert("Brussels");
        s.insert("Amsterdam");
        s.insert("Copenhagen");
        s.insert("Stockholm");
        s.insert("Oslo");
        s.insert("Helsinki");
        s.insert("Reykjavik");
        s.insert("Dublin");
        s.insert("Edinburgh");
        s.insert("Cardiff");
        s.insert("Lisbon");
        s.insert("Athens");
        s.insert("Barcelona");
        s.insert("Milan");
        s.insert("Naples");
        s.insert("Venice");
        s.insert("Florence");
        // US cities and states
        s.insert("York");
        s.insert("California");
        s.insert("Texas");
        s.insert("Florida");
        s.insert("Illinois");
        s.insert("Pennsylvania");
        s.insert("Ohio");
        s.insert("Georgia");
        s.insert("Michigan");
        s.insert("Virginia");
        s.insert("Washington");
        s.insert("Arizona");
        s.insert("Massachusetts");
        s.insert("Tennessee");
        s.insert("Indiana");
        s.insert("Missouri");
        s.insert("Maryland");
        s.insert("Wisconsin");
        s.insert("Colorado");
        s.insert("Minnesota");
        s.insert("Louisiana");
        s.insert("Alabama");
        s.insert("Kentucky");
        s.insert("Oregon");
        s.insert("Oklahoma");
        s.insert("Connecticut");
        s.insert("Iowa");
        s.insert("Mississippi");
        s.insert("Arkansas");
        s.insert("Kansas");
        s.insert("Utah");
        s.insert("Nevada");
        s.insert("Mexico");
        s.insert("Hawaii");
        s.insert("Nebraska");
        s.insert("Idaho");
        s.insert("Maine");
        s.insert("Hampshire");
        s.insert("Rhode");
        s.insert("Montana");
        s.insert("Delaware");
        s.insert("Dakota");
        s.insert("Alaska");
        s.insert("Vermont");
        s.insert("Wyoming");
        s
    };

    // Time indicators
    static ref TIME_PATTERNS: Regex = Regex::new(
        r"(?i)(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    ).unwrap();

    // Quantity patterns
    static ref QUANTITY_PATTERNS: Regex = Regex::new(
        r"(?i)(\d+(?:\.\d+)?)\s*(percent|%|dollars?|\$|euros?|€|pounds?|£|meters?|m|kilometers?|km|miles?|kg|kilograms?|grams?|g|liters?|l|years?|months?|days?|hours?|minutes?|seconds?)"
    ).unwrap();
}

pub struct EntityExtractor {
    // In a production system, we might have ML models here
}

impl EntityExtractor {
    pub fn new() -> Self {
        EntityExtractor {}
    }

    pub fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        // Extract time entities first (to avoid them being extracted as Unknown)
        entities.extend(self.extract_time_entities(text, &mut seen));

        // Extract quantity entities
        entities.extend(self.extract_quantity_entities(text, &mut seen));
        
        // Extract quoted entities
        entities.extend(self.extract_quoted_entities(text, &mut seen));

        // Extract multi-word capitalized sequences (likely proper nouns)
        entities.extend(self.extract_capitalized_sequences(text, &mut seen));

        // Apply entity type classification
        for entity in &mut entities {
            if entity.entity_type == EntityType::Unknown {
                entity.entity_type = self.classify_entity_type(&entity.name, text);
            }
        }

        // Sort by position and deduplicate overlapping entities
        entities.sort_by_key(|e| e.start_pos);
        entities = self.remove_overlapping_entities(entities);
        
        // Post-process to combine related adjacent entities like "Nobel" + "Prize"
        entities = self.combine_adjacent_concept_entities(entities, text);
        
        entities
    }

    fn extract_capitalized_sequences(&self, text: &str, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;

        while i < words.len() {
            let word = words[i];
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            
            // Check if it's an uppercase word (could be an acronym like AI, USA, etc.)
            let is_uppercase = clean_word.len() >= 2 && clean_word.chars().all(|c| !c.is_lowercase());
            let starts_with_capital = clean_word.chars().next().map_or(false, |c| c.is_uppercase());
            
            if clean_word.is_empty() || (!starts_with_capital && !is_uppercase) {
                i += 1;
                continue;
            }
            
            // If it's a standalone uppercase acronym, extract it directly (including 2-letter ones like AI)
            if is_uppercase && clean_word.len() >= 2 {
                if !seen.contains(clean_word) {
                    // Find the actual position of this word instance
                    let position = self.find_word_position(text, word, i, &words);
                    seen.insert(clean_word.to_string());
                    entities.push(Entity {
                        name: clean_word.to_string(),
                        entity_type: EntityType::Unknown,
                        start_pos: position,
                        end_pos: position + clean_word.len(),
                    });
                }
                i += 1;
                continue;
            }

            // Start of a potential multi-word entity
            let mut entity_words = vec![clean_word];
            let start_pos = self.find_word_position(text, word, i, &words);
            let mut j = i + 1;

            // Continue collecting words while they're capitalized or connectors
            while j < words.len() {
                let next_word = words[j];
                let clean_next = next_word.trim_matches(|c: char| !c.is_alphanumeric());
                
                if clean_next.is_empty() {
                    j += 1;
                    continue;
                }

                // Check if the previous word ended with sentence-ending punctuation
                // This indicates we shouldn't continue collecting into the next sentence
                if j > i {
                    let prev_word = words[j - 1];
                    if prev_word.ends_with('.') || prev_word.ends_with('!') || prev_word.ends_with('?') {
                        break; // Stop collecting at sentence boundaries
                    }
                }

                // Check if it's a connector word (and, of, the, etc.)
                // But be more restrictive with "and" - it should only connect if the next word is also capitalized
                if is_connector_word(clean_next) && j + 1 < words.len() {
                    if clean_next == "and" {
                        // Only treat "and" as a connector if the word after it is capitalized
                        // and the current entity doesn't already have common concepts
                        let next_next_word = words.get(j + 1).map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()));
                        let next_is_capitalized = next_next_word.map_or(false, |w| 
                            !w.is_empty() && w.chars().next().unwrap().is_uppercase()
                        );
                        
                        // Don't use "and" as connector if current entity has concept words
                        let has_concept_word = entity_words.iter().any(|w| KNOWN_CONCEPTS.contains(w));
                        
                        if !next_is_capitalized || has_concept_word {
                            break; // Stop collecting here
                        }
                    }
                    entity_words.push(clean_next);
                    j += 1;
                    continue;
                }

                // Check if it's another capitalized word or acronym
                let next_is_uppercase = clean_next.len() >= 2 && clean_next.chars().all(|c| !c.is_lowercase());
                if clean_next.chars().next().unwrap().is_uppercase() || next_is_uppercase {
                    entity_words.push(clean_next);
                    j += 1;
                } else {
                    break;
                }
            }

            // Create entity if we have at least one word and it's not a common word
            let mut entity_name = entity_words.join(" ");
            
            // Remove leading "The" if it's not part of a proper name
            if entity_name.starts_with("The ") && entity_words.len() > 1 {
                let without_the = entity_words[1..].join(" ");
                if without_the.chars().next().map_or(false, |c| c.is_uppercase()) {
                    entity_name = without_the;
                }
            }
            
            if !entity_name.is_empty() && !is_common_word(&entity_name) && !seen.contains(&entity_name) {
                // Check if we should split this entity because it contains known places
                // Pass the actual words from the text (with punctuation) to detect commas
                let original_words: Vec<&str> = words[i..j].to_vec();
                let split_entities = self.try_split_entity_with_places_original(&entity_words, &original_words, text, start_pos, seen);
                
                if !split_entities.is_empty() {
                    // Use the split entities instead
                    entities.extend(split_entities);
                } else {
                    // Calculate actual end position based on the last word included
                    let last_word_idx = j - 1;
                    let end_pos = if last_word_idx < words.len() {
                        let last_word_pos = self.find_word_position(text, words[last_word_idx], last_word_idx, &words);
                        last_word_pos + words[last_word_idx].trim_matches(|c: char| !c.is_alphanumeric()).len()
                    } else {
                        start_pos + entity_name.len()
                    };

                    seen.insert(entity_name.clone());
                    entities.push(Entity {
                        name: entity_name,
                        entity_type: EntityType::Unknown,
                        start_pos,
                        end_pos,
                    });
                }
            }

            i = j;
        }

        entities
    }

    fn extract_quoted_entities(&self, text: &str, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut entities = Vec::new();
        let quote_pattern = Regex::new(r#"["']([^"']+)["']"#).unwrap();

        for cap in quote_pattern.captures_iter(text) {
            if let Some(match_) = cap.get(1) {
                let entity_name = match_.as_str().to_string();
                if !seen.contains(&entity_name) && entity_name.len() > 2 {
                    seen.insert(entity_name.clone());
                    entities.push(Entity {
                        name: entity_name,
                        entity_type: EntityType::Concept, // Quoted items often concepts
                        start_pos: match_.start(),
                        end_pos: match_.end(),
                    });
                }
            }
        }

        entities
    }

    fn extract_time_entities(&self, text: &str, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut entities = Vec::new();

        for cap in TIME_PATTERNS.captures_iter(text) {
            if let Some(match_) = cap.get(0) {
                let entity_name = match_.as_str().to_string();
                if !seen.contains(&entity_name) {
                    seen.insert(entity_name.clone());
                    entities.push(Entity {
                        name: entity_name,
                        entity_type: EntityType::Time,
                        start_pos: match_.start(),
                        end_pos: match_.end(),
                    });
                }
            }
        }

        // Also extract standalone years (4-digit numbers)
        let year_pattern = Regex::new(r"\b(19\d{2}|20\d{2})\b").unwrap();
        for cap in year_pattern.captures_iter(text) {
            if let Some(match_) = cap.get(0) {
                let entity_name = match_.as_str().to_string();
                if !seen.contains(&entity_name) {
                    seen.insert(entity_name.clone());
                    entities.push(Entity {
                        name: entity_name,
                        entity_type: EntityType::Time,
                        start_pos: match_.start(),
                        end_pos: match_.end(),
                    });
                }
            }
        }

        entities
    }

    fn extract_quantity_entities(&self, text: &str, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut entities = Vec::new();

        for cap in QUANTITY_PATTERNS.captures_iter(text) {
            if let Some(match_) = cap.get(0) {
                let entity_name = match_.as_str().to_string();
                if !seen.contains(&entity_name) {
                    seen.insert(entity_name.clone());
                    entities.push(Entity {
                        name: entity_name,
                        entity_type: EntityType::Quantity,
                        start_pos: match_.start(),
                        end_pos: match_.end(),
                    });
                }
            }
        }

        entities
    }

    fn classify_entity_type(&self, entity_name: &str, _context: &str) -> EntityType {
        let words: Vec<&str> = entity_name.split_whitespace().collect();
        
        // Check for person titles
        if let Some(first_word) = words.first() {
            if PERSON_TITLES.contains(first_word) {
                return EntityType::Person;
            }
        }

        // Check for organization indicators
        if let Some(last_word) = words.last() {
            if ORG_INDICATORS.contains(last_word) {
                return EntityType::Organization;
            }
        }

        // Check for place indicators
        for word in &words {
            if PLACE_INDICATORS.contains(word) {
                return EntityType::Place;
            }
        }

        // Check for known place names (before person check)
        for word in &words {
            if KNOWN_PLACES.contains(word) {
                return EntityType::Place;
            }
        }
        
        // Special check for the whole entity name as a known place
        if KNOWN_PLACES.contains(&entity_name) {
            return EntityType::Place;
        }

        // Check for known concepts (before person check)
        for word in &words {
            if KNOWN_CONCEPTS.contains(word) {
                return EntityType::Concept;
            }
        }
        
        // Special check for the whole entity name as a known concept
        if KNOWN_CONCEPTS.contains(&entity_name) {
            return EntityType::Concept;
        }

        // Check if it might be a person name (first and last name pattern)
        // Be more restrictive - only classify as person if it follows typical naming patterns
        if words.len() == 2 || words.len() == 3 {
            let all_capitalized = words.iter().all(|w| {
                w.chars().next().map_or(false, |c| c.is_uppercase())
            });
            
            // Additional checks to be more restrictive about person classification
            let has_common_place_indicators = words.iter().any(|w| {
                // Common words that indicate places, not people
                matches!(w.to_lowercase().as_str(), 
                    "tower" | "building" | "bridge" | "center" | "square" | 
                    "street" | "avenue" | "road" | "park" | "hall" | "station" |
                    "airport" | "port" | "mall" | "market" | "museum" | "library" |
                    "hospital" | "school" | "college" | "university"
                )
            });
            
            // Don't classify as person if it contains place indicators
            if all_capitalized && !words.iter().any(|w| w.len() == 1) && !has_common_place_indicators {
                return EntityType::Person;
            }
        }

        // Default to concept for other multi-word entities
        if words.len() > 1 {
            EntityType::Concept
        } else {
            EntityType::Unknown
        }
    }

    fn find_word_position(&self, text: &str, word: &str, word_index: usize, words: &[&str]) -> usize {
        // Calculate position by reconstructing the text up to this word
        let mut pos = 0;
        
        for (idx, w) in words.iter().enumerate() {
            if idx == word_index {
                // Find this specific occurrence of the word in the remaining text
                if let Some(offset) = text[pos..].find(word) {
                    return pos + offset;
                }
            }
            
            // Move position forward by the word length plus any whitespace
            if let Some(word_pos) = text[pos..].find(w) {
                pos += word_pos + w.len();
                // Skip any whitespace after the word
                while pos < text.len() && text.chars().nth(pos).map_or(false, |c| c.is_whitespace()) {
                    pos += 1;
                }
            }
        }
        
        // Fallback to simple find
        text.find(word).unwrap_or(0)
    }

    fn try_split_entity_with_places(&self, entity_words: &[&str], text: &str, base_start_pos: usize, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut split_entities = Vec::new();
        
        // Only split if we have multiple words and there's a clear reason to split
        if entity_words.len() < 2 {
            return split_entities;
        }
        
        // Check if we have a mix of different entity types that should be split
        let has_known_place = entity_words.iter().any(|word| KNOWN_PLACES.contains(word));
        let has_known_concept = entity_words.iter().any(|word| KNOWN_CONCEPTS.contains(word));
        let has_org_indicator = entity_words.iter().any(|word| ORG_INDICATORS.contains(word));
        
        // Check if we have punctuation that suggests separate entities (like "Paris, France")
        let entity_text = entity_words.join(" ");
        let has_comma_separation = entity_text.contains(',');
        
        // Only split if we have a clear mix of different entity types
        // For example: "Google California" (org + place) should split
        // But "Theory of Relativity" (concept + connector + concept) should NOT split
        let type_count = [has_known_place, has_known_concept, has_org_indicator].iter().filter(|&&x| x).count();
        
        // Don't split if:
        // 1. Only one type is present (like "Theory of Relativity" - all concept related)
        // 2. The entity contains connector words that suggest it's a compound name
        // 3. EXCEPTION: Always split comma-separated places like "Paris, France"
        let has_connectors = entity_words.iter().any(|word| is_connector_word(word));
        
        if (type_count <= 1 || has_connectors) && !has_comma_separation {
            return split_entities; // Don't split, keep as single entity
        }
        
        // Special handling for comma-separated places
        if has_comma_separation && has_known_place {
            return self.split_comma_separated_places(&entity_words, text, base_start_pos, seen);
        }
        
        // Only split when we have genuinely mixed types without connector words
        // This is rare and usually indicates separate entities mentioned together
        let mut current_pos = base_start_pos;
        
        for (_i, word) in entity_words.iter().enumerate() {
            if !is_common_word(word) && !seen.contains(*word) {
                // Find the actual position of this word in the text
                let word_pos = if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos + found_pos
                } else {
                    current_pos
                };
                
                seen.insert(word.to_string());
                split_entities.push(Entity {
                    name: word.to_string(),
                    entity_type: EntityType::Unknown, // Will be classified later
                    start_pos: word_pos,
                    end_pos: word_pos + word.len(),
                });
                
                // Move current position past this word
                current_pos = word_pos + word.len();
            } else {
                // Skip common words but advance position
                if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos += found_pos + word.len();
                }
            }
        }
        
        split_entities
    }

    fn try_split_entity_with_places_original(&self, entity_words: &[&str], original_words: &[&str], text: &str, base_start_pos: usize, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut split_entities = Vec::new();
        
        // Only split if we have multiple words and there's a clear reason to split
        if entity_words.len() < 2 {
            return split_entities;
        }
        
        // Check if we have a mix of different entity types that should be split
        let has_known_place = entity_words.iter().any(|word| KNOWN_PLACES.contains(word));
        let has_known_concept = entity_words.iter().any(|word| KNOWN_CONCEPTS.contains(word));
        let has_org_indicator = entity_words.iter().any(|word| ORG_INDICATORS.contains(word));
        
        // Check if we have punctuation that suggests separate entities (like "Paris, France")
        let has_comma_separation = original_words.iter().any(|word| word.contains(','));
        
        // Only split if we have a clear mix of different entity types
        // For example: "Google California" (org + place) should split
        // But "Theory of Relativity" (concept + connector + concept) should NOT split
        let type_count = [has_known_place, has_known_concept, has_org_indicator].iter().filter(|&&x| x).count();
        
        // Don't split if:
        // 1. Only one type is present (like "Theory of Relativity" - all concept related)
        // 2. The entity contains connector words that suggest it's a compound name
        // 3. EXCEPTION: Always split comma-separated places like "Paris, France"
        let has_connectors = entity_words.iter().any(|word| is_connector_word(word));
        
        if (type_count <= 1 || has_connectors) && !has_comma_separation {
            return split_entities; // Don't split, keep as single entity
        }
        
        // Special handling for comma-separated places
        if has_comma_separation && has_known_place {
            return self.split_comma_separated_places_original(original_words, text, base_start_pos, seen);
        }
        
        // Only split when we have genuinely mixed types without connector words
        // This is rare and usually indicates separate entities mentioned together
        let mut current_pos = base_start_pos;
        
        for (_i, word) in entity_words.iter().enumerate() {
            if !is_common_word(word) && !seen.contains(*word) {
                // Find the actual position of this word in the text
                let word_pos = if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos + found_pos
                } else {
                    current_pos
                };
                
                seen.insert(word.to_string());
                split_entities.push(Entity {
                    name: word.to_string(),
                    entity_type: EntityType::Unknown, // Will be classified later
                    start_pos: word_pos,
                    end_pos: word_pos + word.len(),
                });
                
                // Move current position past this word
                current_pos = word_pos + word.len();
            } else {
                // Skip common words but advance position
                if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos += found_pos + word.len();
                }
            }
        }
        
        split_entities
    }

    fn split_comma_separated_places_original(&self, original_words: &[&str], text: &str, base_start_pos: usize, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut split_entities = Vec::new();
        let mut current_entity_words = Vec::new();
        let mut current_pos = base_start_pos;
        
        for (_i, word) in original_words.iter().enumerate() {
            if word.contains(',') {
                // Add the word without comma to current entity
                let word_without_comma = word.trim_end_matches(',');
                if !word_without_comma.is_empty() && !is_common_word(word_without_comma) {
                    current_entity_words.push(word_without_comma);
                }
                
                // Create entity from current words
                if !current_entity_words.is_empty() {
                    let entity_name = current_entity_words.join(" ");
                    if !seen.contains(&entity_name) {
                        let entity_len = entity_name.len();
                        seen.insert(entity_name.clone());
                        split_entities.push(Entity {
                            name: entity_name,
                            entity_type: EntityType::Unknown,
                            start_pos: current_pos,
                            end_pos: current_pos + entity_len,
                        });
                    }
                }
                
                // Reset for next entity
                current_entity_words.clear();
                current_pos = if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos + found_pos + word.len()
                } else {
                    current_pos + word.len()
                };
            } else {
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean_word.is_empty() && !is_common_word(clean_word) {
                    current_entity_words.push(clean_word);
                }
            }
        }
        
        // Handle the last entity if any words remain
        if !current_entity_words.is_empty() {
            let entity_name = current_entity_words.join(" ");
            if !seen.contains(&entity_name) {
                let entity_len = entity_name.len();
                seen.insert(entity_name.clone());
                split_entities.push(Entity {
                    name: entity_name,
                    entity_type: EntityType::Unknown,
                    start_pos: current_pos,
                    end_pos: current_pos + entity_len,
                });
            }
        }
        
        split_entities
    }

    fn split_comma_separated_places(&self, entity_words: &[&str], text: &str, base_start_pos: usize, seen: &mut HashSet<String>) -> Vec<Entity> {
        let mut split_entities = Vec::new();
        let mut current_entity_words = Vec::new();
        let mut current_pos = base_start_pos;
        
        for (_i, word) in entity_words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != ',');
            
            if word.contains(',') {
                // Add the word without comma to current entity
                let word_without_comma = word.trim_end_matches(',');
                if !word_without_comma.is_empty() {
                    current_entity_words.push(word_without_comma);
                }
                
                // Create entity from current words
                if !current_entity_words.is_empty() {
                    let entity_name = current_entity_words.join(" ");
                    if !seen.contains(&entity_name) && !is_common_word(&entity_name) {
                        let entity_len = entity_name.len();
                        seen.insert(entity_name.clone());
                        split_entities.push(Entity {
                            name: entity_name,
                            entity_type: EntityType::Unknown,
                            start_pos: current_pos,
                            end_pos: current_pos + entity_len,
                        });
                    }
                }
                
                // Reset for next entity
                current_entity_words.clear();
                current_pos = if let Some(found_pos) = text[current_pos..].find(word) {
                    current_pos + found_pos + word.len()
                } else {
                    current_pos + word.len()
                };
            } else if !clean_word.is_empty() && !is_common_word(clean_word) {
                current_entity_words.push(clean_word);
            }
        }
        
        // Handle the last entity if any words remain
        if !current_entity_words.is_empty() {
            let entity_name = current_entity_words.join(" ");
            if !seen.contains(&entity_name) && !is_common_word(&entity_name) {
                let entity_len = entity_name.len();
                seen.insert(entity_name.clone());
                split_entities.push(Entity {
                    name: entity_name,
                    entity_type: EntityType::Unknown,
                    start_pos: current_pos,
                    end_pos: current_pos + entity_len,
                });
            }
        }
        
        split_entities
    }

    fn combine_adjacent_concept_entities(&self, entities: Vec<Entity>, text: &str) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < entities.len() {
            let current = &entities[i];
            
            // Check if current entity can be combined with the next one
            if i + 1 < entities.len() {
                let next = &entities[i + 1];
                
                // Check if they are adjacent (or nearly adjacent with only whitespace/punctuation)
                let gap_text = &text[current.end_pos..next.start_pos];
                let is_adjacent = gap_text.trim_matches(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ':' || c == ';').is_empty();
                
                if is_adjacent {
                    // Check if they should be combined based on concept patterns
                    let should_combine = self.should_combine_entities(current, next);
                    
                    if should_combine {
                        // Combine the entities
                        let combined_name = format!("{} {}", current.name, next.name);
                        let combined_entity = Entity {
                            name: combined_name.clone(),
                            entity_type: self.classify_entity_type(&combined_name, text),
                            start_pos: current.start_pos,
                            end_pos: next.end_pos,
                        };
                        result.push(combined_entity);
                        i += 2; // Skip both entities
                        continue;
                    }
                }
            }
            
            // If we don't combine, just add the current entity
            result.push(current.clone());
            i += 1;
        }
        
        result
    }
    
    fn should_combine_entities(&self, first: &Entity, second: &Entity) -> bool {
        // Combine if first is a proper noun and second is a concept
        // E.g., "Nobel" + "Prize", "Theory" + "Relativity", etc.
        
        // Check for specific combinations
        let first_upper = first.name.chars().next().map_or(false, |c| c.is_uppercase());
        let second_is_concept = second.entity_type == EntityType::Concept || KNOWN_CONCEPTS.contains(&second.name.as_str());
        
        // Nobel + Prize, Peace + Prize, etc.
        if first_upper && second_is_concept {
            return true;
        }
        
        // Theory + of + Something, etc.
        if KNOWN_CONCEPTS.contains(&first.name.as_str()) && first_upper {
            return true;
        }
        
        false
    }

    fn remove_overlapping_entities(&self, mut entities: Vec<Entity>) -> Vec<Entity> {
        if entities.is_empty() {
            return entities;
        }

        let mut result = Vec::new();
        result.push(entities.remove(0));

        for entity in entities {
            let last = result.last().unwrap();
            // If entities don't overlap, add the new one
            if entity.start_pos >= last.end_pos {
                result.push(entity);
            } else if entity.name.len() > last.name.len() {
                // If they overlap and the new one is longer, replace
                result.pop();
                result.push(entity);
            }
            // Otherwise, keep the existing one
        }

        result
    }
}

fn is_connector_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "and" | "of" | "the" | "de" | "del" | "la" | "le" | "von" | "van" | "der"
    )
}

fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" |
        "of" | "with" | "by" | "from" | "as" | "is" | "was" | "are" | "were" |
        "been" | "being" | "have" | "has" | "had" | "do" | "does" | "did" |
        "will" | "would" | "could" | "should" | "may" | "might" | "must" |
        "can" | "this" | "that" | "these" | "those" | "a" | "an" |
        "he" | "she" | "it" | "they" | "we" | "you" | "i" | "me" | "my" |
        "his" | "her" | "its" | "their" | "our" | "your"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_person_entities() {
        let extractor = EntityExtractor::new();
        let text = "Albert Einstein developed the Theory of Relativity in 1905.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "Albert Einstein" && e.entity_type == EntityType::Person));
        assert!(entities.iter().any(|e| e.name == "Theory of Relativity" && e.entity_type == EntityType::Concept));
    }

    #[test]
    fn test_extract_organization_entities() {
        let extractor = EntityExtractor::new();
        let text = "Microsoft Corporation announced a partnership with OpenAI Inc.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "Microsoft Corporation" && e.entity_type == EntityType::Organization));
        assert!(entities.iter().any(|e| e.name == "OpenAI Inc" && e.entity_type == EntityType::Organization));
    }

    #[test]
    fn test_extract_quoted_entities() {
        let extractor = EntityExtractor::new();
        let text = "The concept of 'quantum entanglement' is fascinating.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "quantum entanglement" && e.entity_type == EntityType::Concept));
    }

    #[test]
    fn test_extract_time_entities() {
        let extractor = EntityExtractor::new();
        let text = "The meeting is scheduled for January 15, 2024.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "January" && e.entity_type == EntityType::Time));
        assert!(entities.iter().any(|e| e.name == "2024" && e.entity_type == EntityType::Time));
    }

    #[test]
    fn test_no_overlapping_entities() {
        let extractor = EntityExtractor::new();
        let text = "New York City is in New York State.";
        let entities = extractor.extract_entities(text);

        // Should extract both as separate entities
        assert!(entities.iter().any(|e| e.name == "New York City"));
        assert!(entities.iter().any(|e| e.name == "New York State"));
    }
}