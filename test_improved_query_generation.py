#!/usr/bin/env python3

"""
Test script for improved query generation with better natural language patterns.
"""

def test_query_generation():
    """Test the enhanced query generation patterns"""
    
    # Test cases for the improved functionality
    test_cases = [
        # Case sensitivity fixes
        {
            "query": "Find all facts about einstein",
            "expected_entity": "Einstein",
            "description": "Should fix case sensitivity - Einstein not einstein"
        },
        {
            "query": "Tell me about newton's laws",
            "expected_entity": "Newton",
            "description": "Should extract Newton with proper capitalization"
        },
        # Temporal queries
        {
            "query": "Who invented what after 1900?",
            "expected_pattern": "temporal",
            "description": "Should handle temporal constraint 'after 1900'"
        },
        {
            "query": "What was discovered before 1800?",
            "expected_pattern": "temporal",
            "description": "Should handle temporal constraint 'before 1800'"
        },
        {
            "query": "Scientific breakthroughs during 1920",
            "expected_pattern": "temporal",
            "description": "Should handle temporal constraint 'during 1920'"
        },
        {
            "query": "Inventions between 1850 and 1900",
            "expected_pattern": "temporal",
            "description": "Should handle temporal range constraints"
        },
        # Multi-entity queries
        {
            "query": "What did Einstein and Newton both discover?",
            "expected_entities": ["Einstein", "Newton"],
            "expected_pattern": "multi_entity",
            "description": "Should handle 'both X and Y' patterns"
        },
        {
            "query": "How are Darwin and Mendel both connected to genetics?",
            "expected_entities": ["Darwin", "Mendel"],
            "expected_pattern": "multi_entity", 
            "description": "Should extract multiple entities with relationship"
        },
        # Complex entity patterns
        {
            "query": "Einstein's theory of relativity",
            "expected_entity": "Einstein",
            "expected_context": "possessive: theory",
            "description": "Should handle possessive patterns"
        },
        {
            "query": "Newton who discovered gravity",
            "expected_entity": "Newton",
            "expected_context": "description: discovered gravity",
            "description": "Should handle descriptive clauses"
        },
        # Natural language improvements
        {
            "query": "Tell me about Marie Curie's research",
            "expected_entity": "Marie Curie",
            "description": "Should handle multi-word proper names"
        },
        {
            "query": "What is the theory of Darwin?",
            "expected_entity": "Darwin",
            "description": "Should extract entities from complex sentence structures"
        }
    ]
    
    print("Enhanced Query Generation Test Results")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        # Note: These tests would need to be run against the actual Rust implementation
        # For now, we're documenting the expected behavior
        
        if 'expected_entity' in test_case:
            print(f"Expected Entity: {test_case['expected_entity']}")
            
        if 'expected_entities' in test_case:
            print(f"Expected Entities: {test_case['expected_entities']}")
            
        if 'expected_pattern' in test_case:
            print(f"Expected Pattern: {test_case['expected_pattern']}")
            
        if 'expected_context' in test_case:
            print(f"Expected Context: {test_case['expected_context']}")
        
        print("[OK] Pattern defined")
    
    print(f"\n{len(test_cases)} test cases defined for enhanced query generation")
    
    # Test query improvement suggestions
    print("\nQuery Improvement Examples:")
    print("-" * 30)
    
    improvement_examples = [
        {
            "original": "find facts about einstein",
            "improved": "Find facts about Einstein",
            "fix": "Case sensitivity"
        },
        {
            "original": "AI discoveries", 
            "improved": "AI discoveries (Note: AI could mean Artificial Intelligence)",
            "fix": "Abbreviation expansion"
        },
        {
            "original": "newton and galileo both studied",
            "improved": "Newton and Galileo both studied",
            "fix": "Proper name capitalization"
        }
    ]
    
    for example in improvement_examples:
        print(f"Original: '{example['original']}'")
        print(f"Improved: '{example['improved']}'")
        print(f"Fix Applied: {example['fix']}")
        print()

if __name__ == "__main__":
    test_query_generation()