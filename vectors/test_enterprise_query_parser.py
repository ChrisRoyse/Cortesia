#!/usr/bin/env python3
"""
TDD Tests for Enterprise Query Parser
=====================================

Testing:
- Special character escaping
- Boolean queries
- Regex patterns
- Proximity searches
- Wildcard matching
- Cross-line patterns
- Complex combinations

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import pytest
from enterprise_query_parser import (
    EnterpriseQueryParser, 
    QueryType,
    create_enterprise_query_parser
)


class TestSpecialCharacterHandling:
    """Test special character escaping"""
    
    def test_brackets_escaping(self):
        parser = create_enterprise_query_parser()
        
        # Test square brackets
        result = parser.parse("[dependencies]")
        assert result.query_type == QueryType.EXACT
        assert result.fts_query == '"\\[dependencies\\]"'
        
        # Test parentheses
        result = parser.parse("function(param)")
        assert result.fts_query == '"function\\(param\\)"'
        
        # Test curly braces
        result = parser.parse("{config}")
        assert result.fts_query == '"\\{config\\}"'
    
    def test_operator_characters(self):
        parser = create_enterprise_query_parser()
        
        # Test mathematical operators
        result = parser.parse("a+b")
        assert result.fts_query == '"a\\+b"'
        
        result = parser.parse("x-y")
        assert result.fts_query == '"x\\-y"'
        
        result = parser.parse("foo->bar")
        assert result.fts_query == '"foo\\-\\>bar"'
    
    def test_special_symbols(self):
        parser = create_enterprise_query_parser()
        
        # Test various symbols
        result = parser.parse("@decorator")
        assert result.fts_query == '"\\@decorator"'
        
        result = parser.parse("#define")
        assert result.fts_query == '"\\#define"'
        
        result = parser.parse("$variable")
        assert result.fts_query == '"\\$variable"'
        
        result = parser.parse("module::function")
        assert result.fts_query == '"module\\:\\:function"'
    
    def test_markdown_headers(self):
        parser = create_enterprise_query_parser()
        
        # Test markdown headers
        result = parser.parse("## Header")
        assert result.fts_query == '"\\#\\# Header"'
        
        result = parser.parse("### SubHeader")
        assert result.fts_query == '"\\#\\#\\# SubHeader"'


class TestBooleanQueries:
    """Test boolean query parsing"""
    
    def test_and_operator(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("error AND handling")
        assert result.query_type == QueryType.BOOLEAN
        assert result.tokens == ["error", "handling"]
        assert result.operators == ["AND"]
        assert '"error"' in result.fts_query
        assert ' AND ' in result.fts_query
        assert '"handling"' in result.fts_query
    
    def test_or_operator(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("async OR await")
        assert result.query_type == QueryType.BOOLEAN
        assert result.tokens == ["async", "await"]
        assert result.operators == ["OR"]
        assert ' OR ' in result.fts_query
    
    def test_not_operator(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("function NOT test")
        assert result.query_type == QueryType.BOOLEAN
        assert result.tokens == ["function", "test"]
        assert result.operators == ["NOT"]
        assert ' NOT ' in result.fts_query
    
    def test_complex_boolean(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("(error OR warning) AND NOT test")
        assert result.query_type == QueryType.BOOLEAN
        assert "error" in result.tokens
        assert "warning" in result.tokens
        assert "test" in result.tokens
    
    def test_alternative_operators(self):
        parser = create_enterprise_query_parser()
        
        # Test && for AND
        result = parser.parse("foo && bar")
        assert result.query_type == QueryType.BOOLEAN
        assert ' AND ' in result.fts_query
        
        # Test || for OR
        result = parser.parse("foo || bar")
        assert result.query_type == QueryType.BOOLEAN
        assert ' OR ' in result.fts_query
        
        # Test ! for NOT
        result = parser.parse("foo ! bar")
        assert result.query_type == QueryType.BOOLEAN
        assert ' NOT ' in result.fts_query


class TestRegexQueries:
    """Test regex pattern parsing"""
    
    def test_simple_regex(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("/function.*test/")
        assert result.query_type == QueryType.REGEX
        assert result.regex_pattern is not None
        assert result.metadata['pattern'] == "function.*test"
    
    def test_regex_with_groups(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("/def (\\w+)\\(/")
        assert result.query_type == QueryType.REGEX
        assert result.regex_pattern is not None
    
    def test_multiline_regex(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("/class.*\\n.*def/")
        assert result.query_type == QueryType.REGEX
        assert result.regex_pattern is not None
    
    def test_invalid_regex_fallback(self):
        parser = create_enterprise_query_parser()
        
        # Invalid regex should fall back to exact
        result = parser.parse("/[unclosed/")
        assert result.query_type == QueryType.EXACT


class TestWildcardQueries:
    """Test wildcard pattern matching"""
    
    def test_star_wildcard(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("test*")
        assert result.query_type == QueryType.WILDCARD
        assert result.tokens == ["test"]
        assert '*' in result.fts_query
    
    def test_question_wildcard(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("te?t")
        assert result.query_type == QueryType.WILDCARD
        assert '_' in result.fts_query  # FTS5 uses _ for single char
    
    def test_multiple_wildcards(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("*Service*")
        assert result.query_type == QueryType.WILDCARD
        assert result.tokens == ["Service"]
    
    def test_combined_wildcards(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("get*Value?")
        assert result.query_type == QueryType.WILDCARD
        assert "get" in result.tokens
        assert "Value" in result.tokens


class TestProximityQueries:
    """Test proximity search parsing"""
    
    def test_near_operator(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("error NEAR/5 handling")
        assert result.query_type == QueryType.PROXIMITY
        assert result.tokens == ["error", "handling"]
        assert result.metadata['distance'] == 5
    
    def test_different_distances(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("async NEAR/10 await")
        assert result.query_type == QueryType.PROXIMITY
        assert result.metadata['distance'] == 10


class TestCrossLinePatterns:
    """Test patterns spanning multiple lines"""
    
    def test_literal_newline(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("function\\ndef")
        assert result.query_type == QueryType.CROSS_LINE
        assert ' ' in result.fts_query  # Newline replaced with space
    
    def test_ellipsis_pattern(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("class...end")
        assert result.query_type == QueryType.CROSS_LINE
        # Should convert to wildcard
        assert '*' in result.fts_query
    
    def test_multiline_function(self):
        parser = create_enterprise_query_parser()
        
        query = "def function_name(\\n    param1,\\n    param2\\n)"
        result = parser.parse(query)
        assert result.query_type == QueryType.CROSS_LINE


class TestPhraseQueries:
    """Test exact phrase matching"""
    
    def test_quoted_phrase(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse('"exact phrase match"')
        assert result.query_type == QueryType.PHRASE
        assert result.tokens == ["exact phrase match"]
        assert result.metadata['phrase_length'] == 3
    
    def test_single_quoted_phrase(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("'another phrase'")
        assert result.query_type == QueryType.PHRASE
        assert result.tokens == ["another phrase"]


class TestComplexQueries:
    """Test complex query combinations"""
    
    def test_build_complex_query(self):
        parser = create_enterprise_query_parser()
        
        conditions = [
            ('', 'error'),
            ('AND', 'handling'),
            ('NOT', 'test')
        ]
        
        result = parser.build_complex_query(conditions)
        assert result.query_type == QueryType.BOOLEAN
        assert len(result.tokens) == 3
        assert len(result.operators) == 2
        assert result.metadata['complex'] == True
    
    def test_mixed_special_chars_and_boolean(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("[dependencies] AND version>=1.0")
        assert result.query_type == QueryType.BOOLEAN
        assert '\\[' in result.fts_query
        assert '\\>' in result.fts_query
    
    def test_wildcard_with_special_chars(self):
        parser = create_enterprise_query_parser()
        
        result = parser.parse("@decorator*")
        assert result.query_type == QueryType.WILDCARD
        assert '\\@' in result.fts_query
        assert '*' in result.fts_query  # Wildcard not escaped


class TestCaching:
    """Test query caching mechanism"""
    
    def test_query_caching(self):
        parser = create_enterprise_query_parser()
        
        # First parse
        result1 = parser.parse("test query")
        
        # Second parse (should use cache)
        result2 = parser.parse("test query")
        
        # Should be the same object
        assert result1 is result2
    
    def test_cache_different_queries(self):
        parser = create_enterprise_query_parser()
        
        result1 = parser.parse("query1")
        result2 = parser.parse("query2")
        
        assert result1 is not result2
        assert result1.original != result2.original


def run_all_tests():
    """Run all tests and report results"""
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return result.returncode == 0


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All enterprise query parser tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
    exit(0 if success else 1)