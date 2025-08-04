#!/usr/bin/env python3
"""
Enterprise Query Parser - Advanced Pattern Matching & Special Character Handling
================================================================================

Handles:
- Special character escaping for FTS5
- Complex boolean queries (AND, OR, NOT)
- Regex pattern matching
- Proximity searches
- Cross-line pattern assembly
- Wildcard expansion

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of queries supported"""
    EXACT = "exact"           # Exact string match
    BOOLEAN = "boolean"        # AND/OR/NOT combinations
    REGEX = "regex"           # Regular expression
    PROXIMITY = "proximity"    # Words within N tokens
    PHRASE = "phrase"         # Exact phrase match
    WILDCARD = "wildcard"     # * and ? wildcards
    CROSS_LINE = "cross_line" # Patterns spanning lines


@dataclass
class ParsedQuery:
    """Parsed query structure"""
    original: str
    query_type: QueryType
    tokens: List[str]
    operators: List[str]
    metadata: Dict[str, Any]
    fts_query: str  # Escaped FTS5 query
    regex_pattern: Optional[re.Pattern] = None


class EnterpriseQueryParser:
    """Advanced query parser for enterprise-scale search"""
    
    # FTS5 special characters that need escaping
    FTS5_SPECIAL_CHARS = [
        '"', "'", '(', ')', '[', ']', '{', '}',
        '*', '?', '\\', '/', '-', '+', '=', 
        '<', '>', '!', '@', '#', '$', '%', '^', '&',
        '|', '~', '`', ';', ':', ',', '.'
    ]
    
    # Boolean operators
    BOOLEAN_OPS = {
        'AND': ' AND ',
        '&&': ' AND ',
        'OR': ' OR ', 
        '||': ' OR ',
        'NOT': ' NOT ',
        '!': ' NOT ',
        '-': ' NOT '  # Minus as NOT
    }
    
    def __init__(self):
        self.query_cache = {}
    
    def parse(self, query: str) -> ParsedQuery:
        """Parse a query string into structured format"""
        
        # Check cache
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Parse based on type
        if query_type == QueryType.REGEX:
            result = self._parse_regex(query)
        elif query_type == QueryType.BOOLEAN:
            result = self._parse_boolean(query)
        elif query_type == QueryType.PROXIMITY:
            result = self._parse_proximity(query)
        elif query_type == QueryType.WILDCARD:
            result = self._parse_wildcard(query)
        elif query_type == QueryType.CROSS_LINE:
            result = self._parse_cross_line(query)
        elif query_type == QueryType.PHRASE:
            result = self._parse_phrase(query)
        else:
            result = self._parse_exact(query)
        
        # Cache result
        self.query_cache[query] = result
        return result
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        
        # Check for regex patterns
        if query.startswith('/') and query.endswith('/'):
            return QueryType.REGEX
        
        # Check for boolean operators
        for op in ['AND', 'OR', 'NOT', '&&', '||', '!']:
            if f' {op} ' in query:
                return QueryType.BOOLEAN
        
        # Check for proximity search (e.g., "word1 NEAR/5 word2")
        if 'NEAR/' in query.upper():
            return QueryType.PROXIMITY
        
        # Check for wildcards
        if '*' in query or '?' in query:
            return QueryType.WILDCARD
        
        # Check for cross-line patterns (contains \n or multiline indicators)
        if '\\n' in query or '\n' in query or '...' in query:
            return QueryType.CROSS_LINE
        
        # Check for phrase (quoted)
        if (query.startswith('"') and query.endswith('"')) or \
           (query.startswith("'") and query.endswith("'")):
            return QueryType.PHRASE
        
        return QueryType.EXACT
    
    def _escape_fts_query(self, text: str) -> str:
        """Escape special characters for FTS5"""
        escaped = text
        
        # Escape each special character
        for char in self.FTS5_SPECIAL_CHARS:
            # Don't escape wildcards if in wildcard mode
            if char in ['*', '?'] and self._detect_query_type(text) == QueryType.WILDCARD:
                continue
            escaped = escaped.replace(char, '\\' + char)
        
        # Wrap in quotes for exact matching (unless it's a boolean query)
        if ' AND ' not in escaped and ' OR ' not in escaped and ' NOT ' not in escaped:
            escaped = f'"{escaped}"'
        
        return escaped
    
    def _parse_exact(self, query: str) -> ParsedQuery:
        """Parse exact match query"""
        escaped = self._escape_fts_query(query)
        
        return ParsedQuery(
            original=query,
            query_type=QueryType.EXACT,
            tokens=[query],
            operators=[],
            metadata={},
            fts_query=escaped
        )
    
    def _parse_boolean(self, query: str) -> ParsedQuery:
        """Parse boolean query with AND/OR/NOT"""
        tokens = []
        operators = []
        fts_parts = []
        
        # Split by boolean operators while preserving them
        pattern = r'\s+(AND|OR|NOT|&&|\|\||!|-)\s+'
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Token
                if part.strip():
                    tokens.append(part.strip())
                    fts_parts.append(self._escape_fts_query(part.strip()))
            else:  # Operator
                op_upper = part.upper()
                if op_upper in self.BOOLEAN_OPS:
                    operators.append(op_upper)
                    fts_parts.append(self.BOOLEAN_OPS[op_upper])
        
        return ParsedQuery(
            original=query,
            query_type=QueryType.BOOLEAN,
            tokens=tokens,
            operators=operators,
            metadata={'operator_count': len(operators)},
            fts_query=''.join(fts_parts)
        )
    
    def _parse_regex(self, query: str) -> ParsedQuery:
        """Parse regex query"""
        # Remove regex delimiters
        pattern_str = query.strip('/')
        
        try:
            regex_pattern = re.compile(pattern_str, re.MULTILINE | re.DOTALL)
        except re.error as e:
            # Invalid regex, treat as exact match
            return self._parse_exact(query)
        
        # For FTS, extract literal strings from regex if possible
        literals = self._extract_literals_from_regex(pattern_str)
        fts_query = ' OR '.join([self._escape_fts_query(lit) for lit in literals]) if literals else '""'
        
        return ParsedQuery(
            original=query,
            query_type=QueryType.REGEX,
            tokens=literals,
            operators=[],
            metadata={'pattern': pattern_str},
            fts_query=fts_query,
            regex_pattern=regex_pattern
        )
    
    def _parse_proximity(self, query: str) -> ParsedQuery:
        """Parse proximity search (e.g., 'error NEAR/5 handling')"""
        # Pattern: word1 NEAR/N word2
        match = re.search(r'(\w+)\s+NEAR/(\d+)\s+(\w+)', query, re.IGNORECASE)
        
        if match:
            word1, distance, word2 = match.groups()
            
            # FTS5 doesn't directly support NEAR, so we use a workaround
            fts_query = f'"{word1}" AND "{word2}"'
            
            return ParsedQuery(
                original=query,
                query_type=QueryType.PROXIMITY,
                tokens=[word1, word2],
                operators=['NEAR'],
                metadata={'distance': int(distance)},
                fts_query=fts_query
            )
        
        return self._parse_exact(query)
    
    def _parse_wildcard(self, query: str) -> ParsedQuery:
        """Parse wildcard query (* and ?)"""
        # Convert ? to single char wildcard in FTS5
        fts_query = query.replace('?', '_')
        
        # Don't escape * for wildcards
        for char in self.FTS5_SPECIAL_CHARS:
            if char not in ['*', '?']:
                fts_query = fts_query.replace(char, '\\' + char)
        
        # Extract tokens (parts without wildcards)
        tokens = [part for part in re.split(r'[*?]+', query) if part]
        
        return ParsedQuery(
            original=query,
            query_type=QueryType.WILDCARD,
            tokens=tokens,
            operators=['*', '?'],
            metadata={'pattern': query},
            fts_query=fts_query
        )
    
    def _parse_cross_line(self, query: str) -> ParsedQuery:
        """Parse patterns that span multiple lines"""
        # Replace literal \n with spaces for FTS
        normalized = query.replace('\\n', ' ').replace('\n', ' ')
        
        # Replace ... with wildcard
        if '...' in normalized:
            normalized = normalized.replace('...', '*')
            return self._parse_wildcard(normalized)
        
        return self._parse_exact(normalized)
    
    def _parse_phrase(self, query: str) -> ParsedQuery:
        """Parse exact phrase query"""
        # Remove quotes
        phrase = query.strip('"\'')
        
        # Escape and keep as phrase
        escaped = self._escape_fts_query(phrase)
        
        return ParsedQuery(
            original=query,
            query_type=QueryType.PHRASE,
            tokens=[phrase],
            operators=[],
            metadata={'phrase_length': len(phrase.split())},
            fts_query=escaped
        )
    
    def _extract_literals_from_regex(self, pattern: str) -> List[str]:
        """Extract literal strings from regex pattern"""
        literals = []
        
        # Find literal sequences (alphanumeric + underscore)
        for match in re.finditer(r'[a-zA-Z_]\w+', pattern):
            literal = match.group()
            if len(literal) > 2:  # Only meaningful literals
                literals.append(literal)
        
        return literals
    
    def build_complex_query(self, conditions: List[Tuple[str, str]]) -> ParsedQuery:
        """Build complex query from multiple conditions
        
        Args:
            conditions: List of (operator, query) tuples
            Example: [('', 'error'), ('AND', 'handling'), ('NOT', 'test')]
        """
        if not conditions:
            return self._parse_exact("")
        
        query_parts = []
        fts_parts = []
        all_tokens = []
        all_operators = []
        
        for operator, query_str in conditions:
            parsed = self.parse(query_str)
            all_tokens.extend(parsed.tokens)
            
            if operator:
                all_operators.append(operator)
                query_parts.append(f" {operator} ")
                fts_parts.append(self.BOOLEAN_OPS.get(operator, f" {operator} "))
            
            query_parts.append(query_str)
            fts_parts.append(parsed.fts_query)
        
        return ParsedQuery(
            original=''.join(query_parts),
            query_type=QueryType.BOOLEAN,
            tokens=all_tokens,
            operators=all_operators,
            metadata={'complex': True, 'condition_count': len(conditions)},
            fts_query=''.join(fts_parts)
        )


def create_enterprise_query_parser() -> EnterpriseQueryParser:
    """Factory function to create query parser"""
    return EnterpriseQueryParser()