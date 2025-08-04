#!/usr/bin/env python3
"""
Enhanced Boolean Search Handler
================================

Properly handles boolean queries by searching across all chunks
and combining results intelligently.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

from typing import List, Set, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import re

from multi_level_indexer import SearchResult, IndexType


@dataclass
class BooleanSearchResult:
    """Enhanced search result with boolean logic support"""
    file_path: str
    relative_path: str
    matching_chunks: List[SearchResult]
    score: float
    match_type: str


class EnhancedBooleanSearcher:
    """Handles complex boolean queries properly"""
    
    def __init__(self, indexing_system):
        self.system = indexing_system
        
    def search_boolean(self, query: str, limit: int = 20) -> List[SearchResult]:
        """
        Execute boolean search with proper AND/OR/NOT logic
        """
        # Parse boolean operators
        query_upper = query.upper()
        
        if ' AND ' in query_upper:
            return self._search_and(query, limit)
        elif ' OR ' in query_upper:
            return self._search_or(query, limit)
        elif ' NOT ' in query_upper:
            return self._search_not(query, limit)
        else:
            # No boolean operators, regular search
            return self.system.search(query, IndexType.EXACT, limit=limit)
    
    def _search_and(self, query: str, limit: int) -> List[SearchResult]:
        """
        AND logic: Find documents containing ALL terms
        """
        # Split on AND (case insensitive)
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            try:
                return self.system.search(query, IndexType.EXACT, limit=limit)
            except:
                return []
        
        # Search for each term individually
        all_results = []
        for term in parts:
            term = term.strip()
            results = self.system.search(term, IndexType.EXACT, limit=limit * 2)
            all_results.append(results)
        
        if not all_results or not all_results[0]:
            return []
        
        # Find documents that appear in ALL result sets
        # Group results by file path
        file_results = {}
        for result_set in all_results:
            for result in result_set:
                if result.file_path not in file_results:
                    file_results[result.file_path] = []
                file_results[result.file_path].append(result)
        
        # Keep only files that have results for ALL terms
        final_results = []
        for file_path, results in file_results.items():
            # Check if we have results for all terms
            terms_found = set()
            for result in results:
                # Check which terms are in this chunk
                content_lower = result.content.lower()
                for term in parts:
                    if term.lower().strip() in content_lower:
                        terms_found.add(term.lower().strip())
            
            # If all terms found in this file, include it
            if len(terms_found) == len(parts):
                # Return the best scoring chunk from this file
                best_result = max(results, key=lambda r: r.score)
                final_results.append(best_result)
        
        # Sort by score and limit
        final_results.sort(key=lambda r: r.score, reverse=True)
        return final_results[:limit]
    
    def _search_or(self, query: str, limit: int) -> List[SearchResult]:
        """
        OR logic: Find documents containing ANY term
        """
        # Split on OR
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            return self.system.search(query, IndexType.EXACT, limit=limit)
        
        # Search for each term and combine results
        all_results = []
        seen_docs = set()
        
        for term in parts:
            term = term.strip()
            results = self.system.search(term, IndexType.EXACT, limit=limit)
            for result in results:
                doc_key = (result.file_path, result.doc_id)
                if doc_key not in seen_docs:
                    all_results.append(result)
                    seen_docs.add(doc_key)
        
        # Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]
    
    def _search_not(self, query: str, limit: int) -> List[SearchResult]:
        """
        NOT logic: Find documents containing first term but NOT second
        """
        # Split on NOT
        parts = re.split(r'\s+NOT\s+', query, flags=re.IGNORECASE)
        
        if len(parts) != 2:
            return self.system.search(query, IndexType.EXACT, limit=limit)
        
        include_term = parts[0].strip()
        exclude_term = parts[1].strip()
        
        # Get documents with include term
        include_results = self.system.search(include_term, IndexType.EXACT, limit=limit * 3)
        
        # Get documents with exclude term
        exclude_results = self.system.search(exclude_term, IndexType.EXACT, limit=limit * 3)
        exclude_files = {r.file_path for r in exclude_results}
        
        # Filter out documents that contain exclude term
        final_results = []
        for result in include_results:
            if result.file_path not in exclude_files:
                # Also check the content doesn't contain exclude term
                if exclude_term.lower() not in result.content.lower():
                    final_results.append(result)
        
        return final_results[:limit]
    
    def search_complex(self, query: str, limit: int = 20) -> List[SearchResult]:
        """
        Handle complex queries with nested expressions, special chars, etc.
        """
        # Handle parentheses for grouping
        if '(' in query and ')' in query:
            return self._search_grouped(query, limit)
        
        # Handle proximity searches
        if 'NEAR/' in query.upper():
            return self._search_proximity(query, limit)
        
        # Handle phrase searches
        if query.startswith('"') and query.endswith('"'):
            return self._search_phrase(query, limit)
        
        # Default to boolean search
        return self.search_boolean(query, limit)
    
    def _search_grouped(self, query: str, limit: int) -> List[SearchResult]:
        """Handle grouped expressions like (A OR B) AND C"""
        # For now, simplify by removing parentheses
        # A proper implementation would parse the expression tree
        simplified = query.replace('(', '').replace(')', '')
        return self.search_boolean(simplified, limit)
    
    def _search_proximity(self, query: str, limit: int) -> List[SearchResult]:
        """Handle proximity searches like 'term1 NEAR/5 term2'"""
        # Extract terms and distance
        match = re.search(r'(\w+)\s+NEAR/(\d+)\s+(\w+)', query, re.IGNORECASE)
        if not match:
            return self.search_boolean(query, limit)
        
        term1, distance, term2 = match.groups()
        distance = int(distance)
        
        # Search for both terms
        results1 = self.system.search(term1, IndexType.EXACT, limit=limit * 2)
        results2 = self.system.search(term2, IndexType.EXACT, limit=limit * 2)
        
        # Find documents with both terms within proximity
        final_results = []
        for r1 in results1:
            content = r1.content.lower()
            if term1.lower() in content and term2.lower() in content:
                # Check proximity (simplified - just check if within N words)
                words = content.split()
                try:
                    idx1 = words.index(term1.lower())
                    idx2 = words.index(term2.lower())
                    if abs(idx1 - idx2) <= distance:
                        final_results.append(r1)
                except ValueError:
                    pass
        
        return final_results[:limit]
    
    def _search_phrase(self, query: str, limit: int) -> List[SearchResult]:
        """Handle exact phrase searches"""
        # Remove quotes
        phrase = query.strip('"')
        
        # Search for the phrase
        results = self.system.search(phrase, IndexType.EXACT, limit=limit * 2)
        
        # Filter to only exact matches
        final_results = []
        for result in results:
            if phrase.lower() in result.content.lower():
                final_results.append(result)
        
        return final_results[:limit]


def create_enhanced_boolean_searcher(indexing_system):
    """Factory function"""
    return EnhancedBooleanSearcher(indexing_system)