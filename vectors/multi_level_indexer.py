#!/usr/bin/env python3
"""
Multi-Level Indexing Architecture
=================================

Implements a three-tier indexing system to achieve 100% search accuracy:
1. Exact Index - Direct string matching for keywords, function names, etc.
2. Semantic Index - Vector embeddings for contextual similarity
3. Metadata Index - File types, languages, and structural information

This addresses the core issue where the current system only has semantic indexing,
missing exact match capability that's essential for code search.

Author: Claude (Sonnet 4)  
Date: 2025-08-04
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import chromadb
from chromadb.utils import embedding_functions
import sqlite3
import re
from datetime import datetime

from file_type_classifier import FileTypeClassifier, FileType, create_file_classifier
from cached_embedding_manager import get_cached_embedding_function


class IndexType(Enum):
    """Types of indexes in the multi-level system"""
    EXACT = "exact"
    SEMANTIC = "semantic" 
    METADATA = "metadata"


@dataclass
class IndexedDocument:
    """Document representation in the multi-level index"""
    doc_id: str
    file_path: str
    relative_path: str
    file_type: FileType
    language: Optional[str]
    content: str
    exact_tokens: Set[str]
    metadata: Dict[str, Any]
    chunk_type: str
    chunk_index: int
    parent_doc_id: Optional[str] = None


@dataclass
class SearchQuery:
    """Search query with type specification"""
    query: str
    query_type: IndexType
    file_types: Optional[List[FileType]] = None
    languages: Optional[List[str]] = None
    limit: int = 20


@dataclass
class SearchResult:
    """Result from multi-level search"""
    doc_id: str
    file_path: str
    relative_path: str
    content: str
    score: float
    match_type: IndexType
    file_type: FileType
    language: Optional[str]
    metadata: Dict[str, Any]


class ExactIndexManager:
    """Manages exact string matching index using SQLite"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path / "exact_index.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for exact matching"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            # Create exact tokens table with FTS5 for fast text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS exact_tokens USING fts5(
                    doc_id,
                    token,
                    file_path,
                    relative_path,
                    file_type,
                    language,
                    content,
                    metadata,
                    tokenize='porter'
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def add_document(self, doc: IndexedDocument):
        """Add document to exact index"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            # Extract exact tokens from content
            tokens = self._extract_exact_tokens(doc.content, doc.file_type)
            
            # Insert each token as a separate record for efficient searching
            for token in tokens:
                cursor.execute("""
                    INSERT INTO exact_tokens 
                    (doc_id, token, file_path, relative_path, file_type, language, content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.doc_id,
                    token,
                    doc.file_path,
                    doc.relative_path,
                    doc.file_type.value,
                    doc.language,
                    doc.content,
                    json.dumps(doc.metadata)
                ))
            
            conn.commit()
        finally:
            conn.close()
    
    def _extract_exact_tokens(self, content: str, file_type: FileType) -> Set[str]:
        """Extract exact tokens based on file type"""
        tokens = set()
        
        if file_type == FileType.CODE:
            # Extract code-specific patterns
            # Function definitions
            tokens.update(re.findall(r'\b(?:pub\s+)?fn\s+(\w+)', content))
            tokens.update(re.findall(r'\bfn\s+(\w+)', content))
            
            # Struct/enum/trait names
            tokens.update(re.findall(r'\b(?:pub\s+)?struct\s+(\w+)', content))
            tokens.update(re.findall(r'\b(?:pub\s+)?enum\s+(\w+)', content))
            tokens.update(re.findall(r'\b(?:pub\s+)?trait\s+(\w+)', content))
            
            # Variable and type names
            tokens.update(re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', content))  # CamelCase
            tokens.update(re.findall(r'\b[a-z_][a-z0-9_]*\b', content))    # snake_case
            
            # Keywords and operators
            rust_keywords = {
                'pub', 'fn', 'struct', 'enum', 'trait', 'impl', 'use', 'mod',
                'let', 'mut', 'const', 'static', 'if', 'else', 'match', 'for',
                'while', 'loop', 'return', 'break', 'continue', 'true', 'false'
            }
            tokens.update(word for word in rust_keywords if word in content)
            
            # Common patterns
            tokens.update(re.findall(r'Result<[^>]+>', content))
            tokens.update(re.findall(r'Vec<[^>]+>', content))
            tokens.update(re.findall(r'Option<[^>]+>', content))
        
        # Common exact patterns for all file types
        tokens.update(re.findall(r'\b\w{3,}\b', content))  # Words 3+ chars
        
        # Remove very common words that aren't useful for search
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        tokens = {token for token in tokens if token.lower() not in stop_words and len(token) >= 2}
        
        return tokens
    
    def search(self, query: str, file_types: Optional[List[FileType]] = None, 
               languages: Optional[List[str]] = None, limit: int = 50) -> List[SearchResult]:
        """Search exact index"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = ["exact_tokens MATCH ?"]
            params = [query]
            
            if file_types:
                type_placeholders = ','.join('?' * len(file_types))
                conditions.append(f"file_type IN ({type_placeholders})")
                params.extend(file_type.value for file_type in file_types)
            
            if languages:
                lang_placeholders = ','.join('?' * len(languages))
                conditions.append(f"language IN ({lang_placeholders})")
                params.extend(languages)
            
            where_clause = " AND ".join(conditions)
            params.append(limit)
            
            cursor.execute(f"""
                SELECT DISTINCT doc_id, file_path, relative_path, file_type, language, content, metadata,
                       rank
                FROM exact_tokens
                WHERE {where_clause}
                ORDER BY rank
                LIMIT ?
            """, params)
            
            results = []
            for row in cursor.fetchall():
                doc_id, file_path, relative_path, file_type, language, content, metadata_json, rank = row
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    file_path=file_path,
                    relative_path=relative_path,
                    content=content,
                    score=1.0 / (rank + 1),  # Convert rank to score
                    match_type=IndexType.EXACT,
                    file_type=FileType(file_type),
                    language=language,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                ))
            
            return results
        finally:
            conn.close()
    
    def clear(self):
        """Clear the exact index"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM exact_tokens")
            conn.commit()
        finally:
            conn.close()


class MetadataIndexManager:
    """Manages metadata-based filtering and search"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path / "metadata_index.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize metadata database"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata_index (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    language TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP,
                    chunk_type TEXT,
                    chunk_index INTEGER,
                    parent_doc_id TEXT,
                    metadata_json TEXT
                )
            """)
            
            # Create indexes for fast filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON metadata_index(file_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON metadata_index(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relative_path ON metadata_index(relative_path)")
            
            conn.commit()
        finally:
            conn.close()
    
    def add_document(self, doc: IndexedDocument):
        """Add document metadata to index"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO metadata_index 
                (doc_id, file_path, relative_path, file_type, language, file_size,
                 created_at, modified_at, chunk_type, chunk_index, parent_doc_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id,
                doc.file_path,
                doc.relative_path,
                doc.file_type.value,
                doc.language,
                doc.metadata.get('size', 0),
                datetime.now().isoformat(),
                doc.metadata.get('modified_at'),
                doc.chunk_type,
                doc.chunk_index,
                doc.parent_doc_id,
                json.dumps(doc.metadata)
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    def filter_documents(self, file_types: Optional[List[FileType]] = None,
                        languages: Optional[List[str]] = None,
                        path_pattern: Optional[str] = None) -> List[str]:
        """Filter documents by metadata and return doc_ids"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if file_types:
                type_placeholders = ','.join('?' * len(file_types))
                conditions.append(f"file_type IN ({type_placeholders})")
                params.extend(file_type.value for file_type in file_types)
            
            if languages:
                lang_placeholders = ','.join('?' * len(languages))
                conditions.append(f"language IN ({lang_placeholders})")
                params.extend(languages)
            
            if path_pattern:
                conditions.append("relative_path LIKE ?")
                params.append(f"%{path_pattern}%")
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            cursor.execute(f"""
                SELECT doc_id FROM metadata_index {where_clause}
            """, params)
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            
            # Total documents
            cursor.execute("SELECT COUNT(*) FROM metadata_index")
            total_docs = cursor.fetchone()[0]
            
            # By file type
            cursor.execute("""
                SELECT file_type, COUNT(*) 
                FROM metadata_index 
                GROUP BY file_type
            """)
            by_file_type = dict(cursor.fetchall())
            
            # By language
            cursor.execute("""
                SELECT language, COUNT(*) 
                FROM metadata_index 
                WHERE language IS NOT NULL
                GROUP BY language
            """)
            by_language = dict(cursor.fetchall())
            
            return {
                'total_documents': total_docs,
                'by_file_type': by_file_type,
                'by_language': by_language
            }
        finally:
            conn.close()
    
    def clear(self):
        """Clear metadata index"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metadata_index")
            conn.commit()
        finally:
            conn.close()


class MultiLevelIndexer:
    """
    Multi-level indexing system combining exact, semantic, and metadata indexes
    for achieving 100% search accuracy on dynamic codebases.
    """
    
    def __init__(self, db_path: str = "./multi_level_index"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file type classifier
        self.file_classifier = create_file_classifier()
        
        # Initialize index managers
        self.exact_index = ExactIndexManager(self.db_path)
        self.metadata_index = MetadataIndexManager(self.db_path)
        
        # Initialize semantic index (ChromaDB)
        self.semantic_client = chromadb.PersistentClient(path=str(self.db_path / "semantic"))
        # Use the cached embedding function - loads model only ONCE across all instances
        self.embedding_function = get_cached_embedding_function()
        
        # Get or create semantic collection
        try:
            self.semantic_collection = self.semantic_client.get_collection(
                name="semantic_index",
                embedding_function=self.embedding_function
            )
        except:
            self.semantic_collection = self.semantic_client.create_collection(
                name="semantic_index",
                embedding_function=self.embedding_function
            )
    
    def add_document(self, file_path: Path, content: str, chunk_type: str = "full", 
                    chunk_index: int = 0, parent_doc_id: Optional[str] = None) -> str:
        """
        Add document to all three indexes
        
        Args:
            file_path: Path to the file
            content: File content or chunk content
            chunk_type: Type of chunk (full, function, section, etc.)
            chunk_index: Index of chunk within file
            parent_doc_id: Parent document ID for chunks
            
        Returns:
            Document ID
        """
        # Classify file type
        classification = self.file_classifier.classify_file(file_path)
        
        # Generate document ID
        doc_id = self._generate_doc_id(file_path, chunk_index)
        
        # Create document object
        relative_path = str(file_path.relative_to(file_path.anchor)) if file_path.is_absolute() else str(file_path)
        
        doc = IndexedDocument(
            doc_id=doc_id,
            file_path=str(file_path),
            relative_path=relative_path,
            file_type=classification.file_type,
            language=classification.language,
            content=content,
            exact_tokens=set(),  # Will be filled by exact index
            metadata=classification.metadata,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            parent_doc_id=parent_doc_id
        )
        
        # Add to all indexes
        self.exact_index.add_document(doc)
        self.metadata_index.add_document(doc)
        
        # Add to semantic index
        self.semantic_collection.add(
            documents=[content],
            metadatas=[{
                'doc_id': doc_id,
                'file_path': str(file_path),
                'relative_path': relative_path,
                'file_type': classification.file_type.value,
                'language': classification.language,
                'chunk_type': chunk_type,
                'chunk_index': chunk_index
            }],
            ids=[doc_id]
        )
        
        return doc_id
    
    def _generate_doc_id(self, file_path: Path, chunk_index: int) -> str:
        """Generate unique document ID"""
        path_str = str(file_path)
        content_hash = hashlib.md5(f"{path_str}_{chunk_index}".encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{chunk_index}"
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Multi-level search combining all three indexes
        
        Args:
            query: Search query with type specification
            
        Returns:
            Combined and ranked search results
        """
        all_results = []
        
        if query.query_type == IndexType.EXACT:
            # Use exact index
            results = self.exact_index.search(
                query.query, 
                query.file_types, 
                query.languages, 
                query.limit
            )
            all_results.extend(results)
            
        elif query.query_type == IndexType.SEMANTIC:
            # Use semantic index
            results = self._search_semantic(query)
            all_results.extend(results)
            
        elif query.query_type == IndexType.METADATA:
            # Use metadata filtering + semantic search
            filtered_doc_ids = self.metadata_index.filter_documents(
                query.file_types, 
                query.languages
            )
            if filtered_doc_ids:
                results = self._search_semantic_filtered(query, filtered_doc_ids)
                all_results.extend(results)
        else:
            # Hybrid search - combine all indexes
            # 1. Exact matches (highest priority)
            exact_results = self.exact_index.search(
                query.query, 
                query.file_types, 
                query.languages, 
                min(query.limit // 2, 10)
            )
            
            # 2. Semantic matches
            semantic_results = self._search_semantic(query)
            
            # Combine and deduplicate
            seen_doc_ids = set()
            for result in exact_results:
                if result.doc_id not in seen_doc_ids:
                    all_results.append(result)
                    seen_doc_ids.add(result.doc_id)
            
            for result in semantic_results:
                if result.doc_id not in seen_doc_ids and len(all_results) < query.limit:
                    all_results.append(result)
                    seen_doc_ids.add(result.doc_id)
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:query.limit]
    
    def _search_semantic(self, query: SearchQuery) -> List[SearchResult]:
        """Search semantic index"""
        try:
            # Build where clause for metadata filtering
            where_clause = {}
            if query.file_types:
                where_clause["file_type"] = {"$in": [ft.value for ft in query.file_types]}
            if query.languages:
                where_clause["language"] = {"$in": query.languages}
            
            results = self.semantic_collection.query(
                query_texts=[query.query],
                n_results=query.limit,
                where=where_clause if where_clause else None
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    score = max(0.0, 1.0 - distance)
                    
                    search_results.append(SearchResult(
                        doc_id=metadata['doc_id'],
                        file_path=metadata['file_path'],
                        relative_path=metadata['relative_path'],
                        content=doc,
                        score=score,
                        match_type=IndexType.SEMANTIC,
                        file_type=FileType(metadata['file_type']),
                        language=metadata.get('language'),
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _search_semantic_filtered(self, query: SearchQuery, doc_ids: List[str]) -> List[SearchResult]:
        """Search semantic index with pre-filtered document IDs"""
        # For now, fall back to regular semantic search with metadata filters
        return self._search_semantic(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        metadata_stats = self.metadata_index.get_statistics()
        semantic_count = self.semantic_collection.count()
        
        return {
            'metadata_index': metadata_stats,
            'semantic_index': {'total_documents': semantic_count},
            'exact_index': {'status': 'active'},  # SQLite FTS doesn't provide easy count
            'index_path': str(self.db_path)
        }
    
    def clear_all_indexes(self):
        """Clear all indexes"""
        self.exact_index.clear()
        self.metadata_index.clear()
        
        # Clear semantic collection
        try:
            self.semantic_client.delete_collection("semantic_index")
            self.semantic_collection = self.semantic_client.create_collection(
                name="semantic_index",
                embedding_function=self.embedding_function
            )
        except:
            pass


# Factory function
def create_multi_level_indexer(db_path: str = "./multi_level_index") -> MultiLevelIndexer:
    """Create a new multi-level indexer instance"""
    return MultiLevelIndexer(db_path)


if __name__ == "__main__":
    # Demo usage
    indexer = create_multi_level_indexer()
    
    print("Multi-Level Indexing System Demo")
    print("=" * 50)
    
    # Show statistics
    stats = indexer.get_statistics()
    print(f"Current index statistics:")
    print(f"  Metadata documents: {stats['metadata_index']['total_documents']}")
    print(f"  Semantic documents: {stats['semantic_index']['total_documents']}")
    print(f"  Index path: {stats['index_path']}")