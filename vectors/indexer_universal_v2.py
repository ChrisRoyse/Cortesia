#!/usr/bin/env python3
"""
Universal RAG Indexer V2 - With Git tracking and incremental updates
Production-ready indexing with change detection and cache invalidation
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
from fnmatch import fnmatch
from collections import defaultdict
import gc
import psutil
import yaml
import tomli

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import locale
    if locale.getpreferredencoding().upper() != 'UTF-8':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import click

# Import our enhanced modules
from git_tracker import GitChangeTracker, IncrementalIndexer
from cache_manager import CacheManager, QueryCacheIntegration

# We'll inherit from the original indexer and enhance it
import indexer_universal


class MemoryManager:
    """Monitor and enforce memory limits"""
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.process.memory_info().rss
        
    def check_memory_pressure(self) -> bool:
        """Check if approaching memory limit"""
        current = self.get_memory_usage()
        return current > self.max_memory_bytes * 0.8
        
    def force_cleanup(self):
        """Force garbage collection if needed"""
        if self.check_memory_pressure():
            gc.collect()
            return True
        return False
        
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        current = self.get_memory_usage()
        return {
            'current_mb': current / (1024 * 1024),
            'max_mb': self.max_memory_bytes / (1024 * 1024),
            'usage_percent': (current / self.max_memory_bytes) * 100
        }


class ChunkDeduplicator:
    """Handle chunk deduplication and similarity detection"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.chunk_hashes = {}
        self.chunk_embeddings = {}
        self.stats = {
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'unique_chunks': 0
        }
        
    def get_chunk_hash(self, content: str) -> str:
        """Get hash of chunk content"""
        # Normalize whitespace for better dedup
        normalized = ' '.join(content.split())
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def is_duplicate(self, content: str, embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str]]:
        """Check if chunk is duplicate or near-duplicate"""
        chunk_hash = self.get_chunk_hash(content)
        
        # Check exact duplicate
        if chunk_hash in self.chunk_hashes:
            self.stats['exact_duplicates'] += 1
            return True, self.chunk_hashes[chunk_hash]
            
        # Check near-duplicate using embeddings if provided
        if embedding is not None and len(self.chunk_embeddings) > 0:
            for existing_hash, existing_emb in self.chunk_embeddings.items():
                similarity = cosine_similarity([embedding], [existing_emb])[0][0]
                if similarity > self.similarity_threshold:
                    self.stats['near_duplicates'] += 1
                    return True, existing_hash
                    
        # Not a duplicate
        self.chunk_hashes[chunk_hash] = chunk_hash
        if embedding is not None:
            self.chunk_embeddings[chunk_hash] = embedding
        self.stats['unique_chunks'] += 1
        return False, None
        
    def reset(self):
        """Reset deduplication state"""
        self.chunk_hashes.clear()
        self.chunk_embeddings.clear()
        self.stats = {
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'unique_chunks': 0
        }


class EnhancedConfigParser:
    """Proper parsing for YAML, TOML, and XML config files"""
    
    def parse_yaml(self, content: str) -> List[Dict]:
        """Parse YAML configuration"""
        chunks = []
        try:
            data = yaml.safe_load(content)
            chunks.extend(self._extract_config_sections(data, 'yaml'))
        except:
            # Fallback to text chunking
            chunks.append({
                'content': content,
                'type': 'config_text',
                'metadata': {'config_type': 'yaml', 'parse_failed': True}
            })
        return chunks
        
    def parse_toml(self, content: str) -> List[Dict]:
        """Parse TOML configuration"""
        chunks = []
        try:
            data = tomli.loads(content)
            chunks.extend(self._extract_config_sections(data, 'toml'))
        except:
            # Fallback to text chunking
            chunks.append({
                'content': content,
                'type': 'config_text',
                'metadata': {'config_type': 'toml', 'parse_failed': True}
            })
        return chunks
        
    def _extract_config_sections(self, data: Any, config_type: str, path: str = '') -> List[Dict]:
        """Recursively extract config sections"""
        chunks = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                section_path = f"{path}.{key}" if path else key
                
                # Create chunk for this section
                if isinstance(value, (dict, list)):
                    # Serialize subsection
                    if config_type == 'yaml':
                        content = yaml.dump({key: value}, default_flow_style=False)
                    else:
                        content = json.dumps({key: value}, indent=2)
                else:
                    content = f"{key}: {value}"
                    
                chunks.append({
                    'content': content,
                    'type': 'config_section',
                    'metadata': {
                        'config_type': config_type,
                        'section_path': section_path,
                        'depth': path.count('.') + 1
                    }
                })
                
                # Recurse into nested structures
                if isinstance(value, dict) and len(value) > 3:
                    chunks.extend(self._extract_config_sections(value, config_type, section_path))
                    
        return chunks


class UniversalIndexerV2(indexer_universal.UniversalIndexer):
    """Enhanced indexer with incremental updates and better error handling"""
    
    def __init__(self,
                 root_dir: str = ".",
                 db_dir: str = "./chroma_db_universal",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 incremental: bool = True,
                 max_memory_gb: float = 2.0):
        """Initialize enhanced universal indexer"""
        self.root_dir = Path(root_dir).resolve()
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.incremental = incremental
        
        # Initialize components
        self.code_parser = UniversalCodeParser()
        self.gitignore_parser = GitignoreParser(self.root_dir / '.gitignore')
        self.memory_manager = MemoryManager(max_memory_gb)
        self.deduplicator = ChunkDeduplicator()
        self.config_parser = EnhancedConfigParser()
        
        # Git tracking for incremental updates
        if incremental:
            self.incremental_indexer = IncrementalIndexer(self.root_dir, self.db_dir)
            
        # Cache management
        self.cache_integration = None
        if self.db_dir.exists():
            self.cache_integration = QueryCacheIntegration(self.db_dir)
        
        # Enhanced stats
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "code_files": 0,
            "doc_files": 0,
            "config_files": 0,
            "languages": defaultdict(int),
            "chunk_types": defaultdict(int),
            "processing_time": 0,
            "errors": [],
            "files_processed": [],
            "files_failed": [],
            "incremental_stats": {},
            "memory_stats": {},
            "dedup_stats": {}
        }
        
    def process_file_with_recovery(self, file_path: Path, file_type: str) -> List[Document]:
        """Process file with error recovery"""
        documents = []
        
        try:
            # Check memory before processing
            if self.memory_manager.check_memory_pressure():
                self.memory_manager.force_cleanup()
                
            # Process based on type
            if file_type == 'code':
                documents = self.process_code_file(file_path)
            elif file_type == 'config':
                documents = self.process_config_file(file_path)
            else:
                documents = self.process_document_file(file_path)
                
            self.stats['files_processed'].append(str(file_path))
            
        except Exception as e:
            # Log error but continue processing
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats['errors'].append(error_msg)
            self.stats['files_failed'].append(str(file_path))
            print(f"  Warning: {error_msg}")
            
            # Try fallback processing
            try:
                documents = self.fallback_process(file_path)
            except:
                pass
                
        return documents
        
    def process_config_file(self, file_path: Path) -> List[Document]:
        """Enhanced config file processing"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            file_ext = file_path.suffix.lower()
            
            # Parse based on type
            if file_ext in ['.yaml', '.yml']:
                chunks = self.config_parser.parse_yaml(content)
            elif file_ext == '.toml':
                chunks = self.config_parser.parse_toml(content)
            elif file_ext == '.json':
                # Try JSON parsing
                try:
                    data = json.loads(content)
                    chunks = self.config_parser._extract_config_sections(data, 'json')
                except:
                    chunks = [{'content': content, 'type': 'config_text', 'metadata': {}}]
            else:
                chunks = [{'content': content, 'type': 'config_text', 'metadata': {}}]
                
            # Create documents with deduplication
            for i, chunk in enumerate(chunks):
                # Check for duplicates
                is_dup, dup_id = self.deduplicator.is_duplicate(chunk['content'])
                if not is_dup:
                    doc = Document(
                        page_content=chunk['content'],
                        metadata={
                            "source": str(file_path),
                            "relative_path": str(file_path.relative_to(self.root_dir)),
                            "file_type": file_ext[1:],
                            "chunk_type": chunk['type'],
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            **chunk.get('metadata', {})
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            self.stats['errors'].append(f"Config parsing error for {file_path}: {e}")
            
        return documents
        
    def fallback_process(self, file_path: Path) -> List[Document]:
        """Fallback processing for failed files"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple line-based chunking as fallback
            lines = content.split('\n')
            chunk_size = 50  # lines per chunk
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_content = '\n'.join(chunk_lines)
                
                if chunk_content.strip():
                    doc = Document(
                        page_content=chunk_content,
                        metadata={
                            "source": str(file_path),
                            "relative_path": str(file_path.relative_to(self.root_dir)),
                            "file_type": file_path.suffix[1:],
                            "chunk_type": "fallback",
                            "line_start": i,
                            "line_end": min(i + chunk_size, len(lines))
                        }
                    )
                    documents.append(doc)
                    
        except:
            pass
            
        return documents
        
    def run_incremental(self):
        """Run incremental indexing with git tracking"""
        print("=" * 60)
        print("INCREMENTAL INDEXING MODE")
        print("=" * 60)
        
        # Get supported extensions
        supported_extensions = {
            '.md', '.txt', '.rst', '.markdown',
            '.py', '.rs', '.js', '.jsx', '.ts', '.tsx', '.go', '.java', 
            '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.rb', '.php',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.xml'
        }
        
        # Check if full reindex needed
        if self.incremental_indexer.should_full_reindex():
            print("Full reindex required (no previous state or database)")
            return self.run_full()
            
        # Get incremental changes
        changes = self.incremental_indexer.get_incremental_changes(supported_extensions)
        
        print(f"Changes detected:")
        print(f"  Files to add: {len(changes['add'])}")
        print(f"  Files to update: {len(changes['update'])}")
        print(f"  Files to delete: {len(changes['delete'])}")
        
        if not any([changes['add'], changes['update'], changes['delete']]):
            print("No changes detected. Index is up to date.")
            return True
            
        # Process changes
        start_time = time.time()
        
        # Initialize embeddings
        self.initialize_embeddings()
        
        # Load existing database
        vector_db = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings
        )
        
        # Process new and updated files
        files_to_process = changes['add'] + changes['update']
        
        if files_to_process:
            # Delete old versions of updated files
            for file_path in changes['update']:
                rel_path = str(file_path.relative_to(self.root_dir))
                # Delete existing chunks for this file
                vector_db.delete(where={"relative_path": rel_path})
                
            # Process files
            all_documents = []
            for file_path in files_to_process:
                file_type = self._determine_file_type(file_path)
                docs = self.process_file_with_recovery(file_path, file_type)
                all_documents.extend(docs)
                
                # Batch add to avoid memory issues
                if len(all_documents) >= 100:
                    vector_db.add_documents(all_documents)
                    all_documents.clear()
                    self.memory_manager.force_cleanup()
                    
            # Add remaining documents
            if all_documents:
                vector_db.add_documents(all_documents)
                
        # Handle deletions
        for file_path in changes['delete']:
            rel_path = str(file_path.relative_to(self.root_dir))
            vector_db.delete(where={"relative_path": rel_path})
            
        # Mark as completed
        self.incremental_indexer.mark_completed(
            changes['add'] + changes['update'],
            changes['delete']
        )
        
        # Invalidate cache for changed files
        if self.cache_integration:
            self.cache_integration.invalidate_for_updates(files_to_process)
            
        # Update metadata
        self._update_metadata(incremental=True, changes=changes)
        
        processing_time = time.time() - start_time
        print(f"\nIncremental indexing completed in {processing_time:.2f} seconds")
        
        return True
        
    def run_full(self):
        """Run full indexing (original behavior with enhancements)"""
        # Original run method with error recovery
        return self.run()
        
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type category"""
        ext = file_path.suffix.lower()
        
        doc_extensions = {'.md', '.txt', '.rst', '.markdown'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.xml'}
        
        if ext in doc_extensions:
            return 'doc'
        elif ext in config_extensions:
            return 'config'
        else:
            return 'code'
            
    def _update_metadata(self, incremental: bool = False, changes: Dict = None):
        """Update index metadata"""
        metadata = {
            "version": "universal_2.0",
            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "incremental": incremental,
            "stats": dict(self.stats),
            "memory_stats": self.memory_manager.get_stats(),
            "dedup_stats": self.deduplicator.stats,
            "git_stats": changes.get('stats', {}) if changes else {}
        }
        
        metadata_path = self.db_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def cleanup(self):
        """Enhanced cleanup with memory stats"""
        memory_stats = self.memory_manager.get_stats()
        print(f"[OK] Memory usage: {memory_stats['current_mb']:.1f}MB / {memory_stats['max_mb']:.1f}MB")
        super().cleanup()


@click.command()
@click.option('--root-dir', '-r', default="..", help='Root directory to index')
@click.option('--db-dir', '-o', default="./chroma_db_universal", help='Output database directory')
@click.option('--model', '-m', default="sentence-transformers/all-MiniLM-L6-v2", help='Embedding model')
@click.option('--incremental/--full', default=True, help='Use incremental indexing with git tracking')
@click.option('--max-memory', default=2.0, help='Maximum memory usage in GB')
def main(root_dir: str, db_dir: str, model: str, incremental: bool, max_memory: float):
    """Universal RAG Indexer V2 - With incremental updates and git tracking"""
    
    # Change to vectors directory for database output
    vectors_dir = Path(".")
    if vectors_dir.name == "vectors":
        db_path = Path(db_dir)
    else:
        db_path = Path("vectors") / Path(db_dir).name if Path("vectors").exists() else Path(db_dir)
    
    indexer = UniversalIndexerV2(
        root_dir=root_dir,
        db_dir=str(db_path),
        model_name=model,
        incremental=incremental,
        max_memory_gb=max_memory
    )
    
    try:
        if incremental:
            success = indexer.run_incremental()
        else:
            success = indexer.run_full()
            
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()