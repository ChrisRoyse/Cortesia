#!/usr/bin/env python3
"""
Enterprise Batch Processor - Parallel Processing for Scale
==========================================================

Features:
- Parallel file processing using multiprocessing
- Batch chunking for memory efficiency
- Progress tracking with Redis
- Incremental indexing with change detection
- Large file handling with memory mapping
- Binary file detection and skipping

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import os
import hashlib
import mmap
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import json
import pickle

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available, using in-memory cache")


@dataclass
class FileMetadata:
    """Metadata for tracking file changes"""
    path: str
    size: int
    modified_time: float
    content_hash: str
    is_binary: bool
    encoding: str
    chunk_count: int = 0
    index_time: float = 0.0
    

@dataclass 
class BatchResult:
    """Result from batch processing"""
    files_processed: int
    chunks_created: int
    errors: List[str]
    processing_time: float
    files_skipped: int
    

class RedisProgressTracker:
    """Track indexing progress in Redis for crash recovery"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                self.redis_client.ping()
                self.enabled = True
            except:
                print("Warning: Could not connect to Redis, using fallback")
                self.enabled = False
        else:
            self.enabled = False
    
    def set_progress(self, key: str, value: Any):
        """Set progress value"""
        if self.enabled:
            try:
                self.redis_client.set(key, json.dumps(value) if not isinstance(value, str) else value)
            except:
                pass
    
    def get_progress(self, key: str) -> Optional[Any]:
        """Get progress value"""
        if self.enabled:
            try:
                value = self.redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except:
                        return value
            except:
                pass
        return None
    
    def increment_counter(self, key: str, amount: int = 1):
        """Increment a counter"""
        if self.enabled:
            try:
                self.redis_client.incrby(key, amount)
            except:
                pass


class EnterpriseBatchProcessor:
    """High-performance batch processor for enterprise-scale codebases"""
    
    # File size limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    LARGE_FILE_THRESHOLD = 1 * 1024 * 1024  # 1MB
    
    # Batch settings
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_WORKER_COUNT = max(4, cpu_count() - 2)
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 use_redis: bool = True,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        # Cache directory for file metadata
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'enterprise_indexer'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # File metadata cache
        self.metadata_cache_file = self.cache_dir / 'file_metadata.pkl'
        self.file_metadata = self._load_metadata_cache()
        
        # Progress tracking
        self.progress = RedisProgressTracker(redis_host, redis_port) if use_redis else None
        
        # Binary file detection patterns
        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.zip', '.tar', '.gz', '.7z', '.rar',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
            '.pyc', '.pyo', '.class', '.jar',
            '.o', '.a', '.lib', '.pdb'
        }
    
    def _load_metadata_cache(self) -> Dict[str, FileMetadata]:
        """Load cached file metadata"""
        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def _save_metadata_cache(self):
        """Save file metadata cache"""
        try:
            with open(self.metadata_cache_file, 'wb') as f:
                pickle.dump(self.file_metadata, f)
        except:
            pass
    
    def is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        # Check extension first
        if file_path.suffix.lower() in self.binary_extensions:
            return True
        
        # Check content
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                if b'\0' in chunk:  # Null byte = binary
                    return True
                
                # Check ratio of non-text bytes
                text_chars = bytes(range(32, 127)) + b'\n\r\t\f\b'
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                
                if len(chunk) > 0:
                    ratio = non_text / len(chunk)
                    return ratio > 0.30  # More than 30% non-text = binary
        except:
            return True
        
        return False
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content (efficiently)"""
        hash_md5 = hashlib.md5()
        
        # For large files, use memory mapping
        if file_path.stat().st_size > self.LARGE_FILE_THRESHOLD:
            try:
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmf:
                        # Hash in chunks
                        for i in range(0, len(mmf), 8192):
                            hash_md5.update(mmf[i:i+8192])
            except:
                # Fallback to regular reading
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hash_md5.update(chunk)
        else:
            with open(file_path, 'rb') as f:
                hash_md5.update(f.read())
        
        return hash_md5.hexdigest()
    
    def needs_reindexing(self, file_path: Path) -> bool:
        """Check if file needs reindexing based on changes"""
        path_str = str(file_path)
        
        # Not in cache = needs indexing
        if path_str not in self.file_metadata:
            return True
        
        cached = self.file_metadata[path_str]
        stat = file_path.stat()
        
        # Check if file changed
        if cached.size != stat.st_size:
            return True
        
        if cached.modified_time != stat.st_mtime:
            # Size same but modified time different - check hash
            current_hash = self.get_file_hash(file_path)
            if cached.content_hash != current_hash:
                return True
        
        return False
    
    def discover_files(self, 
                      root_path: Path,
                      patterns: List[str] = None,
                      exclude_dirs: List[str] = None) -> Generator[Path, None, None]:
        """Discover files to process"""
        
        exclude_dirs = exclude_dirs or [
            '.git', '__pycache__', 'node_modules', 
            'target', 'build', 'dist', '.pytest_cache',
            'venv', '.venv', 'env', '.env'
        ]
        
        patterns = patterns or [
            '*.py', '*.rs', '*.js', '*.ts', '*.jsx', '*.tsx',
            '*.java', '*.cpp', '*.c', '*.h', '*.hpp',
            '*.go', '*.rb', '*.php', '*.cs', '*.swift',
            '*.md', '*.txt', '*.toml', '*.json', '*.yaml', '*.yml'
        ]
        
        # Track directories to avoid circular symlinks
        visited_dirs = set()
        
        for pattern in patterns:
            for file_path in root_path.rglob(pattern):
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                
                # Skip symlinks that might be circular
                if file_path.is_symlink():
                    real_path = file_path.resolve()
                    if real_path in visited_dirs:
                        continue
                    visited_dirs.add(real_path)
                
                # Skip if file too large
                try:
                    if file_path.stat().st_size > self.MAX_FILE_SIZE:
                        continue
                except:
                    continue
                
                # Skip binary files
                if self.is_binary_file(file_path):
                    continue
                
                yield file_path
    
    def process_file_batch(self, 
                          files: List[Path],
                          processor_func,
                          batch_id: int = 0) -> BatchResult:
        """Process a batch of files"""
        
        processed = 0
        chunks_created = 0
        errors = []
        skipped = 0
        start_time = time.time()
        
        for file_path in files:
            try:
                # Check if needs reindexing
                if not self.needs_reindexing(file_path):
                    skipped += 1
                    continue
                
                # Process file
                result = processor_func(file_path)
                
                if result:
                    processed += 1
                    chunks_created += result.get('chunks', 0)
                    
                    # Update metadata cache
                    stat = file_path.stat()
                    self.file_metadata[str(file_path)] = FileMetadata(
                        path=str(file_path),
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                        content_hash=self.get_file_hash(file_path),
                        is_binary=False,
                        encoding=result.get('encoding', 'utf-8'),
                        chunk_count=result.get('chunks', 0),
                        index_time=time.time()
                    )
                    
                    # Update progress
                    if self.progress:
                        self.progress.increment_counter('files_processed')
                        self.progress.increment_counter('chunks_created', result.get('chunks', 0))
                    
            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")
        
        return BatchResult(
            files_processed=processed,
            chunks_created=chunks_created,
            errors=errors,
            processing_time=time.time() - start_time,
            files_skipped=skipped
        )
    
    def process_parallel(self,
                        root_path: Path,
                        processor_func,
                        patterns: List[str] = None,
                        batch_size: int = None,
                        max_workers: int = None) -> Dict[str, Any]:
        """Process files in parallel batches"""
        
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        max_workers = max_workers or self.DEFAULT_WORKER_COUNT
        
        print(f"Starting parallel processing with {max_workers} workers")
        
        # Reset progress tracking
        if self.progress:
            self.progress.set_progress('status', 'discovering_files')
            self.progress.set_progress('files_processed', 0)
            self.progress.set_progress('chunks_created', 0)
        
        # Discover all files
        all_files = list(self.discover_files(root_path, patterns))
        total_files = len(all_files)
        
        print(f"Found {total_files} files to process")
        
        if self.progress:
            self.progress.set_progress('total_files', total_files)
            self.progress.set_progress('status', 'processing')
        
        # Split into batches
        batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]
        
        # Process batches in parallel
        total_processed = 0
        total_chunks = 0
        total_errors = []
        total_skipped = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_file_batch, batch, processor_func, i): i
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result()
                    total_processed += result.files_processed
                    total_chunks += result.chunks_created
                    total_errors.extend(result.errors)
                    total_skipped += result.files_skipped
                    
                    print(f"Batch {batch_id} complete: {result.files_processed} files, "
                          f"{result.chunks_created} chunks, {result.files_skipped} skipped")
                    
                except Exception as e:
                    total_errors.append(f"Batch {batch_id} failed: {e}")
        
        # Save metadata cache
        self._save_metadata_cache()
        
        # Final stats
        total_time = time.time() - start_time
        
        if self.progress:
            self.progress.set_progress('status', 'complete')
            self.progress.set_progress('completion_time', total_time)
        
        return {
            'total_files': total_files,
            'files_processed': total_processed,
            'files_skipped': total_skipped,
            'chunks_created': total_chunks,
            'errors': total_errors,
            'processing_time': total_time,
            'files_per_second': total_processed / total_time if total_time > 0 else 0,
            'batches_processed': len(batches)
        }


def create_enterprise_batch_processor(**kwargs) -> EnterpriseBatchProcessor:
    """Factory function to create batch processor"""
    return EnterpriseBatchProcessor(**kwargs)