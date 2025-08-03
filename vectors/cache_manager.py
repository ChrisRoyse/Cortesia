#!/usr/bin/env python3
"""
Cache Manager - Intelligent caching with TTL and version tracking
Handles cache invalidation based on database changes and time
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pickle


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    data: Any
    created_at: float
    ttl_seconds: int
    db_version: str
    query_hash: str
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl_seconds
        
    def is_valid(self, current_db_version: str) -> bool:
        """Check if cache entry is still valid"""
        return not self.is_expired() and self.db_version == current_db_version
        
    def access(self):
        """Record access to this cache entry"""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheManager:
    """Intelligent cache manager with TTL and invalidation"""
    
    def __init__(self, 
                 cache_dir: Path = None,
                 default_ttl: int = 3600,
                 max_cache_size: int = 1000,
                 persist_cache: bool = True):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to persist cache
            default_ttl: Default TTL in seconds (1 hour)
            max_cache_size: Maximum number of cache entries
            persist_cache: Whether to persist cache to disk
        """
        self.cache_dir = cache_dir or Path('.vectors/.cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.persist_cache = persist_cache
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.db_version = None
        self.db_metadata = {}
        
        # Stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        
        # Load persisted cache if exists
        if persist_cache:
            self._load_cache()
            
    def _load_cache(self):
        """Load persisted cache from disk"""
        cache_file = self.cache_dir / 'query_cache.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.stats = data.get('stats', self.stats)
                    # Clean expired entries on load
                    self._clean_expired()
            except:
                pass
                
    def _save_cache(self):
        """Persist cache to disk"""
        if not self.persist_cache:
            return
            
        cache_file = self.cache_dir / 'query_cache.pkl'
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'stats': self.stats,
                    'version': '1.0'
                }, f)
        except:
            pass
            
    def set_db_version(self, db_path: Path):
        """Set database version from metadata"""
        metadata_file = db_path / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.db_metadata = json.load(f)
                    # Create version hash from metadata
                    version_str = f"{self.db_metadata.get('indexed_at', '')}:{self.db_metadata.get('stats', {}).get('total_chunks', 0)}"
                    self.db_version = hashlib.md5(version_str.encode()).hexdigest()
            except:
                self.db_version = str(time.time())
        else:
            self.db_version = str(time.time())
            
    def generate_cache_key(self, 
                          query: str, 
                          k: int, 
                          filter_type: Optional[str] = None,
                          filter_language: Optional[str] = None,
                          **kwargs) -> str:
        """Generate unique cache key for query"""
        # Include all parameters that affect results
        key_parts = [
            query.lower(),
            str(k),
            filter_type or 'all',
            filter_language or 'any',
            str(kwargs.get('rerank', True)),
            str(kwargs.get('exact_match', False))
        ]
        key_str = ':'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(self, 
            key: str, 
            check_version: bool = True) -> Optional[Any]:
        """Get item from cache if valid"""
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
            
        entry = self.cache[key]
        
        # Check validity
        if check_version and not entry.is_valid(self.db_version):
            # Invalid entry, remove it
            del self.cache[key]
            self.stats['invalidations'] += 1
            self.stats['misses'] += 1
            return None
            
        # Valid hit
        entry.access()
        self.stats['hits'] += 1
        return entry.data
        
    def set(self, 
            key: str, 
            data: Any, 
            ttl: Optional[int] = None):
        """Set item in cache"""
        # Check cache size limit
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()
            
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=time.time(),
            ttl_seconds=ttl or self.default_ttl,
            db_version=self.db_version,
            query_hash=key
        )
        
        self.cache[key] = entry
        
        # Persist if enabled
        if self.persist_cache:
            self._save_cache()
            
    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            # Clear all cache
            count = len(self.cache)
            self.cache.clear()
            self.stats['invalidations'] += count
        else:
            # Invalidate matching entries
            keys_to_remove = []
            for key in self.cache:
                if pattern in key:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.cache[key]
                self.stats['invalidations'] += 1
                
    def invalidate_by_file(self, file_path: str):
        """Invalidate cache entries related to a specific file"""
        # This would need to track which queries touched which files
        # For now, invalidate all on file change
        self.invalidate()
        
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
            
        # Find LRU entry
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        self.stats['evictions'] += 1
        
    def _clean_expired(self):
        """Remove all expired entries"""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.cache[key]
            self.stats['invalidations'] += 1
            
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate * 100:.1f}%",
            'evictions': self.stats['evictions'],
            'invalidations': self.stats['invalidations'],
            'db_version': self.db_version,
            'cache_size_bytes': self._estimate_cache_size()
        }
        
    def _estimate_cache_size(self) -> int:
        """Estimate cache memory usage"""
        try:
            return len(pickle.dumps(self.cache))
        except:
            return 0
            
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        if self.persist_cache:
            self._save_cache()


class QueryCacheIntegration:
    """Integration helper for query caching"""
    
    def __init__(self, db_path: Path):
        self.cache_manager = CacheManager(
            cache_dir=db_path / '.cache',
            default_ttl=3600,  # 1 hour
            max_cache_size=500
        )
        self.cache_manager.set_db_version(db_path)
        
    def get_cached_result(self, 
                         query: str,
                         k: int,
                         **kwargs) -> Optional[List]:
        """Get cached query result if available"""
        key = self.cache_manager.generate_cache_key(query, k, **kwargs)
        return self.cache_manager.get(key)
        
    def cache_result(self,
                    query: str,
                    k: int,
                    result: List,
                    **kwargs):
        """Cache query result"""
        key = self.cache_manager.generate_cache_key(query, k, **kwargs)
        # Shorter TTL for larger result sets
        ttl = 3600 if k <= 10 else 1800
        self.cache_manager.set(key, result, ttl)
        
    def invalidate_for_updates(self, changed_files: List[Path]):
        """Invalidate cache based on file changes"""
        if changed_files:
            # For now, invalidate all cache on any file change
            # Could be smarter by tracking which queries touch which files
            self.cache_manager.invalidate()
            
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache_manager.get_stats()