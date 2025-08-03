#!/usr/bin/env python3
"""
MCP Server for Universal RAG Indexing System

Exposes the RAG indexing and query capabilities as an MCP server with two tools:
1. index_codebase: Index any directory with Git tracking
2. query_codebase: Search indexed content with semantic search
"""

import os
import sys
import json
import asyncio
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# MCP imports
try:
    from mcp import Server, Tool
    from mcp.server import StdioServerTransport
    from mcp.types import TextContent, ToolResult, ToolError
except ImportError:
    print("Installing MCP SDK...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp[cli]"])
    from mcp import Server, Tool
    from mcp.server import StdioServerTransport
    from mcp.types import TextContent, ToolResult, ToolError

# Import our RAG system
from indexer_universal import UniversalIndexer
from query_universal import UniversalQuerier
from git_tracker import GitChangeTracker
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / '.mcp-rag-indexer' / 'server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IndexedProject:
    """Track indexed projects"""
    root_dir: str
    db_path: str
    indexed_at: datetime
    file_count: int
    chunk_count: int
    languages: List[str]
    watch_enabled: bool
    last_update: datetime


class GitWatcher:
    """Efficient Git change monitoring with minimal resource usage"""
    
    def __init__(self, root_dir: Path, callback):
        self.root_dir = root_dir
        self.callback = callback
        self.tracker = GitChangeTracker(root_dir)
        self.last_commit = None
        self.watch_thread = None
        self.stop_event = threading.Event()
        self.poll_interval = 30  # seconds
        
    def start(self):
        """Start watching for changes"""
        if self.watch_thread and self.watch_thread.is_alive():
            return
            
        self.last_commit = self.tracker.get_current_commit()
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        logger.info(f"Started Git watcher for {self.root_dir}")
        
    def stop(self):
        """Stop watching"""
        self.stop_event.set()
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        logger.info(f"Stopped Git watcher for {self.root_dir}")
        
    def _watch_loop(self):
        """Main watch loop - runs in background thread"""
        while not self.stop_event.is_set():
            try:
                # Check for changes
                current_commit = self.tracker.get_current_commit()
                
                if current_commit != self.last_commit:
                    # Get changed files
                    changes = self.tracker.get_changed_files()
                    if changes:
                        logger.info(f"Detected {len(changes)} changes in {self.root_dir}")
                        self.callback(self.root_dir, changes)
                        self.last_commit = current_commit
                        
            except Exception as e:
                logger.error(f"Error in Git watcher: {e}")
                
            # Sleep with interruptible wait
            self.stop_event.wait(self.poll_interval)


class MCPRAGServer:
    """MCP Server for RAG Indexing System"""
    
    def __init__(self):
        self.server = Server("mcp-rag-indexer")
        self.projects: Dict[str, IndexedProject] = {}
        self.watchers: Dict[str, GitWatcher] = {}
        self.cache = CacheManager()
        self.data_dir = Path.home() / '.mcp-rag-indexer'
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing projects
        self._load_projects()
        
        # Register tools
        self._register_tools()
        
    def _load_projects(self):
        """Load previously indexed projects"""
        projects_file = self.data_dir / 'projects.json'
        if projects_file.exists():
            try:
                with open(projects_file, 'r') as f:
                    data = json.load(f)
                    for project_id, project_data in data.items():
                        project_data['indexed_at'] = datetime.fromisoformat(project_data['indexed_at'])
                        project_data['last_update'] = datetime.fromisoformat(project_data['last_update'])
                        self.projects[project_id] = IndexedProject(**project_data)
                logger.info(f"Loaded {len(self.projects)} indexed projects")
            except Exception as e:
                logger.error(f"Failed to load projects: {e}")
                
    def _save_projects(self):
        """Save project metadata"""
        projects_file = self.data_dir / 'projects.json'
        try:
            data = {}
            for project_id, project in self.projects.items():
                project_dict = asdict(project)
                project_dict['indexed_at'] = project.indexed_at.isoformat()
                project_dict['last_update'] = project.last_update.isoformat()
                data[project_id] = project_dict
                
            with open(projects_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")
            
    def _get_project_id(self, root_dir: str) -> str:
        """Generate unique project ID from path"""
        return hashlib.md5(root_dir.encode()).hexdigest()[:12]
        
    def _get_db_path(self, project_id: str) -> Path:
        """Get database path for project"""
        return self.data_dir / 'databases' / project_id
        
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.tool()
        async def index_codebase(
            root_dir: str,
            watch: bool = True,
            incremental: bool = True,
            languages: Optional[List[str]] = None
        ) -> str:
            """
            Index a codebase for RAG retrieval with Git tracking.
            
            Args:
                root_dir: Absolute path to codebase root directory
                watch: Enable Git change monitoring
                incremental: Only index changed files
                languages: Languages to index (empty = all)
            
            Returns:
                Status message with indexing statistics
            """
            try:
                # Validate path
                root_path = Path(root_dir)
                if not root_path.exists():
                    raise ValueError(f"Directory not found: {root_dir}")
                if not root_path.is_dir():
                    raise ValueError(f"Not a directory: {root_dir}")
                    
                # Get project ID
                project_id = self._get_project_id(root_dir)
                db_path = self._get_db_path(project_id)
                
                # Check if already indexed
                existing = self.projects.get(project_id)
                if existing and incremental:
                    logger.info(f"Performing incremental update for {root_dir}")
                    # TODO: Implement incremental indexing
                    
                # Create indexer
                logger.info(f"Indexing {root_dir}")
                indexer = UniversalIndexer(
                    root_dir=root_dir,
                    db_dir=str(db_path),
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Run indexing
                start_time = time.time()
                success = await asyncio.to_thread(indexer.run)
                indexing_time = time.time() - start_time
                
                if not success:
                    raise RuntimeError("Indexing failed")
                    
                # Get statistics
                stats = indexer.stats
                
                # Save project metadata
                project = IndexedProject(
                    root_dir=root_dir,
                    db_path=str(db_path),
                    indexed_at=datetime.now(),
                    file_count=stats['total_files'],
                    chunk_count=stats['total_chunks'],
                    languages=list(stats['languages'].keys()),
                    watch_enabled=watch,
                    last_update=datetime.now()
                )
                self.projects[project_id] = project
                self._save_projects()
                
                # Start Git watcher if requested
                if watch:
                    self._start_watcher(root_dir, project_id)
                    
                # Cleanup
                indexer.cleanup()
                
                # Format result
                result = (
                    f"✅ Successfully indexed {root_dir}\n\n"
                    f"📊 Statistics:\n"
                    f"  • Files processed: {stats['total_files']}\n"
                    f"  • Chunks created: {stats['total_chunks']}\n"
                    f"  • Languages: {', '.join(stats['languages'].keys())}\n"
                    f"  • Time: {indexing_time:.2f} seconds\n"
                    f"  • Speed: {stats['total_chunks']/indexing_time:.1f} chunks/sec\n"
                )
                
                if watch:
                    result += f"\n🔄 Git monitoring enabled (checks every 30s)"
                    
                return result
                
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                return f"❌ Indexing failed: {str(e)}"
                
        @self.server.tool()
        async def query_codebase(
            query: str,
            root_dir: str,
            k: int = 5,
            filter_type: str = "all",
            rerank: bool = True
        ) -> str:
            """
            Query indexed codebase with semantic search.
            
            Args:
                query: Search query
                root_dir: Codebase to search (must be indexed)
                k: Number of results (default: 5)
                filter_type: Filter by type: code, docs, config, or all
                rerank: Enable multi-factor reranking
            
            Returns:
                Formatted search results
            """
            try:
                # Get project
                project_id = self._get_project_id(root_dir)
                project = self.projects.get(project_id)
                
                if not project:
                    return (
                        f"❌ Codebase not indexed: {root_dir}\n"
                        f"Please run index_codebase first."
                    )
                    
                # Check if index is recent
                age = datetime.now() - project.last_update
                if age > timedelta(hours=24):
                    logger.warning(f"Index for {root_dir} is {age.days} days old")
                    
                # Create querier
                querier = UniversalQuerier(
                    db_dir=project.db_path,
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                await asyncio.to_thread(querier.initialize)
                
                # Execute query
                logger.info(f"Querying '{query}' in {root_dir}")
                results = await asyncio.to_thread(
                    querier.search,
                    query,
                    k=k,
                    filter_type=filter_type if filter_type != "all" else None,
                    rerank=rerank
                )
                
                # Format results
                if not results:
                    output = f"No results found for: {query}"
                else:
                    output = f"🔍 Search results for: {query}\n"
                    output += f"📁 In: {root_dir}\n\n"
                    
                    for i, (doc, score) in enumerate(results, 1):
                        metadata = doc.metadata
                        file_path = metadata.get('file', 'Unknown')
                        chunk_type = metadata.get('type', 'unknown')
                        lines = metadata.get('lines', '')
                        
                        # Make path relative
                        try:
                            rel_path = Path(file_path).relative_to(root_dir)
                        except:
                            rel_path = Path(file_path).name
                            
                        output += f"{i}. {rel_path}"
                        if lines:
                            output += f":{lines}"
                        output += f" [{chunk_type}] (score: {score:.3f})\n"
                        
                        # Add preview
                        content = doc.page_content[:200]
                        if len(doc.page_content) > 200:
                            content += "..."
                        output += f"   {content}\n\n"
                        
                # Cleanup
                querier.cleanup()
                
                return output
                
            except Exception as e:
                logger.error(f"Query failed: {e}")
                return f"❌ Query failed: {str(e)}"
                
    def _start_watcher(self, root_dir: str, project_id: str):
        """Start Git watcher for a project"""
        if project_id in self.watchers:
            logger.info(f"Watcher already active for {root_dir}")
            return
            
        def on_changes(path, changes):
            """Handle detected changes"""
            logger.info(f"Re-indexing {len(changes)} changed files in {path}")
            # TODO: Implement incremental re-indexing
            project = self.projects.get(project_id)
            if project:
                project.last_update = datetime.now()
                self._save_projects()
                
        watcher = GitWatcher(Path(root_dir), on_changes)
        watcher.start()
        self.watchers[project_id] = watcher
        
    def stop_watchers(self):
        """Stop all Git watchers"""
        for watcher in self.watchers.values():
            watcher.stop()
        self.watchers.clear()
        
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting MCP RAG Indexer Server")
        
        # Use stdio transport for Claude Code
        async with StdioServerTransport() as transport:
            await self.server.run(transport)
            
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down MCP server")
        self.stop_watchers()
        self._save_projects()


async def main():
    """Main entry point"""
    server = MCPRAGServer()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.shutdown()


if __name__ == "__main__":
    # Handle Windows-specific setup
    if sys.platform == "win32":
        # Set up proper async event loop for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    # Run the server
    asyncio.run(main())