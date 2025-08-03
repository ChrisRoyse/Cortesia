#!/usr/bin/env python3
"""
Advanced Query Interface with Hybrid Search
Implements semantic search with reranking and context awareness
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import locale
    if locale.getpreferredencoding().upper() != 'UTF-8':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import click


class AdvancedQuerier:
    def __init__(self, 
                 db_dir: str = "./chroma_db_advanced",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize advanced querier with hybrid search capabilities"""
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.embeddings = None
        self.vector_db = None
        self.metadata = None
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.vector_db is not None:
                del self.vector_db
                self.vector_db = None
            if self.embeddings is not None:
                del self.embeddings
                self.embeddings = None
            import gc
            gc.collect()
            print("[OK] Resources cleaned up")
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
        
    def load_metadata(self):
        """Load indexing metadata"""
        metadata_path = self.db_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                
    def initialize(self):
        """Initialize embeddings and load database"""
        # First check in vectors directory
        if not self.db_dir.exists():
            vectors_db = Path("vectors") / self.db_dir.name
            if vectors_db.exists():
                self.db_dir = vectors_db
            else:
                print("[ERROR] Advanced database not found!")
                print("Please run indexer_advanced.py first.")
                sys.exit(1)
                
        print("Loading advanced vector database...")
        print(f"Model: {self.model_name}")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load database
        self.vector_db = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings
        )
        
        # Load metadata
        self.load_metadata()
        
        if self.metadata:
            stats = self.metadata.get('stats', {})
            print(f"[OK] Database loaded:")
            print(f"  Total chunks: {stats.get('total_chunks', 'unknown')}")
            print(f"  Documentation: {stats.get('semantic_chunks', 0)} semantic chunks")
            print(f"  Code: {stats.get('ast_chunks', 0)} AST-parsed chunks")
        else:
            print("[OK] Database loaded successfully")
            
    def search(self, 
               query: str, 
               k: int = 5,
               filter_type: Optional[str] = None,
               rerank: bool = True,
               show_context: bool = False) -> List[Dict]:
        """Advanced search with filtering and reranking"""
        
        # Build filter if specified
        filter_dict = None
        if filter_type:
            if filter_type == 'code':
                filter_dict = {"$or": [
                    {"chunk_type": {"$contains": "code"}},
                    {"chunk_type": {"$contains": "python"}},
                    {"chunk_type": {"$contains": "rust"}}
                ]}
            elif filter_type == 'docs':
                filter_dict = {"chunk_type": "semantic"}
            elif filter_type in ['python', 'rust']:
                filter_dict = {"file_type": filter_type}
                
        # Perform search
        start_time = time.time()
        
        # Fetch extra results if reranking
        fetch_k = k * 3 if rerank else k
        
        results_with_scores = self.vector_db.similarity_search_with_relevance_scores(
            query=query,
            k=fetch_k,
            filter=filter_dict
        )
        
        # Rerank if requested
        if rerank and len(results_with_scores) > k:
            results_with_scores = self._rerank_results(
                results_with_scores, 
                query, 
                k
            )
        else:
            results_with_scores = results_with_scores[:k]
            
        search_time = time.time() - start_time
        
        # Format results
        results = []
        for doc, score in results_with_scores:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "search_time": search_time
            }
            
            # Add context if requested
            if show_context:
                result["context"] = self._get_context(doc.metadata)
                
            results.append(result)
            
        return results
        
    def _rerank_results(self, results, query, k):
        """Rerank results based on multiple factors"""
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        reranked = []
        for doc, base_score in results:
            metadata = doc.metadata
            content_lower = doc.page_content.lower()
            
            # Calculate bonus scores
            bonus = 0.0
            
            # Exact phrase match bonus
            if query_lower in content_lower:
                bonus += 0.15
                
            # Term frequency bonus
            term_matches = sum(1 for term in query_terms if term in content_lower)
            bonus += (term_matches / len(query_terms)) * 0.1
            
            # File name relevance bonus
            if 'relative_path' in metadata:
                path_lower = metadata['relative_path'].lower()
                if any(term in path_lower for term in query_terms):
                    bonus += 0.1
                    
            # Chunk type preference (prefer semantic chunks for natural queries)
            if metadata.get('chunk_type') == 'semantic':
                bonus += 0.05
                
            # Code-specific bonuses
            if 'code' in metadata.get('chunk_type', ''):
                # Prefer complete functions/classes
                if metadata.get('chunk_type') in ['python_class', 'python_function', 
                                                   'rust_function_item', 'rust_struct_item']:
                    bonus += 0.08
                    
            # Calculate final score
            final_score = min(1.0, base_score + bonus)
            reranked.append((doc, final_score))
            
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]
        
    def _get_context(self, metadata: Dict) -> str:
        """Get additional context for a result"""
        context_parts = []
        
        if metadata.get('chunk_type'):
            context_parts.append(f"Type: {metadata['chunk_type']}")
            
        if metadata.get('language'):
            context_parts.append(f"Language: {metadata['language']}")
            
        if metadata.get('node_type'):
            context_parts.append(f"Node: {metadata['node_type']}")
            
        if metadata.get('methods'):
            context_parts.append(f"Methods: {', '.join(metadata['methods'][:3])}")
            
        return " | ".join(context_parts) if context_parts else ""
        
    def explain_search(self):
        """Explain the advanced search features"""
        print("\n" + "=" * 60)
        print("ADVANCED SEARCH FEATURES")
        print("=" * 60)
        print("\n[SEARCH CAPABILITIES]")
        print("  • Semantic search across code and documentation")
        print("  • AST-aware code search")
        print("  • Automatic reranking for better relevance")
        print("  • Type filtering (code/docs/python/rust)")
        print("  • Context-aware results")
        print("\n[CHUNKING STRATEGIES]")
        print("  • Documentation: Semantic similarity-based chunking")
        print("  • Code: AST parsing with context preservation")
        print("  • Optimal chunk sizes (200-800 chars)")
        print("\n[TIPS]")
        print("  • Use natural language for concept searches")
        print("  • Use technical terms for code searches")
        print("  • Filter by type for focused results")
        print("  • Enable context for detailed information")
        print("=" * 60)


@click.command()
@click.option('--query', '-q', help='Search query')
@click.option('--results', '-k', default=5, help='Number of results')
@click.option('--type', '-t', type=click.Choice(['all', 'code', 'docs', 'python', 'rust']), 
              default='all', help='Filter by content type')
@click.option('--full', '-f', is_flag=True, help='Show full content')
@click.option('--context', '-c', is_flag=True, help='Show additional context')
@click.option('--no-rerank', is_flag=True, help='Disable reranking')
@click.option('--explain', '-e', is_flag=True, help='Explain search features')
@click.option('--db-dir', '-d', default="./chroma_db_advanced", help='Database directory')
def main(query: Optional[str], results: int, type: str, full: bool, 
         context: bool, no_rerank: bool, explain: bool, db_dir: str):
    """Advanced Query Interface with Hybrid Search"""
    
    querier = AdvancedQuerier(
        db_dir=db_dir,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        if explain:
            querier.explain_search()
            return
            
        # Initialize
        querier.initialize()
        
        if query:
            # Determine filter
            filter_type = None if type == 'all' else type
            
            print(f"\n[SEARCH] Query: '{query}'")
            if filter_type:
                print(f"[FILTER] Type: {filter_type}")
            print(f"[CONFIG] Reranking: {'enabled' if not no_rerank else 'disabled'}")
            print("=" * 80)
            
            # Perform search
            search_results = querier.search(
                query, 
                k=results,
                filter_type=filter_type,
                rerank=not no_rerank,
                show_context=context
            )
            
            if not search_results:
                print("No results found.")
                return
                
            print(f"Found {len(search_results)} results in {search_results[0]['search_time']:.3f}s\n")
            
            # Display results
            for i, result in enumerate(search_results, 1):
                print(f"{'='*60}")
                print(f"RESULT {i}/{len(search_results)}")
                print(f"{'='*60}")
                
                # Score and metadata
                print(f"[SCORE] {result['score']:.4f}")
                print(f"[FILE] {result['metadata'].get('relative_path', 'Unknown')}")
                print(f"[TYPE] {result['metadata'].get('chunk_type', 'Unknown')}")
                
                if 'chunk_index' in result['metadata']:
                    chunk_info = f"{result['metadata']['chunk_index'] + 1}/{result['metadata'].get('total_chunks', '?')}"
                    print(f"[CHUNK] {chunk_info}")
                    
                if context and 'context' in result:
                    print(f"[CONTEXT] {result['context']}")
                    
                # Content
                if full:
                    print(f"\n[CONTENT]\n{result['content']}")
                else:
                    preview = result['content'][:400].strip()
                    if len(result['content']) > 400:
                        preview += "..."
                    print(f"\n[PREVIEW]\n{preview}")
                    
                print()
                
        else:
            print("\n" + "=" * 60)
            print("ADVANCED QUERY INTERFACE")
            print("=" * 60)
            print("\nUsage: python query_advanced.py -q 'your query' [options]")
            print("\nOptions:")
            print("  -q, --query      : Search query")
            print("  -k, --results    : Number of results (default: 5)")
            print("  -t, --type       : Filter by type (all/code/docs/python/rust)")
            print("  -f, --full       : Show full content")
            print("  -c, --context    : Show additional context")
            print("  --no-rerank      : Disable reranking")
            print("  -e, --explain    : Explain search features")
            print("\nExamples:")
            print("  python query_advanced.py -q 'temporal memory' -t docs")
            print("  python query_advanced.py -q 'allocation engine' -t code -c")
            print("  python query_advanced.py -q 'def process' -t python -f")
            
    except KeyboardInterrupt:
        print("\nSearch interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        querier.cleanup()


if __name__ == "__main__":
    main()