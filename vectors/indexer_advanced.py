#!/usr/bin/env python3
"""
Advanced RAG Indexer with 2025 Best Practices
Implements semantic chunking, AST-based code parsing, and hybrid strategies
"""

import os
import sys
import ast
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from fnmatch import fnmatch

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

# Try to import tree-sitter for code parsing
try:
    import tree_sitter_rust as tsrust
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available, using fallback code parsing")


@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk"""
    source: str
    file_type: str
    chunk_type: str  # 'semantic', 'code_function', 'code_class', 'documentation'
    chunk_index: int
    total_chunks: int
    semantic_density: float  # Measure of information density
    context_window: Tuple[int, int]  # Start and end positions in original
    parent_context: Optional[str] = None  # For code: class/module context
    dependencies: List[str] = None  # For code: imports/dependencies
    
    
class GitignoreParser:
    """Parse and apply .gitignore rules"""
    
    def __init__(self, gitignore_path: Path):
        self.patterns = []
        self.base_dir = gitignore_path.parent
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Convert gitignore patterns to fnmatch patterns
                        if line.startswith('/'):
                            # Absolute path from repo root
                            pattern = line[1:]
                        else:
                            # Can match anywhere
                            pattern = '**/' + line if not line.startswith('*') else line
                        self.patterns.append(pattern)
                        
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        rel_path = path.relative_to(self.base_dir)
        path_str = str(rel_path).replace('\\', '/')
        
        for pattern in self.patterns:
            if fnmatch(path_str, pattern):
                return True
            # Check if any parent directory matches
            if '/' in path_str:
                parts = path_str.split('/')
                for i in range(len(parts)):
                    partial = '/'.join(parts[:i+1])
                    if fnmatch(partial, pattern.rstrip('/')):
                        return True
        return False


class SemanticChunker:
    """Advanced semantic chunking with sentence embeddings"""
    
    def __init__(self, embedding_model, similarity_threshold=0.7):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
    def chunk_by_semantic_similarity(self, text: str, min_chunk_size=200, max_chunk_size=800) -> List[str]:
        """Split text into semantic chunks based on embedding similarity"""
        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []
            
        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_documents(sentences)
        embeddings = np.array(embeddings)
        
        # Group sentences by semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = cosine_similarity(
                [current_embedding], 
                [embeddings[i]]
            )[0][0]
            
            # Check if we should add to current chunk or start new one
            if (similarity >= self.similarity_threshold and 
                current_size + len(sentences[i]) <= max_chunk_size):
                current_chunk.append(sentences[i])
                current_size += len(sentences[i])
                # Update chunk embedding (weighted average)
                weight = len(sentences[i]) / current_size
                current_embedding = (1 - weight) * current_embedding + weight * embeddings[i]
            else:
                # Save current chunk if it meets minimum size
                if current_size >= min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                elif chunks:
                    # Add to previous chunk if too small
                    chunks[-1] += ' ' + ' '.join(current_chunk)
                else:
                    chunks.append(' '.join(current_chunk))
                    
                # Start new chunk
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
                current_size = len(sentences[i])
                
        # Add final chunk
        if current_chunk:
            if current_size >= min_chunk_size or not chunks:
                chunks.append(' '.join(current_chunk))
            else:
                chunks[-1] += ' ' + ' '.join(current_chunk)
                
        return chunks
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter - can be enhanced with spaCy or NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class CodeChunker:
    """AST-based code chunking for better code understanding"""
    
    def __init__(self):
        self.python_parser = self._create_python_parser()
        self.rust_parser = self._create_rust_parser() if TREE_SITTER_AVAILABLE else None
        
    def _create_python_parser(self):
        """Python AST parser using built-in ast module"""
        return ast
        
    def _create_rust_parser(self):
        """Rust parser using tree-sitter"""
        if not TREE_SITTER_AVAILABLE:
            return None
        parser = tree_sitter.Parser()
        parser.set_language(tree_sitter.Language(tsrust.language(), "rust"))
        return parser
        
    def chunk_python_code(self, code: str, file_path: str) -> List[Dict]:
        """Parse Python code into semantic chunks"""
        chunks = []
        try:
            tree = ast.parse(code)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
                    
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    chunk = self._extract_class_chunk(node, code, imports)
                    chunks.append(chunk)
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    chunk = self._extract_function_chunk(node, code, imports)
                    chunks.append(chunk)
                    
        except SyntaxError:
            # Fallback to simple chunking if parsing fails
            chunks = self._fallback_chunk_code(code, file_path)
            
        return chunks
        
    def chunk_rust_code(self, code: str, file_path: str) -> List[Dict]:
        """Parse Rust code into semantic chunks"""
        if not self.rust_parser:
            return self._fallback_chunk_code(code, file_path)
            
        chunks = []
        tree = self.rust_parser.parse(bytes(code, "utf8"))
        
        # Extract functions, structs, impls, etc.
        # This is a simplified version - can be enhanced
        cursor = tree.walk()
        
        def visit_node(cursor):
            node = cursor.node
            if node.type in ['function_item', 'impl_item', 'struct_item', 'enum_item']:
                start_byte = node.start_byte
                end_byte = node.end_byte
                chunk_code = code[start_byte:end_byte]
                
                chunks.append({
                    'content': chunk_code,
                    'type': f'rust_{node.type}',
                    'start': start_byte,
                    'end': end_byte,
                    'metadata': {
                        'language': 'rust',
                        'node_type': node.type
                    }
                })
                
            if cursor.goto_first_child():
                visit_node(cursor)
                while cursor.goto_next_sibling():
                    visit_node(cursor)
                cursor.goto_parent()
                
        visit_node(cursor)
        
        if not chunks:
            chunks = self._fallback_chunk_code(code, file_path)
            
        return chunks
        
    def _extract_class_chunk(self, node: ast.ClassDef, code: str, imports: List[str]) -> Dict:
        """Extract a class with its methods and context"""
        lines = code.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        class_code = '\n'.join(lines[start_line:end_line])
        
        # Include imports for context
        context = '\n'.join(imports) + '\n\n' if imports else ''
        
        return {
            'content': context + class_code,
            'type': 'python_class',
            'name': node.name,
            'metadata': {
                'language': 'python',
                'node_type': 'class',
                'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            }
        }
        
    def _extract_function_chunk(self, node: ast.FunctionDef, code: str, imports: List[str]) -> Dict:
        """Extract a function with its context"""
        lines = code.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        func_code = '\n'.join(lines[start_line:end_line])
        
        # Include relevant imports
        relevant_imports = self._get_relevant_imports(func_code, imports)
        context = '\n'.join(relevant_imports) + '\n\n' if relevant_imports else ''
        
        return {
            'content': context + func_code,
            'type': 'python_function',
            'name': node.name,
            'metadata': {
                'language': 'python',
                'node_type': 'function'
            }
        }
        
    def _is_method(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if a function is a class method"""
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                if node in class_node.body:
                    return True
        return False
        
    def _get_relevant_imports(self, code: str, imports: List[str]) -> List[str]:
        """Get imports that are actually used in the code"""
        relevant = []
        for imp in imports:
            # Simple heuristic - check if imported name appears in code
            if 'import' in imp:
                parts = imp.split()
                for part in parts:
                    if part != 'import' and part != 'from' and part in code:
                        relevant.append(imp)
                        break
        return relevant
        
    def _fallback_chunk_code(self, code: str, file_path: str) -> List[Dict]:
        """Fallback chunking for code when AST parsing fails"""
        # Use a smaller chunk size for code
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        for i, chunk in enumerate(splitter.split_text(code)):
            chunks.append({
                'content': chunk,
                'type': 'code_block',
                'metadata': {
                    'language': Path(file_path).suffix[1:],
                    'chunk_index': i
                }
            })
        return chunks


class AdvancedIndexer:
    def __init__(self, 
                 root_dir: str = ".",
                 db_dir: str = "./chroma_db_advanced",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the advanced indexer with best practices"""
        self.root_dir = Path(root_dir).resolve()
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        
        # Initialize components
        self.gitignore_parser = GitignoreParser(self.root_dir / '.gitignore')
        self.code_chunker = CodeChunker()
        
        # Stats tracking
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "code_files": 0,
            "doc_files": 0,
            "semantic_chunks": 0,
            "ast_chunks": 0,
            "processing_time": 0
        }
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                del self.embeddings
                self.embeddings = None
            if hasattr(self, 'semantic_chunker') and self.semantic_chunker is not None:
                del self.semantic_chunker
                self.semantic_chunker = None
            import gc
            gc.collect()
            print("[OK] Resources cleaned up")
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
            
    def initialize_embeddings(self):
        """Initialize embeddings and semantic chunker"""
        print(f"Initializing {self.model_name}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            similarity_threshold=0.75
        )
        
        # Test embedding
        test_embedding = self.embeddings.embed_query("test")
        dimensions = len(test_embedding)
        print(f"[OK] Model loaded: {dimensions} dimensions")
        
        return dimensions
        
    def should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed"""
        # Skip vectors directory
        if 'vectors' in file_path.parts:
            return False
            
        # Skip files in gitignore
        if self.gitignore_parser.should_ignore(file_path):
            return False
            
        # Only index specific file types
        valid_extensions = {
            '.md', '.txt', '.rst',  # Documentation
            '.py',                   # Python code
            '.rs',                   # Rust code
            '.toml', '.yaml', '.yml', '.json'  # Config files
        }
        
        return file_path.suffix.lower() in valid_extensions
        
    def process_documentation_file(self, file_path: Path) -> List[Document]:
        """Process documentation files with semantic chunking"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Use semantic chunking for documentation
            chunks = self.semantic_chunker.chunk_by_semantic_similarity(
                content,
                min_chunk_size=200,
                max_chunk_size=800
            )
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "relative_path": str(file_path.relative_to(self.root_dir)),
                        "file_type": file_path.suffix[1:],
                        "chunk_type": "semantic",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "semantic_density": len(chunk.split()) / len(chunk) if chunk else 0
                    }
                )
                documents.append(doc)
                self.stats["semantic_chunks"] += 1
                
        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            
        return documents
        
    def process_code_file(self, file_path: Path) -> List[Document]:
        """Process code files with AST-based chunking"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Choose appropriate chunker based on file extension
            if file_path.suffix == '.py':
                chunks = self.code_chunker.chunk_python_code(content, str(file_path))
            elif file_path.suffix == '.rs':
                chunks = self.code_chunker.chunk_rust_code(content, str(file_path))
            else:
                # Fallback for other code files
                chunks = self.code_chunker._fallback_chunk_code(content, str(file_path))
                
            for i, chunk_data in enumerate(chunks):
                doc = Document(
                    page_content=chunk_data['content'],
                    metadata={
                        "source": str(file_path),
                        "relative_path": str(file_path.relative_to(self.root_dir)),
                        "file_type": file_path.suffix[1:],
                        "chunk_type": chunk_data.get('type', 'code_block'),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **chunk_data.get('metadata', {})
                    }
                )
                documents.append(doc)
                self.stats["ast_chunks"] += 1
                
        except Exception as e:
            print(f"  Warning: Could not process {file_path}: {e}")
            
        return documents
        
    def collect_files(self) -> Tuple[List[Path], List[Path]]:
        """Collect all files to be indexed"""
        doc_files = []
        code_files = []
        
        print("Scanning for files to index...")
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and self.should_index_file(file_path):
                if file_path.suffix in ['.md', '.txt', '.rst']:
                    doc_files.append(file_path)
                elif file_path.suffix in ['.py', '.rs', '.toml', '.yaml', '.yml', '.json']:
                    code_files.append(file_path)
                    
        print(f"Found {len(doc_files)} documentation files")
        print(f"Found {len(code_files)} code files")
        
        return doc_files, code_files
        
    def run(self):
        """Run the advanced indexing pipeline"""
        start_time = time.time()
        vector_db = None
        
        try:
            print("=" * 60)
            print("ADVANCED RAG INDEXER - 2025 BEST PRACTICES")
            print("=" * 60)
            print("Features:")
            print("  - Semantic chunking for documentation")
            print("  - AST-based parsing for code")
            print("  - Context preservation")
            print("  - Gitignore awareness")
            print("=" * 60)
            
            # Initialize embeddings
            dimensions = self.initialize_embeddings()
            
            # Collect files
            doc_files, code_files = self.collect_files()
            
            if not doc_files and not code_files:
                print("No files found to index!")
                return False
                
            # Remove existing database
            if self.db_dir.exists():
                import shutil
                try:
                    shutil.rmtree(self.db_dir)
                    print("Removed existing database")
                except:
                    pass
                    
            # Process documentation files
            all_documents = []
            
            print("\nProcessing documentation files...")
            for i, file_path in enumerate(doc_files, 1):
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(doc_files)} documentation files")
                docs = self.process_documentation_file(file_path)
                all_documents.extend(docs)
                self.stats["doc_files"] += 1
                
            print("\nProcessing code files...")
            for i, file_path in enumerate(code_files, 1):
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(code_files)} code files")
                docs = self.process_code_file(file_path)
                all_documents.extend(docs)
                self.stats["code_files"] += 1
                
            self.stats["total_files"] = len(doc_files) + len(code_files)
            self.stats["total_chunks"] = len(all_documents)
            
            print(f"\nCreating vector database with {len(all_documents)} chunks...")
            
            # Create vector database in batches
            batch_size = 100
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i+batch_size]
                
                if vector_db is None:
                    vector_db = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=str(self.db_dir),
                        collection_metadata={
                            "hnsw:space": "cosine",
                            "hnsw:construction_ef": 100,
                            "hnsw:M": 24  # Balanced for mixed content
                        }
                    )
                else:
                    vector_db.add_documents(batch)
                    
                if (i + batch_size) % 500 == 0:
                    print(f"  Added {min(i + batch_size, len(all_documents))}/{len(all_documents)} chunks")
                    
            # Calculate processing time
            self.stats["processing_time"] = time.time() - start_time
            
            # Save metadata
            metadata = {
                "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": self.stats,
                "root_directory": str(self.root_dir),
                "db_directory": str(self.db_dir),
                "embedding_model": self.model_name,
                "embedding_dimensions": dimensions,
                "chunking_strategies": {
                    "documentation": "Semantic similarity-based chunking",
                    "code": "AST-based parsing with context preservation",
                },
                "features": [
                    "Semantic chunking",
                    "AST code parsing",
                    "Context preservation",
                    "Gitignore awareness",
                    "Hybrid chunking strategies"
                ]
            }
            
            metadata_path = self.db_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Print summary
            print("\n" + "=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"Total files: {self.stats['total_files']}")
            print(f"  Documentation: {self.stats['doc_files']}")
            print(f"  Code files: {self.stats['code_files']}")
            print(f"Total chunks: {self.stats['total_chunks']}")
            print(f"  Semantic chunks: {self.stats['semantic_chunks']}")
            print(f"  AST chunks: {self.stats['ast_chunks']}")
            print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            print(f"Database: {self.db_dir}")
            
            return True
            
        finally:
            # Always cleanup resources
            try:
                if vector_db is not None:
                    vector_db.persist()
                    del vector_db
            except:
                pass
            self.cleanup()


@click.command()
@click.option('--root-dir', '-r', default=".", help='Root directory to index')
@click.option('--db-dir', '-o', default="./chroma_db_advanced", help='Output database directory')
@click.option('--model', '-m', default="sentence-transformers/all-MiniLM-L6-v2", help='Embedding model')
def main(root_dir: str, db_dir: str, model: str):
    """Advanced RAG Indexer with 2025 Best Practices"""
    
    # Change to vectors directory for database output
    vectors_dir = Path("vectors")
    if vectors_dir.exists():
        db_path = vectors_dir / Path(db_dir).name
    else:
        db_path = Path(db_dir)
    
    indexer = AdvancedIndexer(
        root_dir=root_dir,
        db_dir=str(db_path),
        model_name=model
    )
    
    try:
        success = indexer.run()
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