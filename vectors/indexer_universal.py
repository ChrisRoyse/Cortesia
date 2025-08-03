#!/usr/bin/env python3
"""
Universal RAG Indexer - Production-ready indexing system for any codebase
Works with ANY programming language without external parser dependencies
Implements pattern-based extraction with intelligent chunking strategies
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


@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk with comprehensive tracking"""
    source: str
    file_type: str
    language: str
    chunk_type: str  # 'function', 'class', 'method', 'import', 'semantic', 'hierarchical', 'sliding_window'
    chunk_index: int
    total_chunks: int
    semantic_density: float = 0.0
    line_range: Tuple[int, int] = (0, 0)
    parent_class: Optional[str] = None
    method_name: Optional[str] = None
    function_name: Optional[str] = None
    has_imports: bool = False
    has_docstring: bool = False
    dependencies: List[str] = field(default_factory=list)
    hierarchy_level: int = 0
    overlaps_with: List[int] = field(default_factory=list)
    context_preserved: bool = True


class UniversalCodeParser:
    """Universal code parser using pattern-based extraction - NO external dependencies"""
    
    def __init__(self):
        # Language patterns for universal detection
        self.language_patterns = {
            'python': {
                'extensions': ['.py', '.pyw'],
                'function': r'^\s*(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'class': r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))?:',
                'method': r'^\s+(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'import': r'^\s*(from\s+[a-zA-Z0-9_\.]+\s+)?import\s+.+',
                'docstring': r'^\s*["\'][\"\'][\"\']([^"\']+)["\'][\"\'][\"\']',
                'comment': r'^\s*#.*',
                'indent': '    ',
                'block_start': ':',
                'block_end': None
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.mjs'],
                'function': r'^\s*(async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)|^\s*const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(async\s+)?\([^)]*\)\s*=>',
                'class': r'^\s*class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(\s+extends\s+[a-zA-Z_$][a-zA-Z0-9_$]*)?\s*\{',
                'method': r'^\s+(async\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{',
                'import': r'^\s*(import|const|let|var)\s+.*\s+from\s+["\'][^"\']+["\']|^\s*import\s+["\'][^"\']+["\']',
                'export': r'^\s*export\s+(default\s+)?(function|class|const|let|var)',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '  ',
                'block_start': '{',
                'block_end': '}'
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'function': r'^\s*(async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*\([^)]*\)|^\s*const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(async\s+)?(<[^>]*>)?\s*\([^)]*\)\s*=>',
                'class': r'^\s*(export\s+)?(abstract\s+)?class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*(\s+extends\s+[a-zA-Z_$][a-zA-Z0-9_$]*)?\s*(\s+implements\s+[^{]+)?\s*\{',
                'interface': r'^\s*(export\s+)?interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*(\s+extends\s+[^{]+)?\s*\{',
                'method': r'^\s+(public|private|protected|static|readonly|async)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*\([^)]*\)\s*:?[^{]*\{',
                'import': r'^\s*import\s+.*\s+from\s+["\'][^"\']+["\']',
                'type': r'^\s*(export\s+)?type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '  ',
                'block_start': '{',
                'block_end': '}'
            },
            'rust': {
                'extensions': ['.rs'],
                'function': r'^\s*(pub\s+)?(async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?\s*\([^)]*\)',
                'struct': r'^\s*(pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?',
                'impl': r'^\s*impl\s*(<[^>]*>)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?',
                'trait': r'^\s*(pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?',
                'enum': r'^\s*(pub\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?',
                'use': r'^\s*(pub\s+)?use\s+.+;',
                'mod': r'^\s*(pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '    ',
                'block_start': '{',
                'block_end': '}'
            },
            'go': {
                'extensions': ['.go'],
                'function': r'^\s*func\s+(\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'struct': r'^\s*type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+struct\s*\{',
                'interface': r'^\s*type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+interface\s*\{',
                'method': r'^\s*func\s+\([^)]+\)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'import': r'^\s*import\s+(\(|")',
                'package': r'^\s*package\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '\t',
                'block_start': '{',
                'block_end': '}'
            },
            'java': {
                'extensions': ['.java'],
                'class': r'^\s*(public|private|protected)?\s*(abstract|final)?\s*class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*(extends\s+[^{]+)?\s*(implements\s+[^{]+)?\s*\{',
                'interface': r'^\s*(public|private|protected)?\s*interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(<[^>]*>)?\s*(extends\s+[^{]+)?\s*\{',
                'method': r'^\s*(public|private|protected|static|final|synchronized|native|abstract)?\s*(<[^>]*>)?\s*([a-zA-Z_$][a-zA-Z0-9_$<>\[\]]*)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)',
                'import': r'^\s*import\s+(static\s+)?[a-zA-Z0-9_\.]+(\.\*)?;',
                'package': r'^\s*package\s+[a-zA-Z0-9_\.]+;',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '    ',
                'block_start': '{',
                'block_end': '}'
            },
            'cpp': {
                'extensions': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hxx'],
                'class': r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(:.*?)?\s*\{',
                'struct': r'^\s*struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(:.*?)?\s*\{',
                'function': r'^\s*([a-zA-Z_][a-zA-Z0-9_<>:\*&\s]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'method': r'^\s*([a-zA-Z_][a-zA-Z0-9_<>:\*&\s]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*::\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'include': r'^\s*#include\s+[<"][^>"]+[>"]',
                'namespace': r'^\s*namespace\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '    ',
                'block_start': '{',
                'block_end': '}'
            },
            'c': {
                'extensions': ['.c', '.h'],
                'function': r'^\s*([a-zA-Z_][a-zA-Z0-9_\*\s]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'struct': r'^\s*struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{',
                'typedef': r'^\s*typedef\s+.+;',
                'include': r'^\s*#include\s+[<"][^>"]+[>"]',
                'define': r'^\s*#define\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'comment': r'^\s*//.*|^\s*/\*',
                'indent': '    ',
                'block_start': '{',
                'block_end': '}'
            }
        }
        
    def detect_language(self, file_path: Path) -> str:
        """Detect language from file extension and content patterns"""
        ext = file_path.suffix.lower()
        
        # Check by extension first
        for lang, patterns in self.language_patterns.items():
            if ext in patterns.get('extensions', []):
                return lang
                
        # Fallback to content analysis if needed
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars
                
            # Check for language-specific patterns
            if 'def ' in content and 'import ' in content:
                return 'python'
            elif 'function' in content or 'const ' in content:
                return 'javascript'
            elif 'fn ' in content and 'use ' in content:
                return 'rust'
            elif 'func ' in content and 'package ' in content:
                return 'go'
            elif 'public class' in content or 'private class' in content:
                return 'java'
                
        except:
            pass
            
        return 'unknown'
        
    def extract_code_blocks(self, code: str, language: str) -> List[Dict]:
        """Universal pattern-based code extraction without external dependencies"""
        chunks = []
        lines = code.split('\n')
        patterns = self.language_patterns.get(language, {})
        
        if not patterns:
            return self._fallback_extraction(code, language)
            
        # Extract imports/includes/use statements
        imports = self._extract_imports(lines, patterns)
        
        # Extract global context (constants, types, etc)
        globals_context = self._extract_globals(lines, patterns)
        
        # Extract functions and classes
        blocks = self._extract_code_blocks(lines, patterns, language)
        
        # Create chunks with proper context
        for block in blocks:
            # Add necessary imports
            relevant_imports = self._get_relevant_imports(block['content'], imports)
            context = '\n'.join(relevant_imports) + '\n\n' if relevant_imports else ''
            
            # Add global context if needed
            relevant_globals = self._get_relevant_globals(block['content'], globals_context)
            if relevant_globals:
                context += '\n'.join(relevant_globals) + '\n\n'
                
            chunk = {
                'content': context + block['content'],
                'type': block['type'],
                'name': block.get('name', 'unknown'),
                'metadata': {
                    'language': language,
                    'has_imports': len(relevant_imports) > 0,
                    'has_context': len(context) > 0,
                    'line_start': block.get('line_start', 0),
                    'line_end': block.get('line_end', 0),
                    **block.get('metadata', {})
                }
            }
            chunks.append(chunk)
            
        # If no blocks found, use intelligent fallback
        if not chunks:
            chunks = self._fallback_extraction(code, language)
            
        return chunks
        
    def _extract_imports(self, lines: List[str], patterns: Dict) -> List[str]:
        """Extract import/include/use statements"""
        imports = []
        import_patterns = []
        
        # Collect all import-related patterns
        for key in ['import', 'include', 'use', 'require']:
            if key in patterns:
                import_patterns.append(patterns[key])
                
        for line in lines:
            for pattern in import_patterns:
                if re.match(pattern, line):
                    imports.append(line.strip())
                    break
                    
        return imports
        
    def _extract_globals(self, lines: List[str], patterns: Dict) -> List[str]:
        """Extract global variables, constants, and type definitions"""
        globals_context = []
        
        # Pattern for common global definitions
        global_patterns = [
            r'^\s*const\s+[A-Z_][A-Z0-9_]*\s*=',  # Constants
            r'^\s*[A-Z_][A-Z0-9_]*\s*=',  # Python constants
            r'^\s*type\s+\w+\s*=',  # Type aliases
            r'^\s*typedef\s+',  # C/C++ typedefs
        ]
        
        for line in lines:
            for pattern in global_patterns:
                if re.match(pattern, line):
                    globals_context.append(line.strip())
                    break
                    
        return globals_context
        
    def _extract_code_blocks(self, lines: List[str], patterns: Dict, language: str) -> List[Dict]:
        """Extract functions, classes, and other code blocks"""
        blocks = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for class/struct/interface
            for block_type in ['class', 'struct', 'interface', 'trait', 'enum']:
                if block_type in patterns:
                    match = re.match(patterns[block_type], line)
                    if match:
                        block = self._extract_block(lines, i, patterns, block_type)
                        if block:
                            blocks.append(block)
                            
                            # Extract methods from class if applicable
                            if block_type in ['class', 'struct', 'impl']:
                                methods = self._extract_methods_from_block(
                                    block['content'], patterns, block.get('name', 'Unknown')
                                )
                                blocks.extend(methods)
                        i = block.get('line_end', i) if block else i
                        break
                        
            # Check for functions
            if 'function' in patterns:
                match = re.match(patterns['function'], line)
                if match:
                    block = self._extract_block(lines, i, patterns, 'function')
                    if block:
                        blocks.append(block)
                        i = block.get('line_end', i)
                        
            i += 1
            
        return blocks
        
    def _extract_block(self, lines: List[str], start_idx: int, patterns: Dict, block_type: str) -> Dict:
        """Extract a complete code block (function, class, etc.)"""
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        block_lines = [lines[start_idx]]
        
        # Get block markers
        block_start = patterns.get('block_start', '{')
        block_end = patterns.get('block_end', '}')
        indent_char = patterns.get('indent', '    ')
        
        # Extract block name
        name = self._extract_block_name(lines[start_idx], patterns, block_type)
        
        # Handle different block styles
        if block_start == ':':  # Python-style
            i = start_idx + 1
            while i < len(lines):
                if lines[i].strip() == '':
                    block_lines.append(lines[i])
                elif len(lines[i]) - len(lines[i].lstrip()) > indent_level:
                    block_lines.append(lines[i])
                else:
                    break
                i += 1
        else:  # Brace-style languages
            brace_count = lines[start_idx].count(block_start) - lines[start_idx].count(block_end)
            i = start_idx + 1
            
            while i < len(lines) and brace_count > 0:
                block_lines.append(lines[i])
                brace_count += lines[i].count(block_start) - lines[i].count(block_end)
                i += 1
                
        return {
            'content': '\n'.join(block_lines),
            'type': f'{block_type}',
            'name': name,
            'line_start': start_idx,
            'line_end': i,
            'metadata': {
                'block_type': block_type
            }
        }
        
    def _extract_block_name(self, line: str, patterns: Dict, block_type: str) -> str:
        """Extract the name of a code block"""
        if block_type in patterns:
            match = re.match(patterns[block_type], line)
            if match:
                # Try to extract name from groups
                for group in match.groups():
                    if group and not group.startswith('(') and not group.startswith('<'):
                        # Clean up the name
                        name = group.strip()
                        if name and not name in ['public', 'private', 'protected', 'static', 'async', 'const']:
                            return name
        return 'unknown'
        
    def _extract_methods_from_block(self, block_content: str, patterns: Dict, class_name: str) -> List[Dict]:
        """Extract individual methods from a class block"""
        methods = []
        lines = block_content.split('\n')
        
        # Skip the class definition line
        start_line = 1 if len(lines) > 1 else 0
        
        for i in range(start_line, len(lines)):
            # Check for method pattern
            if 'method' in patterns:
                match = re.match(patterns['method'], lines[i])
                if match:
                    method_block = self._extract_method_block(lines, i, patterns)
                    if method_block:
                        method_block['metadata']['parent_class'] = class_name
                        method_block['type'] = 'method'
                        methods.append(method_block)
                        
        return methods
        
    def _extract_method_block(self, lines: List[str], start_idx: int, patterns: Dict) -> Dict:
        """Extract a single method from within a class"""
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        method_lines = [lines[start_idx]]
        
        # Extract method content
        i = start_idx + 1
        while i < len(lines):
            if lines[i].strip() == '':
                method_lines.append(lines[i])
            elif len(lines[i]) - len(lines[i].lstrip()) > indent_level:
                method_lines.append(lines[i])
            else:
                # Check if next line is still part of method (for brace languages)
                if patterns.get('block_end') == '}' and '}' in lines[i]:
                    method_lines.append(lines[i])
                    i += 1
                break
            i += 1
            
        # Extract method name
        method_name = self._extract_method_name(lines[start_idx])
        
        return {
            'content': '\n'.join(method_lines),
            'name': method_name,
            'line_start': start_idx,
            'line_end': i,
            'metadata': {
                'is_method': True
            }
        }
        
    def _extract_method_name(self, line: str) -> str:
        """Extract method name from a line"""
        # Try common patterns
        patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
            r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
                
        return 'unknown_method'
        
    def _get_relevant_imports(self, code: str, imports: List[str]) -> List[str]:
        """Get imports that are actually used in the code"""
        relevant = []
        
        for imp in imports:
            # Extract identifiers from import statement
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', imp)
            
            # Check if any identifier is used in the code
            for identifier in identifiers:
                if identifier not in ['import', 'from', 'as', 'use', 'require', 'include']:
                    if identifier in code:
                        relevant.append(imp)
                        break
                        
        return list(set(relevant))
        
    def _get_relevant_globals(self, code: str, globals_context: List[str]) -> List[str]:
        """Get global variables/constants that are used in the code"""
        relevant = []
        
        for global_var in globals_context:
            # Extract the variable name
            match = re.search(r'([A-Z_][A-Z0-9_]*)\s*=', global_var)
            if match:
                var_name = match.group(1)
                if var_name in code:
                    relevant.append(global_var)
                    
        return relevant
        
    def _fallback_extraction(self, code: str, language: str) -> List[Dict]:
        """Intelligent fallback chunking when patterns don't match"""
        chunks = []
        lines = code.split('\n')
        
        # Use logical boundaries
        current_chunk = []
        current_size = 0
        max_size = 800
        min_size = 200
        
        for i, line in enumerate(lines):
            # Check for logical boundaries
            is_boundary = (
                line.strip().startswith(('def ', 'function ', 'fn ', 'func ', 'class ', 'struct ')) or
                line.strip().startswith(('public ', 'private ', 'protected ')) or
                (i > 0 and line.strip() == '' and lines[i-1].strip() == '')  # Double newline
            )
            
            if is_boundary and current_size > min_size:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'type': 'code_block',
                        'name': f'block_{len(chunks)}',
                        'metadata': {
                            'language': language,
                            'fallback': True,
                            'line_start': i - len(current_chunk),
                            'line_end': i
                        }
                    })
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
                
                # Force chunk break at max size
                if current_size >= max_size:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'type': 'code_block',
                        'name': f'block_{len(chunks)}',
                        'metadata': {
                            'language': language,
                            'fallback': True,
                            'line_start': i - len(current_chunk) + 1,
                            'line_end': i + 1
                        }
                    })
                    current_chunk = []
                    current_size = 0
                    
        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'type': 'code_block',
                'name': f'block_{len(chunks)}',
                'metadata': {
                    'language': language,
                    'fallback': True,
                    'line_start': len(lines) - len(current_chunk),
                    'line_end': len(lines)
                }
            })
            
        return chunks


class UniversalDocumentChunker:
    """Advanced document chunking with hierarchical and semantic strategies"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.header_patterns = [
            (r'^#{1}\s+(.+)$', 1),   # # Header 1
            (r'^#{2}\s+(.+)$', 2),   # ## Header 2
            (r'^#{3}\s+(.+)$', 3),   # ### Header 3
            (r'^#{4,}\s+(.+)$', 4),  # #### Header 4+
            (r'^=+$', 1),             # Underline style headers
            (r'^-+$', 2),             # Underline style headers
        ]
        
    def chunk_document(self, text: str, file_type: str) -> List[Dict]:
        """Choose appropriate chunking strategy based on file type"""
        if file_type == 'md':
            return self.hierarchical_chunk(text)
        elif file_type in ['txt', 'rst']:
            return self.semantic_chunk(text)
        elif file_type in ['json', 'yaml', 'yml', 'toml']:
            return self.config_chunk(text)
        else:
            return self.semantic_chunk(text)
            
    def hierarchical_chunk(self, text: str) -> List[Dict]:
        """Hierarchical chunking for markdown documents"""
        chunks = []
        lines = text.split('\n')
        sections = self._parse_markdown_structure(lines)
        
        for section in sections:
            # Main section chunk
            main_chunk = {
                'content': f"# {section['title']}\n\n{section['content']}",
                'type': 'hierarchical_section',
                'metadata': {
                    'hierarchy_level': section['level'],
                    'section_title': section['title']
                }
            }
            chunks.append(main_chunk)
            
            # Create sliding window chunks for long sections
            if len(section['content']) > 1000:
                sliding_chunks = self.sliding_window_chunk(
                    section['content'],
                    window_size=500,
                    overlap=100,
                    parent_title=section['title']
                )
                chunks.extend(sliding_chunks)
                
        return chunks
        
    def semantic_chunk(self, text: str, min_size=200, max_size=800, overlap=50) -> List[Dict]:
        """Semantic chunking based on paragraph similarity"""
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []
            
        # Get embeddings for all paragraphs
        embeddings = self.embedding_model.embed_documents(paragraphs)
        embeddings = np.array(embeddings)
        
        chunks = []
        current_chunk = [paragraphs[0]]
        current_embedding = embeddings[0]
        current_size = len(paragraphs[0])
        
        for i in range(1, len(paragraphs)):
            # Calculate similarity
            similarity = cosine_similarity(
                [current_embedding],
                [embeddings[i]]
            )[0][0]
            
            # Dynamic threshold based on content
            threshold = self._adjust_similarity_threshold(paragraphs[i])
            
            if similarity >= threshold and current_size + len(paragraphs[i]) <= max_size:
                current_chunk.append(paragraphs[i])
                current_size += len(paragraphs[i])
                # Update embedding (weighted average)
                weight = len(paragraphs[i]) / current_size
                current_embedding = (1 - weight) * current_embedding + weight * embeddings[i]
            else:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'type': 'semantic',
                    'metadata': {
                        'chunk_method': 'semantic_similarity',
                        'has_overlap': False
                    }
                })
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                    current_chunk = [overlap_text, paragraphs[i]]
                    current_size = len(overlap_text) + len(paragraphs[i])
                else:
                    current_chunk = [paragraphs[i]]
                    current_size = len(paragraphs[i])
                    
                current_embedding = embeddings[i]
                
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= min_size or not chunks:
                chunks.append({
                    'content': chunk_text,
                    'type': 'semantic',
                    'metadata': {
                        'chunk_method': 'semantic_similarity',
                        'has_overlap': False
                    }
                })
            elif chunks:
                # Merge with previous if too small
                chunks[-1]['content'] += '\n\n' + chunk_text
                
        return chunks
        
    def sliding_window_chunk(self, text: str, window_size=500, overlap=100, parent_title=None) -> List[Dict]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        words = text.split()
        
        if len(words) <= window_size:
            return []
            
        for i in range(0, len(words) - overlap, window_size - overlap):
            chunk_words = words[i:i + window_size]
            chunk_content = ' '.join(chunk_words)
            
            chunk = {
                'content': chunk_content,
                'type': 'sliding_window',
                'metadata': {
                    'window_position': i,
                    'window_size': window_size,
                    'overlap': overlap
                }
            }
            
            if parent_title:
                chunk['metadata']['parent_section'] = parent_title
                
            chunks.append(chunk)
            
        return chunks
        
    def config_chunk(self, text: str) -> List[Dict]:
        """Special chunking for configuration files"""
        # For config files, try to keep logical sections together
        chunks = []
        
        # Try to parse as JSON/YAML structure
        try:
            import json
            data = json.loads(text)
            # Chunk by top-level keys
            for key, value in data.items():
                chunk_content = json.dumps({key: value}, indent=2)
                chunks.append({
                    'content': chunk_content,
                    'type': 'config_section',
                    'metadata': {
                        'config_key': key,
                        'config_type': 'json'
                    }
                })
        except:
            # Fallback to line-based chunking for config files
            lines = text.split('\n')
            current_section = []
            
            for line in lines:
                if line.strip().startswith('[') or line.strip().startswith('#'):
                    # Section boundary
                    if current_section:
                        chunks.append({
                            'content': '\n'.join(current_section),
                            'type': 'config_section',
                            'metadata': {
                                'config_type': 'ini_style'
                            }
                        })
                    current_section = [line]
                else:
                    current_section.append(line)
                    
            if current_section:
                chunks.append({
                    'content': '\n'.join(current_section),
                    'type': 'config_section',
                    'metadata': {
                        'config_type': 'ini_style'
                    }
                })
                
        return chunks if chunks else self.semantic_chunk(text)
        
    def _parse_markdown_structure(self, lines: List[str]) -> List[Dict]:
        """Parse markdown into hierarchical sections"""
        sections = []
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            is_header = False
            
            # Check for header patterns
            for pattern, level in self.header_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section:
                        current_section['content'] = '\n'.join(current_content).strip()
                        sections.append(current_section)
                        
                    # Start new section
                    if pattern.startswith('^#'):
                        title = match.group(1)
                    else:
                        # Underline style - get title from previous line
                        title = lines[i-1] if i > 0 else 'Section'
                        
                    current_section = {
                        'title': title,
                        'level': level,
                        'content': ''
                    }
                    current_content = []
                    is_header = True
                    break
                    
            if not is_header and current_section:
                current_content.append(line)
            elif not is_header and not current_section:
                # Content before first header
                if not sections:
                    current_section = {
                        'title': 'Introduction',
                        'level': 0,
                        'content': ''
                    }
                    current_content = [line]
                    
        # Add final section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
            
        return sections
        
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Process each paragraph
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Split very long paragraphs
            if len(para) > 1000:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = []
                current_len = 0
                
                for sent in sentences:
                    if current_len + len(sent) > 500:
                        if current:
                            result.append(' '.join(current))
                        current = [sent]
                        current_len = len(sent)
                    else:
                        current.append(sent)
                        current_len += len(sent)
                        
                if current:
                    result.append(' '.join(current))
            else:
                result.append(para)
                
        return result
        
    def _adjust_similarity_threshold(self, text: str, base=0.75) -> float:
        """Adjust similarity threshold based on content type"""
        # Lower threshold for code blocks
        if '```' in text or 'def ' in text or 'class ' in text:
            return base * 0.9
            
        # Higher threshold for very short text
        if len(text) < 100:
            return base * 1.1
            
        return base


class GitignoreParser:
    """Parse and apply .gitignore rules"""
    
    def __init__(self, gitignore_path: Path):
        self.patterns = []
        self.base_dir = gitignore_path.parent
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if line.startswith('/'):
                            pattern = line[1:]
                        else:
                            pattern = '**/' + line if not line.startswith('*') else line
                        self.patterns.append(pattern)
                        
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        try:
            rel_path = path.relative_to(self.base_dir)
        except ValueError:
            return False
            
        path_str = str(rel_path).replace('\\', '/')
        
        for pattern in self.patterns:
            if fnmatch(path_str, pattern):
                return True
            # Check parent directories
            if '/' in path_str:
                parts = path_str.split('/')
                for i in range(len(parts)):
                    partial = '/'.join(parts[:i+1])
                    if fnmatch(partial, pattern.rstrip('/')):
                        return True
        return False


class UniversalIndexer:
    """Main indexer class orchestrating the entire indexing pipeline"""
    
    def __init__(self,
                 root_dir: str = ".",
                 db_dir: str = "./chroma_db_universal",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the universal indexer"""
        self.root_dir = Path(root_dir).resolve()
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        
        # Initialize components
        self.code_parser = UniversalCodeParser()
        self.gitignore_parser = GitignoreParser(self.root_dir / '.gitignore')
        
        # Stats tracking
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "code_files": 0,
            "doc_files": 0,
            "config_files": 0,
            "languages": defaultdict(int),
            "chunk_types": defaultdict(int),
            "processing_time": 0,
            "errors": []
        }
        
    def cleanup(self):
        """Cleanup resources with proper garbage collection"""
        try:
            if hasattr(self, 'embeddings'):
                del self.embeddings
            if hasattr(self, 'document_chunker'):
                del self.document_chunker
            if hasattr(self, 'vector_db'):
                del self.vector_db
            gc.collect()
            print("[OK] Resources cleaned up")
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
        
    def initialize_embeddings(self):
        """Initialize embeddings and chunkers"""
        print(f"Initializing {self.model_name}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        # Initialize document chunker
        self.document_chunker = UniversalDocumentChunker(self.embeddings)
        
        # Test embedding
        test_embedding = self.embeddings.embed_query("test")
        dimensions = len(test_embedding)
        print(f"[OK] Model loaded: {dimensions} dimensions")
        
        return dimensions
        
    def should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed"""
        # Skip vectors directory itself
        if 'vectors' in file_path.parts:
            return False
            
        # Skip binary files
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.db', '.sqlite',
                            '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz'}
        if file_path.suffix.lower() in binary_extensions:
            return False
            
        # Skip files in gitignore
        if self.gitignore_parser.should_ignore(file_path):
            return False
            
        # Supported extensions
        valid_extensions = {
            # Documentation
            '.md', '.txt', '.rst', '.markdown',
            # Code
            '.py', '.rs', '.js', '.jsx', '.ts', '.tsx', '.go', '.java', '.c', '.cpp', '.cc',
            '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.m',
            # Config
            '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.properties', '.xml'
        }
        
        return file_path.suffix.lower() in valid_extensions
        
    def process_code_file(self, file_path: Path) -> List[Document]:
        """Process code file with universal parser"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Detect language
            language = self.code_parser.detect_language(file_path)
            self.stats['languages'][language] += 1
            
            # Extract code blocks
            chunks = self.code_parser.extract_code_blocks(content, language)
            
            # Create documents
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        "source": str(file_path),
                        "relative_path": str(file_path.relative_to(self.root_dir)),
                        "file_type": file_path.suffix[1:],
                        "language": language,
                        "chunk_type": chunk['type'],
                        "chunk_name": chunk.get('name', 'unknown'),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **chunk.get('metadata', {})
                    }
                )
                documents.append(doc)
                self.stats['chunk_types'][chunk['type']] += 1
                
        except Exception as e:
            self.stats['errors'].append(f"Error processing {file_path}: {str(e)}")
            print(f"  Warning: Could not process {file_path}: {e}")
            
        return documents
        
    def process_document_file(self, file_path: Path) -> List[Document]:
        """Process documentation file"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            file_type = file_path.suffix[1:]
            
            # Get chunks using appropriate strategy
            chunks = self.document_chunker.chunk_document(content, file_type)
            
            # Create documents
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        "source": str(file_path),
                        "relative_path": str(file_path.relative_to(self.root_dir)),
                        "file_type": file_type,
                        "chunk_type": chunk['type'],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **chunk.get('metadata', {})
                    }
                )
                documents.append(doc)
                self.stats['chunk_types'][chunk['type']] += 1
                
        except Exception as e:
            self.stats['errors'].append(f"Error processing {file_path}: {str(e)}")
            print(f"  Warning: Could not process {file_path}: {e}")
            
        return documents
        
    def collect_files(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """Collect all files to be indexed"""
        doc_files = []
        code_files = []
        config_files = []
        
        print("Scanning for files to index...")
        
        doc_extensions = {'.md', '.txt', '.rst', '.markdown'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.properties', '.xml'}
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and self.should_index_file(file_path):
                if file_path.suffix.lower() in doc_extensions:
                    doc_files.append(file_path)
                elif file_path.suffix.lower() in config_extensions:
                    config_files.append(file_path)
                else:
                    code_files.append(file_path)
                    
        print(f"Found {len(doc_files)} documentation files")
        print(f"Found {len(code_files)} code files")
        print(f"Found {len(config_files)} config files")
        
        return doc_files, code_files, config_files
        
    def run(self):
        """Run the universal indexing pipeline"""
        start_time = time.time()
        vector_db = None
        
        try:
            print("=" * 60)
            print("UNIVERSAL RAG INDEXER - NO EXTERNAL DEPENDENCIES")
            print("=" * 60)
            print("Features:")
            print("  - Pattern-based code extraction (no tree-sitter)")
            print("  - Multi-language support")
            print("  - Hierarchical document chunking")
            print("  - Semantic similarity chunking")
            print("  - Sliding window with overlap")
            print("  - Intelligent fallback strategies")
            print("  - Production-ready resource management")
            print("=" * 60)
            
            # Initialize embeddings
            dimensions = self.initialize_embeddings()
            
            # Collect files
            doc_files, code_files, config_files = self.collect_files()
            
            if not doc_files and not code_files and not config_files:
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
                    
            # Process files
            all_documents = []
            
            # Process documentation files
            print("\nProcessing documentation files...")
            for i, file_path in enumerate(doc_files, 1):
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(doc_files)} documentation files")
                docs = self.process_document_file(file_path)
                all_documents.extend(docs)
                self.stats["doc_files"] += 1
                
            # Process code files
            print("\nProcessing code files...")
            for i, file_path in enumerate(code_files, 1):
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(code_files)} code files")
                docs = self.process_code_file(file_path)
                all_documents.extend(docs)
                self.stats["code_files"] += 1
                
            # Process config files
            print("\nProcessing config files...")
            for i, file_path in enumerate(config_files, 1):
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(config_files)} config files")
                docs = self.process_document_file(file_path)
                all_documents.extend(docs)
                self.stats["config_files"] += 1
                
            self.stats["total_files"] = len(doc_files) + len(code_files) + len(config_files)
            self.stats["total_chunks"] = len(all_documents)
            
            print(f"\nCreating vector database with {len(all_documents)} chunks...")
            
            # Create vector database with batching
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
                            "hnsw:construction_ef": 128,
                            "hnsw:M": 32
                        }
                    )
                else:
                    vector_db.add_documents(batch)
                    
                if (i + batch_size) % 500 == 0:
                    print(f"  Added {min(i + batch_size, len(all_documents))}/{len(all_documents)} chunks")
                    
            # Persist database
            if vector_db:
                vector_db.persist()
                
            # Calculate processing time
            self.stats["processing_time"] = time.time() - start_time
            
            # Save metadata
            metadata = {
                "version": "universal_1.0",
                "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": dict(self.stats),
                "languages": dict(self.stats['languages']),
                "chunk_types": dict(self.stats['chunk_types']),
                "root_directory": str(self.root_dir),
                "db_directory": str(self.db_dir),
                "embedding_model": self.model_name,
                "embedding_dimensions": dimensions,
                "indexing_strategy": {
                    "code": "Pattern-based universal extraction",
                    "documentation": "Hierarchical + Semantic chunking",
                    "config": "Structure-aware chunking",
                    "fallback": "Intelligent boundary detection"
                },
                "features": [
                    "No external parser dependencies",
                    "Multi-language support",
                    "Context preservation",
                    "Sliding window with overlap",
                    "Method-level granularity",
                    "Production-ready resource management"
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
            print(f"  Config files: {self.stats['config_files']}")
            print(f"Total chunks: {self.stats['total_chunks']}")
            print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            print(f"Chunks/second: {self.stats['total_chunks'] / self.stats['processing_time']:.1f}")
            
            # Language breakdown
            if self.stats['languages']:
                print("\nLanguages detected:")
                for lang, count in sorted(self.stats['languages'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {lang}: {count} files")
                    
            # Chunk type breakdown
            if self.stats['chunk_types']:
                print("\nChunk types created:")
                for chunk_type, count in sorted(self.stats['chunk_types'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {chunk_type}: {count}")
                    
            # Errors if any
            if self.stats['errors']:
                print(f"\nWarnings/Errors: {len(self.stats['errors'])} files had issues")
                
            print(f"\nDatabase location: {self.db_dir}")
            print("\nUse query_universal.py to search the indexed content")
            
            return True
            
        except Exception as e:
            print(f"\nError during indexing: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Always cleanup
            self.cleanup()


@click.command()
@click.option('--root-dir', '-r', default="..", help='Root directory to index')
@click.option('--db-dir', '-o', default="./chroma_db_universal", help='Output database directory')
@click.option('--model', '-m', default="sentence-transformers/all-MiniLM-L6-v2", help='Embedding model')
def main(root_dir: str, db_dir: str, model: str):
    """Universal RAG Indexer - Works with ANY codebase without external dependencies"""
    
    # Change to vectors directory for database output
    vectors_dir = Path(".")
    if vectors_dir.name == "vectors":
        db_path = Path(db_dir)
    else:
        db_path = Path("vectors") / Path(db_dir).name if Path("vectors").exists() else Path(db_dir)
    
    indexer = UniversalIndexer(
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