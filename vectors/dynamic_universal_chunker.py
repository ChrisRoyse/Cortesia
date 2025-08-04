#!/usr/bin/env python3
"""
Dynamic Universal Code Chunking Strategy
========================================

Uses pattern-based detection and structural analysis to identify semantic units
across any programming language without language-specific AST parsers.

This approach identifies universal programming constructs like:
- Functions/methods (any language)
- Classes/structs/types (any language) 
- Documentation blocks (comments, docstrings)
- Import/use statements
- Variable declarations
- Code blocks with semantic meaning

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import json

from file_type_classifier import FileType, create_file_classifier


class SemanticUnitType(Enum):
    """Universal semantic unit types found across languages"""
    FUNCTION = "function"
    CLASS = "class" 
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DOCUMENTATION = "documentation"
    CODE_BLOCK = "code_block"
    FULL_FILE = "full_file"


@dataclass
class SemanticChunk:
    """Universal semantic chunk with overlap support"""
    content: str
    unit_type: SemanticUnitType
    identifier: Optional[str]
    start_line: int
    end_line: int
    overlap_before: int = 0  # Lines of overlap before
    overlap_after: int = 0   # Lines of overlap after
    language: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UniversalPatternDetector:
    """
    Detects programming constructs using universal patterns that work
    across most programming languages without AST parsing.
    """
    
    def __init__(self):
        # Universal function patterns - matches most languages
        self.function_patterns = [
            # Standard function declarations
            r'^\s*(?:pub\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|export\s+)*(?:function\s+)?(?:fn\s+)?(?:def\s+)?(?:func\s+)?(\w+)\s*\(',
            
            # Method patterns (inside classes)
            r'^\s*(?:pub\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*(\w+)\s*\([^)]*\)\s*(?:\{|:|\s*$)',
            
            # Arrow functions and lambdas
            r'^\s*(?:const\s+|let\s+|var\s+)?(\w+)\s*=\s*(?:\([^)]*\)\s*)?=>',
            
            # C-style functions
            r'^\s*(?:static\s+|inline\s+|extern\s+)*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{',
        ]
        
        # Universal class/struct/type patterns
        self.class_patterns = [
            # Class declarations
            r'^\s*(?:pub\s+|public\s+|export\s+)*(?:class\s+|struct\s+|interface\s+|trait\s+|type\s+|enum\s+)(\w+)',
            
            # Object-like patterns
            r'^\s*(\w+)\s*=\s*(?:class|struct|interface)\s*[{\(]',
        ]
        
        # Variable/constant patterns
        self.variable_patterns = [
            # Variable declarations
            r'^\s*(?:const\s+|let\s+|var\s+|val\s+|mut\s+|static\s+|final\s+)(\w+)\s*[=:]',
            
            # Field declarations
            r'^\s*(?:pub\s+|public\s+|private\s+|protected\s+)*(\w+)\s*:\s*\w+',
        ]
        
        # Import/use patterns
        self.import_patterns = [
            r'^\s*(?:import\s+|use\s+|from\s+|#include\s+|require\s*\()',
            r'^\s*(?:extern\s+crate|mod\s+)',
        ]
        
        # Documentation patterns
        self.doc_patterns = [
            # Multi-line comments
            r'/\*[\s\S]*?\*/',
            r'"""[\s\S]*?"""',
            r"'''[\s\S]*?'''",
            
            # Single-line doc comments
            r'^\s*//[/!].*$',
            r'^\s*#.*$',
            r'^\s*///.*$',
            r'^\s*\*.*$',
        ]
        
        # Code block indicators
        self.block_indicators = ['{', '}', 'begin', 'end', 'do', 'done', 'if', 'else', 'elif', 'endif']
        
        # Brace styles for different languages
        self.brace_styles = {
            'curly': ['{', '}'],        # C, Java, JS, Rust, etc.
            'indentation': [':', ''],    # Python
            'keywords': ['begin', 'end'], # Ruby, some others
        }
    
    def detect_semantic_units(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Detect all semantic units in the content using universal patterns.
        Returns list of detected units with their boundaries.
        """
        lines = content.split('\n')
        units = []
        
        # Detect functions
        units.extend(self._detect_functions(content, lines, language))
        
        # Detect classes/structs
        units.extend(self._detect_classes(content, lines, language))
        
        # Detect impl blocks (Rust-style)
        units.extend(self._detect_impl_blocks(content, lines, language))
        
        # Detect variables
        units.extend(self._detect_variables(content, lines, language))
        
        # Detect imports
        units.extend(self._detect_imports(content, lines, language))
        
        # Detect documentation blocks
        units.extend(self._detect_documentation(content, lines, language))
        
        # Sort by start line
        units.sort(key=lambda x: x['start_line'])
        
        return units
    
    def _detect_functions(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect function/method definitions"""
        functions = []
        
        for pattern in self.function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                function_name = match.group(1)
                
                # Find the end of the function
                end_line = self._find_block_end(lines, start_line - 1, language)
                
                # Extract any documentation above the function
                doc_start = self._find_preceding_docs(lines, start_line - 1)
                
                functions.append({
                    'type': SemanticUnitType.FUNCTION,
                    'identifier': function_name,
                    'start_line': doc_start,
                    'end_line': end_line,
                    'signature': lines[start_line - 1].strip(),
                    'has_docs': doc_start < start_line
                })
        
        return functions
    
    def _detect_classes(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect class/struct/type definitions"""
        classes = []
        
        for pattern in self.class_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                class_name = match.group(1)
                
                # Find the end of the class
                end_line = self._find_block_end(lines, start_line - 1, language)
                
                # Find any documentation above
                doc_start = self._find_preceding_docs(lines, start_line - 1)
                
                classes.append({
                    'type': SemanticUnitType.CLASS,
                    'identifier': class_name,
                    'start_line': doc_start,
                    'end_line': end_line,
                    'signature': lines[start_line - 1].strip(),
                    'has_docs': doc_start < start_line
                })
        
        return classes
    
    def _detect_impl_blocks(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect impl blocks (primarily for Rust)"""
        impl_blocks = []
        
        impl_pattern = r'^\s*impl\s+(?:<[^>]*>\s+)?(\w+)'
        
        for match in re.finditer(impl_pattern, content, re.MULTILINE):
            start_line = content[:match.start()].count('\n') + 1
            impl_target = match.group(1) if match.groups() else "unknown"
            
            # Find the end of the impl block
            end_line = self._find_block_end(lines, start_line - 1, language)
            
            # Find any documentation above
            doc_start = self._find_preceding_docs(lines, start_line - 1)
            
            impl_blocks.append({
                'type': SemanticUnitType.CLASS,  # Treat impl as class-like for semantic purposes
                'identifier': f"impl_{impl_target}",
                'start_line': doc_start,
                'end_line': end_line,
                'signature': lines[start_line - 1].strip(),
                'has_docs': doc_start < start_line,
                'impl_target': impl_target
            })
        
        return impl_blocks
    
    def _detect_variables(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect variable/constant declarations"""
        variables = []
        
        for pattern in self.variable_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                var_name = match.group(1)
                
                # Variables are typically single-line, but may have multi-line values
                end_line = start_line
                line_content = lines[start_line - 1]
                
                # Check if it's a multi-line declaration
                if not line_content.rstrip().endswith(';') and not line_content.rstrip().endswith(','):
                    end_line = self._find_statement_end(lines, start_line - 1, language)
                
                variables.append({
                    'type': SemanticUnitType.VARIABLE,
                    'identifier': var_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'signature': lines[start_line - 1].strip()
                })
        
        return variables
    
    def _detect_imports(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect import/use statements"""
        imports = []
        import_blocks = []
        current_block = None
        
        for i, line in enumerate(lines):
            is_import = any(re.match(pattern, line) for pattern in self.import_patterns)
            
            if is_import:
                if current_block is None:
                    current_block = {'start': i + 1, 'lines': []}
                current_block['lines'].append(line)
            else:
                if current_block is not None:
                    # End current import block
                    current_block['end'] = i
                    import_blocks.append(current_block)
                    current_block = None
        
        # Handle case where file ends with imports
        if current_block is not None:
            current_block['end'] = len(lines)
            import_blocks.append(current_block)
        
        # Convert import blocks to semantic units
        for block in import_blocks:
            imports.append({
                'type': SemanticUnitType.IMPORT,
                'identifier': 'imports',
                'start_line': block['start'],
                'end_line': block['end'],
                'content': '\n'.join(block['lines'])
            })
        
        return imports
    
    def _detect_documentation(self, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        """Detect standalone documentation blocks"""
        docs = []
        
        # Multi-line documentation
        for pattern in self.doc_patterns[:3]:  # Multi-line patterns
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1
                
                docs.append({
                    'type': SemanticUnitType.DOCUMENTATION,
                    'identifier': 'documentation',
                    'start_line': start_line,
                    'end_line': end_line,
                    'content': match.group(0)
                })
        
        # Consecutive single-line comments as blocks
        comment_block = None
        for i, line in enumerate(lines):
            is_comment = any(re.match(pattern, line) for pattern in self.doc_patterns[3:])
            
            if is_comment:
                if comment_block is None:
                    comment_block = {'start': i + 1, 'lines': []}
                comment_block['lines'].append(line)
            else:
                if comment_block is not None and len(comment_block['lines']) >= 2:
                    # Only consider blocks of 2+ comment lines
                    docs.append({
                        'type': SemanticUnitType.DOCUMENTATION,
                        'identifier': 'comments',
                        'start_line': comment_block['start'],
                        'end_line': i,
                        'content': '\n'.join(comment_block['lines'])
                    })
                comment_block = None
        
        return docs
    
    def _find_block_end(self, lines: List[str], start_line_idx: int, language: str) -> int:
        """
        Universal block end detection using multiple strategies:
        1. Brace counting for C-style languages
        2. Indentation for Python-style languages
        3. Keyword matching for begin/end languages
        """
        start_line = lines[start_line_idx].strip()
        
        # Strategy 1: Brace counting (most common)
        if '{' in start_line or self._is_likely_brace_language(language):
            return self._find_brace_block_end(lines, start_line_idx)
        
        # Strategy 2: Indentation-based (Python, YAML, etc.)
        elif ':' in start_line and self._is_likely_indented_language(language):
            return self._find_indented_block_end(lines, start_line_idx)
        
        # Strategy 3: Keyword-based (Ruby, some others)
        elif self._has_begin_keyword(start_line):
            return self._find_keyword_block_end(lines, start_line_idx)
        
        # Fallback: heuristic-based detection
        else:
            return self._find_heuristic_block_end(lines, start_line_idx)
    
    def _find_brace_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end using brace counting"""
        brace_count = 0
        paren_count = 0
        in_string = False
        string_char = None
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            for char in line:
                # Handle string literals
                if char in ['"', "'"] and not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and in_string:
                    in_string = False
                    string_char = None
                elif in_string:
                    continue
                
                # Count braces and parentheses
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            # If we've closed all braces and we had some
            if brace_count == 0 and i > start_idx:
                # Make sure we actually had braces
                if any('{' in lines[j] for j in range(start_idx, i + 1)):
                    return i + 1
        
        return len(lines)
    
    def _find_indented_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end using indentation level"""
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
            
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            
            # If we're back to original indentation or less, we're done
            if current_indent <= start_indent:
                return i
        
        return len(lines)
    
    def _find_keyword_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end using begin/end keywords"""
        keyword_count = 0
        begin_keywords = ['begin', 'do', 'if', 'while', 'for', 'def', 'class']
        end_keywords = ['end', 'done', 'endif', 'endwhile', 'endfor']
        
        for i in range(start_idx, len(lines)):
            line = lines[i].lower()
            
            # Count begin keywords
            for keyword in begin_keywords:
                if re.search(r'\b' + keyword + r'\b', line):
                    keyword_count += 1
            
            # Count end keywords
            for keyword in end_keywords:
                if re.search(r'\b' + keyword + r'\b', line):
                    keyword_count -= 1
            
            # If we've matched all begins with ends
            if keyword_count == 0 and i > start_idx:
                return i + 1
        
        return len(lines)
    
    def _find_heuristic_block_end(self, lines: List[str], start_idx: int) -> int:
        """Fallback heuristic for finding block end"""
        # Look for patterns that suggest end of block:
        # 1. Return to same or lesser indentation
        # 2. Empty line followed by new declaration
        # 3. Maximum reasonable function size
        
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        empty_line_count = 0
        
        for i in range(start_idx + 1, min(len(lines), start_idx + 100)):  # Max 100 lines
            line = lines[i]
            stripped = line.strip()
            
            if not stripped:
                empty_line_count += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # If back to original indentation and we've seen some content
            if current_indent <= start_indent and i > start_idx + 2:
                return i
            
            # If we see a new declaration pattern
            if any(pattern in stripped.lower() for pattern in ['def ', 'function ', 'class ', 'struct ']):
                if current_indent <= start_indent:
                    return i
            
            empty_line_count = 0
        
        return min(len(lines), start_idx + 50)  # Default max size
    
    def _find_statement_end(self, lines: List[str], start_idx: int, language: str) -> int:
        """Find end of a multi-line statement"""
        # Look for statement terminators
        terminators = [';', ',', '}', ')']
        paren_count = 0
        brace_count = 0
        
        for i in range(start_idx, min(len(lines), start_idx + 10)):  # Max 10 lines for statements
            line = lines[i]
            
            for char in line:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            # If line ends with terminator and counts are balanced
            if (line.rstrip().endswith(';') or line.rstrip().endswith(',')) and paren_count == 0 and brace_count == 0:
                return i + 1
        
        return start_idx + 1
    
    def _find_preceding_docs(self, lines: List[str], start_idx: int) -> int:
        """Find documentation that precedes a code element"""
        doc_start = start_idx + 1  # Default to no docs
        
        # Look backwards for documentation
        for i in range(start_idx - 1, max(-1, start_idx - 10), -1):
            line = lines[i].strip()
            
            if not line:  # Empty line
                continue
            
            # Check if it's a documentation line
            is_doc = any(re.match(pattern, lines[i]) for pattern in self.doc_patterns)
            
            if is_doc:
                doc_start = i + 1
            else:
                break  # Stop at first non-doc line
        
        return doc_start
    
    def _is_likely_brace_language(self, language: str) -> bool:
        """Check if language likely uses braces"""
        brace_languages = ['rust', 'javascript', 'typescript', 'java', 'c', 'cpp', 'csharp', 'go', 'kotlin', 'swift']
        return language.lower() in brace_languages
    
    def _is_likely_indented_language(self, language: str) -> bool:
        """Check if language likely uses indentation"""
        indented_languages = ['python', 'yaml', 'coffeescript']
        return language.lower() in indented_languages
    
    def _has_begin_keyword(self, line: str) -> bool:
        """Check if line has begin-style keywords"""
        begin_keywords = ['begin', 'do', 'if', 'while', 'for']
        return any(keyword in line.lower() for keyword in begin_keywords)


class DynamicUniversalChunker:
    """
    Dynamic universal chunker that creates semantic chunks with overlap
    across any programming language.
    """
    
    def __init__(self, overlap_percentage: float = 0.1):
        self.overlap_percentage = overlap_percentage
        self.file_classifier = create_file_classifier()
        self.pattern_detector = UniversalPatternDetector()
    
    def chunk_file(self, file_path: Path, content: str) -> List[SemanticChunk]:
        """
        Create semantic chunks with 10% overlap from any code file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of semantic chunks with overlap
        """
        # Classify the file
        classification = self.file_classifier.classify_file(file_path)
        language = classification.language or "unknown"
        
        if classification.file_type == FileType.CODE:
            return self._chunk_code_file(file_path, content, language)
        elif classification.file_type == FileType.DOCUMENTATION:
            return self._chunk_documentation_file(file_path, content, language)
        else:
            return self._chunk_generic_file(file_path, content, language)
    
    def _chunk_code_file(self, file_path: Path, content: str, language: str) -> List[SemanticChunk]:
        """Chunk code file using universal semantic detection"""
        lines = content.split('\n')
        chunks = []
        
        # Always include full file chunk
        chunks.append(SemanticChunk(
            content=content,
            unit_type=SemanticUnitType.FULL_FILE,
            identifier=file_path.stem,
            start_line=1,
            end_line=len(lines),
            language=language,
            metadata={"file_path": str(file_path)}
        ))
        
        # Detect semantic units
        units = self.pattern_detector.detect_semantic_units(content, language)
        
        # Create chunks with overlap
        for unit in units:
            chunk_content, overlap_before, overlap_after = self._extract_chunk_with_overlap(
                lines, unit['start_line'], unit['end_line']
            )
            
            chunks.append(SemanticChunk(
                content=chunk_content,
                unit_type=unit['type'],
                identifier=unit['identifier'],
                start_line=unit['start_line'] - overlap_before,
                end_line=unit['end_line'] + overlap_after,
                overlap_before=overlap_before,
                overlap_after=overlap_after,
                language=language,
                metadata={
                    "file_path": str(file_path),
                    "signature": unit.get('signature', ''),
                    "has_docs": unit.get('has_docs', False)
                }
            ))
        
        return chunks
    
    def _chunk_documentation_file(self, file_path: Path, content: str, language: str) -> List[SemanticChunk]:
        """Chunk documentation with semantic awareness"""
        lines = content.split('\n')
        chunks = []
        
        # Full file
        chunks.append(SemanticChunk(
            content=content,
            unit_type=SemanticUnitType.FULL_FILE,
            identifier=file_path.stem,
            start_line=1,
            end_line=len(lines),
            language=language,
            metadata={"file_path": str(file_path)}
        ))
        
        # For markdown, detect sections
        if language == "markdown":
            sections = self._detect_markdown_sections(content)
            for section in sections:
                chunk_content, overlap_before, overlap_after = self._extract_chunk_with_overlap(
                    lines, section['start_line'], section['end_line']
                )
                
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    unit_type=SemanticUnitType.DOCUMENTATION,
                    identifier=section['title'],
                    start_line=section['start_line'] - overlap_before,
                    end_line=section['end_line'] + overlap_after,
                    overlap_before=overlap_before,
                    overlap_after=overlap_after,
                    language=language,
                    metadata={
                        "file_path": str(file_path),
                        "header_level": section['level']
                    }
                ))
        
        return chunks
    
    def _chunk_generic_file(self, file_path: Path, content: str, language: str) -> List[SemanticChunk]:
        """Chunk generic files"""
        lines = content.split('\n')
        
        return [SemanticChunk(
            content=content,
            unit_type=SemanticUnitType.FULL_FILE,
            identifier=file_path.stem,
            start_line=1,
            end_line=len(lines),
            language=language,
            metadata={"file_path": str(file_path)}
        )]
    
    def _extract_chunk_with_overlap(self, lines: List[str], start_line: int, end_line: int) -> Tuple[str, int, int]:
        """
        Extract chunk content with 10% overlap before and after.
        
        Returns:
            (chunk_content, overlap_lines_before, overlap_lines_after)
        """
        total_lines = end_line - start_line + 1
        overlap_lines = max(1, int(total_lines * self.overlap_percentage))
        
        # Calculate overlap boundaries
        overlap_before = min(overlap_lines, start_line - 1)
        overlap_after = min(overlap_lines, len(lines) - end_line)
        
        # Extract content with overlap
        start_idx = start_line - 1 - overlap_before
        end_idx = end_line + overlap_after
        
        chunk_lines = lines[start_idx:end_idx]
        chunk_content = '\n'.join(chunk_lines)
        
        return chunk_content, overlap_before, overlap_after
    
    def _detect_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """Detect markdown sections by headers"""
        sections = []
        lines = content.split('\n')
        
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        current_sections = []
        
        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Close previous sections at same or higher level
                current_sections = [s for s in current_sections if s['level'] < level]
                
                # Start new section
                section = {
                    'level': level,
                    'title': title,
                    'start_line': i + 1,
                    'end_line': len(lines)  # Will be updated
                }
                
                # Update end line of previous section at this level or higher
                for prev_section in sections:
                    if prev_section['level'] >= level and prev_section['end_line'] == len(lines):
                        prev_section['end_line'] = i
                
                sections.append(section)
                current_sections.append(section)
        
        return sections
    
    def get_chunking_statistics(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get comprehensive chunking statistics"""
        stats = {
            "total_chunks": len(chunks),
            "by_unit_type": {},
            "by_language": {},
            "overlap_stats": {
                "chunks_with_overlap": 0,
                "average_overlap_before": 0,
                "average_overlap_after": 0
            },
            "size_stats": {
                "average_chunk_size": 0,
                "min_chunk_size": float('inf'),
                "max_chunk_size": 0
            }
        }
        
        total_size = 0
        total_overlap_before = 0
        total_overlap_after = 0
        chunks_with_overlap = 0
        
        for chunk in chunks:
            # Count by unit type
            unit_type = chunk.unit_type.value
            stats["by_unit_type"][unit_type] = stats["by_unit_type"].get(unit_type, 0) + 1
            
            # Count by language
            stats["by_language"][chunk.language] = stats["by_language"].get(chunk.language, 0) + 1
            
            # Size statistics
            chunk_size = len(chunk.content)
            total_size += chunk_size
            stats["size_stats"]["min_chunk_size"] = min(stats["size_stats"]["min_chunk_size"], chunk_size)
            stats["size_stats"]["max_chunk_size"] = max(stats["size_stats"]["max_chunk_size"], chunk_size)
            
            # Overlap statistics
            if chunk.overlap_before > 0 or chunk.overlap_after > 0:
                chunks_with_overlap += 1
                total_overlap_before += chunk.overlap_before
                total_overlap_after += chunk.overlap_after
        
        if len(chunks) > 0:
            stats["size_stats"]["average_chunk_size"] = total_size // len(chunks)
            
        if chunks_with_overlap > 0:
            stats["overlap_stats"]["chunks_with_overlap"] = chunks_with_overlap
            stats["overlap_stats"]["average_overlap_before"] = total_overlap_before / chunks_with_overlap
            stats["overlap_stats"]["average_overlap_after"] = total_overlap_after / chunks_with_overlap
        
        if stats["size_stats"]["min_chunk_size"] == float('inf'):
            stats["size_stats"]["min_chunk_size"] = 0
        
        return stats


def create_dynamic_universal_chunker(overlap_percentage: float = 0.1) -> DynamicUniversalChunker:
    """Create a new dynamic universal chunker"""
    return DynamicUniversalChunker(overlap_percentage)


if __name__ == "__main__":
    # Demo the dynamic universal chunker
    chunker = create_dynamic_universal_chunker()
    
    print("Dynamic Universal Code Chunking Demo")
    print("=" * 60)
    
    # Test with Rust code
    rust_code = '''
use std::collections::HashMap;

/// Spiking cortical column implementation
pub struct SpikingCorticalColumn {
    neurons: Vec<SpikingNeuron>,
    lateral_inhibition: bool,
}

impl SpikingCorticalColumn {
    /// Create new cortical column
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            lateral_inhibition: true,
        }
    }
    
    /// Process temporal patterns with lateral inhibition
    pub fn process_temporal_patterns(&mut self) -> Result<(), Error> {
        if self.lateral_inhibition {
            self.apply_lateral_inhibition();
        }
        Ok(())
    }
    
    fn apply_lateral_inhibition(&mut self) {
        // Apply inhibition to neighboring neurons
        for neuron in &mut self.neurons {
            neuron.suppress_weak_signals();
        }
    }
}
'''
    
    chunks = chunker.chunk_file(Path("neural.rs"), rust_code)
    
    print(f"Generated {len(chunks)} chunks with 10% overlap:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk.unit_type.value}: {chunk.identifier}")
        print(f"     Lines {chunk.start_line}-{chunk.end_line} (overlap: -{chunk.overlap_before}/+{chunk.overlap_after})")
        print(f"     Size: {len(chunk.content)} chars")
    
    stats = chunker.get_chunking_statistics(chunks)
    print(f"\nStatistics:")
    print(f"  Chunks with overlap: {stats['overlap_stats']['chunks_with_overlap']}")
    print(f"  Average overlap before: {stats['overlap_stats']['average_overlap_before']:.1f} lines")
    print(f"  Average overlap after: {stats['overlap_stats']['average_overlap_after']:.1f} lines")
    print(f"  Average chunk size: {stats['size_stats']['average_chunk_size']} chars")