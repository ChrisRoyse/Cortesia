#!/usr/bin/env python3
"""
Code-Specific Chunking Strategies
=================================

Implements intelligent chunking strategies for different programming languages
to preserve code structure and context while maintaining semantic coherence.

This addresses the critical issue where previous chunking treated code as 
documentation, destroying exact match capability and reducing accuracy.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from file_type_classifier import FileType, FileTypeClassifier, create_file_classifier


class ChunkType(Enum):
    """Types of code chunks"""
    FULL_FILE = "full_file"
    FUNCTION = "function"
    CLASS = "class"
    STRUCT = "struct"
    ENUM = "enum"
    TRAIT = "trait"
    IMPL = "impl"
    MODULE = "module"
    IMPORT_SECTION = "import_section"
    DOCUMENTATION_SECTION = "documentation_section"
    CODE_BLOCK = "code_block"
    HIERARCHICAL_SECTION = "hierarchical_section"


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    chunk_type: ChunkType
    chunk_index: int
    start_line: int
    end_line: int
    language: str
    identifier: Optional[str] = None  # Function name, class name, etc.
    parent_identifier: Optional[str] = None  # Parent class/module
    metadata: Dict[str, Any] = None
    context_lines: int = 0  # Number of context lines included
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RustChunkingStrategy:
    """Chunking strategy specifically for Rust code"""
    
    def __init__(self):
        self.patterns = {
            'function': re.compile(r'^(\s*)(pub\s+)?fn\s+(\w+)\s*[(<]', re.MULTILINE),
            'struct': re.compile(r'^(\s*)(pub\s+)?struct\s+(\w+)', re.MULTILINE),
            'enum': re.compile(r'^(\s*)(pub\s+)?enum\s+(\w+)', re.MULTILINE),
            'trait': re.compile(r'^(\s*)(pub\s+)?trait\s+(\w+)', re.MULTILINE),
            'impl': re.compile(r'^(\s*)impl\s+(?:<[^>]*>\s+)?(\w+)', re.MULTILINE),
            'mod': re.compile(r'^(\s*)(pub\s+)?mod\s+(\w+)', re.MULTILINE),
            'use': re.compile(r'^(\s*)use\s+', re.MULTILINE),
            'doc_comment': re.compile(r'^(\s*)///.*$', re.MULTILINE),
        }
    
    def chunk_code(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk Rust code into logical units"""
        chunks = []
        lines = content.split('\n')
        
        # First, create a full file chunk for global search
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language="rust",
            identifier=file_path.stem,
            metadata={"file_path": str(file_path)}
        ))
        
        # Find all code structures
        structures = self._find_rust_structures(content)
        
        chunk_index = 1
        for structure in structures:
            chunk_content = self._extract_chunk_content(
                lines, structure['start_line'], structure['end_line']
            )
            
            chunks.append(CodeChunk(
                content=chunk_content,
                chunk_type=ChunkType(structure['type']),
                chunk_index=chunk_index,
                start_line=structure['start_line'],
                end_line=structure['end_line'],
                language="rust",
                identifier=structure['identifier'],
                parent_identifier=structure.get('parent'),
                metadata={
                    "visibility": structure.get('visibility', 'private'),
                    "signature": structure.get('signature', ''),
                    "file_path": str(file_path)
                }
            ))
            chunk_index += 1
        
        # Add import section if exists
        import_section = self._extract_import_section(content)
        if import_section:
            chunks.append(CodeChunk(
                content=import_section,
                chunk_type=ChunkType.IMPORT_SECTION,
                chunk_index=chunk_index,
                start_line=1,
                end_line=import_section.count('\n') + 1,
                language="rust",
                identifier="imports",
                metadata={"file_path": str(file_path)}
            ))
        
        return chunks
    
    def _find_rust_structures(self, content: str) -> List[Dict[str, Any]]:
        """Find all Rust code structures"""
        structures = []
        lines = content.split('\n')
        
        # Find functions
        for match in self.patterns['function'].finditer(content):
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)
            
            visibility = "public" if match.group(2) else "private"
            function_name = match.group(3)
            
            structures.append({
                'type': 'function',
                'identifier': function_name,
                'start_line': start_line,
                'end_line': end_line,
                'visibility': visibility,
                'signature': lines[start_line - 1].strip()
            })
        
        # Find structs
        for match in self.patterns['struct'].finditer(content):
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)
            
            visibility = "public" if match.group(2) else "private"
            struct_name = match.group(3)
            
            structures.append({
                'type': 'struct',
                'identifier': struct_name,
                'start_line': start_line,
                'end_line': end_line,
                'visibility': visibility,
                'signature': lines[start_line - 1].strip()
            })
        
        # Find enums
        for match in self.patterns['enum'].finditer(content):
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)
            
            visibility = "public" if match.group(2) else "private"
            enum_name = match.group(3)
            
            structures.append({
                'type': 'enum',
                'identifier': enum_name,
                'start_line': start_line,
                'end_line': end_line,
                'visibility': visibility,
                'signature': lines[start_line - 1].strip()
            })
        
        # Find traits
        for match in self.patterns['trait'].finditer(content):
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)
            
            visibility = "public" if match.group(2) else "private"
            trait_name = match.group(3)
            
            structures.append({
                'type': 'trait',
                'identifier': trait_name,
                'start_line': start_line,
                'end_line': end_line,
                'visibility': visibility,
                'signature': lines[start_line - 1].strip()
            })
        
        # Find impl blocks
        for match in self.patterns['impl'].finditer(content):
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)
            
            impl_target = match.group(2)
            
            structures.append({
                'type': 'impl',
                'identifier': f"impl_{impl_target}",
                'start_line': start_line,
                'end_line': end_line,
                'signature': lines[start_line - 1].strip(),
                'target': impl_target
            })
        
        # Sort by start line
        structures.sort(key=lambda x: x['start_line'])
        
        return structures
    
    def _find_block_end(self, lines: List[str], start_line_idx: int) -> int:
        """Find the end of a code block starting from start_line_idx"""
        brace_count = 0
        paren_count = 0
        in_block = False
        
        for i in range(start_line_idx, len(lines)):
            line = lines[i]
            
            # Count braces and parentheses
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_block = True
                elif char == '}':
                    brace_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            # If we've closed all braces and we were in a block
            if in_block and brace_count == 0:
                return i + 1
            
            # For single-line items (like struct without body)
            if not in_block and ';' in line:
                return i + 1
        
        return len(lines)
    
    def _extract_chunk_content(self, lines: List[str], start_line: int, end_line: int, 
                              context_lines: int = 2) -> str:
        """Extract chunk content with optional context"""
        start_idx = max(0, start_line - 1 - context_lines)
        end_idx = min(len(lines), end_line + context_lines)
        
        chunk_lines = lines[start_idx:end_idx]
        
        # Add context markers if we included context
        if context_lines > 0 and start_idx < start_line - 1:
            chunk_lines.insert(0, "// ... (context)")
        if context_lines > 0 and end_idx > end_line:
            chunk_lines.append("// ... (context)")
        
        return '\n'.join(chunk_lines)
    
    def _extract_import_section(self, content: str) -> Optional[str]:
        """Extract use/import statements section"""
        import_lines = []
        
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('use ') or stripped.startswith('extern crate'):
                import_lines.append(line)
            elif stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                # Stop at first non-import, non-comment line
                break
        
        return '\n'.join(import_lines) if import_lines else None


class PythonChunkingStrategy:
    """Chunking strategy for Python code"""
    
    def chunk_code(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk Python code using AST parsing"""
        chunks = []
        
        try:
            # Parse the AST
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Full file chunk
            chunks.append(CodeChunk(
                content=content,
                chunk_type=ChunkType.FULL_FILE,
                chunk_index=0,
                start_line=1,
                end_line=len(lines),
                language="python",
                identifier=file_path.stem,
                metadata={"file_path": str(file_path)}
            ))
            
            chunk_index = 1
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk_content = self._extract_python_function(lines, node)
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        chunk_type=ChunkType.FUNCTION,
                        chunk_index=chunk_index,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        language="python",
                        identifier=node.name,
                        metadata={
                            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                            "file_path": str(file_path)
                        }
                    ))
                    chunk_index += 1
                
                elif isinstance(node, ast.ClassDef):
                    chunk_content = self._extract_python_class(lines, node)
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        chunk_type=ChunkType.CLASS,
                        chunk_index=chunk_index,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        language="python",
                        identifier=node.name,
                        metadata={
                            "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                            "file_path": str(file_path)
                        }
                    ))
                    chunk_index += 1
            
        except SyntaxError:
            # Fallback to simple line-based chunking
            chunks = self._fallback_python_chunking(content, file_path)
        
        return chunks
    
    def _extract_python_function(self, lines: List[str], node: ast.FunctionDef) -> str:
        """Extract function with some context"""
        start_line = max(0, node.lineno - 3)  # Include some context
        end_line = min(len(lines), (node.end_lineno or node.lineno) + 1)
        
        return '\n'.join(lines[start_line:end_line])
    
    def _extract_python_class(self, lines: List[str], node: ast.ClassDef) -> str:
        """Extract class with some context"""
        start_line = max(0, node.lineno - 2)
        end_line = min(len(lines), (node.end_lineno or node.lineno) + 1)
        
        return '\n'.join(lines[start_line:end_line])
    
    def _fallback_python_chunking(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback chunking when AST parsing fails"""
        chunks = []
        lines = content.split('\n')
        
        # Full file
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language="python",
            identifier=file_path.stem,
            metadata={"file_path": str(file_path), "ast_failed": True}
        ))
        
        return chunks


class GenericChunkingStrategy:
    """Generic chunking strategy for unsupported languages"""
    
    def chunk_code(self, content: str, file_path: Path, language: str) -> List[CodeChunk]:
        """Generic chunking based on patterns"""
        chunks = []
        lines = content.split('\n')
        
        # Always include full file
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language=language,
            identifier=file_path.stem,
            metadata={"file_path": str(file_path)}
        ))
        
        # Try to identify function-like structures
        function_patterns = [
            r'^\s*function\s+(\w+)',  # JavaScript
            r'^\s*def\s+(\w+)',       # Python (backup)
            r'^\s*public\s+\w+\s+(\w+)\s*\(',  # Java/C#
            r'^\s*(\w+)\s*\(',        # Generic function call pattern
        ]
        
        chunk_index = 1
        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n') + 1
                # Simple heuristic for end line
                end_line = min(len(lines), start_line + 20)  # Assume max 20 lines
                
                chunk_content = '\n'.join(lines[start_line-1:end_line])
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    chunk_type=ChunkType.FUNCTION,
                    chunk_index=chunk_index,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    identifier=match.group(1) if match.groups() else f"func_{chunk_index}",
                    metadata={"file_path": str(file_path), "pattern_match": pattern}
                ))
                chunk_index += 1
                
                if chunk_index > 50:  # Prevent too many chunks
                    break
        
        return chunks


class DocumentationChunkingStrategy:
    """Chunking strategy for documentation files"""
    
    def chunk_documentation(self, content: str, file_path: Path, format_type: str) -> List[CodeChunk]:
        """Chunk documentation by sections"""
        chunks = []
        
        if format_type == "markdown":
            chunks = self._chunk_markdown(content, file_path)
        elif format_type == "restructuredtext":
            chunks = self._chunk_rst(content, file_path)
        else:
            chunks = self._chunk_plain_text(content, file_path, format_type)
        
        return chunks
    
    def _chunk_markdown(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk markdown by headers"""
        chunks = []
        lines = content.split('\n')
        
        # Full document
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language="markdown",
            identifier=file_path.stem,
            metadata={"file_path": str(file_path)}
        ))
        
        # Find headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        sections = []
        
        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            line_num = content[:match.start()].count('\n') + 1
            
            sections.append({
                'level': level,
                'title': title,
                'line': line_num,
                'start_pos': match.start()
            })
        
        # Create chunks for each section
        chunk_index = 1
        for i, section in enumerate(sections):
            start_line = section['line']
            
            # Find end line (next section of same or higher level)
            end_line = len(lines)
            for j in range(i + 1, len(sections)):
                if sections[j]['level'] <= section['level']:
                    end_line = sections[j]['line'] - 1
                    break
            
            section_content = '\n'.join(lines[start_line-1:end_line])
            
            chunks.append(CodeChunk(
                content=section_content,
                chunk_type=ChunkType.HIERARCHICAL_SECTION,
                chunk_index=chunk_index,
                start_line=start_line,
                end_line=end_line,
                language="markdown",
                identifier=section['title'],
                metadata={
                    "header_level": section['level'],
                    "file_path": str(file_path)
                }
            ))
            chunk_index += 1
        
        return chunks
    
    def _chunk_rst(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Basic RST chunking"""
        return self._chunk_plain_text(content, file_path, "restructuredtext")
    
    def _chunk_plain_text(self, content: str, file_path: Path, format_type: str) -> List[CodeChunk]:
        """Chunk plain text by paragraphs"""
        chunks = []
        lines = content.split('\n')
        
        # Full document
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language=format_type,
            identifier=file_path.stem,
            metadata={"file_path": str(file_path)}
        ))
        
        return chunks


class CodeChunkingManager:
    """
    Main manager for code-specific chunking strategies.
    Routes different file types to appropriate chunking strategies.
    """
    
    def __init__(self):
        self.file_classifier = create_file_classifier()
        self.rust_strategy = RustChunkingStrategy()
        self.python_strategy = PythonChunkingStrategy()
        self.generic_strategy = GenericChunkingStrategy()
        self.doc_strategy = DocumentationChunkingStrategy()
    
    def chunk_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """
        Chunk a file using the appropriate strategy based on file type.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of code chunks with preserved structure and context
        """
        # Classify the file
        classification = self.file_classifier.classify_file(file_path)
        
        if classification.file_type == FileType.CODE:
            return self._chunk_code_file(file_path, content, classification.language)
        elif classification.file_type == FileType.DOCUMENTATION:
            return self._chunk_documentation_file(file_path, content, classification.language)
        elif classification.file_type == FileType.CONFIG:
            return self._chunk_config_file(file_path, content, classification.language)
        else:
            # Binary or unknown files - return single chunk
            return [CodeChunk(
                content=content,
                chunk_type=ChunkType.FULL_FILE,
                chunk_index=0,
                start_line=1,
                end_line=content.count('\n') + 1,
                language=classification.language or "unknown",
                identifier=file_path.stem,
                metadata={"file_path": str(file_path), "file_type": classification.file_type.value}
            )]
    
    def _chunk_code_file(self, file_path: Path, content: str, language: Optional[str]) -> List[CodeChunk]:
        """Chunk code files using language-specific strategies"""
        if language == "rust":
            return self.rust_strategy.chunk_code(content, file_path)
        elif language == "python":
            return self.python_strategy.chunk_code(content, file_path)
        else:
            return self.generic_strategy.chunk_code(content, file_path, language or "unknown")
    
    def _chunk_documentation_file(self, file_path: Path, content: str, format_type: Optional[str]) -> List[CodeChunk]:
        """Chunk documentation files"""
        return self.doc_strategy.chunk_documentation(content, file_path, format_type or "plain_text")
    
    def _chunk_config_file(self, file_path: Path, content: str, format_type: Optional[str]) -> List[CodeChunk]:
        """Chunk configuration files"""
        lines = content.split('\n')
        
        return [CodeChunk(
            content=content,
            chunk_type=ChunkType.FULL_FILE,
            chunk_index=0,
            start_line=1,
            end_line=len(lines),
            language=format_type or "config",
            identifier=file_path.stem,
            metadata={"file_path": str(file_path), "file_type": "config"}
        )]
    
    def get_chunking_statistics(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        stats = {
            "total_chunks": len(chunks),
            "by_type": {},
            "by_language": {},
            "average_chunk_size": 0,
            "total_lines": 0
        }
        
        total_chars = 0
        
        for chunk in chunks:
            # Count by type
            chunk_type = chunk.chunk_type.value
            stats["by_type"][chunk_type] = stats["by_type"].get(chunk_type, 0) + 1
            
            # Count by language
            language = chunk.language
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            
            # Size statistics
            total_chars += len(chunk.content)
            stats["total_lines"] += chunk.end_line - chunk.start_line + 1
        
        if len(chunks) > 0:
            stats["average_chunk_size"] = total_chars // len(chunks)
        
        return stats


# Factory function
def create_code_chunking_manager() -> CodeChunkingManager:
    """Create a new code chunking manager instance"""
    return CodeChunkingManager()


if __name__ == "__main__":
    # Demo usage
    manager = create_code_chunking_manager()
    
    print("Code-Specific Chunking Strategies Demo")
    print("=" * 50)
    
    # Test Rust code chunking
    rust_code = '''
pub struct SpikingCorticalColumn {
    neurons: Vec<SpikingNeuron>,
    lateral_connections: NetworkTopology,
}

impl SpikingCorticalColumn {
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            lateral_connections: NetworkTopology::default(),
        }
    }
    
    pub fn process_temporal_patterns(&mut self) -> Result<(), Error> {
        // Apply lateral inhibition
        self.apply_lateral_inhibition();
        Ok(())
    }
    
    fn apply_lateral_inhibition(&mut self) {
        // Implementation here
    }
}

pub fn create_neural_network() -> SpikingCorticalColumn {
    SpikingCorticalColumn::new()
}
'''
    
    chunks = manager.chunk_file(Path("neural_network.rs"), rust_code)
    
    print(f"Rust code chunked into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk.chunk_type.value}: {chunk.identifier} (lines {chunk.start_line}-{chunk.end_line})")
    
    stats = manager.get_chunking_statistics(chunks)
    print(f"\nChunking statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Average chunk size: {stats['average_chunk_size']} chars")