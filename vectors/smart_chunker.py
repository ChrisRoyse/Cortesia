#!/usr/bin/env python3
"""
SmartChunker - Declaration-Centric Chunking Engine
Creates chunks that preserve documentation-code relationships
Addresses the 55.7% missed documentation problem in traditional text chunking
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from ultra_reliable_core import UniversalDocumentationDetector


@dataclass
class Declaration:
    """Represents a code declaration (function, class, method, etc.)"""
    declaration_type: str  # 'function', 'class', 'method', 'struct', 'enum', etc.
    name: str
    line_number: int
    full_signature: str
    scope_start: int
    scope_end: int
    language: str
    parent_class: Optional[str] = None
    visibility: Optional[str] = None  # 'public', 'private', 'protected'


@dataclass
class SmartChunk:
    """Represents a smart chunk with preserved doc-code relationships"""
    content: str
    declaration: Optional[Declaration]
    documentation_lines: List[int] = field(default_factory=list)
    has_documentation: bool = False
    confidence: float = 0.0
    chunk_type: str = "declaration"  # 'declaration', 'standalone_doc', 'code_only'
    line_range: Tuple[int, int] = (0, 0)
    size_chars: int = 0
    relationship_preserved: bool = True


class SmartChunker:
    """Declaration-centric chunker that preserves documentation-code relationships"""
    
    def __init__(self, max_chunk_size: int = 4000, min_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.doc_detector = UniversalDocumentationDetector()
        
        # Language-specific declaration patterns (extended from ultra_reliable_core)
        self.declaration_patterns = {
            'rust': {
                'patterns': [
                    r'^\s*(pub\s+)?(struct|enum|trait|impl|mod|const|static|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    r'^\s*(pub\s+)?(async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<(]',
                ],
                'scope_markers': ['{', '}'],
                'doc_range': 20,  # Lines to search backwards for documentation
            },
            'python': {
                'patterns': [
                    r'^\s*(def|class|async\s+def)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                ],
                'scope_markers': [':', 'def ', 'class '],
                'doc_range': 15,
            },
            'javascript': {
                'patterns': [
                    r'^\s*(export\s+)?(async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                    r'^\s*(export\s+)?(default\s+)?class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                    r'^\s*const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(async\s+)?\([^)]*\)\s*=>',
                ],
                'scope_markers': ['{', '}'],
                'doc_range': 10,
            },
            'typescript': {
                'patterns': [
                    r'^\s*(export\s+)?(async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                    r'^\s*(export\s+)?(abstract\s+)?class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                    r'^\s*(export\s+)?interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                    r'^\s*(export\s+)?type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=',
                ],
                'scope_markers': ['{', '}'],
                'doc_range': 10,
            }
        }
    
    def chunk_by_declarations(self, content: str, language: str, file_path: str) -> List[SmartChunk]:
        """
        Main chunking method that creates chunks around code declarations
        while preserving documentation-code relationships
        """
        if not content.strip():
            return []
        
        lines = content.split('\n')
        declarations = self.find_declarations(lines, language)
        
        if not declarations:
            # No declarations found - create semantic chunks
            return self._create_semantic_chunks(lines, language, file_path)
        
        chunks = []
        processed_lines = set()
        
        # Process each declaration
        for declaration in declarations:
            if declaration.line_number in processed_lines:
                continue
                
            chunk = self.create_declaration_chunk(lines, declaration, language)
            if chunk and self.validate_chunk_quality(chunk):
                chunks.append(chunk)
                # Mark lines as processed
                for line_num in range(chunk.line_range[0], chunk.line_range[1] + 1):
                    processed_lines.add(line_num)
        
        # Handle remaining unprocessed lines
        remaining_chunks = self._process_remaining_lines(lines, processed_lines, language, file_path)
        chunks.extend(remaining_chunks)
        
        return self._merge_small_chunks(chunks)
    
    def find_declarations(self, lines: List[str], language: str) -> List[Declaration]:
        """Identify code declarations in the content"""
        declarations = []
        lang_config = self.declaration_patterns.get(language, {})
        patterns = lang_config.get('patterns', [])
        
        if not patterns:
            return declarations
        
        current_class = None
        
        # First pass: find all potential declarations
        potential_declarations = []
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    declaration_type = self._extract_declaration_type(line, language)
                    name = self._extract_declaration_name(match, declaration_type)
                    potential_declarations.append((i, declaration_type, name, line.strip()))
                    break
        
        # Second pass: determine scopes and create declarations
        for i, declaration_type, name, signature in potential_declarations:
            scope_start, scope_end = self._find_declaration_scope(lines, i, language)
            
            # Track class context for methods
            if declaration_type == 'class':
                current_class = name
            
            parent_class = current_class if declaration_type == 'method' else None
            
            declaration = Declaration(
                declaration_type=declaration_type,
                name=name,
                line_number=i,
                full_signature=signature,
                scope_start=scope_start,
                scope_end=scope_end,
                language=language,
                parent_class=parent_class,
                visibility=self._extract_visibility(signature, language)
            )
            
            declarations.append(declaration)
        
        # Sort declarations by line number for consistent processing
        declarations.sort(key=lambda d: d.line_number)
        return declarations
    
    def create_declaration_chunk(self, lines: List[str], declaration: Declaration, language: str) -> Optional[SmartChunk]:
        """Create a chunk for a single declaration with its documentation"""
        
        # Find documentation using the ultra-reliable detector
        doc_detection = self.doc_detector.detect_documentation_multi_pass(
            '\n'.join(lines), language, declaration.line_number
        )
        
        # Determine chunk boundaries
        chunk_start = declaration.line_number
        chunk_end = declaration.scope_end
        
        # Find documentation that's actually associated with THIS declaration
        # Look for documentation in a reasonable range before the declaration
        doc_search_start = max(0, declaration.line_number - 15)
        associated_doc_lines = []
        
        if doc_detection['has_documentation'] and doc_detection['documentation_lines']:
            # Filter documentation lines to only those that are reasonably close to this declaration
            for doc_line in doc_detection['documentation_lines']:
                # Only include documentation that's close to and before this declaration
                if (doc_search_start <= doc_line < declaration.line_number):
                    associated_doc_lines.append(doc_line)
                # Also include documentation that's within the first few lines after declaration start
                # (for inline documentation)
                elif (declaration.line_number <= doc_line <= declaration.line_number + 2):
                    associated_doc_lines.append(doc_line)
        
        # Include associated documentation if found
        if associated_doc_lines:
            doc_start = min(associated_doc_lines)
            
            # Only extend backwards if the documentation is immediately before the declaration
            # Check for gaps - if there's a significant gap, don't include the documentation
            gap_found = False
            for line_num in range(max(associated_doc_lines) + 1, declaration.line_number):
                if line_num < len(lines):
                    line_content = lines[line_num].strip()
                    if line_content and not self._is_documentation_line(lines[line_num], language):
                        # Check if this is a language-specific annotation or attribute that's part of the declaration
                        if self._is_declaration_annotation(line_content, language):
                            continue  # This is part of the declaration, not a gap
                        
                        # Special handling for JavaScript - */ is part of JSDoc block
                        if language in ['javascript', 'typescript'] and line_content == '*/':
                            continue  # This is the end of a JSDoc block, not a gap
                        
                        # Found non-empty, non-documentation line between doc and declaration
                        gap_found = True
                        break
            
            if not gap_found:
                chunk_start = min(chunk_start, doc_start)
                # Find the actual start of the documentation block
                chunk_start = self._find_documentation_start(lines, chunk_start, language)
        
        # Ensure minimum viable chunk but don't extend too far
        if chunk_end - chunk_start < 3:
            chunk_end = min(len(lines) - 1, chunk_start + 5)
        
        # Extract chunk content
        chunk_lines = lines[chunk_start:chunk_end + 1]
        content = '\n'.join(chunk_lines)
        
        # Apply size constraints
        if len(content) > self.max_chunk_size:
            content, chunk_end = self._trim_chunk_to_size(lines, chunk_start, chunk_end)
        
        # Update associated_doc_lines to reflect actual lines in the chunk
        final_doc_lines = [line_num for line_num in associated_doc_lines if chunk_start <= line_num <= chunk_end]
        
        chunk = SmartChunk(
            content=content,
            declaration=declaration,
            documentation_lines=final_doc_lines,
            has_documentation=len(final_doc_lines) > 0,
            confidence=doc_detection['confidence'] if final_doc_lines else 0.0,
            chunk_type="declaration",
            line_range=(chunk_start, chunk_end),
            size_chars=len(content),
            relationship_preserved=len(final_doc_lines) > 0
        )
        
        return chunk
    
    def validate_chunk_quality(self, chunk: SmartChunk) -> bool:
        """Ensure chunk preserves doc-code relationships and meets quality standards"""
        
        # Content validation - must have some content
        if not chunk.content.strip():
            return False
        
        # Size validation - be more flexible for declaration chunks
        if chunk.chunk_type == "declaration" and chunk.declaration:
            # For declaration chunks, allow smaller sizes if they contain a complete declaration
            min_size = max(50, self.min_chunk_size // 4)  # Much more flexible minimum
            
            # Even more flexibility for very small but meaningful declarations
            # Check if it's a type alias, constant, or other brief but complete declaration
            if (chunk.declaration.declaration_type in ['type', 'const', 'constant', 'enum'] or
                len(chunk.content.strip().split('\n')) <= 3):  # Very brief declarations
                min_size = 30  # Allow very small meaningful declarations
        else:
            min_size = self.min_chunk_size
        
        if chunk.size_chars < min_size:
            return False
        
        if chunk.size_chars > self.max_chunk_size * 1.2:  # Allow 20% overage for relationship preservation
            return False
        
        # Documentation relationship validation
        if chunk.has_documentation:
            # Ensure documentation is actually in the chunk
            lines = chunk.content.split('\n')
            doc_found = False
            
            for line in lines:
                if self._is_documentation_line(line, chunk.declaration.language if chunk.declaration else 'unknown'):
                    doc_found = True
                    break
            
            if not doc_found:
                chunk.relationship_preserved = False
                chunk.confidence *= 0.5
        
        # Declaration validation
        if chunk.declaration:
            # Ensure declaration is in the chunk
            if chunk.declaration.full_signature not in chunk.content:
                return False
        
        # Additional quality check: ensure chunk has meaningful content
        non_empty_lines = [line.strip() for line in chunk.content.split('\n') if line.strip()]
        if len(non_empty_lines) < 2:  # At least 2 non-empty lines
            return False
        
        return True
    
    def _extract_declaration_type(self, line: str, language: str) -> str:
        """Extract the type of declaration from the line"""
        line_lower = line.strip().lower()
        
        # Language-specific type detection
        if language == 'rust':
            if 'fn ' in line_lower:
                return 'function'
            elif 'struct ' in line_lower:
                return 'struct'
            elif 'enum ' in line_lower:
                return 'enum'
            elif 'trait ' in line_lower:
                return 'trait'
            elif 'impl ' in line_lower:
                return 'impl'
            elif 'mod ' in line_lower:
                return 'module'
            elif 'const ' in line_lower or 'static ' in line_lower:
                return 'constant'
            elif 'type ' in line_lower:
                return 'type'
        
        elif language in ['python']:
            if line_lower.strip().startswith('def ') or 'def ' in line_lower:
                return 'function'
            elif line_lower.strip().startswith('class '):
                return 'class'
            elif line_lower.strip().startswith('async def '):
                return 'async_function'
        
        elif language in ['javascript', 'typescript']:
            if 'function ' in line_lower:
                return 'function'
            elif 'class ' in line_lower:
                return 'class'
            elif 'interface ' in line_lower:
                return 'interface'
            elif 'type ' in line_lower and '=' in line:
                return 'type'
            elif '=>' in line:
                return 'arrow_function'
        
        return 'unknown'
    
    def _extract_declaration_name(self, match: re.Match, declaration_type: str) -> str:
        """Extract the name of the declaration from the regex match"""
        # Try different group positions based on common patterns
        for i in range(1, min(len(match.groups()) + 1, 5)):
            group = match.group(i)
            if group and group.strip():
                # Clean up the group - remove keywords and get identifier
                clean_name = group.strip()
                
                # Skip keywords and modifiers
                keywords_to_skip = ['pub', 'async', 'const', 'let', 'var', 'export', 'default', 'function', 'class', 'struct', 'enum', 'trait', 'impl', 'interface', 'type']
                if clean_name.lower() in keywords_to_skip:
                    continue
                
                # Extract identifier from potential complex expressions
                # Look for word characters (identifier pattern)
                identifier_match = re.search(r'([a-zA-Z_$][a-zA-Z0-9_$]*)', clean_name)
                if identifier_match:
                    potential_name = identifier_match.group(1)
                    if potential_name.lower() not in keywords_to_skip:
                        return potential_name
                
                # If it looks like an identifier itself, use it
                if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', clean_name):
                    return clean_name
        
        # Fallback - try to find identifier in the last group
        if match.groups():
            last_group = match.groups()[-1] or ''
            identifier_match = re.search(r'([a-zA-Z_$][a-zA-Z0-9_$]*)', last_group)
            if identifier_match:
                return identifier_match.group(1)
        
        return 'unnamed'
    
    def _find_declaration_scope(self, lines: List[str], start_line: int, language: str) -> Tuple[int, int]:
        """Find the scope boundaries of a declaration"""
        lang_config = self.declaration_patterns.get(language, {})
        scope_markers = lang_config.get('scope_markers', ['{', '}'])
        
        if not scope_markers:
            # For languages without clear scope markers, use indentation
            return self._find_scope_by_indentation(lines, start_line, language)
        
        # Find scope using braces/markers
        start_marker = scope_markers[0] if len(scope_markers) > 0 else '{'
        end_marker = scope_markers[1] if len(scope_markers) > 1 else '}'
        
        # Find the opening marker on the same line or next few lines
        brace_count = 0
        found_start = False
        
        # Check if the opening marker is on the same line as the declaration
        if start_marker in lines[start_line]:
            found_start = True
            brace_count += lines[start_line].count(start_marker)
            brace_count -= lines[start_line].count(end_marker)
            
            # If balanced on same line, it's a one-liner
            if brace_count <= 0:
                return start_line, start_line
        
        # Look for opening marker in next few lines
        search_limit = min(len(lines), start_line + 3)
        for i in range(start_line if not found_start else start_line + 1, search_limit):
            line = lines[i]
            
            if start_marker in line:
                found_start = True
                brace_count += line.count(start_marker)
            
            if found_start and end_marker in line:
                brace_count -= line.count(end_marker)
                
                if brace_count <= 0:
                    return start_line, i
        
        # If we found start marker, continue searching for end
        if found_start:
            for i in range(search_limit, len(lines)):
                line = lines[i]
                
                if start_marker in line:
                    brace_count += line.count(start_marker)
                if end_marker in line:
                    brace_count -= line.count(end_marker)
                    
                    if brace_count <= 0:
                        return start_line, i
        
        # Fallback: limited scope based on declaration type
        declaration_type = self._extract_declaration_type(lines[start_line], language)
        if declaration_type in ['struct', 'class', 'enum', 'impl']:
            # These usually have larger scopes
            return start_line, min(len(lines) - 1, start_line + 30)
        else:
            # Functions and other declarations have smaller scopes
            return start_line, min(len(lines) - 1, start_line + 15)
    
    def _find_scope_by_indentation(self, lines: List[str], start_line: int, language: str) -> Tuple[int, int]:
        """Find scope using indentation (for Python-like languages)"""
        if start_line >= len(lines):
            return start_line, start_line
        
        base_line = lines[start_line]
        base_indent = len(base_line) - len(base_line.lstrip())
        
        # Find the end of the indented block
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # If we find a line with same or less indentation, scope ends
            if current_indent <= base_indent:
                return start_line, i - 1
        
        # Scope extends to end of file
        return start_line, len(lines) - 1
    
    def _extract_visibility(self, line: str, language: str) -> Optional[str]:
        """Extract visibility modifier (public, private, etc.)"""
        line_lower = line.strip().lower()
        
        if language == 'rust':
            if 'pub ' in line_lower:
                return 'public'
            return 'private'
        
        elif language in ['javascript', 'typescript']:
            if 'private ' in line_lower:
                return 'private'
            elif 'protected ' in line_lower:
                return 'protected'
            elif 'public ' in line_lower:
                return 'public'
            return 'public'  # Default in JS/TS
        
        elif language == 'python':
            if line_lower.strip().startswith('def __') or line_lower.strip().startswith('class __'):
                return 'private'
            elif line_lower.strip().startswith('def _') or line_lower.strip().startswith('class _'):
                return 'protected'
            return 'public'
        
        return None
    
    def _find_documentation_start(self, lines: List[str], suggested_start: int, language: str) -> int:
        """Find the actual start of documentation block"""
        # Look backwards for the start of a documentation block
        for i in range(suggested_start, max(0, suggested_start - 10), -1):
            if i >= len(lines):
                continue
                
            line = lines[i].strip()
            if not line:
                continue
            
            # Check if this is the start of a documentation block
            if self._is_documentation_start(line, language):
                return i
            
            # If we hit non-documentation, stop
            if not self._is_documentation_line(line, language):
                return suggested_start
        
        return suggested_start
    
    def _is_documentation_line(self, line: str, language: str) -> bool:
        """Check if a line is part of documentation"""
        line = line.strip()
        if not line:
            return False
        
        # Special handling for JavaScript JSDoc closing tags
        if language in ['javascript', 'typescript']:
            if line == '*/' or line.startswith('*/'):
                return True  # JSDoc closing tag is part of documentation
            if line.startswith('*') and not line.startswith('*/'):  # JSDoc content lines
                return True
        
        # Use the existing doc detector patterns
        lang_config = self.doc_detector.language_patterns.get(language, {})
        doc_patterns = lang_config.get('doc_patterns', [])
        
        for pattern in doc_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Universal patterns
        for pattern in self.doc_detector.universal_patterns['line_doc']:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_documentation_start(self, line: str, language: str) -> bool:
        """Check if a line starts a documentation block"""
        line = line.strip()
        
        # Language-specific documentation starters
        if language == 'python':
            return line.startswith('"""') or line.startswith("'''")
        elif language == 'rust':
            return line.startswith('///') or line.startswith('//!')
        elif language in ['javascript', 'typescript']:
            return line.startswith('/**')
        
        return self._is_documentation_line(line, language)
    
    def _trim_chunk_to_size(self, lines: List[str], start: int, end: int) -> Tuple[str, int]:
        """Trim chunk to fit within size constraints while preserving relationships"""
        current_size = 0
        trimmed_end = start
        
        for i in range(start, end + 1):
            if i >= len(lines):
                break
            
            line_size = len(lines[i]) + 1  # +1 for newline
            if current_size + line_size > self.max_chunk_size:
                break
            
            current_size += line_size
            trimmed_end = i
        
        content = '\n'.join(lines[start:trimmed_end + 1])
        return content, trimmed_end
    
    def _create_semantic_chunks(self, lines: List[str], language: str, file_path: str) -> List[SmartChunk]:
        """Create semantic chunks when no declarations are found"""
        chunks = []
        current_chunk_lines = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1
            
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Create chunk from accumulated lines
                content = '\n'.join(current_chunk_lines)
                chunk = SmartChunk(
                    content=content,
                    declaration=None,
                    chunk_type="semantic",
                    line_range=(i - len(current_chunk_lines), i - 1),
                    size_chars=len(content),
                    relationship_preserved=False
                )
                chunks.append(chunk)
                
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Handle remaining lines
        if current_chunk_lines:
            content = '\n'.join(current_chunk_lines)
            chunk = SmartChunk(
                content=content,
                declaration=None,
                chunk_type="semantic",
                line_range=(len(lines) - len(current_chunk_lines), len(lines) - 1),
                size_chars=len(content),
                relationship_preserved=False
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_remaining_lines(self, lines: List[str], processed_lines: Set[int], language: str, file_path: str) -> List[SmartChunk]:
        """Process lines that weren't included in declaration chunks"""
        remaining_chunks = []
        current_chunk_lines = []
        current_size = 0
        
        for i, line in enumerate(lines):
            if i in processed_lines:
                # Finalize current chunk if we have one
                if current_chunk_lines:
                    content = '\n'.join(current_chunk_lines)
                    if len(content.strip()) > self.min_chunk_size:
                        chunk = SmartChunk(
                            content=content,
                            declaration=None,
                            chunk_type="orphaned_code",
                            line_range=(i - len(current_chunk_lines), i - 1),
                            size_chars=len(content),
                            relationship_preserved=False
                        )
                        remaining_chunks.append(chunk)
                    
                    current_chunk_lines = []
                    current_size = 0
                continue
            
            line_size = len(line) + 1
            
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Create chunk
                content = '\n'.join(current_chunk_lines)
                chunk = SmartChunk(
                    content=content,
                    declaration=None,
                    chunk_type="orphaned_code",
                    line_range=(i - len(current_chunk_lines), i - 1),
                    size_chars=len(content),
                    relationship_preserved=False
                )
                remaining_chunks.append(chunk)
                
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Handle final chunk
        if current_chunk_lines:
            content = '\n'.join(current_chunk_lines)
            if len(content.strip()) > self.min_chunk_size:
                chunk = SmartChunk(
                    content=content,
                    declaration=None,
                    chunk_type="orphaned_code",
                    line_range=(len(lines) - len(current_chunk_lines), len(lines) - 1),
                    size_chars=len(content),
                    relationship_preserved=False
                )
                remaining_chunks.append(chunk)
        
        return remaining_chunks
    
    def _merge_small_chunks(self, chunks: List[SmartChunk]) -> List[SmartChunk]:
        """Merge chunks that are too small to be useful"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next
            if (current_chunk.size_chars < self.min_chunk_size and 
                i + 1 < len(chunks) and
                chunks[i + 1].size_chars < self.min_chunk_size):
                
                next_chunk = chunks[i + 1]
                
                # Only merge if they're compatible
                if self._can_merge_chunks(current_chunk, next_chunk):
                    merged_content = current_chunk.content + '\n\n' + next_chunk.content
                    
                    merged_chunk = SmartChunk(
                        content=merged_content,
                        declaration=current_chunk.declaration or next_chunk.declaration,
                        documentation_lines=current_chunk.documentation_lines + next_chunk.documentation_lines,
                        has_documentation=current_chunk.has_documentation or next_chunk.has_documentation,
                        confidence=max(current_chunk.confidence, next_chunk.confidence),
                        chunk_type="merged",
                        line_range=(current_chunk.line_range[0], next_chunk.line_range[1]),
                        size_chars=len(merged_content),
                        relationship_preserved=current_chunk.relationship_preserved and next_chunk.relationship_preserved
                    )
                    
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk since we merged it
                    continue
            
            merged_chunks.append(current_chunk)
            i += 1
        
        return merged_chunks
    
    def _is_declaration_annotation(self, line: str, language: str) -> bool:
        """Check if a line is a language-specific annotation or attribute that's part of a declaration"""
        line = line.strip()
        
        if language == 'rust':
            # Rust attributes like #[derive(...), #[cfg(...), etc.
            if line.startswith('#[') and line.endswith(']'):
                return True
        
        elif language == 'python':
            # Python decorators
            if line.startswith('@'):
                return True
        
        elif language in ['javascript', 'typescript']:
            # TypeScript/JavaScript decorators
            if line.startswith('@'):
                return True
            # Export statements that might precede declarations
            if line.startswith('export ') and not any(keyword in line for keyword in ['function', 'class', 'const', 'let', 'var']):
                return True
        
        return False
    
    def _can_merge_chunks(self, chunk1: SmartChunk, chunk2: SmartChunk) -> bool:
        """Check if two chunks can be safely merged"""
        # Don't merge if combined size would be too large
        if chunk1.size_chars + chunk2.size_chars > self.max_chunk_size:
            return False
        
        # Don't merge chunks with different declarations
        if (chunk1.declaration and chunk2.declaration and 
            chunk1.declaration.name != chunk2.declaration.name):
            return False
        
        # Don't merge if line ranges are not adjacent
        if chunk2.line_range[0] - chunk1.line_range[1] > 5:
            return False
        
        return True


# Convenience function for easy usage
def smart_chunk_content(content: str, language: str, file_path: str = "unknown", 
                       max_chunk_size: int = 4000, min_chunk_size: int = 200) -> List[SmartChunk]:
    """
    Convenience function to chunk content using SmartChunker
    
    Args:
        content: The source code content to chunk
        language: Programming language (rust, python, javascript, typescript)
        file_path: Path to the source file (for metadata)
        max_chunk_size: Maximum size per chunk in characters
        min_chunk_size: Minimum size per chunk in characters
    
    Returns:
        List of SmartChunk objects with preserved doc-code relationships
    """
    chunker = SmartChunker(max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size)
    return chunker.chunk_by_declarations(content, language, file_path)


if __name__ == "__main__":
    # Example usage and testing
    
    # Test with Rust code
    rust_code = '''/*
 * Rust Microservice for Product Recommendation Engine
 * High-performance service using Actix-web
 */

/// This struct represents a neural network
/// with multiple layers and activation functions
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

/// Represents a single layer in the network
/// Contains the number of neurons
pub struct Layer {
    neurons: u32,
}

/// Initialize a new neural network
/// with the specified number of layers
pub fn create_network(layer_count: usize) -> NeuralNetwork {
    let layers = vec![];
    NeuralNetwork { layers }
}
'''
    
    print("Testing SmartChunker with Rust code...")
    chunks = smart_chunk_content(rust_code, "rust", "test.rs")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Has documentation: {chunk.has_documentation}")
        print(f"Confidence: {chunk.confidence:.2f}")
        print(f"Size: {chunk.size_chars} chars")
        print(f"Line range: {chunk.line_range}")
        if chunk.declaration:
            print(f"Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
        print(f"Content preview: {chunk.content[:200]}...")