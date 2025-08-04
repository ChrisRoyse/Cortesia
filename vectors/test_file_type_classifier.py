#!/usr/bin/env python3
"""
Comprehensive Test Suite for File Type Classifier
=================================================

Tests the FileTypeClassifier for 100% accuracy in file type detection.
Validates all supported file types, edge cases, and performance requirements.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from file_type_classifier import FileTypeClassifier, FileType, FileClassification, create_file_classifier


class TestFileTypeClassifier:
    """Comprehensive test suite for FileTypeClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create a fresh classifier instance for each test"""
        return create_file_classifier()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_code_file_classification_accuracy(self, classifier):
        """Test 100% accuracy for all supported code file types"""
        
        # Test all supported code extensions
        code_tests = [
            # Systems programming
            ('main.rs', FileType.CODE, 'rust'),
            ('hello.c', FileType.CODE, 'c'),
            ('app.cpp', FileType.CODE, 'cpp'),
            ('service.go', FileType.CODE, 'go'),
            
            # Web development
            ('app.js', FileType.CODE, 'javascript'),
            ('component.jsx', FileType.CODE, 'javascript_react'),
            ('types.ts', FileType.CODE, 'typescript'),
            ('page.tsx', FileType.CODE, 'typescript_react'),
            ('index.html', FileType.CODE, 'html'),
            ('styles.css', FileType.CODE, 'css'),
            
            # Backend languages
            ('script.py', FileType.CODE, 'python'),
            ('Application.java', FileType.CODE, 'java'),
            ('service.kt', FileType.CODE, 'kotlin'),
            ('server.rb', FileType.CODE, 'ruby'),
            ('api.php', FileType.CODE, 'php'),
            ('Program.cs', FileType.CODE, 'csharp'),
            
            # Shell scripting
            ('deploy.sh', FileType.CODE, 'bash'),
            ('setup.bash', FileType.CODE, 'bash'),
            ('install.ps1', FileType.CODE, 'powershell'),
            ('build.bat', FileType.CODE, 'batch'),
            
            # Data science
            ('analysis.r', FileType.CODE, 'r'),
            ('model.R', FileType.CODE, 'r'),
            ('notebook.ipynb', FileType.CODE, 'jupyter_notebook'),
            
            # Mobile
            ('ViewController.swift', FileType.CODE, 'swift'),
            ('widget.dart', FileType.CODE, 'dart'),
            
            # Database
            ('schema.sql', FileType.CODE, 'sql'),
        ]
        
        for filename, expected_type, expected_language in code_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == expected_type, f"Failed for {filename}: got {result.file_type}, expected {expected_type}"
            assert result.language == expected_language, f"Failed language for {filename}: got {result.language}, expected {expected_language}"
            assert result.confidence == 1.0, f"Confidence not 100% for {filename}: got {result.confidence}"
            assert result.detected_by == "extension", f"Wrong detection method for {filename}: got {result.detected_by}"
    
    def test_documentation_file_classification_accuracy(self, classifier):
        """Test 100% accuracy for all supported documentation file types"""
        
        doc_tests = [
            ('README.md', FileType.DOCUMENTATION, 'markdown'),
            ('CHANGELOG.markdown', FileType.DOCUMENTATION, 'markdown'),
            ('docs.rst', FileType.DOCUMENTATION, 'restructuredtext'),
            ('notes.txt', FileType.DOCUMENTATION, 'plain_text'),
            ('paper.tex', FileType.DOCUMENTATION, 'latex'),
            ('guide.adoc', FileType.DOCUMENTATION, 'asciidoc'),
            ('TODO.org', FileType.DOCUMENTATION, 'org_mode'),
        ]
        
        for filename, expected_type, expected_format in doc_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == expected_type, f"Failed for {filename}"
            assert result.language == expected_format, f"Failed format for {filename}"
            assert result.confidence == 1.0, f"Confidence not 100% for {filename}"
    
    def test_config_file_classification_accuracy(self, classifier):
        """Test 100% accuracy for all supported config file types"""
        
        config_tests = [
            ('package.json', FileType.CONFIG, 'json'),
            ('docker-compose.yaml', FileType.CONFIG, 'yaml'),
            ('config.yml', FileType.CONFIG, 'yaml'),
            ('Cargo.toml', FileType.CONFIG, 'toml'),
            ('settings.ini', FileType.CONFIG, 'ini'),
            ('app.conf', FileType.CONFIG, 'config'),
            ('database.properties', FileType.CONFIG, 'properties'),
            ('schema.xml', FileType.CONFIG, 'xml'),
            ('.env', FileType.CONFIG, 'environment'),
        ]
        
        for filename, expected_type, expected_format in config_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == expected_type, f"Failed for {filename}"
            assert result.language == expected_format, f"Failed format for {filename}"
            assert result.confidence == 1.0, f"Confidence not 100% for {filename}"
    
    def test_binary_file_classification(self, classifier):
        """Test correct identification of binary files"""
        
        binary_tests = [
            'image.jpg', 'photo.png', 'icon.svg', 'video.mp4', 
            'archive.zip', 'program.exe', 'library.dll', 'data.db'
        ]
        
        for filename in binary_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == FileType.BINARY, f"Failed to identify binary file: {filename}"
            assert result.language is None, f"Binary file should have no language: {filename}"
    
    def test_fallback_classification_by_filename_patterns(self, classifier):
        """Test fallback classification using filename patterns"""
        
        pattern_tests = [
            # Config files by name
            ('Makefile', FileType.CONFIG),
            ('Dockerfile', FileType.CONFIG),
            ('requirements.txt', FileType.CONFIG),
            ('.gitignore', FileType.CONFIG),
            
            # Documentation by name  
            ('README', FileType.DOCUMENTATION),
            ('CHANGELOG', FileType.DOCUMENTATION),
            ('LICENSE', FileType.DOCUMENTATION),
            ('AUTHORS', FileType.DOCUMENTATION),
        ]
        
        for filename, expected_type in pattern_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == expected_type, f"Failed pattern match for {filename}"
            assert result.detected_by in ["filename_pattern"], f"Wrong detection method for {filename}"
            assert result.confidence >= 0.8, f"Low confidence for pattern match: {filename}"
    
    def test_unknown_file_handling(self, classifier):
        """Test handling of completely unknown file types"""
        
        unknown_tests = ['file.xyz', 'data.unknown', 'test.weird']
        
        for filename in unknown_tests:
            result = classifier.classify_file(Path(filename))
            # Should either be classified by fallback rules or marked as unknown
            assert result.file_type in [FileType.UNKNOWN, FileType.DOCUMENTATION], f"Unexpected classification for {filename}"
    
    def test_case_insensitive_extensions(self, classifier):
        """Test that extension matching is case-insensitive"""
        
        case_tests = [
            ('File.RS', FileType.CODE, 'rust'),
            ('README.MD', FileType.DOCUMENTATION, 'markdown'),
            ('Config.JSON', FileType.CONFIG, 'json'),
            ('Script.PY', FileType.CODE, 'python'),
        ]
        
        for filename, expected_type, expected_lang in case_tests:
            result = classifier.classify_file(Path(filename))
            assert result.file_type == expected_type, f"Case sensitivity failed for {filename}"
            assert result.language == expected_lang, f"Language failed for {filename}"
    
    def test_file_metadata_extraction(self, classifier, temp_dir):
        """Test that file metadata is correctly extracted"""
        
        # Create a test file
        test_file = temp_dir / "test.rs"
        test_content = "fn main() { println!(\"Hello, world!\"); }"
        test_file.write_text(test_content)
        
        result = classifier.classify_file(test_file)
        
        assert "extension" in result.metadata
        assert "filename" in result.metadata
        assert "size" in result.metadata
        assert result.metadata["extension"] == ".rs"
        assert result.metadata["filename"] == "test.rs"
        assert result.metadata["size"] > 0
    
    def test_is_indexable_method(self, classifier):
        """Test the is_indexable method for filtering"""
        
        indexable_tests = [
            ('main.rs', True),
            ('README.md', True),
            ('config.json', True),
            ('image.jpg', False),
            ('program.exe', False),
        ]
        
        for filename, should_be_indexable in indexable_tests:
            result = classifier.is_indexable(Path(filename))
            assert result == should_be_indexable, f"Indexable check failed for {filename}"
    
    def test_supported_languages_methods(self, classifier):
        """Test the getter methods for supported file types"""
        
        languages = classifier.get_supported_languages()
        assert 'rust' in languages
        assert 'python' in languages
        assert 'javascript' in languages
        assert len(languages) > 20  # Should support many languages
        
        doc_formats = classifier.get_supported_doc_formats()
        assert 'markdown' in doc_formats
        assert 'plain_text' in doc_formats
        
        config_formats = classifier.get_supported_config_formats()
        assert 'json' in config_formats
        assert 'yaml' in config_formats
    
    def test_performance_requirements(self, classifier):
        """Test that classification meets performance requirements"""
        import time
        
        # Test files
        test_files = [f"file_{i}.rs" for i in range(1000)]
        
        start_time = time.time()
        for filename in test_files:
            classifier.classify_file(Path(filename))
        end_time = time.time()
        
        # Should classify 1000 files in under 100ms
        assert (end_time - start_time) < 0.1, "Classification too slow"
    
    def test_file_stats_method(self, classifier, temp_dir):
        """Test directory statistics functionality"""
        
        # Create test files
        test_files = [
            ('main.rs', 'fn main() {}'),
            ('README.md', '# Project'),
            ('config.json', '{}'),
            ('image.jpg', b'\xff\xd8\xff'),  # Binary content
        ]
        
        for filename, content in test_files:
            file_path = temp_dir / filename
            if isinstance(content, str):
                file_path.write_text(content)
            else:
                file_path.write_bytes(content)
        
        stats = classifier.get_file_stats(temp_dir)
        
        assert stats['code'] == 1
        assert stats['documentation'] == 1
        assert stats['config'] == 1
        assert stats['binary'] == 1
        assert stats['unknown'] == 0
    
    def test_edge_cases(self, classifier):
        """Test various edge cases"""
        
        # Files with no extensions
        result = classifier.classify_file(Path("Makefile"))
        assert result.file_type == FileType.CONFIG
        
        # Very long filenames
        long_name = "a" * 200 + ".py"
        result = classifier.classify_file(Path(long_name))
        assert result.file_type == FileType.CODE
        assert result.language == 'python'
        
        # Files with multiple dots
        result = classifier.classify_file(Path("file.name.with.dots.rs"))
        assert result.file_type == FileType.CODE
        assert result.language == 'rust'
    
    def test_error_handling(self, classifier):
        """Test error handling for invalid inputs"""
        
        # Non-existent file (should still classify by extension)
        result = classifier.classify_file(Path("/nonexistent/file.rs"))
        assert result.file_type == FileType.CODE
        assert result.language == 'rust'
        
        # Empty path
        result = classifier.classify_file(Path(""))
        assert result.file_type == FileType.UNKNOWN
    
    @patch('mimetypes.guess_type')
    def test_mime_type_fallback(self, mock_guess_type, classifier):
        """Test MIME type fallback functionality"""
        
        # Mock MIME type detection
        mock_guess_type.return_value = ('text/plain', None)
        
        result = classifier.classify_file(Path("unknownfile"))
        assert result.file_type == FileType.DOCUMENTATION
        assert result.detected_by == 'mime_type'


class TestIntegration:
    """Integration tests for the complete file classification system"""
    
    def test_real_codebase_classification(self):
        """Test classification on a real codebase structure"""
        classifier = create_file_classifier()
        
        # Test on the actual vectors directory
        vectors_dir = Path(".")
        
        if vectors_dir.exists():
            stats = classifier.get_file_stats(vectors_dir)
            
            # Should find some Python files
            assert stats['code'] > 0, "Should find code files in vectors directory"
            
            # Check specific files if they exist
            for filename in ['indexer_universal.py', 'query_universal.py']:
                if (vectors_dir / filename).exists():
                    result = classifier.classify_file(vectors_dir / filename)
                    assert result.file_type == FileType.CODE
                    assert result.language == 'python'
    
    def test_classification_consistency(self):
        """Test that classification is consistent across multiple calls"""
        classifier = create_file_classifier()
        
        test_file = Path("test.rs")
        
        # Classify the same file multiple times
        results = [classifier.classify_file(test_file) for _ in range(10)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.file_type == first_result.file_type
            assert result.language == first_result.language
            assert result.confidence == first_result.confidence
            assert result.detected_by == first_result.detected_by


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])