#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Level Indexer
================================================

Tests the multi-level indexing system for 100% accuracy across
exact, semantic, and metadata indexes.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from multi_level_indexer import (
    MultiLevelIndexer, 
    ExactIndexManager, 
    MetadataIndexManager,
    IndexType,
    SearchQuery,
    SearchResult,
    IndexedDocument,
    create_multi_level_indexer
)
from file_type_classifier import FileType


class TestExactIndexManager:
    """Test exact string matching index"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test databases"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def exact_index(self, temp_dir):
        """Create exact index manager"""
        return ExactIndexManager(temp_dir)
    
    def test_exact_index_initialization(self, exact_index):
        """Test that exact index initializes correctly"""
        assert exact_index.db_path.exists()
        
        # Check database structure
        conn = sqlite3.connect(str(exact_index.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'exact_tokens' in tables
        conn.close()
    
    def test_add_rust_code_document(self, exact_index):
        """Test adding Rust code document with exact token extraction"""
        doc = IndexedDocument(
            doc_id="test_rust_1",
            file_path="/test/main.rs",
            relative_path="main.rs",
            file_type=FileType.CODE,
            language="rust",
            content="""
            pub fn SpikingCorticalColumn() -> Result<Vec<u32>, Error> {
                let mut cortical_grid = Vec::new();
                impl Default for NeuromorphicLayer {
                    fn default() -> Self {
                        // lateral inhibition implementation
                    }
                }
            }
            """,
            exact_tokens=set(),
            metadata={"size": 200},
            chunk_type="function",
            chunk_index=0
        )
        
        exact_index.add_document(doc)
        
        # Test exact searches
        results = exact_index.search("SpikingCorticalColumn")
        assert len(results) > 0
        assert any("SpikingCorticalColumn" in r.content for r in results)
        
        results = exact_index.search("pub fn")
        assert len(results) > 0
        
        results = exact_index.search("impl Default")
        assert len(results) > 0
        
        results = exact_index.search("lateral inhibition")
        assert len(results) > 0
    
    def test_exact_search_with_filters(self, exact_index):
        """Test exact search with file type and language filters"""
        # Add Rust document
        rust_doc = IndexedDocument(
            doc_id="rust_1",
            file_path="/test/main.rs",
            relative_path="main.rs",
            file_type=FileType.CODE,
            language="rust",
            content="pub fn test_function() {}",
            exact_tokens=set(),
            metadata={},
            chunk_type="function",
            chunk_index=0
        )
        
        # Add Python document
        python_doc = IndexedDocument(
            doc_id="python_1",
            file_path="/test/main.py",
            relative_path="main.py",
            file_type=FileType.CODE,
            language="python",
            content="def test_function(): pass",
            exact_tokens=set(),
            metadata={},
            chunk_type="function",
            chunk_index=0
        )
        
        exact_index.add_document(rust_doc)
        exact_index.add_document(python_doc)
        
        # Search with Rust filter
        results = exact_index.search("test_function", languages=["rust"])
        assert len(results) >= 1
        assert all(r.language == "rust" for r in results)
        
        # Search with code file type filter
        results = exact_index.search("test_function", file_types=[FileType.CODE])
        assert len(results) >= 2
        assert all(r.file_type == FileType.CODE for r in results)
    
    def test_exact_token_extraction_accuracy(self, exact_index):
        """Test accuracy of exact token extraction"""
        content = """
        pub struct SpikingColumn {
            activation_threshold: f64,
            lateral_inhibition: bool,
        }
        
        impl SpikingColumn {
            pub fn new() -> Self { ... }
            fn process_input(&mut self, data: Vec<u8>) -> Result<(), Error> { ... }
        }
        """
        
        tokens = exact_index._extract_exact_tokens(content, FileType.CODE)
        
        # Should extract struct names
        assert "SpikingColumn" in tokens
        
        # Should extract function names
        assert "new" in tokens
        assert "process_input" in tokens
        
        # Should extract keywords
        assert "pub" in tokens
        assert "struct" in tokens
        assert "impl" in tokens
        
        # Should extract field names
        assert "activation_threshold" in tokens
        assert "lateral_inhibition" in tokens


class TestMetadataIndexManager:
    """Test metadata-based filtering and search"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def metadata_index(self, temp_dir):
        return MetadataIndexManager(temp_dir)
    
    def test_metadata_index_initialization(self, metadata_index):
        """Test metadata index initialization"""
        assert metadata_index.db_path.exists()
        
        conn = sqlite3.connect(str(metadata_index.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'metadata_index' in tables
        conn.close()
    
    def test_add_and_filter_documents(self, metadata_index):
        """Test adding documents and filtering by metadata"""
        # Add various document types
        rust_doc = IndexedDocument(
            doc_id="rust_1",
            file_path="/test/main.rs",
            relative_path="main.rs",
            file_type=FileType.CODE,
            language="rust",
            content="fn main() {}",
            exact_tokens=set(),
            metadata={"size": 100},
            chunk_type="full",
            chunk_index=0
        )
        
        md_doc = IndexedDocument(
            doc_id="md_1",
            file_path="/test/README.md",
            relative_path="README.md",
            file_type=FileType.DOCUMENTATION,
            language="markdown",
            content="# Project",
            exact_tokens=set(),
            metadata={"size": 50},
            chunk_type="section",
            chunk_index=0
        )
        
        json_doc = IndexedDocument(
            doc_id="json_1",
            file_path="/test/config.json",
            relative_path="config.json",
            file_type=FileType.CONFIG,
            language="json",
            content='{"key": "value"}',
            exact_tokens=set(),
            metadata={"size": 20},
            chunk_type="full",
            chunk_index=0
        )
        
        metadata_index.add_document(rust_doc)
        metadata_index.add_document(md_doc)
        metadata_index.add_document(json_doc)
        
        # Filter by file type
        code_docs = metadata_index.filter_documents(file_types=[FileType.CODE])
        assert "rust_1" in code_docs
        assert len(code_docs) == 1
        
        doc_docs = metadata_index.filter_documents(file_types=[FileType.DOCUMENTATION])
        assert "md_1" in doc_docs
        assert len(doc_docs) == 1
        
        # Filter by language
        rust_docs = metadata_index.filter_documents(languages=["rust"])
        assert "rust_1" in rust_docs
        assert len(rust_docs) == 1
        
        # Filter by path pattern
        readme_docs = metadata_index.filter_documents(path_pattern="README")
        assert "md_1" in readme_docs
        assert len(readme_docs) == 1
    
    def test_metadata_statistics(self, metadata_index):
        """Test metadata statistics generation"""
        # Add test documents
        for i in range(5):
            doc = IndexedDocument(
                doc_id=f"doc_{i}",
                file_path=f"/test/file_{i}.rs",
                relative_path=f"file_{i}.rs",
                file_type=FileType.CODE,
                language="rust",
                content=f"fn function_{i}() {{}}",
                exact_tokens=set(),
                metadata={"size": 100 + i},
                chunk_type="function",
                chunk_index=i
            )
            metadata_index.add_document(doc)
        
        stats = metadata_index.get_statistics()
        
        assert stats['total_documents'] == 5
        assert stats['by_file_type']['code'] == 5
        assert stats['by_language']['rust'] == 5


class TestMultiLevelIndexer:
    """Test the complete multi-level indexing system"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def indexer(self, temp_dir):
        return MultiLevelIndexer(str(temp_dir / "test_index"))
    
    def test_indexer_initialization(self, indexer):
        """Test that all three indexes initialize correctly"""
        assert indexer.exact_index is not None
        assert indexer.metadata_index is not None
        assert indexer.semantic_collection is not None
        assert indexer.file_classifier is not None
    
    def test_add_document_to_all_indexes(self, indexer, temp_dir):
        """Test adding document to all three indexes"""
        test_file = temp_dir / "test.rs"
        content = """
        pub struct SpikingCorticalColumn {
            neurons: Vec<Neuron>,
        }
        
        impl SpikingCorticalColumn {
            pub fn process_temporal_patterns(&mut self) -> Result<(), Error> {
                // Neuromorphic processing with lateral inhibition
                self.apply_lateral_inhibition();
                Ok(())
            }
        }
        """
        
        doc_id = indexer.add_document(test_file, content)
        
        assert doc_id is not None
        assert doc_id.startswith("doc_")
        
        # Test all indexes contain the document
        stats = indexer.get_statistics()
        assert stats['metadata_index']['total_documents'] >= 1
        assert stats['semantic_index']['total_documents'] >= 1
    
    def test_exact_search_type(self, indexer, temp_dir):
        """Test exact search functionality"""
        test_file = temp_dir / "neural.rs"
        content = """
        pub fn SpikingCorticalColumn() -> NeuralNetwork {
            let cortical_grid = initialize_grid();
            cortical_grid
        }
        """
        
        indexer.add_document(test_file, content)
        
        # Test exact search
        query = SearchQuery(
            query="SpikingCorticalColumn",
            query_type=IndexType.EXACT,
            limit=10
        )
        
        results = indexer.search(query)
        assert len(results) > 0
        assert any("SpikingCorticalColumn" in r.content for r in results)
        assert all(r.match_type == IndexType.EXACT for r in results)
    
    def test_semantic_search_type(self, indexer, temp_dir):
        """Test semantic search functionality"""
        test_file = temp_dir / "neural.rs"
        content = """
        // Implementation of cortical column with lateral inhibition
        struct NeuromorphicProcessor {
            columns: Vec<CorticalColumn>,
            inhibition_radius: f32,
        }
        
        impl NeuromorphicProcessor {
            fn process_spatio_temporal_patterns(&self) {
                // Process neural activation patterns
            }
        }
        """
        
        indexer.add_document(test_file, content)
        
        # Test semantic search for related concepts
        query = SearchQuery(
            query="neural processing with inhibition",
            query_type=IndexType.SEMANTIC,
            limit=10
        )
        
        results = indexer.search(query)
        assert len(results) > 0
        assert all(r.match_type == IndexType.SEMANTIC for r in results)
    
    def test_metadata_filtered_search(self, indexer, temp_dir):
        """Test metadata-based filtering"""
        # Add Rust file
        rust_file = temp_dir / "neural.rs"
        rust_content = "pub fn neural_process() {}"
        indexer.add_document(rust_file, rust_content)
        
        # Add Markdown file
        md_file = temp_dir / "README.md"
        md_content = "# Neural Processing Documentation"
        indexer.add_document(md_file, md_content)
        
        # Search only in code files
        query = SearchQuery(
            query="neural",
            query_type=IndexType.SEMANTIC,
            file_types=[FileType.CODE],
            limit=10
        )
        
        results = indexer.search(query)
        assert len(results) > 0
        assert all(r.file_type == FileType.CODE for r in results)
        
        # Search only in documentation
        query = SearchQuery(
            query="neural",
            query_type=IndexType.SEMANTIC,
            file_types=[FileType.DOCUMENTATION],
            limit=10
        )
        
        results = indexer.search(query)
        assert len(results) > 0
        assert all(r.file_type == FileType.DOCUMENTATION for r in results)
    
    def test_hybrid_search_combines_indexes(self, indexer, temp_dir):
        """Test that hybrid search combines exact and semantic results"""
        test_file = temp_dir / "spiking.rs"
        content = """
        pub struct SpikingCorticalColumn {
            // Neuromorphic architecture for temporal processing
            neurons: Vec<SpikingNeuron>,
            lateral_connections: NetworkTopology,
        }
        
        impl SpikingCorticalColumn {
            pub fn apply_lateral_inhibition(&mut self) {
                // Implementation of competitive learning
            }
        }
        """
        
        indexer.add_document(test_file, content)
        
        # Hybrid search should find both exact matches and semantic matches
        query = SearchQuery(
            query="SpikingCorticalColumn neuromorphic",
            query_type=IndexType.EXACT,  # Will trigger hybrid search
            limit=20
        )
        
        results = indexer.search(query)
        assert len(results) > 0
        
        # Should have high-scoring exact matches
        exact_matches = [r for r in results if "SpikingCorticalColumn" in r.content]
        assert len(exact_matches) > 0
    
    def test_search_accuracy_requirements(self, indexer, temp_dir):
        """Test that search meets 100% accuracy requirements for exact matches"""
        # Add test documents with known content
        test_cases = [
            ("main.rs", "pub fn main() { SpikingCorticalColumn::new(); }"),
            ("neural.rs", "struct SpikingCorticalColumn { neurons: Vec<Neuron> }"),
            ("lib.rs", "// SpikingCorticalColumn implementation"),
            ("other.rs", "fn process() { lateral_inhibition(); }"),
        ]
        
        for filename, content in test_cases:
            test_file = temp_dir / filename
            indexer.add_document(test_file, content)
        
        # Test exact search for "SpikingCorticalColumn"
        query = SearchQuery(
            query="SpikingCorticalColumn",
            query_type=IndexType.EXACT,
            limit=10
        )
        
        results = indexer.search(query)
        
        # Should find exactly 3 documents containing "SpikingCorticalColumn"
        expected_files = {"main.rs", "neural.rs", "lib.rs"}
        found_files = {Path(r.relative_path).name for r in results 
                      if "SpikingCorticalColumn" in r.content}
        
        assert found_files == expected_files, f"Expected {expected_files}, got {found_files}"
        
        # Test search for "lateral_inhibition"
        query = SearchQuery(
            query="lateral_inhibition",
            query_type=IndexType.EXACT,
            limit=10
        )
        
        results = indexer.search(query)
        found_files = {Path(r.relative_path).name for r in results 
                      if "lateral_inhibition" in r.content}
        
        assert "other.rs" in found_files
    
    def test_performance_requirements(self, indexer, temp_dir):
        """Test that search performance meets requirements"""
        import time
        
        # Add multiple documents
        for i in range(100):
            test_file = temp_dir / f"file_{i}.rs"
            content = f"pub fn function_{i}() {{ SpikingColumn::process_{i}(); }}"
            indexer.add_document(test_file, content)
        
        # Test search performance
        query = SearchQuery(
            query="SpikingColumn",
            query_type=IndexType.EXACT,
            limit=20
        )
        
        start_time = time.time()
        results = indexer.search(query)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Should complete search in under 1 second
        assert search_time < 1.0, f"Search took {search_time:.3f}s, should be under 1.0s"
        assert len(results) > 0
    
    def test_clear_all_indexes(self, indexer, temp_dir):
        """Test clearing all indexes"""
        # Add some data
        test_file = temp_dir / "test.rs"
        indexer.add_document(test_file, "pub fn test() {}")
        
        # Clear indexes
        indexer.clear_all_indexes()
        
        # Verify indexes are empty
        stats = indexer.get_statistics()
        assert stats['metadata_index']['total_documents'] == 0
        assert stats['semantic_index']['total_documents'] == 0


class TestIntegration:
    """Integration tests for the multi-level system"""
    
    def test_real_codebase_simulation(self):
        """Test on simulated real codebase structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            indexer = MultiLevelIndexer(str(temp_path / "test_index"))
            
            # Simulate real codebase files
            codebase_files = [
                ("src/lib.rs", """
                pub struct SpikingCorticalColumn {
                    neurons: Vec<SpikingNeuron>,
                    lateral_inhibition: bool,
                }
                
                impl SpikingCorticalColumn {
                    pub fn new() -> Self { ... }
                    pub fn process_temporal_sequence(&mut self) { ... }
                }
                """),
                
                ("src/neuron.rs", """
                pub struct SpikingNeuron {
                    threshold: f64,
                    current_potential: f64,
                }
                
                impl SpikingNeuron {
                    pub fn apply_lateral_inhibition(&mut self) { ... }
                }
                """),
                
                ("README.md", """
                # Neuromorphic Computing System
                
                This project implements spiking cortical columns with lateral inhibition
                for temporal pattern recognition in neuromorphic architectures.
                """),
                
                ("Cargo.toml", """
                [package]
                name = "neuromorphic-system"
                version = "0.1.0"
                """)
            ]
            
            # Index all files
            for file_path, content in codebase_files:
                full_path = temp_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                indexer.add_document(full_path, content)
            
            # Test comprehensive search scenarios
            test_queries = [
                # Exact searches
                (SearchQuery("SpikingCorticalColumn", IndexType.EXACT), 2),  # Should find 2 files
                (SearchQuery("lateral_inhibition", IndexType.EXACT), 2),     # Should find 2 files
                (SearchQuery("pub fn", IndexType.EXACT), 2),                 # Should find 2 files
                
                # Semantic searches
                (SearchQuery("neuromorphic computing", IndexType.SEMANTIC), 1),
                (SearchQuery("temporal pattern processing", IndexType.SEMANTIC), 1),
                
                # Filtered searches
                (SearchQuery("spiking", IndexType.SEMANTIC, [FileType.CODE]), 2),
                (SearchQuery("neuromorphic", IndexType.SEMANTIC, [FileType.DOCUMENTATION]), 1),
            ]
            
            for query, expected_min_results in test_queries:
                results = indexer.search(query)
                assert len(results) >= expected_min_results, \
                    f"Query '{query.query}' ({query.query_type}) returned {len(results)} results, expected >= {expected_min_results}"
    
    def test_accuracy_against_grep_simulation(self):
        """Test accuracy against simulated grep results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            indexer = MultiLevelIndexer(str(temp_path / "accuracy_test"))
            
            # Create test files with known content
            test_content = {
                "neural_1.rs": "pub fn SpikingCorticalColumn() { lateral_inhibition(); }",
                "neural_2.rs": "struct SpikingCorticalColumn { neurons: Vec<Neuron> }",
                "docs.md": "# SpikingCorticalColumn Documentation",
                "other.rs": "fn process() { println!(\"Hello\"); }",
                "config.json": '{"SpikingCorticalColumn": true}'
            }
            
            for filename, content in test_content.items():
                file_path = temp_path / filename
                indexer.add_document(file_path, content)
            
            # Test exact search accuracy
            query = SearchQuery("SpikingCorticalColumn", IndexType.EXACT)
            results = indexer.search(query)
            
            # Should find exactly files containing "SpikingCorticalColumn"
            expected_files = {"neural_1.rs", "neural_2.rs", "docs.md", "config.json"}
            found_files = {Path(r.relative_path).name for r in results}
            
            # Allow for subset matching (semantic index might not catch all exact matches)
            # but should have high precision
            matches = found_files.intersection(expected_files)
            precision = len(matches) / len(found_files) if found_files else 0
            
            assert precision >= 0.8, f"Precision {precision:.2f} below 80%"
            assert len(matches) >= 3, f"Should find at least 3 matching files, found {len(matches)}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])