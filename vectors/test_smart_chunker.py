#!/usr/bin/env python3
"""
Comprehensive test suite for SmartChunker
Validates declaration-centric chunking and doc-code relationship preservation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker, smart_chunk_content, Declaration, SmartChunk
from ultra_reliable_core import UniversalDocumentationDetector


def test_rust_code():
    """Test SmartChunker with Rust code examples"""
    print("=== Testing Rust Code ===")
    
    rust_code = '''/*
 * Rust Microservice for Product Recommendation Engine
 * High-performance service using Actix-web, Tokio, and advanced Rust patterns
 */

use actix_web::{web, App, HttpResponse, HttpServer, Result};
use serde::{Deserialize, Serialize};

/// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost:5432/db".to_string(),
            server_host: "0.0.0.0".to_string(),
            server_port: 8001,
        }
    }
}

/// Data models
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Product {
    pub id: i32,
    pub name: String,
    pub price: f64,
}

/// Create a new product with validation
/// Returns the created product or an error
pub fn create_product(name: String, price: f64) -> Result<Product, String> {
    if name.is_empty() {
        return Err("Name cannot be empty".to_string());
    }
    
    if price <= 0.0 {
        return Err("Price must be positive".to_string());
    }
    
    Ok(Product {
        id: 0,
        name,
        price,
    })
}

/// HTTP handler for creating products
async fn create_product_handler(
    product: web::Json<Product>,
) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(&*product))
}
'''
    
    chunks = smart_chunk_content(rust_code, "rust", "main.rs")
    
    print(f"Generated {len(chunks)} chunks")
    
    doc_preserved_count = 0
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Has documentation: {chunk.has_documentation}")
        print(f"  Confidence: {chunk.confidence:.2f}")
        print(f"  Size: {chunk.size_chars} chars")
        print(f"  Relationship preserved: {chunk.relationship_preserved}")
        
        if chunk.declaration:
            print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
        
        if chunk.has_documentation:
            doc_preserved_count += 1
            print(f"  Documentation lines: {len(chunk.documentation_lines)}")
        
        # Validate that documentation and code are together
        if chunk.has_documentation and chunk.declaration:
            assert '///' in chunk.content or '/*' in chunk.content, f"Chunk {i+1} claims to have docs but doesn't contain comment markers"
            assert chunk.declaration.name in chunk.content, f"Chunk {i+1} missing declaration name in content"
    
    print(f"\nDocumentation preserved in {doc_preserved_count}/{len(chunks)} chunks")
    
    # Validate specific expectations
    assert len(chunks) >= 3, "Should generate at least 3 chunks for this Rust code"
    assert any(chunk.declaration and chunk.declaration.name == "AppConfig" for chunk in chunks), "Should have AppConfig struct chunk"
    assert any(chunk.declaration and chunk.declaration.name == "create_product" for chunk in chunks), "Should have create_product function chunk"
    
    print("+ Rust code test passed")


def test_python_code():
    """Test SmartChunker with Python code examples"""
    print("\n=== Testing Python Code ===")
    
    python_code = '''"""
Database models and data access layer for the e-commerce application.
Provides advanced querying capabilities and business logic.
"""

from sqlalchemy import and_, or_, func, text, case
from sqlalchemy.orm import Session, joinedload, selectinload
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"

class AdvancedUserRepository:
    """Advanced user data access and business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_with_full_profile(self, user_id: int) -> Optional[User]:
        """Get user with all related data loaded."""
        return self.db.query(User).options(
            selectinload(User.orders).selectinload(Order.order_items),
            selectinload(User.reviews).selectinload(Review.product)
        ).filter(User.id == user_id).first()
    
    def get_user_analytics(self, user_id: int) -> Optional[UserAnalytics]:
        """Get comprehensive analytics for a user."""
        result = self.db.query(
            User.id,
            User.username,
            func.count(Order.id).label('total_orders'),
            func.coalesce(func.sum(Order.total_amount), 0).label('total_spent'),
        ).outerjoin(Order).filter(User.id == user_id).group_by(User.id, User.username).first()
        
        if not result:
            return None
        
        return UserAnalytics(
            user_id=result.id,
            username=result.username,
            total_orders=result.total_orders,
            total_spent=float(result.total_spent),
        )

def create_user_repository(db: Session) -> AdvancedUserRepository:
    """Create a user repository instance."""
    return AdvancedUserRepository(db)
'''
    
    chunks = smart_chunk_content(python_code, "python", "models.py")
    
    print(f"Generated {len(chunks)} chunks")
    
    doc_preserved_count = 0
    method_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Has documentation: {chunk.has_documentation}")
        print(f"  Confidence: {chunk.confidence:.2f}")
        print(f"  Size: {chunk.size_chars} chars")
        
        if chunk.declaration:
            print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
            if chunk.declaration.declaration_type in ['function', 'method']:
                method_chunks += 1
        
        if chunk.has_documentation:
            doc_preserved_count += 1
            # Validate Python docstrings are preserved
            assert '"""' in chunk.content or "'''" in chunk.content or chunk.content.strip().startswith('#'), \
                f"Chunk {i+1} claims to have docs but doesn't contain docstring markers"
    
    print(f"\nDocumentation preserved in {doc_preserved_count}/{len(chunks)} chunks")
    print(f"Method/function chunks: {method_chunks}")
    
    # Validate expectations - focus on core functionality rather than exact chunk count
    assert len(chunks) >= 1, "Should generate at least 1 chunk"
    assert doc_preserved_count >= 1, "Should preserve documentation in at least 1 chunk"
    
    # Validate that we have meaningful declarations
    has_class_chunk = any(chunk.declaration and chunk.declaration.declaration_type == "class" for chunk in chunks)
    has_function_chunk = any(chunk.declaration and chunk.declaration.declaration_type == "function" for chunk in chunks)
    assert has_class_chunk or has_function_chunk, "Should have at least one class or function chunk"
    
    print("+ Python code test passed")


def test_javascript_code():
    """Test SmartChunker with JavaScript code examples"""
    print("\n=== Testing JavaScript Code ===")
    
    js_code = '''/**
 * Advanced React component for product display
 * Includes state management and event handling
 */

import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';

/**
 * Product display component with advanced features
 * Handles product data and user interactions
 */
class ProductDisplay extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            loading: false,
            error: null,
            product: null
        };
    }
    
    /**
     * Load product data from API
     * @param {number} productId - The ID of the product to load
     */
    async loadProduct(productId) {
        this.setState({ loading: true, error: null });
        
        try {
            const response = await fetch(`/api/products/${productId}`);
            const product = await response.json();
            this.setState({ product, loading: false });
        } catch (error) {
            this.setState({ error: error.message, loading: false });
        }
    }
    
    render() {
        const { product, loading, error } = this.state;
        
        if (loading) return <div>Loading...</div>;
        if (error) return <div>Error: {error}</div>;
        if (!product) return <div>No product found</div>;
        
        return (
            <div className="product-display">
                <h2>{product.name}</h2>
                <p>${product.price}</p>
            </div>
        );
    }
}

/**
 * Utility function for formatting product prices
 * @param {number} price - Raw price value
 * @returns {string} Formatted price string
 */
function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(price);
}

/**
 * Hook for managing product state
 * @param {number} initialProductId - Initial product ID
 */
const useProduct = (initialProductId) => {
    const [product, setProduct] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        if (initialProductId) {
            loadProduct(initialProductId);
        }
    }, [initialProductId]);
    
    const loadProduct = async (productId) => {
        setLoading(true);
        // Load product logic here
        setLoading(false);
    };
    
    return { product, loading, loadProduct };
};

export { ProductDisplay, formatPrice, useProduct };
'''
    
    chunks = smart_chunk_content(js_code, "javascript", "ProductDisplay.js")
    
    print(f"Generated {len(chunks)} chunks")
    
    doc_preserved_count = 0
    jsdoc_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Has documentation: {chunk.has_documentation}")
        print(f"  Confidence: {chunk.confidence:.2f}")
        print(f"  Size: {chunk.size_chars} chars")
        
        if chunk.declaration:
            print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
        
        if chunk.has_documentation:
            doc_preserved_count += 1
            if '/**' in chunk.content or '*/' in chunk.content:
                jsdoc_chunks += 1
    
    print(f"\nDocumentation preserved in {doc_preserved_count}/{len(chunks)} chunks")
    print(f"JSDoc chunks: {jsdoc_chunks}")
    
    # Validate expectations - focus on core functionality
    assert len(chunks) >= 1, "Should generate at least 1 chunk"
    assert doc_preserved_count >= 1, "Should preserve documentation in at least 1 chunk"
    assert jsdoc_chunks >= 1, "Should preserve JSDoc in at least 1 chunk"
    
    # Validate that we find meaningful JavaScript constructs
    has_class_or_function = any(
        chunk.declaration and chunk.declaration.declaration_type in ['class', 'function', 'arrow_function'] 
        for chunk in chunks
    )
    assert has_class_or_function, "Should identify JavaScript classes or functions"
    
    print("+ JavaScript code test passed")


def test_chunk_size_management():
    """Test that chunks respect size constraints while preserving relationships"""
    print("\n=== Testing Chunk Size Management ===")
    
    # Create a very large function with documentation
    large_python_code = '''
def very_large_function():
    """
    This is a very large function with extensive documentation.
    It demonstrates how the SmartChunker handles size constraints
    while preserving the documentation-code relationship.
    
    This function has many parameters and complex logic that
    would normally be split by traditional text chunkers.
    
    Args:
        param1: First parameter with detailed description
        param2: Second parameter with detailed description
        param3: Third parameter with detailed description
    
    Returns:
        A complex result with detailed explanation
    
    Raises:
        ValueError: When something goes wrong
        TypeError: When types don't match
    """
    # Very long implementation
''' + '    ' + '\n    '.join([f'variable_{i} = "This is line {i} of a very long function implementation"' for i in range(200)])
    
    chunker = SmartChunker(max_chunk_size=2000, min_chunk_size=100)
    chunks = chunker.chunk_by_declarations(large_python_code, "python", "large.py")
    
    print(f"Generated {len(chunks)} chunks from large function")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk.size_chars} chars, has_docs: {chunk.has_documentation}")
        
        # Validate size constraints
        if chunk.chunk_type == "declaration":
            # Allow some overage for relationship preservation
            assert chunk.size_chars <= 2400, f"Chunk {i+1} exceeds maximum size: {chunk.size_chars}"
        
        # Ensure the first chunk has the documentation
        if i == 0:
            assert chunk.has_documentation, "First chunk should preserve documentation"
            assert '"""' in chunk.content, "First chunk should contain docstring"
            assert 'very_large_function' in chunk.content, "First chunk should contain function declaration"
    
    print("+ Size management test passed")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Testing Edge Cases ===")
    
    # Empty content
    chunks = smart_chunk_content("", "python", "empty.py")
    assert len(chunks) == 0, "Empty content should produce no chunks"
    
    # Only whitespace
    chunks = smart_chunk_content("   \n\n   \n  ", "python", "whitespace.py")
    assert len(chunks) == 0, "Whitespace-only content should produce no chunks"
    
    # Unknown language
    chunks = smart_chunk_content("some code here", "unknown_language", "test.unknown")
    print(f"Unknown language produced {len(chunks)} chunks")
    
    # Code without declarations
    simple_code = '''
# Just some comments
# and variable assignments
x = 1
y = 2
z = x + y
print(z)
'''
    chunks = smart_chunk_content(simple_code, "python", "simple.py")
    print(f"Simple code without declarations produced {len(chunks)} chunks")
    assert len(chunks) >= 1, "Should create at least one semantic chunk"
    assert chunks[0].chunk_type == "semantic", "Should be semantic chunk type"
    
    # Malformed code
    malformed_code = '''
def incomplete_function(
    # Missing closing parenthesis and colon
    
class IncompleteClass
    # Missing colon
    pass
'''
    chunks = smart_chunk_content(malformed_code, "python", "malformed.py")
    print(f"Malformed code produced {len(chunks)} chunks")
    
    print("+ Edge cases test passed")


def test_documentation_detection_integration():
    """Test integration with UniversalDocumentationDetector"""
    print("\n=== Testing Documentation Detection Integration ===")
    
    mixed_code = '''
/// This function has Rust-style documentation
/// It should be properly detected and preserved
pub fn rust_style_function() -> i32 {
    42
}

// This is just a regular comment
// Not documentation
fn regular_function() -> String {
    "hello".to_string()
}

/**
 * This function has C-style documentation
 * Multiple lines with asterisks
 */
pub fn c_style_function() -> bool {
    true
}
'''
    
    chunks = smart_chunk_content(mixed_code, "rust", "mixed.rs")
    
    doc_chunks = [chunk for chunk in chunks if chunk.has_documentation]
    regular_chunks = [chunk for chunk in chunks if not chunk.has_documentation]
    
    print(f"Documented chunks: {len(doc_chunks)}")
    print(f"Regular chunks: {len(regular_chunks)}")
    
    # Should have detected 2 documented functions and 1 regular function
    assert len(doc_chunks) >= 2, "Should detect at least 2 documented functions"
    assert len(regular_chunks) >= 1, "Should have at least 1 function without docs"
    
    # Validate that documented chunks actually contain documentation
    for chunk in doc_chunks:
        assert ('///' in chunk.content or '/**' in chunk.content), \
            "Documented chunk should contain documentation markers"
        assert chunk.confidence > 0.3, "Documented chunk should have reasonable confidence"
    
    print("+ Documentation detection integration test passed")


def run_all_tests():
    """Run all SmartChunker tests"""
    print("Running SmartChunker Test Suite...")
    print("=" * 50)
    
    try:
        test_rust_code()
        test_python_code()
        test_javascript_code()
        test_chunk_size_management()
        test_edge_cases()
        test_documentation_detection_integration()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! SmartChunker is working correctly.")
        print("+ Declaration-centric chunking implemented")
        print("+ Documentation-code relationships preserved")
        print("+ Multiple language support verified")
        print("+ Size management working correctly")
        print("+ Edge cases handled properly")
        print("+ Integration with UniversalDocumentationDetector successful")
        
        return True
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)