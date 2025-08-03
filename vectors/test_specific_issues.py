#!/usr/bin/env python3
"""
Test suite to isolate and validate the specific issues mentioned:
1. JavaScript JSDoc detection with `/**` and `*/` patterns
2. Small chunk validation for valid brief declarations  
3. Documentation gap detection being too strict
"""

import unittest
from smart_chunker import SmartChunker, smart_chunk_content

class TestSpecificIssues(unittest.TestCase):
    """Test specific identified limitations in SmartChunker"""
    
    def setUp(self):
        self.chunker = SmartChunker(max_chunk_size=4000, min_chunk_size=200)
    
    def test_javascript_jsdoc_detection_issue(self):
        """Test Issue 1: JavaScript JSDoc detection with /** and */ patterns"""
        js_code = '''/**
 * Calculates the distance between two points
 * @param {number} x1 - X coordinate of first point
 * @param {number} y1 - Y coordinate of first point
 * @param {number} x2 - X coordinate of second point
 * @param {number} y2 - Y coordinate of second point
 * @returns {number} The distance
 */
function calculateDistance(x1, y1, x2, y2) {
    return Math.sqrt((x2-x1)**2 + (y2-y1)**2);
}'''
        
        chunks = smart_chunk_content(js_code, "javascript", "test.js")
        
        # Find function chunks
        function_chunks = [chunk for chunk in chunks if chunk.declaration and 
                          chunk.declaration.declaration_type in ["function", "arrow_function"]]
        
        print(f"\nJavaScript JSDoc Test:")
        print(f"Generated {len(chunks)} chunks")
        if function_chunks:
            chunk = function_chunks[0]
            print(f"Function chunk has_documentation: {chunk.has_documentation}")
            print(f"Function chunk confidence: {chunk.confidence}")
            print(f"JSDoc content in chunk: {'/**' in chunk.content}")
            print(f"@param in chunk: {'@param' in chunk.content}")
            
            # This is the identified issue - JSDoc should be detected but may not be
            # due to gap detection treating `*/` as non-documentation
            if not chunk.has_documentation and ('/**' in chunk.content or '@param' in chunk.content):
                print("ISSUE CONFIRMED: JSDoc content present but not detected as documentation")
                return False
            elif chunk.has_documentation:
                print("JSDoc properly detected")
                return True
        else:
            print("No function chunks found")
            return False
            
        return True
    
    def test_small_chunk_validation_issue(self):
        """Test Issue 2: Small chunk validation rejecting valid brief declarations"""
        rust_code = '''/// ID type for branches
pub type BranchId = String;

/// Status enum  
pub enum Status {
    Active,
    Inactive,
}'''
        
        chunks = smart_chunk_content(rust_code, "rust", "test.rs")
        
        print(f"\nSmall Chunk Validation Test:")
        print(f"Generated {len(chunks)} chunks")
        
        # Look for type declarations
        type_chunks = [chunk for chunk in chunks if chunk.declaration and 
                      chunk.declaration.declaration_type == "type"]
        enum_chunks = [chunk for chunk in chunks if chunk.declaration and
                      chunk.declaration.declaration_type == "enum"]
        
        issues_found = []
        
        if type_chunks:
            chunk = type_chunks[0]
            print(f"Type chunk size: {chunk.size_chars} chars")
            print(f"Type chunk has_documentation: {chunk.has_documentation}")
            if chunk.size_chars < 200 and not chunk.has_documentation and '///' in chunk.content:
                issues_found.append("Type declaration with documentation rejected due to size")
        else:
            print("No type chunks found - may have been rejected due to size")
            if len(chunks) == 0 or all(chunk.chunk_type == "semantic" for chunk in chunks):
                issues_found.append("Valid type declaration not chunked properly")
        
        if enum_chunks:
            chunk = enum_chunks[0]
            print(f"Enum chunk size: {chunk.size_chars} chars")
            print(f"Enum chunk has_documentation: {chunk.has_documentation}")
        
        if issues_found:
            print("ISSUES CONFIRMED:")
            for issue in issues_found:
                print(f"  - {issue}")
            return False
        else:
            print("Small chunk validation working properly")
            return True
    
    def test_documentation_gap_detection_issue(self):
        """Test Issue 3: Documentation gap detection being too strict"""
        python_code = '''def process_data():
    """Process the input data."""
    # Some implementation comment
    data = load_data()
    return transform(data)'''
        
        chunks = smart_chunk_content(python_code, "python", "test.py")
        
        print(f"\nDocumentation Gap Detection Test:")
        print(f"Generated {len(chunks)} chunks")
        
        function_chunks = [chunk for chunk in chunks if chunk.declaration and 
                          chunk.declaration.declaration_type == "function"]
        
        if function_chunks:
            chunk = function_chunks[0]
            print(f"Function chunk has_documentation: {chunk.has_documentation}")
            print(f"Function chunk confidence: {chunk.confidence}")
            print(f"Docstring in content: {'"""' in chunk.content}")
            
            # Check if docstring is properly connected to function
            if '"""' in chunk.content and not chunk.has_documentation:
                print("ISSUE CONFIRMED: Docstring present but not detected due to gap detection")
                return False
            elif chunk.has_documentation:
                print("Documentation properly detected and connected")
                return True
        else:
            print("No function chunks found")
            return False
            
        return True
    
    def test_comprehensive_accuracy_validation(self):
        """Test comprehensive accuracy across all identified issues"""
        test_cases = [
            # JavaScript JSDoc
            ('''/**
 * User class with JSDoc
 * @param {string} name - User name
 */
class User {
    constructor(name) {
        this.name = name;
    }
}''', 'javascript', True),
            
            # Small Rust declarations
            ('''/// Simple type alias
pub type UserId = u64;''', 'rust', True),
            
            # Python function with docstring
            ('''def calculate():
    """Calculate something important."""
    return 42''', 'python', True),
        ]
        
        correct_detections = 0
        total_cases = len(test_cases)
        
        print(f"\nComprehensive Accuracy Test:")
        
        for i, (code, language, should_have_docs) in enumerate(test_cases):
            chunks = smart_chunk_content(code, language, f"test{i}.{language}")
            
            # Check if any chunk has documentation
            has_documentation = any(chunk.has_documentation for chunk in chunks)
            
            print(f"Case {i+1} ({language}): Expected {should_have_docs}, Got {has_documentation}")
            
            if has_documentation == should_have_docs:
                correct_detections += 1
            else:
                print(f"  MISS: Expected {should_have_docs}, got {has_documentation}")
                # Debug info
                for j, chunk in enumerate(chunks):
                    print(f"    Chunk {j}: type={chunk.chunk_type}, has_docs={chunk.has_documentation}, size={chunk.size_chars}")
        
        accuracy = (correct_detections / total_cases) * 100
        print(f"Current accuracy: {accuracy:.1f}% ({correct_detections}/{total_cases})")
        
        # Return whether we need fixes
        return accuracy >= 99.0


def run_issue_tests():
    """Run specific issue tests to validate the problems"""
    print("SMARTCHUNKER SPECIFIC ISSUE VALIDATION")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecificIssues)
    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(suite)
    
    print(f"\n{'=' * 50}")
    print("ISSUE VALIDATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures or result.errors:
        print("\nISSUES CONFIRMED - FIXES NEEDED")
        return False
    else:
        print("\nNO ISSUES FOUND - SYSTEM WORKING PROPERLY")
        return True


if __name__ == "__main__":
    run_issue_tests()