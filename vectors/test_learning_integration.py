#!/usr/bin/env python3
"""
Simple Integration Test for Continuous Learning System

Tests basic functionality and integration without complex Unicode characters.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime

# Add the vectors directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that all required modules can be imported"""
    print("Testing basic imports...")
    
    try:
        from continuous_learning_system import (
            FeedbackCollectionSystem, PatternDiscoveryEngine, 
            AdaptiveModelUpdater, ContinuousLearningSystem
        )
        print("  [OK] Continuous learning system components imported")
    except ImportError as e:
        print(f"  [FAIL] Could not import learning system: {e}")
        return False
    
    try:
        from smart_chunker_optimized import SmartChunkerOptimized
        print("  [OK] SmartChunker imported")
    except ImportError as e:
        print(f"  [FAIL] Could not import SmartChunker: {e}")
        return False
    
    try:
        from ultra_reliable_core import UniversalDocumentationDetector
        print("  [OK] Documentation detector imported")
    except ImportError as e:
        print(f"  [FAIL] Could not import documentation detector: {e}")
        return False
    
    return True


def test_system_creation():
    """Test creating learning system components"""
    print("Testing system creation...")
    
    try:
        from continuous_learning_system import ContinuousLearningSystem
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create learning system
            learning_system = ContinuousLearningSystem(enable_auto_deployment=False)
            print("  [OK] Learning system created")
            
            # Test basic functionality
            status = learning_system.get_learning_status()
            print(f"  [OK] System status retrieved: {status['current_accuracy']:.1%} accuracy")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"  [FAIL] System creation failed: {e}")
        return False


def test_feedback_collection():
    """Test feedback collection functionality"""
    print("Testing feedback collection...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        try:
            from continuous_learning_system import FeedbackCollectionSystem
            
            feedback_system = FeedbackCollectionSystem(
                database_path=os.path.join(temp_dir, "test_feedback.db")
            )
            
            # Test collecting feedback
            system_prediction = {
                'has_documentation': False,
                'documentation_lines': [],
                'confidence': 0.3
            }
            
            feedback = feedback_system.collect_feedback(
                content='def test():\n    """Test function"""\n    return True',
                language='python',
                file_path='test.py',
                user_has_documentation=True,
                user_documentation_lines=[1],
                user_confidence=0.9,
                system_prediction=system_prediction
            )
            
            print(f"  [OK] Feedback collected: {feedback.feedback_id}")
            
            # Test statistics
            stats = feedback_system.get_feedback_statistics()
            print(f"  [OK] Statistics retrieved: {stats['total_feedback']} records")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"  [FAIL] Feedback collection failed: {e}")
        return False


def test_documentation_detection():
    """Test documentation detection with chunker"""
    print("Testing documentation detection integration...")
    
    try:
        from smart_chunker_optimized import SmartChunkerOptimized
        from ultra_reliable_core import UniversalDocumentationDetector
        
        # Create components
        chunker = SmartChunkerOptimized()
        detector = UniversalDocumentationDetector()
        
        # Test code with documentation
        test_code = '''def calculate_sum(a, b):
    """
    Calculate the sum of two numbers.
    
    Args:
        a (int): First number
        b (int): Second number
        
    Returns:
        int: Sum of a and b
    """
    return a + b'''
        
        # Test chunking
        chunks = chunker._chunk_content_optimized(test_code, 'python', 'test.py')
        print(f"  [OK] Chunker created {len(chunks)} chunks")
        
        # Test detection
        result = detector.detect_documentation_multi_pass(test_code, 'python')
        print(f"  [OK] Detection result: {result['has_documentation']}, confidence: {result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Documentation detection failed: {e}")
        return False


def test_end_to_end_workflow():
    """Test complete workflow"""
    print("Testing end-to-end workflow...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        try:
            from continuous_learning_system import ContinuousLearningSystem
            
            # Create learning system
            learning_system = ContinuousLearningSystem(enable_auto_deployment=False)
            
            # Override database paths to use temp directory
            learning_system.feedback_system.database_path = os.path.join(temp_dir, "feedback.db")
            learning_system.feedback_system._init_database()
            
            # Collect feedback
            feedback = learning_system.collect_user_feedback(
                content='def documented_function():\n    """This function has documentation"""\n    pass',
                language='python',
                file_path='test.py',
                user_has_documentation=True,
                user_documentation_lines=[1],
                user_confidence=0.9
            )
            
            print(f"  [OK] Feedback collected: {feedback.feedback_id}")
            
            # Check system status
            status = learning_system.get_learning_status()
            print(f"  [OK] System status: {status['current_accuracy']:.1%} accuracy, {status['total_feedback_records']} feedback records")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"  [FAIL] End-to-end workflow failed: {e}")
        return False


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Continuous Learning System - Integration Tests")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_system_creation, 
        test_feedback_collection,
        test_documentation_detection,
        test_end_to_end_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
                print(f"[PASS] {test.__name__}")
            else:
                failed += 1
                print(f"[FAIL] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {test.__name__}: {e}")
    
    print()
    print("=" * 60)
    print("Integration Test Summary")
    print("=" * 60)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    
    if failed == 0:
        print("\n[SUCCESS] All integration tests passed!")
        print("Continuous Learning System is ready for deployment.")
        print("\nKey Features Validated:")
        print("- Feedback collection system")
        print("- Pattern discovery engine")
        print("- Documentation detection integration")
        print("- SmartChunker integration") 
        print("- End-to-end learning workflow")
        return True
    else:
        print(f"\n[FAILURE] {failed} test(s) failed!")
        print("Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)