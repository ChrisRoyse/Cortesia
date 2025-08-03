"""
Unit tests for the calculator application.
"""

import unittest
import math
from calculator import Calculator
from advanced_operations import AdvancedCalculator
from utils import format_result, validate_input, is_number

class TestCalculator(unittest.TestCase):
    """Test cases for basic calculator operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = Calculator()
    
    def test_addition(self):
        """Test addition operation."""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0.1, 0.2), 0.30000000000000004)  # Floating point precision
    
    def test_subtraction(self):
        """Test subtraction operation."""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(0, 5), -5)
        self.assertEqual(self.calc.subtract(-2, -3), 1)
    
    def test_multiplication(self):
        """Test multiplication operation."""
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(-2, 5), -10)
        self.assertEqual(self.calc.multiply(0, 100), 0)
    
    def test_division(self):
        """Test division operation."""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertEqual(self.calc.divide(7, 3), 7/3)
        
        # Test division by zero
        with self.assertRaises(ValueError):
            self.calc.divide(5, 0)
    
    def test_history(self):
        """Test calculation history."""
        self.calc.add(2, 3)
        self.calc.multiply(4, 5)
        
        history = self.calc.get_history()
        self.assertEqual(len(history), 2)
        self.assertIn("2 + 3 = 5", history[0])
        self.assertIn("4 * 5 = 20", history[1])
        
        self.calc.clear_history()
        self.assertEqual(len(self.calc.get_history()), 0)

class TestAdvancedCalculator(unittest.TestCase):
    """Test cases for advanced calculator operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = AdvancedCalculator()
    
    def test_power(self):
        """Test power operation."""
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertAlmostEqual(self.calc.power(4, 0.5), 2)
    
    def test_square_root(self):
        """Test square root operation."""
        self.assertEqual(self.calc.square_root(4), 2)
        self.assertEqual(self.calc.square_root(9), 3)
        self.assertAlmostEqual(self.calc.square_root(2), math.sqrt(2))
        
        # Test negative number
        with self.assertRaises(ValueError):
            self.calc.square_root(-1)
    
    def test_factorial(self):
        """Test factorial operation."""
        self.assertEqual(self.calc.factorial(0), 1)
        self.assertEqual(self.calc.factorial(1), 1)
        self.assertEqual(self.calc.factorial(5), 120)
        
        # Test negative number
        with self.assertRaises(ValueError):
            self.calc.factorial(-1)
        
        # Test non-integer
        with self.assertRaises(ValueError):
            self.calc.factorial(3.5)
    
    def test_logarithm(self):
        """Test logarithm operations."""
        self.assertAlmostEqual(self.calc.logarithm(math.e), 1)
        self.assertAlmostEqual(self.calc.logarithm(1), 0)
        self.assertAlmostEqual(self.calc.logarithm(10, 10), 1)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            self.calc.logarithm(0)
        
        with self.assertRaises(ValueError):
            self.calc.logarithm(-1)
        
        with self.assertRaises(ValueError):
            self.calc.logarithm(10, 1)
    
    def test_trigonometric_functions(self):
        """Test trigonometric functions."""
        # Test well-known values
        self.assertAlmostEqual(self.calc.sin(0), 0, places=10)
        self.assertAlmostEqual(self.calc.sin(90), 1, places=10)
        self.assertAlmostEqual(self.calc.cos(0), 1, places=10)
        self.assertAlmostEqual(self.calc.cos(90), 0, places=10)
        self.assertAlmostEqual(self.calc.tan(45), 1, places=10)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_is_number(self):
        """Test number validation."""
        self.assertTrue(is_number("123"))
        self.assertTrue(is_number("123.45"))
        self.assertTrue(is_number("-123"))
        self.assertTrue(is_number("0"))
        self.assertFalse(is_number("abc"))
        self.assertFalse(is_number("12.34.56"))
    
    def test_format_result(self):
        """Test result formatting."""
        self.assertEqual(format_result(5.0), "Result: 5")
        self.assertEqual(format_result(3.14159), "Result: 3.14159")
        self.assertEqual(format_result(42), "Result: 42")
    
    def test_validate_input(self):
        """Test input validation."""
        self.assertEqual(validate_input("123"), 123.0)
        self.assertEqual(validate_input("123", "integer"), 123)
        self.assertEqual(validate_input("123.5", "positive"), 123.5)
        
        with self.assertRaises(ValueError):
            validate_input("abc")
        
        with self.assertRaises(ValueError):
            validate_input("-5", "positive")

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)