"""
Advanced mathematical operations for the calculator.
"""

import math

class AdvancedCalculator:
    """Calculator with advanced mathematical operations."""
    
    def power(self, base, exponent):
        """Calculate base raised to the power of exponent."""
        return base ** exponent
    
    def square_root(self, number):
        """Calculate square root of a number."""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(number)
    
    def factorial(self, n):
        """Calculate factorial of a number."""
        if n < 0:
            raise ValueError("Cannot calculate factorial of negative number")
        if not isinstance(n, int):
            raise ValueError("Factorial requires an integer")
        
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    def logarithm(self, number, base=math.e):
        """Calculate logarithm of a number with given base."""
        if number <= 0:
            raise ValueError("Logarithm requires positive number")
        if base <= 0 or base == 1:
            raise ValueError("Invalid logarithm base")
        
        if base == math.e:
            return math.log(number)
        else:
            return math.log(number) / math.log(base)
    
    def sin(self, angle_degrees):
        """Calculate sine of angle in degrees."""
        return math.sin(math.radians(angle_degrees))
    
    def cos(self, angle_degrees):
        """Calculate cosine of angle in degrees."""
        return math.cos(math.radians(angle_degrees))
    
    def tan(self, angle_degrees):
        """Calculate tangent of angle in degrees."""
        return math.tan(math.radians(angle_degrees))