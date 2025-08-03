"""
File with intentional syntax errors to test error handling in indexing systems.
This tests how robust the code extraction is when encountering invalid syntax.
"""

import json
from typing import Dict, List

class BrokenSyntaxDemo:
    """Class with various syntax errors for testing."""
    
    def __init__(self):
        self.data = {}
        
    def function_with_missing_colon(self)  # Missing colon here
        return "This function has a syntax error"
    
    def function_with_unmatched_parentheses(self):
        result = some_function(
            param1="value1",
            param2="value2"
            # Missing closing parenthesis
        
        return result
    
    def function_with_wrong_indentation(self):
        if True:
        print("Wrong indentation")  # Should be indented
            return "error"
    
    def function_with_unclosed_string(self):
        message = "This string is never closed
        return message
    
    def function_with_invalid_dict_syntax(self):
        data = {
            "key1": "value1",
            "key2": "value2"
            "key3": "value3"  # Missing comma
        }
        return data
    
    def function_with_mixed_quotes(self):
        text = 'Single quote start but double quote end"
        return text
    
    def function_with_invalid_variable_name(self):
        123invalid_name = "Can't start with number"
        return 123invalid_name

# Missing class definition closing
class IncompleteClass:
    def __init__(self):
        self.value = 1
    
    def incomplete_method(self):
        # Method never returns anything and has incomplete logic
        if self.value > 0:
            print("Positive")
        else

# Incomplete if-else statement above

# Function with incorrect try-except syntax
def broken_exception_handling():
    try:
        result = 10 / 0
    except ZeroDivisionError as e
        print(f"Error: {e}")  # Missing colon after except
    finally:
        print("Cleanup")

# Function with incorrect lambda syntax
def lambda_errors():
    # Missing colon in lambda
    square = lambda x x * x
    
    # Invalid lambda syntax
    invalid_lambda = lambda: return 5
    
    return square, invalid_lambda

# Class with inheritance syntax error
class BadInheritance(object, str, int, float, dict):  # Too many base classes
    def __init__(self):
        super().__init__()

# Function with generator syntax error
def broken_generator():
    for i in range(10):
        yield i
        yield from  # Incomplete yield from statement

# Decorator syntax error
@property
@staticmethod  # Can't combine these decorators
def conflicting_decorators():
    return "This won't work"

# Invalid comprehension syntax
def broken_comprehensions():
    # Missing 'for' keyword
    list_comp = [x x in range(10)]
    
    # Invalid dict comprehension
    dict_comp = {k: v for k in range(5)}  # Missing v definition
    
    # Nested comprehension error
    nested = [[y for y in x] for x in if x > 0]  # Missing iterable after 'in'
    
    return list_comp, dict_comp, nested

# Import syntax errors
import json, os,  # Trailing comma in import
from typing import Dict List  # Missing comma between imports
import sys as  # Missing alias name

# Global variable with syntax error
GLOBAL_DICT = {
    "key1": "value1"
    "key2": "value2"  # Missing comma
}

# Function with f-string syntax error
def broken_fstring():
    name = "World"
    greeting = f"Hello {name  # Missing closing brace
    return greeting

# Async/await syntax errors
async def broken_async():
    await  # Incomplete await
    
    # Can't use await outside async function (this would be caught at runtime)
    def inner():
        result = await some_async_function()  # await in non-async function
        return result
    
    return inner()

# Context manager syntax error
def broken_context_manager():
    with open("file.txt") as f  # Missing colon
        content = f.read()
    return content

# Match statement syntax error (Python 3.10+)
def broken_match_statement(value):
    match value:
        case 1:
            return "one"
        case 2:
            return "two"
        case  # Incomplete case statement
            return "default"

# Class with property syntax error
class PropertyErrors:
    def __init__(self):
        self._value = 0
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def set_value(self, val):  # Wrong method name for setter
        self._value = val

# Function with annotation syntax error
def type_annotation_errors(param1: int, param2: str -> str:  # Mixed syntax
    return f"{param1}: {param2}"

# Multiple inheritance with syntax error
class MultipleInheritanceError(dict, list):  # Conflicting base classes
    def __init__(self):
        super().__init__()  # Which super() to call?

# Incomplete function definition
def incomplete_function(param1, param2,):  # Trailing comma with no parameter
    pass

# Invalid operator usage
def operator_errors():
    result1 = 5 ++ 3  # Invalid operator
    result2 = "string" - "str"  # Invalid operation
    result3 = [] * * 5  # Double asterisk without valid syntax
    
    return result1, result2, result3

# Incorrect use of walrus operator
def walrus_operator_errors():
    # Can't use walrus operator in function signature
    def inner(x := 5):  # Invalid syntax
        return x
    
    # Invalid walrus operator usage
    if x := y := 5:  # Chained assignment not allowed
        return x
    
    return inner()

# Missing parentheses in print (Python 3)
def python2_syntax():
    print "This is Python 2 syntax"  # Invalid in Python 3
    exec "print('executed')"  # exec without parentheses
    
# Incorrect exception syntax
def old_exception_syntax():
    try:
        1 / 0
    except ZeroDivisionError, e:  # Old Python 2 syntax
        print("Error:", e)

# This file intentionally contains many syntax errors to test
# how well the indexing system handles malformed code.
# Real indexing systems should be robust enough to extract
# what they can and skip problematic sections.